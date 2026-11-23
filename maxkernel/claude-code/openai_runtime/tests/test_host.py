"""Offline checks with real SDK serialization and real filesystem/shell tools."""

import hashlib
import json
import re
import sys
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from openai_runtime.__main__ import (
  execute,
  inspect_result,
  main,
  parse_args,
  validate_job,
)
from openai_runtime.prompts import ROOT, Prompts, body
from openai_runtime.runtime import ModelConfig, ResponsesHost
from openai_runtime.tools import NeedsInput, ToolHost


def message(text):
  return {
    "type": "message",
    "id": "msg_test",
    "role": "assistant",
    "status": "completed",
    "phase": "final_answer",
    "content": [{"type": "output_text", "text": text, "annotations": []}],
  }


def call(name, **arguments):
  return {
    "type": "function_call",
    "id": "fc_test",
    "call_id": "call_test",
    "name": name,
    "arguments": json.dumps(arguments),
    "status": "completed",
  }


def response(items, status="completed", tokens=100):
  return {
    "id": "resp_test",
    "object": "response",
    "created_at": 1,
    "model": "test-model",
    "status": status,
    "output": items,
    "usage": {
      "input_tokens": tokens,
      "output_tokens": 10,
      "total_tokens": tokens + 10,
    },
  }


@pytest.fixture
def prompts(tmp_path):
  return Prompts(tmp_path, Path(sys.executable))


def host(prompts, handler, **settings):
  client = AsyncOpenAI(
    api_key="test-only",
    max_retries=0,
    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
  )
  tools = ToolHost(prompts.run_dir, prompts)
  return ResponsesHost(
    ModelConfig(model="test-model", **settings), prompts, tools, client
  )


def test_all_original_prompts_and_guides_are_rendered(prompts):
  originals = list((ROOT / "agents").glob("maxkernel-*.md"))
  assert len(prompts.agents) == len(originals) + 1
  for source in originals:
    rendered = (prompts.directory / "agents" / source.name).read_text()
    assert body(rendered) in prompts.instructions(source.stem)
    assert "{{MAXKERNEL_ROOT}}" not in rendered
    assert (
      prompts.manifest[str(source.relative_to(ROOT))]
      == hashlib.sha256(source.read_bytes()).hexdigest()
    )
  assert (
    prompts.directory / "skills/maxkernel/guides/general_autotune.md"
  ).is_file()
  assert "up to 6 attempts" in prompts.instructions("maxkernel-worker")
  assert "Phase 5: Profiling Loop" in prompts.instructions("maxkernel-worker")
  assert "Phase 0.15" in prompts.instructions("maxkernel-worker")
  assert "Agent" not in prompts.allowed["maxkernel-implement-kernel"]


@pytest.mark.asyncio
async def test_sdk_timeout_retries_preserve_native_state(prompts, monkeypatch):
  import openai
  from openai import APITimeoutError

  attempts = []
  state = prompts.run_dir / "state.json"
  state.write_text('{"iteration": 1}')
  monkeypatch.setenv("OPENAI_API_KEY", "test-only")

  def handler(request):
    attempts.append(request)
    assert request.extensions["timeout"]["read"] == 360
    raise httpx.ReadTimeout("simulated slow model", request=request)

  def client_factory(**kwargs):
    return AsyncOpenAI(
      **kwargs,
      http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

  monkeypatch.setattr(openai, "AsyncOpenAI", client_factory)
  args = parse_args(["--run-dir", str(prompts.run_dir)])
  with pytest.raises(APITimeoutError):
    await execute(
      args, ModelConfig(model="test-model"), prompts, "Continue", []
    )
  assert len(attempts) == 3
  assert state.read_text() == '{"iteration": 1}'
  status = json.loads((prompts.run_dir / "openai_host/status.json").read_text())
  assert status["status"] == "failed"
  assert status["error_type"] == "APITimeoutError"


@pytest.mark.asyncio
async def test_native_hierarchy_all_optimization_phases_and_tool_replay(
  prompts,
):
  phases = [
    "plan-kernel",
    "implement-kernel",
    "compilation-summary",
    "summarize-test-results",
    "autotune-planner",
    "autotune-summary",
    "generate-profile-script",
    "summarize-profile",
  ]
  counts, seen = {}, []
  private = "parent-only-message-not-in-child"
  opaque = {
    "type": "reasoning",
    "id": "rs_test",
    "summary": [],
    "encrypted_content": "opaque",
  }

  def handler(request):
    data = json.loads(request.content)
    role = re.search(r"Registered role: ([\w-]+)", data["instructions"])[1]
    step = counts.get(role, 0)
    counts[role] = step + 1
    assert data["store"] is False
    assert data["include"] == ["reasoning.encrypted_content"]
    assert data["parallel_tool_calls"] is False
    if role == "maxkernel":
      if step == 0:
        items = [
          opaque,
          call(
            "Agent",
            subagent_type="maxkernel-worker",
            prompt="Run one iteration",
          ),
        ]
      else:
        assert opaque in data["input"]
        assert data["input"][-1]["type"] == "function_call_output"
        items = [message("Orchestrator finished")]
    elif role == "maxkernel-worker":
      assert private not in json.dumps(data["input"])
      items = (
        [
          call(
            "Agent",
            subagent_type="maxkernel-" + phases[step],
            prompt="Execute the assigned phase",
          )
        ]
        if step < len(phases)
        else [message("Worker finished")]
      )
    else:
      assert private not in json.dumps(data["input"])
      if step == 0:
        seen.append(role)
        items = [
          call(
            "Write",
            file_path=str(prompts.run_dir / (role + ".md")),
            content="offline phase artifact",
          )
        ]
      else:
        assert "written" in data["input"][-1]["output"]
        items = [message("Phase artifact written")]
    return httpx.Response(200, json=response(items))

  runtime = host(prompts, handler)
  try:
    assert (
      await runtime.run_agent("maxkernel", private) == "Orchestrator finished"
    )
  finally:
    await runtime.client.close()
  assert seen == ["maxkernel-" + phase for phase in phases]
  assert all((prompts.run_dir / (name + ".md")).is_file() for name in seen)
  events = [
    json.loads(line) for line in runtime.events.read_text().splitlines()
  ]
  assert any(
    e["event"] == "llm_end" and e["usage"]["input_tokens"] == 100
    for e in events
  )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "failed"])
async def test_incomplete_response_cannot_execute_partial_tools(
  prompts, status
):
  target = prompts.run_dir / "must-not-exist"
  runtime = host(
    prompts,
    lambda _: httpx.Response(
      200,
      json=response(
        [call("Write", file_path=str(target), content="partial")], status
      ),
    ),
  )
  try:
    with pytest.raises(RuntimeError, match="no tools executed"):
      await runtime.run_agent("maxkernel", "test")
  finally:
    await runtime.client.close()
  assert not target.exists()


@pytest.mark.asyncio
async def test_nested_question_pauses_before_further_mutations(prompts):
  def handler(request):
    data = json.loads(request.content)
    if "Registered role: maxkernel\n" in data["instructions"]:
      items = [
        call("Agent", subagent_type="maxkernel-worker", prompt="Read state")
      ]
    else:
      items = [
        call(
          "AskUserQuestion",
          questions=[{"question": "Approve extracted source?"}],
        ),
        call(
          "Write", file_path=str(prompts.run_dir / "approved"), content="yes"
        ),
      ]
    return httpx.Response(200, json=response(items))

  runtime = host(prompts, handler)
  try:
    with pytest.raises(NeedsInput):
      await runtime.run_agent("maxkernel", "test")
  finally:
    await runtime.client.close()
  assert not (prompts.run_dir / "approved").exists()
  pending = json.loads(
    (runtime.directory / "pending_question.json").read_text()
  )
  assert pending["agent"] == "maxkernel-worker"


@pytest.mark.asyncio
async def test_compaction_retains_matched_tool_turns_and_initial_constraints(
  prompts,
):
  requests = []

  def handler(request):
    data = json.loads(request.content)
    requests.append(data)
    return httpx.Response(
      200, json=response([message("Paths, failures and next step.")])
    )

  runtime = host(prompts, handler)
  seed = [{"role": "user", "content": "Never change shape 512x512"}]
  turns = [
    [
      call("Read", file_path="state.json"),
      {"type": "function_call_output", "call_id": "call_test", "output": "{}"},
    ]
    for _ in range(7)
  ]
  try:
    new_seed, new_turns = await runtime.compact("maxkernel-worker", seed, turns)
  finally:
    await runtime.client.close()
  assert new_seed[0] == seed[0]
  assert new_turns == turns[-5:]
  old_calls = [
    i for i in requests[0]["input"] if i.get("type") == "function_call"
  ]
  old_results = [
    i for i in requests[0]["input"] if i.get("type") == "function_call_output"
  ]
  assert len(old_calls) == len(old_results) == 2


@pytest.mark.asyncio
async def test_specialist_cannot_delegate(prompts):
  n = 0

  def handler(request):
    nonlocal n
    n += 1
    if n == 1:
      return httpx.Response(
        200,
        json=response(
          [call("Agent", subagent_type="maxkernel-worker", prompt="bad")]
        ),
      )
    assert "not allowed" in json.loads(request.content)["input"][-1]["output"]
    return httpx.Response(200, json=response([message("No delegation")]))

  runtime = host(prompts, handler)
  try:
    await runtime.run_agent("maxkernel-implement-kernel", "test")
  finally:
    await runtime.client.close()
  assert runtime.counter == 1


@pytest.mark.asyncio
async def test_shell_timeout_redaction_file_edits_and_read_paging(
  prompts, monkeypatch
):
  monkeypatch.setenv("OPENAI_API_KEY", "never-log-this-value")
  tools = ToolHost(prompts.run_dir, prompts)
  result = await tools.execute("Bash", {"command": "sleep 30", "timeout": 20})
  assert result["timed_out"] and result["exit_code"] != 0
  result = await tools.execute(
    "Bash", {"command": 'test -z "$OPENAI_API_KEY" && printf ready'}
  )
  assert result["exit_code"] == 0 and result["output"] == "ready"
  await tools.execute(
    "Write", {"file_path": "state.txt", "content": "one\none\nthree\n"}
  )
  with pytest.raises(ValueError, match="unique"):
    await tools.execute(
      "Edit",
      {"file_path": "state.txt", "old_string": "one", "new_string": "two"},
    )
  await tools.execute(
    "Edit",
    {
      "file_path": "state.txt",
      "old_string": "one",
      "new_string": "two",
      "replace_all": True,
    },
  )
  result = await tools.execute("Read", {"file_path": "state.txt", "limit": 1})
  assert result["next_offset"] == 2 and "two" in result["content"]
  with pytest.raises(ValueError, match="outside"):
    await tools.execute(
      "Write",
      {"file_path": str(prompts.run_dir.parent / "escape"), "content": "no"},
    )
  assert "never-log-this-value" not in tools.redact("never-log-this-value")


@pytest.mark.asyncio
async def test_original_guard_is_invoked(prompts):
  tools = ToolHost(prompts.run_dir, prompts, guard=True)
  await tools.check_guard("Read", {"file_path": str(ROOT / "README.md")})
  with pytest.raises(ValueError, match="workspace guard"):
    await tools.check_guard(
      "Read",
      {"file_path": "/opt/lib/python/site-packages/jax/_src/pallas/kernel.py"},
    )


def test_result_uses_correct_candidates_and_requires_full_pipeline_files(
  tmp_path,
):
  rows = []
  for i, (correct, timing) in enumerate([(True, 2.0), (False, 0.01)], 1):
    directory = tmp_path / f"iter{i}"
    directory.mkdir()
    row = {
      "iteration": i,
      "compile_ok": True,
      "test_ok": correct,
      "optimized_time_ms": timing,
      "speedup": 3 / timing,
    }
    for key, filename in [
      ("optimized_kernel", "optimized.py"),
      ("kernel_plan", "kernel_plan.md"),
      ("autotune_summary", "autotune_summary.md"),
      ("profile_summary", "profile_summary.md"),
    ]:
      path = directory / filename
      path.write_text(f"artifact {i}")
      row[key] = str(path)
    rows.append(row)
  (tmp_path / "optimized.py").write_text("artifact 1")
  state = {
    "iteration": 2,
    "max_iterations": 2,
    "history": rows,
    "best_code_path": rows[0]["optimized_kernel"],
  }
  (tmp_path / "state.json").write_text(json.dumps(state))
  assert (
    inspect_result(tmp_path)["best_code_path"] == rows[0]["optimized_kernel"]
  )
  Path(rows[0]["profile_summary"]).unlink()
  with pytest.raises(ValueError, match="profile_summary"):
    inspect_result(tmp_path)


def test_defaults_and_real_job_validation(monkeypatch):
  for key in [
    "MAXKERNEL_API_TIMEOUT",
    "MAXKERNEL_MODEL",
    "MODEL_NAME",
    "OPENAI_BASE_URL",
  ]:
    monkeypatch.delenv(key, raising=False)
  monkeypatch.setenv("MODEL_NAME", "fallback-model")
  config = ModelConfig.from_env()
  assert (
    config.timeout == 360
    and config.max_retries == 2
    and config.model == "fallback-model"
  )
  monkeypatch.setenv("MAXKERNEL_MODEL", "selected-model")
  monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1/responses")
  assert ModelConfig.from_env().base_url == "https://proxy.example/v1"
  assert ModelConfig.from_env().model == "selected-model"
  _, report = validate_job(
    ROOT / "openai_runtime/examples/matmul_bias_relu.job.json"
  )
  assert (
    report["plan"]["input_type"] == "jax" and report["plan"]["internal_harness"]
  )


def test_missing_job_source_fails_before_execution(tmp_path):
  job = json.loads(
    (ROOT / "openai_runtime/examples/matmul_bias_relu.job.json").read_text()
  )
  job["input"]["path"] = "missing.py"
  path = tmp_path / "job.json"
  path.write_text(json.dumps(job))
  with pytest.raises(ValueError, match="Missing job paths"):
    validate_job(path)


def test_dry_run_and_resume_do_not_replace_native_state(tmp_path, monkeypatch):
  import openai_runtime.__main__ as cli

  monkeypatch.setattr(cli, "ROOT", tmp_path)
  source = tmp_path / "source.py"
  source.write_text("def computation(x): return x\n")
  run = tmp_path / "workspace/run"
  args = ["--run-dir", str(run), "--model", "test-model", "--dry-run"]
  assert main(args + ["--source", str(source)]) == 0
  state = run / "state.json"
  state.write_text('{"iteration": 1}')
  assert main(args + ["--resume"]) == 0
  assert state.read_text() == '{"iteration": 1}'
  assert len(list((run / "openai_host/prompts").iterdir())) == 2
  pending = run / "openai_host/pending_question.json"
  pending.write_text('{"questions": [{"question": "Approve source?"}]}')
  assert main(args + ["--resume"]) == 1
  reply = tmp_path / "reply.txt"
  reply.write_text("Please revise the extraction; I do not approve it yet.")
  assert main(args + ["--resume", "--reply-file", str(reply)]) == 0
  assert "do not approve" in (run / "openai_host/request.json").read_text()
  assert state.read_text() == '{"iteration": 1}'
