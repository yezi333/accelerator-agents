"""Offline adapter tests; optional integration uses the real CLI and local SSE."""

import json
import os
import re
import shlex
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import tomllib

from codex_cli.__main__ import main, run_cli
from codex_cli.prepare import ROOT, prepare
from codex_cli.request_input import main as request_input


def test_render_preserves_roles_and_settings(tmp_path, monkeypatch):
  monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example/v1")
  monkeypatch.setenv("MAXKERNEL_API_TIMEOUT", "360")
  monkeypatch.setenv("MAXKERNEL_API_MAX_RETRIES", "2")
  home, metadata = prepare(
    tmp_path, Path(sys.executable), sandbox="workspace-write"
  )
  assert len(metadata["agents"]) == 22
  config = tomllib.loads((home / "config.toml").read_text())
  assert (
    config["model_providers"]["maxkernel"]["stream_idle_timeout_ms"] == 360000
  )
  assert config["model_providers"]["maxkernel"]["request_max_retries"] == 2
  assert config["shell_environment_policy"]["exclude"] == [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
  ]
  assert config["agents"]["max_concurrent_threads_per_session"] == 4
  for name in metadata["agents"]:
    role = tomllib.loads((home / "agents" / (name + ".toml")).read_text())
    original = (
      (home / "agents" / (name + ".md")).read_text().split("---", 2)[2].lstrip()
    )
    assert original in role["developer_instructions"]
    assert 'fork_turns="none"' in role["developer_instructions"]
    assert "{{" not in role["developer_instructions"]
  worker = tomllib.loads((home / "agents/maxkernel-worker.toml").read_text())[
    "developer_instructions"
  ]
  for marker in [
    "up to 6 attempts",
    "Phase 0.15",
    "Phase 4: Autotuning Loop",
    "Phase 5: Profiling Loop",
  ]:
    assert marker in worker
  assert (home / "skills/maxkernel/guides/general_autotune.md").is_file()


def test_pending_question_durable_without_self_approval(tmp_path):
  host = tmp_path / "codex_host"
  host.mkdir()
  (host / "request.json").write_text("{}")
  state = tmp_path / "state.json"
  state.write_text('{"approved": null}')
  assert (
    request_input(
      [
        "--run-dir",
        str(tmp_path),
        "--question",
        "Approve extraction?",
        "--artifact",
        "source.py",
      ]
    )
    == 0
  )
  pending = (host / "pending_question.json").read_text()
  request_input(["--run-dir", str(tmp_path), "--question", "Other question"])
  assert (host / "pending_question.json").read_text() == pending
  assert "Approve extraction?" in pending
  assert state.read_text() == '{"approved": null}'


def test_resume_preserves_pending_and_state(tmp_path, monkeypatch):
  import codex_cli.__main__ as cli

  monkeypatch.setattr(cli, "ROOT", tmp_path)
  monkeypatch.setattr(cli, "check_cli", lambda *args: None)
  run = tmp_path / "workspace/run"
  host = run / "codex_host"
  host.mkdir(parents=True)
  (host / "request.json").write_text('{"task": "Original workload"}')
  (host / "pending_question.json").write_text('{"question": "Approve?"}')
  (run / "state.json").write_text('{"iteration": 1}')
  args = ["--codex", sys.executable, "--run-dir", str(run), "--resume"]
  assert main(args) == 1
  reply = tmp_path / "reply.txt"
  reply.write_text("Do not approve; revise it.")
  assert main(args + ["--reply-file", str(reply), "--dry-run"]) == 1
  assert (host / "pending_question.json").exists()
  monkeypatch.setenv("OPENAI_API_KEY", "test-only")
  captured = []
  monkeypatch.setattr(
    cli, "run_cli", lambda exe, home, host, task: captured.append(task) or 0
  )
  assert main(args + ["--reply-file", str(reply)]) == 0
  assert "Do not approve; revise it." in captured[0]
  assert (run / "state.json").read_text() == '{"iteration": 1}'
  assert not (host / "pending_question.json").exists()


def tool(name, namespace, **arguments):
  return {
    "type": "function_call",
    "id": "fc_mock",
    "call_id": "call_mock",
    "name": name,
    "namespace": namespace,
    "arguments": json.dumps(arguments),
  }


def message(text):
  return {
    "type": "message",
    "id": "msg_mock",
    "role": "assistant",
    "status": "completed",
    "phase": "final_answer",
    "content": [{"type": "output_text", "text": text, "annotations": []}],
  }


@pytest.mark.skipif(
  not os.environ.get("MAXKERNEL_TEST_CODEX"),
  reason="Set MAXKERNEL_TEST_CODEX; integration uses localhost only",
)
@pytest.mark.parametrize("scenario", ["hierarchy", "pause", "api_error"])
def test_real_cli_nested_roles_and_slot_reuse(tmp_path, monkeypatch, scenario):
  counts, requests, errors = {}, [], []
  marker = tmp_path / "native-marker.txt"

  class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
      pass

    def do_POST(self):
      try:
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if scenario == "api_error":
          self.send_error(503, "Simulated API outage")
          return
        roles = re.findall(r"registered role: ([\w-]+)", json.dumps(data))
        role = roles[-1] if roles else "maxkernel"
        requests.append((role, data))
        step = counts.get(role, 0)
        counts[role] = step + 1
        if role == "maxkernel":
          if counts.get("maxkernel-worker", 0) >= 18:
            item = message("Native pipeline mock finished")
          elif step % 2 == 0:
            item = tool(
              "spawn_agent",
              "collaboration",
              agent_type="maxkernel-worker",
              task_name=f"worker_{step}",
              fork_turns="none",
              message="CLI integration: run one iteration.",
            )
          else:
            item = tool("wait_agent", "collaboration", timeout_ms=10000)
        elif role == "maxkernel-worker":
          if step % 3 == 0:
            item = tool(
              "spawn_agent",
              "collaboration",
              agent_type="maxkernel-implement-kernel",
              task_name=f"implement_{step}",
              fork_turns="none",
              message="CLI integration: write marker and report.",
            )
          elif step % 3 == 1:
            item = tool("wait_agent", "collaboration", timeout_ms=10000)
          else:
            item = message("Worker completed")
        elif role == "maxkernel-implement-kernel":
          if step % 2 == 0:
            command = shlex.join(
              [
                sys.executable,
                "-c",
                f"import os; from pathlib import Path; assert 'OPENAI_API_KEY' not in os.environ; Path({str(marker)!r}).write_text('native-cli-ok')",
              ]
            )
            if scenario == "pause":
              command = shlex.join(
                [
                  sys.executable,
                  str(ROOT / "codex_cli/request_input.py"),
                  "--run-dir",
                  str(tmp_path),
                  "--question",
                  "Review the extracted source before proceeding",
                ]
              )
            item = {
              "type": "custom_tool_call",
              "id": "ct_mock",
              "call_id": "call_mock",
              "name": "exec",
              "namespace": "functions",
              "input": "text(await tools.exec_command("
              + json.dumps({"cmd": command})
              + "));",
            }
          else:
            item = message("Implementation completed")
        else:
          raise AssertionError("Unexpected role " + role)
        response = {
          "id": "resp_mock",
          "object": "response",
          "created_at": 1,
          "model": "gpt-5.6-sol",
          "status": "completed",
          "output": [item],
          "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
          },
        }
        events = [
          {
            "type": "response.created",
            "response": {**response, "status": "in_progress", "output": []},
          },
          {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": item,
          },
          {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": item,
          },
          {"type": "response.completed", "response": response},
        ]
        encoded = "".join(
          "data: " + json.dumps(e) + "\n\n" for e in events
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
      except Exception as exc:
        errors.append(str(exc))
        self.send_error(500)

  server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
  threading.Thread(target=server.serve_forever, daemon=True).start()
  monkeypatch.setenv(
    "OPENAI_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1"
  )
  monkeypatch.setenv("OPENAI_API_KEY", "offline-test-only")
  monkeypatch.setenv("MAXKERNEL_API_TIMEOUT", "5")
  monkeypatch.setenv("MAXKERNEL_API_MAX_RETRIES", "0")
  home, _ = prepare(tmp_path, Path(sys.executable), sandbox="workspace-write")
  # The marker lives in the already-writable system temporary directory.
  (tmp_path / "codex_host/request.json").write_text("{}")
  if scenario != "hierarchy":
    (tmp_path / "state.json").write_text('{"iteration": 0}')
  binary = shutil.which(os.environ["MAXKERNEL_TEST_CODEX"])
  timer = threading.Timer(
    55,
    lambda: (tmp_path / "codex_host/pending_question.json").write_text(
      '{"question": "Test watchdog"}'
    ),
  )
  timer.start()
  try:
    code = run_cli(
      binary, home, tmp_path / "codex_host", "Run native CLI integration test."
    )
  finally:
    timer.cancel()
    server.shutdown()
  (tmp_path / "requests.json").write_text(json.dumps(requests))
  assert not errors, errors
  if scenario == "pause":
    assert code == 2
    assert (
      "Review the extracted source"
      in (tmp_path / "codex_host/pending_question.json").read_text()
    )
    assert (tmp_path / "state.json").read_text() == '{"iteration": 0}'
    assert not marker.exists()
    return
  if scenario == "api_error":
    assert code == 1
    assert (tmp_path / "state.json").read_text() == '{"iteration": 0}'
    assert (
      json.loads((tmp_path / "codex_host/status.json").read_text())["status"]
      == "failed"
    )
    return
  assert code == 1  # No fake TPU state is accepted as successful completion.
  assert marker.read_text() == "native-cli-ok"
  assert counts["maxkernel-worker"] == 18
  assert counts["maxkernel-implement-kernel"] == 12
  assert not (tmp_path / "codex_host/pending_question.json").exists()
  for role, data in requests:
    assert data["model"] == "gpt-5.6-sol"
    assert data["stream"] is True
    if role != "maxkernel":
      assert "Run native CLI integration test." not in json.dumps(data["input"])
