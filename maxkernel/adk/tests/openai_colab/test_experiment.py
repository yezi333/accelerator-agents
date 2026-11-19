import json
import os
import subprocess
import sys
import time

import pytest

from experiments.colab.core import (
  DEFAULT_LIMITS,
  Budget,
  BudgetExhausted,
  apply_edits,
  meaningful_gain,
  run_process,
  tuning_candidates,
)
from experiments.colab.run import profile_program


def test_budget_is_charged_before_execution_and_survives_resume(tmp_path):
  limits = DEFAULT_LIMITS | {"tpu_jobs": 1}
  budget = Budget(tmp_path, limits)
  budget.reserve("tpu_jobs", profile=True)
  resumed = Budget(tmp_path, limits)
  assert resumed.state["profile_jobs"] == 1
  with pytest.raises(BudgetExhausted):
    resumed.reserve("tpu_jobs")


def test_wall_deadline_does_not_reset_on_resume(tmp_path):
  budget = Budget(tmp_path, DEFAULT_LIMITS)
  budget.state["started_at"] -= 3601
  budget.save()
  with pytest.raises(BudgetExhausted):
    Budget(tmp_path, DEFAULT_LIMITS).reserve("model_calls")


def test_patch_rejects_ambiguous_or_invalid_edits():
  with pytest.raises(ValueError, match="exactly once"):
    apply_edits("x = 1\nx = 1\n", [{"old": "x = 1", "new": "x = 2"}])
  with pytest.raises(SyntaxError):
    apply_edits("x = 1\n", [{"old": "x = 1", "new": "def ("}])
  assert apply_edits("x = 1\n", [{"old": "x = 1", "new": "x = 2"}]) == "x = 2\n"


def test_tuning_does_not_change_workload_and_is_bounded():
  base = {"block_b": 8, "block_kv": 1024}
  candidates = tuning_candidates("gqa_decode", base, 3)
  assert len(candidates) == 3
  assert all(set(c) == set(base) for c in candidates)
  assert base == {"block_b": 8, "block_kv": 1024}


def test_host_noise_is_not_promoted_as_a_gain():
  before = {"median_ms": 1, "mad_ms": 0.04}
  assert not meaningful_gain(before, {"median_ms": 0.95, "mad_ms": 0.02})
  assert meaningful_gain(before, {"median_ms": 0.8, "mad_ms": 0.02})


def test_device_program_not_sum_of_custom_calls():
  profile = {
    "calls": 2,
    "errors": [],
    "groups": [
      {"kind": "kernel", "count": 2, "samples": [{"duration_ps": 1e8}] * 2},
      {"kind": "program", "count": 2, "samples": [{"duration_ps": 3e8}] * 2},
    ],
  }
  assert profile_program(profile)["median_ms"] == 0.3
  profile["errors"] = ["missing capture"]
  assert profile_program(profile) is None


def test_timeout_kills_child_process_group(tmp_path):
  marker = tmp_path / "survived"
  child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
  parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1]]); time.sleep(30)"
  with pytest.raises(subprocess.TimeoutExpired):
    run_process(
      [sys.executable, "-c", parent, child], directory=tmp_path, timeout=0.2
    )
  time.sleep(1.1)
  assert not marker.exists()


@pytest.mark.parametrize(
  "provider,model,expected",
  [
    ("openai", "test-model", "OpenAIResponses"),
    ("gemini", "gemini-3.1-pro-preview", "TimeoutGemini"),
  ],
)
def test_provider_and_summarizer_selection_without_credentials(
  provider, model, expected
):
  env = os.environ | {
    "MAXKERNEL_LLM_PROVIDER": provider,
    "MAXKERNEL_MODEL": model,
  }
  env.pop("OPENAI_API_KEY", None)
  code = """
from auto_agent.config import compaction_model, get_thinking_planner
from auto_agent.custom_types import CustomLlmAgent
import json
agent = CustomLlmAgent(name='test_agent', model='test-model')
print(json.dumps([type(agent.model).__name__,type(compaction_model).__name__,get_thinking_planner() is None]))
"""
  result = subprocess.run(
    [sys.executable, "-c", code],
    env=env,
    capture_output=True,
    text=True,
    check=True,
  )
  assert json.loads(result.stdout.strip()) == [
    expected,
    expected,
    provider == "openai",
  ]


@pytest.fixture
def prepared_run(tmp_path):
  from argparse import Namespace

  import yaml

  from experiments.colab.run import prepare

  repo = tmp_path / "repo"
  (repo / "pallas_kernels/kernels").mkdir(parents=True)
  (repo / "pallas_kernels/configs").mkdir()
  (repo / "pyproject.toml").write_text('[project]\nname="test"\n')
  (repo / "pallas_kernels/kernels/gqa_decode.py").write_text("VALUE = 1\n")
  config = {
    "name": "gqa_decode",
    "kernel": "gqa_decode",
    "shape": {"batch": 8, "seq_len": 1024},
    "dtype": "bfloat16",
    "seed": 0,
    "tiling": {"block_b": 8, "block_kv": 1024},
    "devices": {"num_devices": 1},
    "tolerance": {"atol": 0.01, "rtol": 0.01},
  }
  (repo / "pallas_kernels/configs/gqa_decode.yaml").write_text(
    yaml.safe_dump(config)
  )
  directory = tmp_path / "run"
  prepare(
    Namespace(
      repo=repo,
      kernel="gqa_decode",
      run_dir=directory,
      seq_len=None,
      wall_seconds=60,
      tpu_jobs=12,
      model_calls=10,
    )
  )
  return directory


def fake_result(speed=0.8, device_speed=0.8, profile=False):
  def timing(x):
    return {"median_ms": x, "mad_ms": 0.001, "samples_ms": [x] * 12}

  result = {
    "correct": True,
    "hardware": {"device_kind": "TEST"},
    "baseline": timing(1),
    "candidate": timing(speed),
  }
  if profile:

    def trace(x):
      return {
        "calls": 12,
        "errors": [],
        "groups": [
          {
            "kind": "program",
            "count": 12,
            "samples": [{"duration_ps": x * 1e9}] * 12,
          }
        ],
      }

    result["profiles"] = {
      "baseline": trace(1),
      "candidate": trace(device_speed),
    }
  return result


@pytest.mark.asyncio
@pytest.mark.parametrize("device_speed,published", [(0.7, True), (1.1, False)])
async def test_complete_budgeted_workflow_and_device_publication(
  prepared_run, monkeypatch, device_speed, published
):
  from experiments.colab import agents
  from experiments.colab import run as driver
  from experiments.colab.core import write_json

  monkeypatch.setenv("MAXKERNEL_MODEL", "test-model")
  monkeypatch.setattr(agents, "context", lambda *a: "fixed experiment context")
  calls = []

  async def ask(budget, role, prompt):
    budget.reserve("model_calls")
    calls.append(role)
    if role == "plan":
      return "Reduce redundant work."
    return json.dumps({"edits": [{"old": "VALUE = 1", "new": "VALUE = 2"}]})

  def run_process(command, *, directory, timeout, env):
    assert "OPENAI_API_KEY" not in env
    folder = Path(directory)
    request = json.loads((folder / "request.json").read_text())
    rewritten = "VALUE = 2" in Path(request["source"]).read_text()
    speed = 0.6 if rewritten else 0.85
    # Original baseline at original tiling should compare equal.
    if (
      request["source"].endswith("baseline.py")
      and request["tiling"]["block_b"] == 8
    ):
      speed = 1
    write_json(
      folder / "result.json",
      fake_result(speed, device_speed, request["profile"]),
    )
    return 0

  from pathlib import Path

  monkeypatch.setattr(agents, "ask", ask)
  monkeypatch.setattr(driver, "run_process", run_process)
  experiment = driver.Experiment(prepared_run, mode="agent")
  await experiment.run()
  report = json.loads((prepared_run / "report.json").read_text())
  assert report["status"] == "complete"
  assert report["device_gain_confirmed"] is published
  assert ("VALUE = 2" in (prepared_run / "best.py").read_text()) is published
  assert report["budget"]["tpu_jobs"] <= 12
  assert report["budget"]["profile_jobs"] == 2
  assert calls[:2] == ["plan", "implement"]
  assert (prepared_run / "best_config.json").exists()
  used = experiment.budget.state.copy()
  await driver.Experiment(prepared_run, mode="agent", resume=True).run()
  assert json.loads((prepared_run / "budget.json").read_text()) == used


def test_resume_rejects_changed_task_and_repository(prepared_run):
  from experiments.colab.core import write_json
  from experiments.colab.run import Experiment

  exp = Experiment(prepared_run, mode="autotune-only")
  task = json.loads((prepared_run / "task.json").read_text())
  task["limits"]["tpu_jobs"] += 100
  write_json(prepared_run / "task.json", task)
  with pytest.raises(ValueError, match="Frozen task"):
    Experiment(prepared_run, mode="autotune-only", resume=True)
  from pathlib import Path

  (Path(exp.task["repo"]) / "pallas_kernels/kernels/gqa_decode.py").write_text(
    "VALUE = 9"
  )
  with pytest.raises(ValueError, match="source/config changed"):
    exp.check_inputs()


@pytest.mark.asyncio
async def test_failed_baseline_stops_before_agent_or_search(
  prepared_run, monkeypatch
):
  from pathlib import Path

  from experiments.colab import run as driver
  from experiments.colab.core import write_json

  def failed(command, *, directory, timeout, env):
    write_json(
      Path(directory) / "result.json", {"correct": False, "error": "No TPU"}
    )
    return 1

  monkeypatch.setattr(driver, "run_process", failed)
  experiment = driver.Experiment(prepared_run, mode="autotune-only")
  await experiment.run()
  assert experiment.state["status"] == "baseline_failed"
  assert experiment.budget.state["tpu_jobs"] == 1
  assert experiment.budget.state["model_calls"] == 0
  assert (prepared_run / "best.py").read_text() == "VALUE = 1\n"
