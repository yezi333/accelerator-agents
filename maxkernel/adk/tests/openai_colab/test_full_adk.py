"""Checks for the launcher that delegates to the original ADK pipeline."""

import json
from pathlib import Path

import pytest
from google.adk.agents import BaseAgent
from google.adk.events import Event, EventActions
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from experiments.full_adk.run import prepare_input


def test_prepare_input_preserves_kernel_and_uses_registry(tmp_path):
  repo = tmp_path / "repo"
  config = {
    "kernel": "gqa_decode",
    "devices": {"num_devices": 1},
    "tiling": {"block_b": 8, "block_kv": 1024},
    "seed": 0,
  }
  files = {
    "pallas_kernels/configs/gqa_decode.yaml": json.dumps(config),
    "pallas_kernels/kernels/gqa_decode.py": "# existing math\ndef kernel():\n    return 42\n",
    "pallas_kernels/kernels/gqa_common.py": "# shared helper\n",
    "jax_ops/gqa_decode.py": "# reference\n",
    "context/programming.md": "guidance",
  }
  for name, contents in files.items():
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
  directory = tmp_path / "session"
  directory.mkdir()
  assert prepare_input(repo, directory) == config
  seed = (directory / "base_kernel.py").read_text()
  assert seed.startswith(files["pallas_kernels/kernels/gqa_decode.py"])
  assert seed == (directory / "optimized_kernel.py").read_text()
  assert "runner.make_public_inputs" in seed
  assert "block_b=8" in seed and "block_kv=1024" in seed
  for name, contents in files.items():
    assert (repo / name).read_text() == contents


@pytest.mark.asyncio
async def test_original_pipeline_profiles_each_valid_iteration(
  tmp_path, monkeypatch
):
  # Import only in this test: original profiling modules need TensorFlow/XProf.
  import auto_agent.subagents.pipeline_agent as native

  monkeypatch.setattr(native, "WORKDIR", str(tmp_path))
  calls = []
  plans = []

  class Stage(BaseAgent):
    async def _run_async_impl(self, ctx):
      iteration = ctx.session.state.get("iteration", -1)
      calls.append((self.name, iteration))
      delta = {}
      if self.name == "prepare":
        Path(ctx.session.state["base_kernel_path"]).write_text("# base")
      elif self.name == "test_gen":
        delta = {"validation_loop_status": {"success": True}}
      elif self.name == "plan":
        plans.append(ctx.session.state.get("profiling_summary"))
      elif self.name == "implement":
        Path(ctx.session.state["optimized_kernel_path"]).write_text(
          "# candidate"
        )
      elif self.name == "validate":
        delta = {"kernel_compilation_status": {"success": True}}
      elif self.name == "test":
        delta = {"test_results": {"success": True, "output": ""}}
      elif self.name == "autotune":
        delta = {"autotune_results": {"status": "success", "best_speedup": 1.1}}
      elif self.name == "profile":
        delta = {"profiling_summary": f"measured iteration {iteration}"}
      yield Event(author=self.name, actions=EventActions(state_delta=delta))

  agent = native.AutonomousPipelineAgent(
    name="native_pipeline",
    max_iterations=2,
    session_dir=str(tmp_path / "run"),
    prepare_base_kernel_agent=Stage(name="prepare"),
    test_gen_agent=Stage(name="test_gen"),
    plan_agent=Stage(name="plan"),
    implement_agent=Stage(name="implement"),
    validate_agent=Stage(name="validate"),
    test_run_agent=Stage(name="test"),
    autotune_agent=Stage(name="autotune"),
    profile_agent=Stage(name="profile"),
  )
  service = InMemorySessionService()
  session = await service.create_session(app_name="native_test", user_id="test")
  runner = Runner(app_name="native_test", agent=agent, session_service=service)
  try:
    async for _ in runner.run_async(
      user_id="test",
      session_id=session.id,
      new_message=types.Content(
        role="user", parts=[types.Part(text="Optimize")]
      ),
    ):
      pass
  finally:
    await runner.close()
  assert [name for name, _ in calls] == ["prepare", "test_gen"] + [
    "plan",
    "implement",
    "validate",
    "test",
    "autotune",
    "profile",
  ] * 2
  assert plans == [None, "measured iteration 0"]
