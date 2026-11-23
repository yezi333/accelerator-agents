"""Run the original AutonomousPipelineAgent with its original local services."""

import argparse
import asyncio
import fcntl
import hashlib
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


def save_json(path, data):
  path = Path(path)
  temporary = path.with_suffix(path.suffix + ".tmp")
  temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
  temporary.replace(path)


def prepare_input(repo, directory):
  """Add the native harness entry/input contract to unchanged decode source."""
  import yaml

  config = yaml.safe_load(
    (repo / "pallas_kernels/configs/gqa_decode.yaml").read_text()
  )
  if config["kernel"] != "gqa_decode" or config["devices"]["num_devices"] != 1:
    raise ValueError("Expected a single-device gqa_decode config")
  source = (repo / "pallas_kernels/kernels/gqa_decode.py").read_text()
  tiling = config["tiling"]
  extra = f"""

# Native ADK harness contract. The implementation above is unmodified.
# Workload and tolerances: {config!r}
def computation(q, k_cache, v_cache, cache_lengths):
    return _gqa_decode(
        q, k_cache, v_cache, cache_lengths,
        block_sizes=BlockSizes(block_b={tiling["block_b"]!r},
                              block_q={tiling.get("block_q", 128)!r},
                              block_kv={tiling["block_kv"]!r}),
    )

def get_inputs():
    from pallas_kernels.kernels import get_kernel
    from pallas_kernels.kernels.config_loader import KernelSpec
    runner = get_kernel("gqa_decode", KernelSpec(**{config!r}))
    values = runner.make_public_inputs(jax.random.key({config["seed"]!r}))
    return [(list(values), [])]
"""
  seed = source + extra
  compile(seed, "source_kernel.py", "exec")
  inputs = directory / "inputs"
  inputs.mkdir()
  (inputs / "source_kernel.py").write_text(seed)
  for relative in (
    "pallas_kernels/kernels/gqa_common.py",
    "jax_ops/gqa_decode.py",
    "context/programming.md",
    "docs/kernels/gqa_decode.md",
  ):
    if relative.endswith(".md") and not (repo / relative).is_file():
      logging.info("Optional input documentation absent: %s", relative)
      continue
    target = inputs / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(repo / relative, target)
  # Same initial-seed convention as auto_search.worker._prepare_inputs.
  (directory / "base_kernel.py").write_text(seed)
  (directory / "optimized_kernel.py").write_text(seed)
  save_json(directory / "input_config.json", config)
  save_json(
    directory / "input_provenance.json",
    {
      "source_path": str(repo / "pallas_kernels/kernels/gqa_decode.py"),
      "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
      "note": "Original Pallas implementation plus computation/get_inputs adapters. "
      "Native ADK generates tests and benchmarks against this Pallas baseline; "
      "this is not the bounded experiment's fixed JAX oracle.",
    },
  )
  return config


def stop_process_group(process):
  # Kill only groups created by this launcher, including execution children.
  try:
    os.killpg(process.pid, signal.SIGTERM)
  except ProcessLookupError:
    pass
  try:
    process.wait(timeout=10)
  except subprocess.TimeoutExpired:
    pass
  try:
    os.killpg(process.pid, signal.SIGKILL)
  except ProcessLookupError:
    pass
  process.wait()


def wait_healthy(process, port, log_path):
  deadline = time.monotonic() + 120
  while time.monotonic() < deadline:
    if process.poll() is not None:
      break
    try:
      with urllib.request.urlopen(
        f"http://127.0.0.1:{port}/health", timeout=2
      ) as response:
        if response.status == 200:
          return
    except (OSError, urllib.error.URLError):
      pass
    time.sleep(1)
  raise RuntimeError(f"Service on {port} failed. See {log_path}")


async def run_agent(directory, iterations, config):
  from auto_agent.agent import create_root_agent
  from auto_agent.agent_client.auto_agent_client import AutoAgentClient
  from auto_agent.tools.file_tools import filesystem_tool_r, filesystem_tool_rw

  agent = create_root_agent(
    max_iterations=iterations,
    session_dir=str(directory),
    atol=config["tolerance"]["atol"],
    rtol=config["tolerance"]["rtol"],
  )
  query = (
    f"Optimize the existing Pallas GQA decode implementation at "
    f"{directory / 'inputs/source_kernel.py'}. Copy it unchanged to base_kernel.py "
    "during preparation. An identical initial candidate is at optimized_kernel.py. "
    "The public entry is computation(q, k_cache, v_cache, cache_lengths). "
    "Use its existing get_inputs() workload and specified tolerances. "
    "Preserve the numerical contract and optimize the actual Pallas body. "
    "Additional dependency source and documentation are in the inputs directory. "
    "Run the full native pipeline including per-iteration autotuning and profiling."
  )
  client = AutoAgentClient(
    user_id="colab",
    session_id=directory.name,
    query=query,
    agent=agent,
    events_compaction=True,
  )
  await client.create_session()
  with (directory / "events.jsonl").open("a") as events:

    def on_event(event):
      events.write(event.model_dump_json(exclude_none=True) + "\n")
      events.flush()
      calls = event.get_function_calls()
      names = ", ".join(call.name for call in calls)
      print(
        f"[{event.author}]" + (f" tools: {names}" if names else ""), flush=True
      )

    try:
      await client.run_async(on_event=on_event)
    finally:
      save_json(directory / "session.json", client.get_session_data())
      state = client.get_state()
      save_json(directory / "state.json", state)
      save_json(
        directory / "native_summary.json",
        {
          "pipeline_status": state.get("pipeline_status"),
          "best_iteration": state.get("best_iteration"),
          "iterations": [
            {
              "iteration": row.get("iteration"),
              "speedup": row.get("speedup"),
              "compilation_status": row.get("compilation_status"),
              "test_status": row.get("test_status"),
              "autotuning_summary": row.get("autotuning_summary"),
              "profiling_summary": row.get("profiling_summary"),
            }
            for row in state.get("history", [])
          ],
          "note": "Native ADK metrics/selection, not device_gain_confirmed from the "
          "bounded experiment. speedup follows the native test/autotune harness.",
        },
      )
      for toolset in (filesystem_tool_r, filesystem_tool_rw):
        try:
          await toolset.close()
        except Exception:
          logging.exception("MCP toolset cleanup failed")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--repo", required=True, type=Path)
  parser.add_argument("--session-dir", required=True, type=Path)
  parser.add_argument("--iterations", type=int, default=5)
  parser.add_argument(
    "--strategies",
    nargs="+",
    help="Run one original Parallel Search worker per explicit strategy",
  )
  parser.add_argument("--max-concurrency", type=int, default=2)
  args = parser.parse_args()
  # Native agents resolve their knowledge files relative to the ADK directory.
  adk_dir = Path(os.environ["MAXKERNEL_ADK_DIR"]).resolve()
  if not (adk_dir / "auto_agent/tpu_specs.json").is_file():
    parser.error("MAXKERNEL_ADK_DIR must point to maxkernel/adk")
  os.chdir(adk_dir)
  if args.iterations < 1:
    parser.error("--iterations must be positive")
  if args.max_concurrency < 1:
    parser.error("--max-concurrency must be positive")
  if args.strategies and (
    any(not strategy.strip() for strategy in args.strategies)
    or len(set(args.strategies)) != len(args.strategies)
  ):
    parser.error("--strategies must contain distinct non-empty directions")
  if not os.environ.get("OPENAI_API_KEY") or not (
    os.environ.get("MAXKERNEL_MODEL") or os.environ.get("MODEL_NAME")
  ):
    parser.error("Set OPENAI_API_KEY and MAXKERNEL_MODEL (or MODEL_NAME)")
  if os.environ.get("MAXKERNEL_LLM_PROVIDER") != "openai":
    parser.error("Set MAXKERNEL_LLM_PROVIDER=openai before launch")
  directory, repo = args.session_dir.resolve(), args.repo.resolve()
  workdir = Path(os.environ["WORKDIR"]).resolve()
  if not directory.is_relative_to(workdir) or directory == workdir:
    parser.error("session-dir must be below WORKDIR")
  if directory.exists():
    parser.error(
      "Use a new session directory; native pipeline is not stage-resumable"
    )
  if not shutil.which("npx"):
    parser.error("npx is required for the original MCP filesystem tools")
  # The driver may import JAX via API lookup, but must never acquire the TPU.
  os.environ["JAX_PLATFORMS"] = "cpu"
  logging.basicConfig(level=logging.INFO)
  lock_path = Path(tempfile.gettempdir()) / "maxkernel-colab-tpu.lock"
  with lock_path.open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for port in (5463, 5464, 1245):
      with socket.socket() as probe:
        probe.bind(("127.0.0.1", port))
    directory.mkdir(parents=True)
    config = prepare_input(repo, directory)
    services_dir = directory / "services"
    services_dir.mkdir()
    save_json(
      services_dir / "eval_config.yaml",
      {
        "backends": [
          {"name": "local_tpu", "type": "tpu", "ip": "127.0.0.1", "port": 5463},
          {"name": "local_cpu", "type": "cpu", "ip": "127.0.0.1", "port": 5464},
        ]
      },
    )  # JSON is also valid YAML.
    processes, logs = [], []
    try:
      for name, port in (("tpu", 5463), ("cpu", 5464), ("eval", 1245)):
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("GITHUB_TOKEN", None)
        env["JAX_PLATFORMS"] = "tpu,cpu" if name == "tpu" else "cpu"
        module = f"auto_agent.server_utils.{name}_server"
        command = [sys.executable, "-u", "-m", module]
        if name == "eval":
          command = [
            sys.executable,
            "-m",
            "uvicorn",
            module + ":app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
          ]
        log_path = services_dir / f"{name}.log"
        log = log_path.open("w")
        logs.append(log)
        process = subprocess.Popen(
          command,
          cwd=services_dir,
          env=env,
          stdout=log,
          stderr=subprocess.STDOUT,
          start_new_session=True,
        )
        processes.append(process)
        wait_healthy(process, port, log_path)
        print(f"Native {name} service ready: {port}", flush=True)
      if args.strategies:
        from experiments.full_adk.search import run_search

        asyncio.run(
          run_search(
            directory,
            args.iterations,
            config,
            args.strategies,
            args.max_concurrency,
          )
        )
      else:
        asyncio.run(run_agent(directory, args.iterations, config))
    finally:
      for process in reversed(processes):
        stop_process_group(process)
      for log in logs:
        log.close()


if __name__ == "__main__":
  main()
