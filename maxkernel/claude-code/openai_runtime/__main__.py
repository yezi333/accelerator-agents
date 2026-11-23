"""Run from maxkernel/claude-code: python -m openai_runtime --help."""

import argparse
import asyncio
import fcntl
import json
import math
import os
import sys
from pathlib import Path

from .prompts import ROOT, Prompts
from .runtime import ModelConfig, ResponsesHost, save_json
from .tools import NeedsInput, ToolHost


def validate_job(path):
  import importlib.util

  spec = importlib.util.spec_from_file_location(
    "maxkernel_validate_job", ROOT / "tools/validate_job.py"
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  job = json.loads(path.read_text())
  errors, warnings, plan = module.validate(job, path)
  if errors:
    raise ValueError("Invalid job:\n" + "\n".join(errors))
  if plan.get("missing_paths"):
    raise ValueError(
      "Missing job paths: " + ", ".join(map(str, plan["missing_paths"]))
    )
  if (
    plan["external_eval"]
    and not (ROOT / "evaluation/compare_kernel.py").is_file()
  ):
    raise ValueError(
      "Job requests external evaluation, but this checkout lacks evaluation/compare_kernel.py"
    )
  if (
    plan["input_type"] == "specification"
    and not (
      ROOT / "evaluation/templates/reference_model.py.template"
    ).is_file()
  ):
    raise ValueError(
      "Specification synthesis requires the upstream template evaluation/templates/reference_model.py.template, missing in this checkout"
    )
  if not plan["internal_harness"]:
    raise ValueError(
      "This launcher runs the original internal pipeline; this job requires external evaluation instead"
    )
  return job, {"valid": True, "warnings": warnings, "plan": plan}


def inspect_result(run_dir):
  """Check completion against native state and files, never model prose."""
  path = run_dir / "state.json"
  if not path.is_file():
    return {"status": "incomplete", "reason": "No native state.json"}
  state = json.loads(path.read_text())
  iteration, limit = state.get("iteration", 0), state.get("max_iterations", 5)
  history = state.get("history", [])
  completed = (
    type(iteration) is int
    and type(limit) is int
    and limit > 0
    and iteration >= limit
    and {h.get("iteration") for h in history} >= set(range(1, limit + 1))
  )
  result = {
    "status": "completed" if completed else "incomplete",
    "iteration": iteration,
    "max_iterations": limit,
    "best_code_path": None,
    "best_speedup": None,
  }
  valid = [
    h
    for h in history
    if h.get("compile_ok") is True
    and h.get("test_ok") is True
    and type(h.get("optimized_time_ms")) in {int, float}
    and math.isfinite(h["optimized_time_ms"])
    and h["optimized_time_ms"] > 0
  ]
  if valid:
    best = min(valid, key=lambda h: h["optimized_time_ms"])
    expected = Path(best["optimized_kernel"]).resolve()
    if not expected.is_relative_to(run_dir) or not expected.is_file():
      raise ValueError(
        "Best candidate must be an existing kernel inside this run"
      )
    if Path(state.get("best_code_path") or "").resolve() != expected:
      raise ValueError(
        "Native best_code_path disagrees with measured valid history"
      )
    if completed:
      final = run_dir / "optimized.py"
      if not final.is_file() or final.read_bytes() != expected.read_bytes():
        raise ValueError(
          "Final optimized.py is missing or does not match the native winner"
        )
      for key in ("kernel_plan", "autotune_summary", "profile_summary"):
        artifact = Path(best.get(key) or "").resolve()
        if (
          not artifact.is_relative_to(run_dir)
          or not artifact.is_file()
          or not artifact.stat().st_size
        ):
          raise ValueError(f"Full pipeline winner is missing {key}")
    result.update(
      best_code_path=str(expected), best_speedup=best.get("speedup")
    )
  result["has_valid_candidate"] = bool(valid)
  return result


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  inputs = parser.add_mutually_exclusive_group()
  inputs.add_argument(
    "--source",
    type=Path,
    help="Self-contained input; original classifier determines its language",
  )
  inputs.add_argument(
    "--job",
    type=Path,
    help="Original MaxKernel job.json (recommended for exact workload/tolerances)",
  )
  parser.add_argument(
    "--run-dir",
    required=True,
    type=Path,
    help="New directory below this checkout's workspace/",
  )
  parser.add_argument(
    "--python",
    default=sys.executable,
    type=Path,
    help="Prepared pipeline environment's bin/python",
  )
  parser.add_argument(
    "--model", help="Defaults to MAXKERNEL_MODEL, then MODEL_NAME"
  )
  parser.add_argument(
    "--resume",
    action="store_true",
    help="Continue from on-disk state, with fresh agent contexts",
  )
  parser.add_argument(
    "--reply-file",
    type=Path,
    help="User-authored reply to pending_question.json; requires --resume",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate inputs/render all prompts; no API or TPU calls",
  )
  parser.add_argument(
    "--with-guard",
    action="store_true",
    help="Run the original optional workspace guard before tools",
  )
  parser.add_argument(
    "--tpu-version",
    help="Explicit target, e.g. 'TPU v6e'; otherwise use original hardware discovery",
  )
  return parser.parse_args(argv)


async def execute(args, config, prompts, task, read_roots):
  from openai import AsyncOpenAI

  if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Set OPENAI_API_KEY before launching")
  tools = ToolHost(
    args.run_dir, prompts, read_roots=read_roots, guard=args.with_guard
  )
  status = prompts.run_dir / "openai_host/status.json"
  save_json(status, {"status": "running"})
  try:
    async with AsyncOpenAI(
      base_url=config.base_url,
      timeout=config.timeout,
      max_retries=config.max_retries,
    ) as client:
      host = ResponsesHost(config, prompts, tools, client)
      report = await host.run_agent("maxkernel", task)
      (prompts.run_dir / "openai_host/report.md").write_text(
        tools.redact(report)
      )
      result = inspect_result(prompts.run_dir)
      save_json(status, result)
      print(tools.redact(report))
      print(json.dumps(result, indent=2))
      return 0 if result["status"] == "completed" else 1
  except NeedsInput as exc:
    save_json(
      status,
      {"status": "needs_input", "agent": exc.agent, "questions": exc.questions},
    )
    print(
      "Run paused. Review openai_host/pending_question.json and the referenced artifacts; resume with --reply-file."
    )
    return 2
  except (Exception, asyncio.CancelledError) as exc:
    save_json(
      status,
      {
        "status": "interrupted"
        if isinstance(exc, asyncio.CancelledError)
        else "failed",
        "error_type": type(exc).__name__,
        "error": tools.redact(str(exc)),
      },
    )
    raise


def main(argv=None):
  args = parse_args(argv)
  try:
    config = ModelConfig.from_env(args.model)
    args.run_dir = args.run_dir.expanduser().resolve()
    workspace = ROOT / "workspace"
    if not args.run_dir.is_relative_to(workspace) or args.run_dir == workspace:
      raise ValueError("--run-dir must be below " + str(workspace))
    if args.reply_file and not args.resume:
      raise ValueError("--reply-file requires --resume")
    if args.resume and (args.source or args.job):
      raise ValueError(
        "Resume reuses the saved request; do not replace --source or --job"
      )
    if not args.resume and not (args.source or args.job):
      raise ValueError("Provide --source or --job for a new run")
    args.python = args.python.expanduser().absolute()
    if not args.python.is_file():
      raise ValueError("Prepared interpreter is missing: " + str(args.python))
    if args.resume:
      if not (args.run_dir / "openai_host/request.json").is_file():
        raise ValueError("No saved request to resume")
    else:
      # Fail before creating artifacts if the user supplied a nonexistent source.
      for path in (args.source, args.job):
        if path and not path.expanduser().is_file():
          raise ValueError("Input is not a file: " + str(path))
      if args.job:
        validate_job(args.job.expanduser().resolve())
      args.run_dir.mkdir(parents=True, exist_ok=False)
    host_dir = args.run_dir / "openai_host"
    host_dir.mkdir(exist_ok=True)
    with (host_dir / "run.lock").open("w") as lock:
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      if args.resume:
        request = json.loads((host_dir / "request.json").read_text())
        task, read_roots = request["task"], request["read_roots"]
        task += "\nRESUME: Read existing state.json and artifacts; do not reinitialize or overwrite completed work."
        pending_path = host_dir / "pending_question.json"
        if pending_path.is_file() and not args.reply_file:
          raise ValueError(
            "Run is waiting for user input; review pending_question.json and supply --reply-file"
          )
        if args.reply_file:
          if not pending_path.is_file():
            raise ValueError("No pending question to answer")
          task += "\nPrevious question:\n" + pending_path.read_text()
          task += (
            "\nUser's explicit reply:\n"
            + args.reply_file.expanduser().read_text()
          )
          save_json(
            host_dir / "request.json", {"task": task, "read_roots": read_roots}
          )
          import uuid

          pending_path.rename(
            host_dir / f"answered_question_{uuid.uuid4().hex[:12]}.json"
          )
      else:
        source = (args.source or args.job).expanduser().resolve()
        read_roots = [str(source.parent)]
        task = f"Optimize using the full original MaxKernel pipeline. Use run_dir = {args.run_dir}.\n"
        task += f"{'Job file (validate first and obey its declarations)' if args.job else 'Source file'}: {source}\n"
        if args.job:
          task += f"Resolve the job's relative file paths against {source.parent}, not the run directory; store absolute paths in state.json.\n"
          job, validation = validate_job(source)
          declared = [job["input"].get("path")]
          declared += [ref.get("path") for ref in job.get("references", [])]
          declared += [(job.get("backward") or {}).get("forward_context")]
          for value in filter(None, declared):
            path = (source.parent / value).resolve()
            read_roots.append(str(path if path.is_dir() else path.parent))
          save_json(host_dir / "job_validation.json", validation)
        if args.tpu_version:
          task += f"User-specified target TPU version: {args.tpu_version}\n"
        task += "Keep all original numerical gates, per-iteration autotuning and profiling. AskUserQuestion for any required user decision."
        save_json(
          host_dir / "request.json", {"task": task, "read_roots": read_roots}
        )
      prompts = Prompts(args.run_dir, args.python)
      snapshot = {
        **config.public(),
        "with_guard": args.with_guard,
        "python": str(args.python),
        "prompt_directory": str(prompts.directory),
        "prompt_sha256": prompts.manifest,
      }
      save_json(host_dir / "config.json", snapshot)
      save_json(prompts.directory / "config.json", snapshot)
      if args.dry_run:
        save_json(host_dir / "status.json", {"status": "prepared"})
        print(
          f"Rendered {len(prompts.agents) - 1} original subagents + orchestrator. No API/TPU requests."
        )
        print(f"Run --resume with this --run-dir to execute: {args.run_dir}")
        return 0
      return asyncio.run(execute(args, config, prompts, task, read_roots))
  except KeyboardInterrupt:
    print(
      "Interrupted; native state and host logs remain in the run directory.",
      file=sys.stderr,
    )
    return 130
  except Exception as exc:
    message = str(exc)
    if os.environ.get("OPENAI_API_KEY"):
      message = message.replace(os.environ["OPENAI_API_KEY"], "[REDACTED]")
    print(f"{type(exc).__name__}: {message}", file=sys.stderr)
    return 1


if __name__ == "__main__":
  raise SystemExit(main())
