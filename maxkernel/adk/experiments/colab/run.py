"""Prepare/run a bounded ADK experiment without modifying PallasKernels."""

import argparse
import asyncio
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from experiments.colab.core import (
  DEFAULT_LIMITS,
  KERNELS,
  Budget,
  BudgetExhausted,
  apply_edits,
  digest,
  meaningful_gain,
  repository_fingerprint,
  run_process,
  tuning_candidates,
  validate_limits,
  write_json,
)


def prepare(args):
  import yaml

  repo, directory = args.repo.resolve(), args.run_dir.resolve()
  source = repo / f"pallas_kernels/kernels/{args.kernel}.py"
  config_path = repo / f"pallas_kernels/configs/{args.kernel}.yaml"
  config = yaml.safe_load(config_path.read_text())
  if config["kernel"] != args.kernel or config["devices"]["num_devices"] != 1:
    raise ValueError("Expected a matching single-device config")
  if args.seq_len:
    config["shape"]["seq_len"] = args.seq_len
  if config.get("params", {}).get("interpret_mode") or config.get(
    "params", {}
  ).get("interpret"):
    raise ValueError("Interpret mode is not a performance experiment")
  if args.kernel == "gqa_bwd":
    config.setdefault("params", {})["use_fused"] = False
  limits = DEFAULT_LIMITS | {
    "wall_seconds": args.wall_seconds,
    "tpu_jobs": args.tpu_jobs,
    "model_calls": args.model_calls,
  }
  validate_limits(limits)
  directory.mkdir(parents=True, exist_ok=False)
  shutil.copyfile(source, directory / "baseline.py")
  task = {
    "schema": 1,
    "repo": str(repo),
    "kernel": args.kernel,
    "config": config,
    "limits": limits,
    "repository_fingerprint": repository_fingerprint(repo),
    "baseline_sha256": digest(source),
    "evaluator_sha256": digest(Path(__file__).with_name("evaluate.py")),
  }
  write_json(directory / "task.json", task)
  print(f"Prepared {args.kernel}: {directory}\nNo TPU or API calls made.")


def profile_program(profile):
  if profile.get("errors"):
    return None
  groups = [g for g in profile.get("groups", []) if g["kind"] == "program"]
  if len(groups) != 1 or groups[0].get("count") != profile["calls"]:
    return None
  group = groups[0]
  samples = [s["duration_ps"] / 1e9 for s in group["samples"]]
  import statistics

  median = statistics.median(samples)
  return {
    "median_ms": median,
    "mad_ms": statistics.median(abs(v - median) for v in samples),
  }


class Experiment:
  def __init__(self, directory, *, mode, resume=False):
    self.directory = Path(directory).resolve()
    self.task = json.loads((self.directory / "task.json").read_text())
    self.check_inputs()
    path = self.directory / "state.json"
    if path.exists() and not resume:
      raise ValueError("Existing run: use --resume; budgets will not reset")
    if resume and not path.exists():
      raise ValueError("No saved run to resume")
    self.state = (
      json.loads(path.read_text())
      if path.exists()
      else {
        "task_sha256": digest(self.directory / "task.json"),
        "mode": mode,
        "model": os.environ.get("MAXKERNEL_MODEL") if mode == "agent" else None,
        "history": [],
        "best_source": "baseline.py",
        "best_tiling": self.task["config"]["tiling"],
        "best_ratio": 1.0,
        "tune_index": 0,
        "round": 0,
        "status": "ready",
        "autotune_ratio": 1.0,
      }
    )
    if self.state["task_sha256"] != digest(self.directory / "task.json"):
      raise ValueError(
        "Frozen task/config/budget was changed; prepare a new experiment"
      )
    if self.state["mode"] != mode:
      raise ValueError("Cannot change experiment mode during resume")
    if mode == "agent" and self.state["model"] != os.environ.get(
      "MAXKERNEL_MODEL"
    ):
      raise ValueError("Cannot change model during resume")
    self.budget = Budget(self.directory, self.task["limits"])
    self.save()

  def check_inputs(self):
    if (
      digest(Path(__file__).with_name("evaluate.py"))
      != self.task["evaluator_sha256"]
    ):
      raise ValueError("Fixed evaluator changed; prepare a new experiment")
    if digest(self.directory / "baseline.py") != self.task["baseline_sha256"]:
      raise ValueError("Frozen baseline was modified")
    if (
      repository_fingerprint(self.task["repo"])
      != self.task["repository_fingerprint"]
    ):
      raise ValueError(
        "PallasKernels source/config changed; prepare a new experiment"
      )

  def save(self):
    write_json(self.directory / "state.json", self.state)

  def evaluate(self, source, tiling, *, label, profile=False):
    self.check_inputs()
    timeout = self.budget.reserve("tpu_jobs", profile=profile)
    index = self.budget.state["tpu_jobs"]
    folder = self.directory / "jobs" / f"{index:03d}_{label}"
    folder.mkdir(parents=True)
    request = self.task | {
      "source": str(self.directory / source),
      "tiling": tiling,
      "profile": profile,
      "cache_dir": str(self.directory / "compilation_cache"),
    }
    write_json(folder / "request.json", request)
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    command = [
      sys.executable,
      "-m",
      "experiments.colab.evaluate",
      str(folder / "request.json"),
      str(folder / "result.json"),
    ]
    try:
      code = run_process(command, directory=folder, timeout=timeout, env=env)
      result_file = folder / "result.json"
      result = (
        json.loads(result_file.read_text())
        if result_file.exists()
        else {
          "correct": False,
          "error": (folder / "stderr.log").read_text()[-6000:],
        }
      )
      if code != 0:
        result["correct"] = False
    except subprocess.TimeoutExpired:
      result = {"correct": False, "error": f"TPU job exceeded {timeout:.1f}s"}
    self.check_inputs()
    record = {
      "label": label,
      "source": source,
      "tiling": tiling,
      "directory": str(folder.relative_to(self.directory)),
      "result": result,
    }
    self.state["history"].append(record)
    if result.get("correct"):
      hardware = result["hardware"]
      if "hardware" in self.state and hardware != self.state["hardware"]:
        raise ValueError(
          "Hardware/environment changed; start a new experiment and rebaseline"
        )
      self.state["hardware"] = hardware
    self.save()
    print(
      f"{label}: {'correct' if result.get('correct') else 'failed'}", flush=True
    )
    return result

  def consider(self, source, tiling, result):
    if not result.get("correct"):
      return
    # Compare normalized, freshly paired latencies, not cross-session absolute times.
    ratio = result["candidate"]["median_ms"] / result["baseline"]["median_ms"]
    threshold = dict(result["baseline"])
    threshold["median_ms"] *= self.state["best_ratio"]
    if meaningful_gain(threshold, result["candidate"]):
      self.state.update(
        best_source=source, best_tiling=tiling, best_ratio=ratio
      )
      self.save()

  def feedback(self):
    records = []
    for row in self.state["history"]:
      result = row["result"]
      summary = {
        k: result[k]
        for k in ("correct", "error", "hardware", "baseline", "candidate")
        if k in result
      }
      if "profiles" in result:
        summary["device_profiles"] = {
          name: {
            "errors": profile["errors"],
            "groups": [
              {
                k: v
                for k, v in group.items()
                if k not in {"samples", "samples_ps"}
              }
              for group in profile["groups"]
            ],
          }
          for name, profile in result["profiles"].items()
        }
      records.append(
        {"label": row["label"], "tiling": row["tiling"], "result": summary}
      )
    return records

  async def run(self):
    from experiments.colab.agents import ask, context, parse_edits

    if self.state["status"] == "complete":
      return
    try:
      if not any(
        r["label"] == "baseline" and r["result"].get("correct")
        for r in self.state["history"]
      ):
        result = self.evaluate(
          "baseline.py",
          self.task["config"]["tiling"],
          label="baseline",
          profile=True,
        )
        if not result.get("correct"):
          self.state["status"] = "baseline_failed"
          return
      choices = tuning_candidates(
        self.task["kernel"],
        self.task["config"]["tiling"],
        self.task["limits"]["tune_configs"],
      )
      for i in range(self.state["tune_index"], len(choices)):
        result = self.evaluate("baseline.py", choices[i], label=f"autotune_{i}")
        self.consider("baseline.py", choices[i], result)
        self.state["tune_index"] = i + 1
        self.state["autotune_ratio"] = self.state["best_ratio"]
        self.save()
      if self.state["mode"] == "agent":
        for round_id in range(
          self.state["round"], self.task["limits"]["rounds"]
        ):
          # Reserve final confirmation time/jobs; never sacrifice all budget to repairs.
          if (
            self.budget.state["tpu_jobs"] >= self.task["limits"]["tpu_jobs"] - 1
          ):
            break
          self.state["round"] = round_id + 1
          self.save()
          original = (self.directory / self.state["best_source"]).read_text()
          current_task = self.task | {
            "config": self.task["config"]
            | {"tiling": self.state["best_tiling"]}
          }
          prefix = context(current_task, original, self.feedback())
          plan = await ask(
            self.budget,
            "plan",
            prefix
            + (
              "\nPropose ONE structural/algorithm optimization. State measured evidence, "
              "expected compute/traffic/wait reduction, legality constraints, and how to falsify it. "
              "Do not write code yet. Keep the plan under 500 words."
            ),
          )
          (self.directory / f"plan_{round_id}.md").write_text(plan)
          source = original
          error = ""
          for attempt in range(self.task["limits"]["repairs"] + 1):
            if (
              self.budget.state["tpu_jobs"]
              >= self.task["limits"]["tpu_jobs"] - 1
            ):
              break
            instruction = (
              '\nReturn JSON only: {"edits": [{"old": "exact unique substring", "new": "replacement"}]}. '
              "At most 8 edits. Preserve exported functions and the configured registry interface. "
              "Only edit the kernel module; no changes to shape, dtype, tolerance or external files.\n"
              f"Plan: {plan}\nLast failure: {error}\n"
            )
            response = await ask(
              self.budget,
              "implement" if not attempt else "repair",
              context(current_task, source, self.feedback()) + instruction,
            )
            try:
              source = apply_edits(source, parse_edits(response))
            except (ValueError, SyntaxError, TypeError, KeyError) as exc:
              error = str(exc)
              continue
            name = f"candidate_{round_id}_{attempt}.py"
            (self.directory / name).write_text(source)
            tiling = self.state["best_tiling"]
            result = self.evaluate(
              name, tiling, label=f"round_{round_id}_attempt_{attempt}"
            )
            if result.get("correct"):
              self.consider(name, tiling, result)
              break
            error = result.get("error", "Evaluation failed")
      if (
        self.budget.state["profile_jobs"] < self.task["limits"]["profile_jobs"]
      ):
        final = self.evaluate(
          self.state["best_source"],
          self.state["best_tiling"],
          label="confirmation",
          profile=True,
        )
        self.state["confirmation"] = final
      self.state["status"] = "complete"
    except BudgetExhausted as exc:
      self.state.update(status="budget_exhausted", stop_reason=str(exc))
    except BaseException as exc:
      self.state.update(
        status="interrupted"
        if isinstance(exc, KeyboardInterrupt)
        else "failed",
        stop_reason=f"{type(exc).__name__}: {exc}",
      )
      raise
    finally:
      self.save()
      self.report()

  def report(self):
    final = self.state.get("confirmation", {})
    profiles = final.get("profiles", {})
    base = profile_program(profiles.get("baseline", {}))
    candidate = profile_program(profiles.get("candidate", {}))
    verified = bool(
      final.get("correct")
      and base
      and candidate
      and meaningful_gain(base, candidate)
    )
    published = self.state["best_source"] if verified else "baseline.py"
    tiling = (
      self.state["best_tiling"] if verified else self.task["config"]["tiling"]
    )
    shutil.copyfile(self.directory / published, self.directory / "best.py")
    write_json(
      self.directory / "best_config.json",
      self.task["config"] | {"tiling": tiling},
    )
    report = {
      "status": self.state["status"],
      "stop_reason": self.state.get("stop_reason"),
      "device_gain_confirmed": verified,
      "device_program_speedup": base["median_ms"] / candidate["median_ms"]
      if base and candidate
      else None,
      "screening_host_speedup": 1 / self.state["best_ratio"],
      "autotune_host_speedup": 1 / self.state["autotune_ratio"],
      "screening_source": self.state["best_source"],
      "published_source": published,
      "budget": self.budget.state,
      "hardware": self.state.get("hardware"),
      "note": "Host timing only screens candidates. best.py falls back to the original unless "
      "a correct candidate shows a >3% and >3*MAD full-device-program gain. "
      "Tests are finite evidence, not a proof of correctness.",
    }
    write_json(self.directory / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  prep = commands.add_parser(
    "prepare", help="Freeze inputs; no API/TPU required"
  )
  prep.add_argument("--repo", type=Path, required=True)
  prep.add_argument("--kernel", choices=sorted(KERNELS), required=True)
  prep.add_argument("--run-dir", type=Path, required=True)
  prep.add_argument("--seq-len", type=int)
  for name in ("wall_seconds", "tpu_jobs", "model_calls"):
    prep.add_argument(
      "--" + name.replace("_", "-"), type=int, default=DEFAULT_LIMITS[name]
    )
  run = commands.add_parser("run")
  run.add_argument("--run-dir", type=Path, required=True)
  run.add_argument(
    "--mode", choices=("agent", "autotune-only"), default="agent"
  )
  run.add_argument("--resume", action="store_true")
  args = parser.parse_args()
  if args.command == "prepare":
    if args.seq_len is not None and args.seq_len < 1:
      parser.error("--seq-len must be positive")
    prepare(args)
    return
  if args.mode == "agent" and (
    not os.environ.get("OPENAI_API_KEY")
    or not os.environ.get("MAXKERNEL_MODEL")
  ):
    parser.error(
      "Set OPENAI_API_KEY and MAXKERNEL_MODEL, or select --mode autotune-only"
    )
  host_lock = Path(tempfile.gettempdir()) / "maxkernel-colab-tpu.lock"
  with (
    host_lock.open("w") as global_lock,
    (args.run_dir / ".run.lock").open("w") as lock,
  ):
    try:
      fcntl.flock(global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
      fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      parser.error("A Colab experiment is already running on this host")
    experiment = Experiment(args.run_dir, mode=args.mode, resume=args.resume)
    asyncio.run(experiment.run())
    if experiment.state["status"] == "baseline_failed":
      raise SystemExit(1)


if __name__ == "__main__":
  main()
