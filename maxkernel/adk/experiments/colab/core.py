"""Dependency-light task contracts, patching, and process budget enforcement."""

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

KERNELS = {"gqa_decode", "gqa_bwd"}
DEFAULT_LIMITS = {
  "wall_seconds": 3600,
  "job_seconds": 300,
  "tpu_jobs": 12,
  "model_calls": 10,
  "profile_jobs": 2,
  "rounds": 2,
  "repairs": 2,
  "tune_configs": 4,
}


class BudgetExhausted(RuntimeError):
  pass


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
  path = Path(path)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(value, indent=2) + "\n")
  tmp.replace(path)


def validate_limits(limits):
  if set(limits) != set(DEFAULT_LIMITS):
    raise ValueError("Unknown or missing budget fields")
  for key, value in limits.items():
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
      raise ValueError(f"{key} must be a positive integer")


def apply_edits(source, edits):
  if not isinstance(edits, list) or not 1 <= len(edits) <= 8:
    raise ValueError("Expected 1 to 8 exact source replacements")
  for edit in edits:
    if set(edit) != {"old", "new"} or not all(
      isinstance(v, str) for v in edit.values()
    ):
      raise ValueError("Each edit requires string old/new fields")
    if not edit["old"] or source.count(edit["old"]) != 1:
      raise ValueError("Each old fragment must match exactly once")
    source = source.replace(edit["old"], edit["new"], 1)
  compile(source, "candidate.py", "exec")
  return source


def tuning_candidates(kernel, base, count):
  if kernel == "gqa_decode":
    choices = [
      {"block_b": b, "block_kv": k}
      for b, k in [(1, 256), (2, 256), (1, 512), (2, 512)]
    ]
  else:
    choices = [
      {"block_q_dq": q, "block_kv_dq": k, "block_q_dkv": q, "block_kv_dkv": k}
      for q, k in [(128, 128), (256, 128), (256, 256), (128, 512)]
    ]
  return [base | values for values in choices if base | values != base][:count]


def meaningful_gain(before, after, minimum=0.03):
  """Wall-clock screening only; final device confirmation is separate."""
  if not all(
    math.isfinite(x) and x > 0
    for x in [before["median_ms"], after["median_ms"]]
  ):
    return False
  noise = max(
    before["mad_ms"] / before["median_ms"], after["mad_ms"] / after["median_ms"]
  )
  return after["median_ms"] < before["median_ms"] * (
    1 - max(minimum, 3 * noise)
  )


class Budget:
  def __init__(self, directory, limits):
    self.directory = Path(directory)
    self.limits = limits
    validate_limits(limits)
    self.path = self.directory / "budget.json"
    self.state = (
      json.loads(self.path.read_text())
      if self.path.exists()
      else {
        "started_at": time.time(),
        "tpu_jobs": 0,
        "model_calls": 0,
        "profile_jobs": 0,
      }
    )
    self.save()

  def save(self):
    write_json(self.path, self.state)

  def remaining(self):
    return max(
      0, self.limits["wall_seconds"] - (time.time() - self.state["started_at"])
    )

  def reserve(self, kind, *, profile=False):
    if self.remaining() <= 0 or self.state[kind] >= self.limits[kind]:
      raise BudgetExhausted(f"{kind} or wall-time budget exhausted")
    if profile and self.state["profile_jobs"] >= self.limits["profile_jobs"]:
      raise BudgetExhausted("profile budget exhausted")
    self.state[kind] += 1
    if profile:
      self.state["profile_jobs"] += 1
    # Charge before dispatch; failed/interrupted work is never free on resume.
    self.save()
    return min(self.limits["job_seconds"], self.remaining())


def run_process(command, *, directory, timeout, env=None):
  """One child process group; timeout/interruption kills its descendants too."""
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  with (
    (directory / "stdout.log").open("w") as out,
    (directory / "stderr.log").open("w") as err,
  ):
    process = subprocess.Popen(
      command, stdout=out, stderr=err, env=env, start_new_session=True
    )
    try:
      code = process.wait(timeout=timeout)
    except BaseException:
      try:
        os.killpg(process.pid, signal.SIGKILL)
      except ProcessLookupError:
        pass
      process.wait()
      raise
  return code


def repository_fingerprint(repo):
  repo = Path(repo)
  paths = [repo / "pyproject.toml"]
  for folder in ("pallas_kernels", "jax_ops", "tools"):
    paths.extend(sorted((repo / folder).rglob("*.py")))
  paths.extend(sorted((repo / "pallas_kernels/configs").glob("*.yaml")))
  payload = {str(path.relative_to(repo)): digest(path) for path in paths}
  return hashlib.sha256(
    json.dumps(payload, sort_keys=True).encode()
  ).hexdigest()
