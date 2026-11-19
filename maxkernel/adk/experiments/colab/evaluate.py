"""Fixed TPU evaluator. Run only as a child of the bounded experiment driver."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path


def load_candidate(repo, kernel, source):
  name = f"pallas_kernels.kernels.{kernel}"
  spec = importlib.util.spec_from_file_location(name, source)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)


def evaluate(request, out_dir):
  repo = Path(request["repo"])
  sys.path.insert(0, str(repo))
  import jax
  import jax.numpy as jnp
  import numpy as np
  from pallas_kernels.kernels import get_kernel
  from pallas_kernels.kernels.config_loader import KernelSpec

  devices = jax.devices("tpu")
  if not devices:
    raise RuntimeError(
      "A TPU is required; CPU/interpret timing is not accepted"
    )
  device = devices[0]
  if jax.__version__ != "0.8.1":
    raise RuntimeError("PallasKernels experiment requires its pinned JAX 0.8.1")
  jax.config.update("jax_compilation_cache_dir", request["cache_dir"])
  baseline_cfg = KernelSpec(**request["config"])
  candidate_cfg = KernelSpec(
    **(request["config"] | {"tiling": request["tiling"]})
  )
  baseline = get_kernel(request["kernel"], baseline_cfg)
  # Generate a fixed suite and CPU reference before importing candidate code.
  cpu = jax.devices("cpu")[0]
  suite = []
  with jax.default_device(cpu):
    for seed, stress in [
      (baseline_cfg.seed, False),
      (baseline_cfg.seed + 1, False),
      (baseline_cfg.seed + 2, True),
    ]:
      values = list(baseline.make_public_inputs(jax.random.key(seed)))
      if stress:
        values[0] = values[0] * jnp.asarray(50, values[0].dtype)
        values[1] = values[1] * jnp.asarray(50, values[1].dtype)
        if request["kernel"] == "gqa_bwd":
          from jax_ops.gqa_fwd import gqa_fwd_reference

          values[3], values[4] = gqa_fwd_reference(*values[:3], return_lse=True)
        else:
          lengths = np.array(values[3])
          lengths[0] = 1
          if len(lengths) > 1:
            lengths[1] = min(129, baseline_cfg.shape["seq_len"])
          values[3] = jnp.asarray(lengths)
      expected = baseline.reference_functions()[0](*values)
      expected = jax.tree.map(lambda x: np.asarray(x).copy(), expected)
      suite.append((tuple(values), expected))
  suite = [
    (jax.device_put(values, device), expected) for values, expected in suite
  ]
  jax.block_until_ready([values for values, _ in suite])
  tolerance = baseline_cfg.get_tolerance(baseline_cfg.dtype)

  def build(runner, label):
    def entry(*xs):
      return runner(*xs)

    entry.__name__ = f"maxkernel_{label}"
    with jax.default_device(device):
      lowered = jax.jit(entry).lower(*suite[0][0])
      (out_dir / f"{label}.hlo.txt").write_text(lowered.as_text())
      t0 = time.monotonic()
      compiled = lowered.compile()
      compilation_seconds = time.monotonic() - t0
    # Require a real TPU executable (no reference fallback).
    used = compiled.runtime_executable().local_devices()
    if len(used) != 1 or used[0].platform != "tpu":
      raise RuntimeError("Expected a single-TPU executable")
    for values, expected in suite:
      got = jax.block_until_ready(compiled(*values))
      if jax.tree.structure(got) != jax.tree.structure(expected):
        raise AssertionError("Output tree differs from reference")
      for actual, ref in zip(
        jax.tree.leaves(got), jax.tree.leaves(expected), strict=True
      ):
        actual = np.asarray(actual)
        if actual.shape != ref.shape or actual.dtype != ref.dtype:
          raise AssertionError("Output shape/dtype differs from reference")
        if not np.isfinite(actual).all():
          raise AssertionError("Non-finite kernel output")
        np.testing.assert_allclose(actual, ref, **tolerance)
    for _ in range(3):
      jax.block_until_ready(compiled(*suite[0][0]))
    return compiled, compilation_seconds, entry.__name__

  base_fn, base_compile, base_tag = build(baseline, "baseline")
  load_candidate(repo, request["kernel"], request["source"])
  candidate = get_kernel(request["kernel"], candidate_cfg)
  cand_fn, cand_compile, cand_tag = build(candidate, "candidate")
  samples = {"baseline": [], "candidate": []}
  # Alternate order to reduce drift; this is host synchronized timing, NOT device time.
  for i in range(12):
    order = [("baseline", base_fn), ("candidate", cand_fn)]
    for label, fn in order if i % 2 == 0 else reversed(order):
      t0 = time.perf_counter()
      jax.block_until_ready(fn(*suite[0][0]))
      samples[label].append((time.perf_counter() - t0) * 1000)
  result = {
    "correct": True,
    "cases": len(suite),
    "hardware": {
      "device_kind": device.device_kind,
      "device_id": device.id,
      "visible_devices": len(devices),
      "versions": {
        p: importlib.metadata.version(p) for p in ("jax", "jaxlib", "libtpu")
      },
      "compiler_flags": {
        k: os.environ.get(k, "") for k in ("XLA_FLAGS", "LIBTPU_INIT_ARGS")
      },
    },
    "compilation_seconds": {
      "baseline": base_compile,
      "candidate": cand_compile,
    },
    "timing_scope": "synchronized host latency of the full public registry entry",
  }
  for label, values in samples.items():
    median = statistics.median(values)
    result[label] = {
      "median_ms": median,
      "samples_ms": values,
      "mad_ms": statistics.median(abs(x - median) for x in values),
    }
  if request["profile"]:
    from tools.get_tpu_duration_info import _capture_entry

    result["profiles"] = {}
    for label, fn, tag in [
      ("baseline", base_fn, base_tag),
      ("candidate", cand_fn, cand_tag),
    ]:
      result["profiles"][label] = _capture_entry(
        fn,
        suite[0][0],
        directory=out_dir / f"profile_{label}",
        tag=tag,
        calls=12,
        warmup=3,
        device_ids=[device.id],
      )
    result["profile_scope"] = (
      "full public entry; use Program time, do not sum custom calls"
    )
  return result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("request", type=Path)
  parser.add_argument("output", type=Path)
  args = parser.parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  try:
    result = evaluate(json.loads(args.request.read_text()), args.output.parent)
  except Exception as exc:
    result = {
      "correct": False,
      "error": f"{type(exc).__name__}: {exc}",
      "traceback": traceback.format_exc(),
    }
    args.output.write_text(json.dumps(result, indent=2))
    raise
  args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
