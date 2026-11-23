"""Colab input/output adapters around the original Parallel Search."""

import json
import logging
import shutil
from pathlib import Path

from auto_agent.tools.file_tools import filesystem_tool_r, filesystem_tool_rw
from auto_search.algorithms.parallel_search import (
  SimpleParallelSearchOrchestrator,
)
from auto_search.worker import ADKSessionWorker
from experiments.full_adk.run import save_json


class ColabSessionWorker(ADKSessionWorker):
  """Give each native worker the same supplemental source documentation."""

  def __init__(self, inputs_dir):
    super().__init__()
    self.inputs_dir = inputs_dir

  def _prepare_inputs(self, session_dir, parent_node, reference_code):
    super()._prepare_inputs(session_dir, parent_node, reference_code)
    shutil.copytree(
      self.inputs_dir, Path(session_dir) / "inputs", dirs_exist_ok=True
    )


def create_search(directory, iterations, config, strategies, max_concurrency):
  reference_code = (directory / "base_kernel.py").read_text()
  search = SimpleParallelSearchOrchestrator(
    problem_id="gqa_decode",
    reference_code=reference_code,
    initial_kernel_code=reference_code,
    graph_db_path=str(directory / "search_graph.json"),
    num_parallel_runs=len(strategies),
    strategies=strategies,
    max_concurrency=max_concurrency,
    max_worker_retries=1,
    agent_config={
      "max_iterations": iterations,
      "atol": config["tolerance"]["atol"],
      "rtol": config["tolerance"]["rtol"],
    },
    events_compaction=True,
  )
  search.worker = ColabSessionWorker(directory / "inputs")
  return search


def save_search_results(search, directory, completed):
  """Report native worker histories and the native graph's selected candidate."""
  routes = []
  for node in search.graph.nodes.values():
    if node.parent_id is None:
      continue
    session_path = Path(node.session_dir) / "session.json"
    state = {}
    if session_path.is_file():
      state = json.loads(session_path.read_text()).get("state", {})
    routes.append(
      {
        "node_id": node.node_id,
        "strategy": node.strategy_applied,
        "session_dir": node.session_dir,
        "execution_status": node.execution_status,
        "execution_error": node.execution_error,
        "evaluation": node.evaluation.model_dump(),
        "best_iteration": state.get("best_iteration", -1),
        "iterations": state.get("history", []),
      }
    )
  best_id = search.graph.best_node_id
  best = search.graph.get_node(best_id) if best_id else None
  valid_best = best is not None and best.is_valid_candidate
  if valid_best:
    (directory / "optimized_kernel.py").write_text(best.code)
    (directory / "base_kernel_plan.md").write_text(best.plan)
  summary = {
    "status": "completed" if completed else "interrupted",
    "best_node_id": best_id if valid_best else None,
    "best_speedup": best.evaluation.speedup if valid_best else None,
    "optimized_kernel_path": (
      str(directory / "optimized_kernel.py") if valid_best else None
    ),
    "routes": routes,
    "note": "Original Parallel Search and native test/autotune selection. "
    "No valid best_node_id means no measured correct candidate was selected. "
    "Native speedup is not a separate device-only confirmation.",
  }
  save_json(directory / "search_summary.json", summary)
  return summary


async def run_search(
  directory, iterations, config, strategies, max_concurrency
):
  search = create_search(
    directory, iterations, config, strategies, max_concurrency
  )
  save_json(
    directory / "search_config.json",
    {
      "algorithm": "parallel",
      "strategies": strategies,
      "num_parallel_runs": len(strategies),
      "max_concurrency": max_concurrency,
      "max_iterations": iterations,
    },
  )
  completed = False
  try:
    await search.run()
    completed = True
  finally:
    try:
      search.graph.save()
      save_search_results(search, directory, completed)
    finally:
      for toolset in (filesystem_tool_r, filesystem_tool_rw):
        try:
          await toolset.close()
        except Exception:
          logging.exception("MCP toolset cleanup failed")
