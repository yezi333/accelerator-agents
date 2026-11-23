"""Exercise Colab adapters with native scheduling, workers and selection."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from experiments.full_adk import search as colab


def make_search(tmp_path):
  (tmp_path / "base_kernel.py").write_text("# unchanged baseline")
  (tmp_path / "inputs").mkdir()
  (tmp_path / "inputs" / "programming.md").write_text("source guidance")
  return colab.create_search(
    tmp_path,
    iterations=5,
    config={"tolerance": {"atol": 0.01, "rtol": 0.02}},
    strategies=[f"strategy {i}" for i in range(5)],
    max_concurrency=2,
  )


@pytest.mark.asyncio
@pytest.mark.parametrize("all_invalid", [False, True])
async def test_five_native_routes_and_valid_selection(
  tmp_path, monkeypatch, all_invalid
):
  search = make_search(tmp_path)
  active = 0
  peak_active = 0
  seen = []

  async def fake_agent(**kwargs):
    nonlocal active, peak_active
    active += 1
    peak_active = max(peak_active, active)
    seen.append(kwargs["strategy"])
    directory = Path(kwargs["session_dir"])
    assert (directory / "base_kernel.py").read_text() == "# unchanged baseline"
    assert (
      directory / "optimized_kernel.py"
    ).read_text() == "# unchanged baseline"
    assert (
      directory / "inputs/programming.md"
    ).read_text() == "source guidance"
    assert kwargs["agent_config"] == {
      "max_iterations": 5,
      "atol": 0.01,
      "rtol": 0.02,
    }
    assert kwargs["events_compaction"] is True
    await asyncio.sleep(0)
    active -= 1
    index = int(kwargs["strategy"].split()[-1])
    state = {
      "best_iteration": 0,
      "history": [
        {
          "iteration": 0,
          "compilation_status": {"success": index != 4},
          "test_status": {"success": index != 3 and not all_invalid},
          "speedup": 1 + index,
          "kernel_code": f"# candidate {index}",
          "plan": f"plan {index}",
        }
      ],
    }
    (directory / "session.json").write_text(json.dumps({"state": state}))
    return state

  monkeypatch.setattr(search.worker, "_run_agent", fake_agent)
  await search.run()
  summary = colab.save_search_results(search, tmp_path, completed=True)
  assert sorted(seen) == search.strategies
  assert peak_active == 2
  assert len(summary["routes"]) == 5
  assert len(search.graph.nodes) == 6  # seed plus five routes
  assert all(route["iterations"] for route in summary["routes"])
  if all_invalid:
    assert summary["best_node_id"] is None
    assert summary["optimized_kernel_path"] is None
    assert not (tmp_path / "optimized_kernel.py").exists()
  else:
    # Faster candidates 3 and 4 fail correctness / compilation respectively.
    assert summary["best_speedup"] == 3
    assert (tmp_path / "optimized_kernel.py").read_text() == "# candidate 2"
    assert (tmp_path / "base_kernel_plan.md").read_text() == "plan 2"


@pytest.mark.asyncio
async def test_interruption_saves_partial_graph_and_closes_tools(
  tmp_path, monkeypatch
):
  from auto_search.graph import EvaluationResult, Node

  search = make_search(tmp_path)

  async def interrupted():
    search.graph.add_node(
      Node(
        node_id="partial",
        parent_id=search.graph.root_id,
        session_dir=str(tmp_path / "partial"),
        depth=1,
        code="# measured before interruption",
        plan="partial plan",
        strategy_applied="strategy 0",
        execution_status="INTERRUPTED",
        evaluation=EvaluationResult(compiled=True, correct=True, speedup=1.2),
      )
    )
    raise asyncio.CancelledError()

  monkeypatch.setattr(search, "run", interrupted)
  monkeypatch.setattr(colab, "create_search", lambda *args: search)
  read_tools, write_tools = AsyncMock(), AsyncMock()
  monkeypatch.setattr(colab, "filesystem_tool_r", read_tools)
  monkeypatch.setattr(colab, "filesystem_tool_rw", write_tools)
  with pytest.raises(asyncio.CancelledError):
    await colab.run_search(tmp_path, 5, {}, search.strategies, 2)
  summary = json.loads((tmp_path / "search_summary.json").read_text())
  assert summary["status"] == "interrupted"
  assert summary["best_node_id"] == "partial"
  assert summary["best_speedup"] == 1.2
  graph = json.loads((tmp_path / "search_graph.json").read_text())
  assert graph["best_node_id"] == "partial"
  read_tools.close.assert_awaited_once()
  write_tools.close.assert_awaited_once()
