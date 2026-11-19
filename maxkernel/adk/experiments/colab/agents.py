"""Small ADK planning/editing sessions with no shell or file-write tools."""

import asyncio
import json
import os
from pathlib import Path

from experiments.colab.core import write_json


async def ask(budget, role, prompt):
  from google.adk.agents import LlmAgent
  from google.adk.runners import Runner
  from google.adk.sessions import InMemorySessionService
  from google.genai import types

  from auto_agent.openai_model import OpenAIResponses

  model_name = os.environ.get("MAXKERNEL_MODEL")
  if not model_name or not os.environ.get("OPENAI_API_KEY"):
    raise ValueError(
      "Set MAXKERNEL_MODEL and OPENAI_API_KEY before running the agent"
    )
  timeout = budget.reserve("model_calls")
  directory = (
    budget.directory
    / "model_calls"
    / f"{budget.state['model_calls']:03d}_{role}"
  )
  directory.mkdir(parents=True)
  (directory / "prompt.txt").write_text(prompt)
  model = OpenAIResponses(
    model=model_name,
    max_retries=0,
    timeout_seconds=timeout,
    max_output_tokens=8000,
  )
  agent = LlmAgent(
    name="kernel_experiment",
    model=model,
    instruction=(
      "You optimize an existing TPU Pallas kernel. Preserve public APIs, dtype, "
      "causal semantics and exact output contracts. Never alter tests, references "
      "or tolerances. Do not import a reference to replace computation. "
      "Work only on the requested stage. Performance hypotheses require measurement."
    ),
  )
  service = InMemorySessionService()
  session = await service.create_session(
    app_name="colab_experiment", user_id="local"
  )
  runner = Runner(
    app_name="colab_experiment", agent=agent, session_service=service
  )
  texts, usages = [], []

  async def consume():
    async for event in runner.run_async(
      user_id="local",
      session_id=session.id,
      new_message=types.Content(role="user", parts=[types.Part(text=prompt)]),
    ):
      if event.usage_metadata:
        usages.append(
          event.usage_metadata.model_dump(mode="json", exclude_none=True)
        )
      if event.content:
        if any(p.function_call for p in event.content.parts or []):
          raise RuntimeError("Experiment agents cannot invoke tools")
        texts.extend(
          p.text for p in event.content.parts or [] if p.text and not p.thought
        )

  try:
    await asyncio.wait_for(consume(), timeout=timeout)
    if not texts:
      raise RuntimeError("Agent returned no text")
    # A Responses turn can contain commentary followed by a final message.
    # Each message is already assembled by the adapter; parse only the last.
    output = texts[-1]
    (directory / "response.txt").write_text(output)
    return output
  finally:
    write_json(directory / "usage.json", usages)
    await model.client.close()
    await runner.close()


def parse_edits(text):
  text = text.strip()
  if text.startswith("```json") and text.endswith("```"):
    text = text[7:-3].strip()
  value = json.loads(text)
  if not isinstance(value, dict) or set(value) != {"edits"}:
    raise ValueError('Return exactly {"edits": [{"old": "...", "new": "..."}]}')
  return value["edits"]


def context(task, source, feedback):
  repo = Path(task["repo"])
  knowledge = []
  for name in [
    "context/programming.md",
    f"docs/kernels/{task['kernel']}.md",
    "pallas_kernels/kernels/gqa_common.py",
  ]:
    path = repo / name
    if path.exists():
      knowledge.append(f"\n{name}\n{path.read_text()}")
  # Reuse ADK's actual hardware table; the device reported by evaluation is
  # authoritative. Never infer a TPU generation from the example name.
  specs = Path(__file__).resolve().parents[2] / "auto_agent/tpu_specs.json"
  knowledge.append(
    "\nHardware table (use only matching device):\n" + specs.read_text()
  )
  return (
    f"Kernel: {task['kernel']}\nFrozen workload: {json.dumps(task['config'])}\n"
    "Budget: two structural rounds, few repairs, one TPU worker. "
    "The config is a correctness workload; do not assume compute/bandwidth-bound. "
    "gqa_decode uses contiguous KV, and block_kv does not shrink its full KV DMA window. "
    "For gqa_bwd, O/LSE are fixed inputs and the objective is full dQ+dK+dV latency.\n"
    f"Measured feedback (host timing is only a screen):\n{json.dumps(feedback)}\n"
    + "\n".join(knowledge)
    + f"\nCurrent editable module:\n```python\n{source}\n```\n"
  )
