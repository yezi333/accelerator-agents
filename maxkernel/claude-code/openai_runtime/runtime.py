"""Responses API conversations, isolated delegation and context compaction."""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .tools import SCHEMAS, NeedsInput


def save_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(path.name + ".tmp")
  temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
  temporary.replace(path)


@dataclass
class ModelConfig:
  model: str
  base_url: str = "https://api.openai.com/v1"
  timeout: float = 360
  max_retries: int = 2
  max_output_tokens: int = 8000
  reasoning_effort: str = "high"
  compaction_tokens: int = 32000
  max_turns: int = 500

  @classmethod
  def from_env(cls, model=None):
    config = cls(
      model=model
      or os.environ.get("MAXKERNEL_MODEL")
      or os.environ.get("MODEL_NAME", ""),
      base_url=os.environ.get(
        "OPENAI_BASE_URL", "https://api.openai.com/v1"
      ).rstrip("/"),
      timeout=float(os.environ.get("MAXKERNEL_API_TIMEOUT", "360")),
      max_retries=int(os.environ.get("MAXKERNEL_API_MAX_RETRIES", "2")),
      max_output_tokens=int(
        os.environ.get("MAXKERNEL_MAX_OUTPUT_TOKENS", "8000")
      ),
      reasoning_effort=os.environ.get("MAXKERNEL_REASONING_EFFORT", "high"),
      compaction_tokens=int(
        os.environ.get("MAXKERNEL_COMPACTION_TOKENS", "32000")
      ),
    )
    config.validate()
    return config

  def validate(self):
    if self.base_url.endswith("/responses"):
      self.base_url = self.base_url[: -len("/responses")]
    url = urlsplit(self.base_url)
    if (
      url.scheme not in {"http", "https"}
      or not url.hostname
      or url.username
      or url.password
      or url.query
      or url.fragment
      or url.path.endswith("/chat/completions")
    ):
      raise ValueError(
        "OPENAI_BASE_URL must be a Responses API base URL without credentials/query"
      )
    if not self.model.strip():
      raise ValueError("Set MAXKERNEL_MODEL (or MODEL_NAME), or use --model")
    if (
      min(
        self.timeout,
        self.max_output_tokens,
        self.compaction_tokens,
        self.max_turns,
      )
      <= 0
      or self.max_retries < 0
    ):
      raise ValueError(
        "Timeout, token limits and turn limit must be positive; retries must be non-negative"
      )

  def public(self):
    return asdict(self)


class ResponsesHost:
  def __init__(self, config, prompts, tools, client):
    self.config, self.prompts, self.tools, self.client = (
      config,
      prompts,
      tools,
      client,
    )
    self.directory = tools.run_dir / "openai_host"
    self.counter = 0
    self.events = self.directory / "events.jsonl"

  def event(self, kind, **fields):
    entry = {"time": time.time(), "event": kind, **fields}
    with self.events.open("a") as file:
      file.write(
        self.tools.redact(json.dumps(entry, ensure_ascii=False)) + "\n"
      )
    if kind in {
      "agent_start",
      "agent_end",
      "llm_start",
      "llm_end",
      "llm_error",
      "tool_start",
      "tool_end",
    }:
      info = fields.get("tool", fields.get("purpose", ""))
      elapsed = (
        f" {fields['elapsed_seconds']:.1f}s"
        if "elapsed_seconds" in fields
        else ""
      )
      print(f"[{fields.get('agent', '')}] {kind} {info}{elapsed}", flush=True)

  async def request(
    self, agent, inputs, instructions, schemas, purpose="generate"
  ):
    started = time.monotonic()
    self.event("llm_start", agent=agent, purpose=purpose)
    kwargs = dict(
      model=self.config.model,
      instructions=instructions,
      input=inputs,
      store=False,
      include=["reasoning.encrypted_content"],
      max_output_tokens=self.config.max_output_tokens,
    )
    if self.config.reasoning_effort:
      kwargs["reasoning"] = {"effort": self.config.reasoning_effort}
    if schemas:
      kwargs.update(tools=schemas, parallel_tool_calls=False)
    try:
      result = await self.client.responses.create(**kwargs)
      self.event(
        "llm_end",
        agent=agent,
        purpose=purpose,
        elapsed_seconds=time.monotonic() - started,
        status=result.status,
        response_id=result.id,
        request_id=getattr(result, "_request_id", None),
        usage=result.usage.model_dump() if result.usage else None,
      )
      if result.status != "completed":
        raise RuntimeError(
          f"Model response {result.status}: {result.incomplete_details or result.error}; no tools executed"
        )
      for item in result.output:
        if item.type == "message" and any(
          chunk.type == "refusal" for chunk in item.content
        ):
          raise RuntimeError("Model refused this request")
      return result
    except Exception as exc:
      self.event(
        "llm_error",
        agent=agent,
        purpose=purpose,
        elapsed_seconds=time.monotonic() - started,
        error_type=type(exc).__name__,
        error=str(exc),
      )
      raise

  async def compact(self, agent, seed, turns):
    # Keep complete turns, including each call and its matching result.
    old = [item for turn in turns[:-5] for item in turn]
    response = await self.request(
      agent,
      seed
      + old
      + [
        {
          "role": "user",
          "content": "Summarize the completed work above for continuation.",
        }
      ],
      "Write a factual continuation note for a MaxKernel agent. Preserve exact artifact paths, "
      "user constraints, workload/tolerances, approvals or pending questions, failed checks, "
      "retry counts, commands/results and remaining work. Never invent successful tests. "
      "State and artifacts on disk remain authoritative. Do not execute or propose tools now.",
      [],
      purpose="compact",
    )
    if not response.output_text.strip():
      raise RuntimeError("Context compaction returned no summary")
    self.event("context_compacted", agent=agent, turns_removed=len(turns) - 5)
    return [
      seed[0],
      {
        "role": "user",
        "content": "Earlier context summary (re-read state on disk):\n"
        + response.output_text,
      },
    ], turns[-5:]

  async def run_agent(self, name, task, *, parent=None):
    if name not in self.prompts.agents:
      raise ValueError(f"Unknown registered agent: {name}")
    self.counter += 1
    identifier = f"{self.counter:04d}_{name}_{uuid.uuid4().hex[:8]}"
    session_path = self.directory / "sessions" / f"{identifier}.json"
    allowed = self.prompts.allowed[name]
    schemas = [schema for schema in SCHEMAS if schema["name"] in allowed]
    seed = [{"role": "user", "content": task}]
    turns = []
    instructions = self.prompts.instructions(name)
    self.event("agent_start", agent=name, session=identifier, parent=parent)
    try:
      for _ in range(self.config.max_turns):
        response = await self.request(
          name,
          seed + [item for turn in turns for item in turn],
          instructions,
          schemas,
        )
        turn = [
          item.model_dump(mode="json", exclude_none=True)
          for item in response.output
        ]
        turns.append(turn)
        calls = [
          item for item in response.output if item.type == "function_call"
        ]
        if not calls:
          if not response.output_text.strip():
            raise RuntimeError(
              f"{name} returned neither tool calls nor final text"
            )
          self.event("agent_end", agent=name, session=identifier)
          return response.output_text
        for call in calls:
          args = None
          started = time.monotonic()
          self.event(
            "tool_start",
            agent=name,
            session=identifier,
            tool=call.name,
            call_id=call.call_id,
          )
          try:
            if call.name not in allowed:
              raise ValueError(f"Tool {call.name} is not allowed for {name}")
            args = json.loads(call.arguments)
            if not isinstance(args, dict):
              raise ValueError("Tool arguments must be an object")
            await self.tools.check_guard(call.name, args)
            if call.name == "AskUserQuestion":
              if not args.get("questions"):
                raise ValueError("Provide at least one question")
              raise NeedsInput(args["questions"], name)
            if call.name == "Agent":
              child = args["subagent_type"]
              if args.get("run_in_background"):
                raise ValueError("Agent calls must be synchronous")
              if (
                (name == "maxkernel" and child != "maxkernel-worker")
                or child in {"maxkernel", "maxkernel-worker"}
                and name != "maxkernel"
              ):
                raise ValueError(
                  "Dispatch must follow orchestrator -> worker -> specialist"
                )
              result = {
                "report": await self.run_agent(
                  child, args["prompt"], parent=identifier
                )
              }
            else:
              result = await self.tools.execute(call.name, args)
          except NeedsInput as exc:
            save_json(
              self.directory / "pending_question.json",
              {"agent": exc.agent, "questions": exc.questions},
            )
            raise
          except Exception as exc:
            result = {
              "error": self.tools.redact(str(exc)),
              "error_type": type(exc).__name__,
            }
          self.event(
            "tool_end",
            agent=name,
            session=identifier,
            tool=call.name,
            call_id=call.call_id,
            elapsed_seconds=time.monotonic() - started,
            arguments=args if args is not None else call.arguments,
            result=result,
          )
          turn.append(
            {
              "type": "function_call_output",
              "call_id": call.call_id,
              "output": self.tools.redact(
                json.dumps(result, ensure_ascii=False)
              ),
            }
          )
        save_json(
          session_path,
          json.loads(
            self.tools.redact(
              json.dumps({"agent": name, "seed": seed, "turns": turns})
            )
          ),
        )
        if (
          response.usage
          and response.usage.input_tokens >= self.config.compaction_tokens
          and len(turns) > 5
        ):
          seed, turns = await self.compact(name, seed, turns)
      raise RuntimeError(
        f"{name} exhausted the host turn limit ({self.config.max_turns}); pipeline is unfinished"
      )
    finally:
      save_json(
        session_path,
        json.loads(
          self.tools.redact(
            json.dumps({"agent": name, "seed": seed, "turns": turns})
          )
        ),
      )
