"""Text/function-call ADK backend using the OpenAI Responses API.

The SDK is optional and loaded only when this backend makes a request. Native
Gemini tools and multimodal inputs are deliberately rejected, not discarded.
"""

import json
import os
from functools import cached_property

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field

_REPLAY_PREFIX = b"maxkernel-openai-response-v1:"


def _schema(value):
  """Translate genai Schema objects, preserving property names and enums."""
  if hasattr(value, "model_dump"):
    value = value.model_dump(exclude_none=True, mode="json", by_alias=True)
  result = dict(value)
  result.pop("propertyOrdering", None)
  if isinstance(result.get("type"), str):
    result["type"] = result["type"].lower()
  for key in ("properties", "$defs"):
    if key in result:
      result[key] = {
        name: _schema(child) for name, child in result[key].items()
      }
  for key in ("items", "additionalProperties", "not"):
    if isinstance(result.get(key), dict):
      result[key] = _schema(result[key])
  for key in ("anyOf", "allOf", "oneOf"):
    if key in result:
      result[key] = [_schema(child) for child in result[key]]
  if result.pop("nullable", False):
    return {"anyOf": [result, {"type": "null"}]}
  return result


def _text(instruction):
  if instruction is None:
    return ""
  if isinstance(instruction, str):
    return instruction
  if isinstance(instruction, types.Content):
    instruction = instruction.parts or []
  if isinstance(instruction, types.Part):
    instruction = [instruction]
  if isinstance(instruction, list):
    return "\n".join(
      _text(part) if isinstance(part, str) else _part_text(part)
      for part in instruction
    )
  raise ValueError("OpenAI backend requires text system instructions")


def _part_text(part):
  if part.text is None:
    raise ValueError("OpenAI backend requires text system instructions")
  return part.text


def _inputs(contents):
  items = []
  for content in contents:
    for part in content.parts or []:
      if any(
        getattr(part, name, None) is not None
        for name in (
          "inline_data",
          "file_data",
          "executable_code",
          "code_execution_result",
        )
      ):
        raise ValueError(
          "OpenAI backend currently supports text and function calls only"
        )
      signature = part.thought_signature or b""
      if signature.startswith(_REPLAY_PREFIX) and content.role == "model":
        replay = json.loads(signature[len(_REPLAY_PREFIX) :])
        # The signature follows its part through ADK session serialization.
        # If a callback edits the visible content, do not replay stale output.
        visible = part.model_dump(
          mode="json", exclude_none=True, exclude={"thought_signature"}
        )
        if visible == replay["visible"]:
          items.extend(replay["items"])
          continue
      if part.function_call:
        call = part.function_call
        if not call.id:
          raise ValueError("OpenAI tool history requires function call IDs")
        items.append(
          {
            "type": "function_call",
            "call_id": call.id,
            "name": call.name,
            "arguments": json.dumps(call.args or {}),
          }
        )
      elif part.function_response:
        response = part.function_response
        if not response.id:
          raise ValueError("OpenAI tool history requires function response IDs")
        items.append(
          {
            "type": "function_call_output",
            "call_id": response.id,
            "output": json.dumps(response.response or {}),
          }
        )
      elif part.thought:
        continue
      elif part.text is not None:
        items.append(
          {
            "role": "assistant" if content.role == "model" else "user",
            "content": part.text,
          }
        )
      else:
        raise ValueError(
          "OpenAI backend currently supports text and function calls only"
        )
  return items or [{"role": "user", "content": "Follow the instructions."}]


class OpenAIResponses(BaseLlm):
  """Buffered responses: a single final ADK event, even in streaming mode."""

  max_output_tokens: int = Field(
    default_factory=lambda: int(
      os.environ.get("MAXKERNEL_MAX_OUTPUT_TOKENS", "16000")
    ),
    gt=0,
  )
  timeout_seconds: float = Field(
    default_factory=lambda: float(
      os.environ.get("MAXKERNEL_API_TIMEOUT", "180")
    ),
    gt=0,
  )
  max_retries: int = Field(default=2, ge=0)
  reasoning_effort: str | None = Field(
    default_factory=lambda: os.environ.get("MAXKERNEL_REASONING_EFFORT") or None
  )

  @cached_property
  def client(self):
    from openai import AsyncOpenAI

    return AsyncOpenAI(
      timeout=self.timeout_seconds, max_retries=self.max_retries
    )

  async def generate_content_async(self, llm_request, stream=False):
    config = llm_request.config
    if config.response_schema or config.response_json_schema:
      raise ValueError("OpenAI backend does not yet support ADK output_schema")
    declarations = []
    for tool in config.tools or []:
      if not tool.function_declarations:
        raise ValueError(
          "OpenAI backend supports function tools, not native Gemini tools"
        )
      for declaration in tool.function_declarations:
        params = declaration.parameters_json_schema
        if params is None:
          params = declaration.parameters or {
            "type": "object",
            "properties": {},
          }
        declarations.append(
          {
            "type": "function",
            "name": declaration.name,
            "description": declaration.description or "",
            "parameters": _schema(params),
            # ADK tool arguments may be optional; do not silently make them required.
            "strict": False,
          }
        )
    request = {
      "model": self.model,
      "instructions": _text(config.system_instruction),
      "input": _inputs(llm_request.contents),
      "store": False,
      "include": ["reasoning.encrypted_content"],
      "max_output_tokens": config.max_output_tokens or self.max_output_tokens,
    }
    if self.reasoning_effort:
      request["reasoning"] = {"effort": self.reasoning_effort}
    if declarations:
      request["tools"] = declarations
    tool_config = config.tool_config
    if tool_config and tool_config.function_calling_config:
      calling = tool_config.function_calling_config
      if calling.allowed_function_names:
        request["tools"] = [
          d for d in declarations if d["name"] in calling.allowed_function_names
        ]
      mode = getattr(calling.mode, "value", calling.mode)
      if mode in {"AUTO", "NONE", "ANY"}:
        request["tool_choice"] = {
          "AUTO": "auto",
          "NONE": "none",
          "ANY": "required",
        }[mode]
    response = await self.client.responses.create(**request)
    if response.status != "completed":
      raise RuntimeError(
        f"OpenAI response did not complete: {response.status}; "
        f"{response.incomplete_details or response.error}"
      )
    parts, pending = [], []
    for item in response.output:
      raw = item.model_dump(mode="json", exclude_none=True)
      if item.type == "reasoning":
        pending.append(raw)
        continue
      if item.type == "function_call":
        args = json.loads(item.arguments)
        if not isinstance(args, dict):
          raise ValueError("Function arguments must be a JSON object")
        part = types.Part(
          function_call=types.FunctionCall(
            id=item.call_id, name=item.name, args=args
          )
        )
      elif item.type == "message":
        chunks = []
        for chunk in item.content:
          if chunk.type == "refusal":
            raise RuntimeError(f"OpenAI refused the request: {chunk.refusal}")
          if chunk.type != "output_text":
            raise ValueError(
              f"Unsupported OpenAI message content: {chunk.type}"
            )
          chunks.append(chunk.text)
        part = types.Part(text="".join(chunks))
      else:
        raise ValueError(f"Unsupported OpenAI response item: {item.type}")
      visible = part.model_dump(mode="json", exclude_none=True)
      part.thought_signature = (
        _REPLAY_PREFIX
        + json.dumps(
          {
            "visible": visible,
            "items": pending + [raw],
          }
        ).encode()
      )
      pending = []
      parts.append(part)
    if pending or not parts:
      raise RuntimeError("OpenAI response has no complete visible output")
    usage = response.usage
    yield LlmResponse(
      content=types.Content(role="model", parts=parts),
      partial=False,
      usage_metadata=types.GenerateContentResponseUsageMetadata(
        prompt_token_count=usage.input_tokens,
        candidates_token_count=usage.output_tokens,
        total_token_count=usage.total_tokens,
        thoughts_token_count=usage.output_tokens_details.reasoning_tokens,
      )
      if usage
      else None,
    )
