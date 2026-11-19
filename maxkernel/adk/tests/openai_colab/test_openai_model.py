"""Exercise real SDK serialization and ADK tool execution without network/API keys."""

import json

import httpx
import pytest
from google.adk.agents import LlmAgent
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from openai import AsyncOpenAI

from auto_agent.openai_model import OpenAIResponses, _inputs, _schema


def response(output, status="completed"):
  return {
    "id": "resp_test",
    "object": "response",
    "created_at": 1,
    "model": "test-model",
    "status": status,
    "output": output,
    "usage": {
      "input_tokens": 10,
      "output_tokens": 5,
      "total_tokens": 15,
      "input_tokens_details": {"cached_tokens": 0},
      "output_tokens_details": {"reasoning_tokens": 2},
    },
  }


def message(text, phase="final_answer"):
  return {
    "type": "message",
    "id": "msg_test",
    "role": "assistant",
    "status": "completed",
    "phase": phase,
    "content": [{"type": "output_text", "text": text, "annotations": []}],
  }


def backend(handler):
  model = OpenAIResponses(model="test-model")
  model.__dict__["client"] = AsyncOpenAI(
    api_key="test-not-a-secret",
    max_retries=0,
    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
  )
  return model


@pytest.mark.asyncio
async def test_adk_tool_round_trip_preserves_reasoning_and_phase():
  requests, executions = [], []
  reasoning = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [],
    "encrypted_content": "opaque",
  }

  def handler(request):
    data = json.loads(request.content)
    requests.append(data)
    if len(requests) == 1:
      return httpx.Response(
        200,
        json=response(
          [
            reasoning,
            message("Checking.", "commentary"),
            {
              "type": "function_call",
              "id": "fc_1",
              "call_id": "call_1",
              "name": "measure",
              "arguments": '{"tile": 128}',
              "status": "completed",
            },
          ]
        ),
      )
    return httpx.Response(200, json=response([message("Measured 7 us.")]))

  def measure(tile: int) -> dict:
    """Measure a tile."""
    executions.append(tile)
    return {"time_us": 7}

  model = backend(handler)
  service = InMemorySessionService()
  session = await service.create_session(app_name="test", user_id="u")
  runner = Runner(
    app_name="test",
    session_service=service,
    agent=LlmAgent(name="tester", model=model, tools=[measure]),
  )
  try:
    events = [
      event
      async for event in runner.run_async(
        user_id="u",
        session_id=session.id,
        new_message=types.Content(
          role="user", parts=[types.Part(text="Measure tile 128")]
        ),
      )
    ]
    assert executions == [128]
    assert len(requests) == 2
    assert (
      requests[0]["tools"][0]["parameters"]["properties"]["tile"]["type"]
      == "integer"
    )
    items = requests[1]["input"]
    assert reasoning in items
    assert any(i.get("phase") == "commentary" for i in items)
    assert any(
      i.get("type") == "function_call_output"
      and i["call_id"] == "call_1"
      and json.loads(i["output"])["time_us"] == 7
      for i in items
    )
    assert any(
      e.usage_metadata and e.usage_metadata.total_token_count == 15
      for e in events
    )
  finally:
    await model.client.close()
    await runner.close()


@pytest.mark.asyncio
async def test_replay_survives_json_session_roundtrip_and_respects_edits():
  model = backend(
    lambda _: httpx.Response(200, json=response([message("original")]))
  )
  try:
    result = [r async for r in model.generate_content_async(LlmRequest())][0]
    content = types.Content.model_validate_json(
      result.content.model_dump_json()
    )
    assert _inputs([content])[0]["phase"] == "final_answer"
    content.parts[0].text = "edited"
    assert _inputs([content]) == [{"role": "assistant", "content": "edited"}]
  finally:
    await model.client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "failed"])
async def test_incomplete_response_is_not_a_successful_candidate(status):
  model = backend(
    lambda _: httpx.Response(200, json=response([message("partial")], status))
  )
  try:
    with pytest.raises(RuntimeError, match="did not complete"):
      _ = [r async for r in model.generate_content_async(LlmRequest())]
  finally:
    await model.client.close()


def test_schema_property_names_and_enum_values_are_not_lowercased():
  schema = types.Schema(
    type="OBJECT",
    properties={
      "type": types.Schema(type="STRING", enum=["VMEM", "HBM"]),
      "optional": types.Schema(type="INTEGER", nullable=True),
    },
  )
  result = _schema(schema)
  assert result["properties"]["type"]["enum"] == ["VMEM", "HBM"]
  assert result["properties"]["optional"]["anyOf"][1] == {"type": "null"}


@pytest.mark.asyncio
async def test_native_tools_fail_before_api_call():
  model = OpenAIResponses(model="test-model")
  request = LlmRequest(
    config=types.GenerateContentConfig(
      tools=[types.Tool(google_search=types.GoogleSearch())]
    )
  )
  with pytest.raises(ValueError, match="native Gemini"):
    _ = [r async for r in model.generate_content_async(request)]


def test_parallel_tool_results_keep_distinct_ids():
  content = types.Content(
    role="user",
    parts=[
      types.Part(
        function_response=types.FunctionResponse(
          id=f"call_{i}", name="measure", response={"value": i}
        )
      )
      for i in range(2)
    ],
  )
  items = _inputs([content])
  assert [i["call_id"] for i in items] == ["call_0", "call_1"]


@pytest.mark.asyncio
async def test_experiment_uses_final_message_not_commentary(
  tmp_path, monkeypatch
):
  from auto_agent import openai_model
  from experiments.colab.agents import ask, parse_edits
  from experiments.colab.core import DEFAULT_LIMITS, Budget

  model = backend(
    lambda _: httpx.Response(
      200,
      json=response(
        [
          message("I will change the loop.", "commentary"),
          message('{"edits": [{"old": "x=1", "new": "x=2"}]}'),
        ]
      ),
    )
  )
  monkeypatch.setenv("OPENAI_API_KEY", "test-only")
  monkeypatch.setenv("MAXKERNEL_MODEL", "test-model")
  monkeypatch.setattr(openai_model, "OpenAIResponses", lambda **kwargs: model)
  budget = Budget(tmp_path, DEFAULT_LIMITS)
  output = await ask(budget, "implement", "Return edits.")
  assert parse_edits(output) == [{"old": "x=1", "new": "x=2"}]
  assert budget.state["model_calls"] == 1
  usage = json.loads(
    (tmp_path / "model_calls/001_implement/usage.json").read_text()
  )
  assert usage[0]["total_token_count"] == 15
