import os

LLM_PROVIDER = os.environ.get("MAXKERNEL_LLM_PROVIDER", "gemini").lower()
if LLM_PROVIDER not in {"gemini", "openai"}:
  raise ValueError("MAXKERNEL_LLM_PROVIDER must be gemini or openai")
# Keep the fork's explicit setting ahead of the upstream MODEL_NAME alias.
MODEL_NAME = (
  os.environ.get("MAXKERNEL_MODEL")
  or os.environ.get("MODEL_NAME")
  or ("gemini-3.1-pro-preview" if LLM_PROVIDER == "gemini" else "")
)
if not MODEL_NAME:
  raise ValueError(
    "Set MAXKERNEL_MODEL or MODEL_NAME to an available OpenAI Responses model ID"
  )
MAX_ITERATIONS = 5
LLM_GEN_RETRY_COUNT = 3
TEMPERATURE = 0.5
TOP_P = 0.9
TOP_K = 5
REQUEST_TIMEOUT = 3600 * 3
TPU_SERVER_PORT = 5463
CPU_SERVER_PORT = 5464
EVAL_SERVER_PORT = 1245
EVENTS_COMPACTION = True
