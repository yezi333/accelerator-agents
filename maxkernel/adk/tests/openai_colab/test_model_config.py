"""Model configuration compatibility after merging upstream MODEL_NAME support."""

import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


ADK_ROOT = Path(__file__).resolve().parents[2]


class ModelConfigTests(unittest.TestCase):
  def load_constants(self, agent, env):
    with patch.dict(os.environ, env, clear=True):
      return runpy.run_path(str(ADK_ROOT / agent / "constants.py"))

  def test_auto_model_selection(self):
    cases = [
      ({}, "gemini-3.1-pro-preview"),
      ({"MODEL_NAME": "custom-gemini"}, "custom-gemini"),
      (
        {"MODEL_NAME": "", "MAXKERNEL_MODEL": ""},
        "gemini-3.1-pro-preview",
      ),
      (
        {"MAXKERNEL_LLM_PROVIDER": "openai", "MODEL_NAME": "custom-openai"},
        "custom-openai",
      ),
      (
        {"MAXKERNEL_LLM_PROVIDER": "openai", "MAXKERNEL_MODEL": "fork-model"},
        "fork-model",
      ),
      (
        {
          "MAXKERNEL_LLM_PROVIDER": "openai",
          "MAXKERNEL_MODEL": "fork-model",
          "MODEL_NAME": "upstream-model",
        },
        "fork-model",
      ),
      ({"MAXKERNEL_MODEL": "", "MODEL_NAME": "upstream-model"}, "upstream-model"),
    ]
    for env, expected in cases:
      with self.subTest(env=env):
        self.assertEqual(
          self.load_constants("auto_agent", env)["MODEL_NAME"], expected
        )

  def test_openai_requires_explicit_model(self):
    for model_env in ({}, {"MAXKERNEL_MODEL": "", "MODEL_NAME": ""}):
      with self.subTest(model_env=model_env):
        with self.assertRaisesRegex(
          ValueError, "available OpenAI Responses model"
        ):
          self.load_constants(
            "auto_agent", {"MAXKERNEL_LLM_PROVIDER": "openai", **model_env}
          )

  def test_invalid_provider_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "must be gemini or openai"):
      self.load_constants("auto_agent", {"MAXKERNEL_LLM_PROVIDER": "invalid"})

  def test_hitl_keeps_upstream_model_setting(self):
    for env, expected in [
      ({}, "gemini-3.1-pro-preview"),
      ({"MODEL_NAME": "custom-gemini"}, "custom-gemini"),
      ({"MODEL_NAME": ""}, "gemini-3.1-pro-preview"),
    ]:
      with self.subTest(env=env):
        self.assertEqual(
          self.load_constants("hitl_agent", env)["MODEL_NAME"], expected
        )
