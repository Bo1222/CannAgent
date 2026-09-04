from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from llm_config import ENV_FILE, get_env, get_optional_env


class LLMConfigTests(unittest.TestCase):
    def test_repository_env_is_present(self) -> None:
        self.assertTrue(ENV_FILE.is_file())
        self.assertTrue((Path(__file__).resolve().parent.parent / ".env.example").is_file())

    def test_process_environment_has_precedence(self) -> None:
        with patch.dict(os.environ, {"TEST_LLM_MODEL": "runtime-model"}):
            self.assertEqual(get_env("TEST_LLM_MODEL", "fallback"), "runtime-model")

    def test_empty_value_uses_default_or_fails_clearly(self) -> None:
        with patch.dict(os.environ, {"TEST_EMPTY_LLM_VALUE": ""}):
            self.assertEqual(get_env("TEST_EMPTY_LLM_VALUE", "fallback"), "fallback")
            self.assertIsNone(get_optional_env("TEST_EMPTY_LLM_VALUE"))
            with self.assertRaisesRegex(RuntimeError, "Please create .env from .env.example"):
                get_env("TEST_EMPTY_LLM_VALUE")


if __name__ == "__main__":
    unittest.main()
