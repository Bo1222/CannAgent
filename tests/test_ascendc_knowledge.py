from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ascendc_multi_turn.knowledge import (
    KnowledgeVersion,
    detect_cann_version,
    parse_knowledge_selection,
    render_knowledge,
    select_knowledge_version,
)
from ascendc_multi_turn.llm import OpenAICompatibleProvider


class KnowledgeVersionTests(unittest.TestCase):
    def test_exact_version(self) -> None:
        selected = select_knowledge_version("8.5.0")
        self.assertEqual(selected.status, "exact")
        self.assertEqual(selected.knowledge_version, "8.5.0")

    def test_same_major_fallback(self) -> None:
        selected = select_knowledge_version("8.6.0")
        self.assertEqual(selected.status, "same-major-fallback")
        self.assertEqual(selected.knowledge_version, "8.5.0")
        self.assertTrue(selected.warning)

    def test_cross_major_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Cross-major"):
            select_knowledge_version("9.0.0")

    def test_explicit_and_mock_detection(self) -> None:
        self.assertEqual(detect_cann_version("8.5", mock=False), "8.5.0")
        self.assertEqual(detect_cann_version("auto", mock=True), "8.5.0")


class KnowledgeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.version = KnowledgeVersion("8.5.0", "8.5.0", "exact")

    def test_allowlisted_api_and_supplement(self) -> None:
        response = json.dumps(
            {
                "skill": "ascendc-translator",
                "topics": ["vector"],
                "api_names": ["Relu", "DataCopy简介"],
                "supplements": ["dsl2Ascendc_compute_vector.md"],
                "reason": "elementwise implementation",
            }
        )
        selection = parse_knowledge_selection(
            response,
            version=self.version,
            max_api_docs=2,
            fallback_text="",
        )
        self.assertFalse(selection.fallback)
        self.assertLessEqual(len(selection.selected_files), 2)
        rendered = render_knowledge(selection, version=self.version, max_chars=60000)
        self.assertIn("Runtime CANN: 8.5.0", rendered)
        self.assertIn("AscendC Kernel 转译 Skill", rendered)
        self.assertLessEqual(len(rendered), 60100)

    def test_invalid_router_output_uses_deterministic_fallback(self) -> None:
        selection = parse_knowledge_selection(
            "not-json",
            version=self.version,
            max_api_docs=8,
            fallback_text="Use Relu and DataCopy for the candidate kernel",
        )
        self.assertTrue(selection.fallback)
        self.assertIn("Relu", selection.api_names)


class ProviderTests(unittest.TestCase):
    def _config(self, provider: str, base_url: str) -> SimpleNamespace:
        return SimpleNamespace(
            provider=provider,
            model="test-model",
            base_url=base_url,
            temperature=0.2,
            max_tokens=100,
            timeout=30,
        )

    def test_openai_provider_reads_only_openai_api_key(self) -> None:
        config = self._config("openai", "https://api.openai.com/v1")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-test-key", "DEEPSEEK_API_KEY": ""}):
            provider = OpenAICompatibleProvider.from_env(config)
        self.assertEqual(provider.api_key, "openai-test-key")
        self.assertEqual(provider.url, "https://api.openai.com/v1/chat/completions")

    def test_deepseek_provider_reads_only_deepseek_api_key(self) -> None:
        config = self._config("deepseek", "https://api.deepseek.com")
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "deepseek-test-key", "OPENAI_API_KEY": ""}):
            provider = OpenAICompatibleProvider.from_env(config)
        self.assertEqual(provider.api_key, "deepseek-test-key")
        self.assertEqual(provider.url, "https://api.deepseek.com/chat/completions")

    def test_provider_captures_finish_reason(self) -> None:
        provider = OpenAICompatibleProvider(
            model="test-model",
            base_url="https://example.invalid/v1",
            api_key="test-key",
        )
        http_response = MagicMock()
        http_response.read.return_value = json.dumps(
            {
                "id": "test-request",
                "model": "served-model",
                "choices": [
                    {
                        "message": {"content": "partial response"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"total_tokens": 10},
            }
        ).encode("utf-8")
        context = MagicMock()
        context.__enter__.return_value = http_response
        context.__exit__.return_value = False

        with patch("ascendc_multi_turn.llm.urllib.request.urlopen", return_value=context):
            response = provider.generate("test prompt")

        self.assertEqual(response.finish_reason, "length")
        self.assertEqual(response.model, "served-model")


if __name__ == "__main__":
    unittest.main()
