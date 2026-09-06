from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ascendc_multi_turn.knowledge import (
    MANIFEST_PATH,
    KnowledgeSelection,
    KnowledgeVersion,
    candidate_doc_ids,
    detect_cann_version,
    load_knowledge_state,
    parse_knowledge_selection,
    render_knowledge,
    select_knowledge_version,
)
from ascendc_multi_turn.llm import OpenAICompatibleProvider
from ascendc_multi_turn.models import LLMCallConfig, RunConfig
from ascendc_multi_turn.runtime_knowledge import collect_runtime_facts


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
        self.assertIn("AscendC Direct-LLM Core Rules", rendered)
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

    def test_manifest_uses_unique_ids_even_for_duplicate_api_names(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        documents = manifest["documents"]
        ids = [item["doc_id"] for item in documents]
        names = [item["name"] for item in documents]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreater(names.count("Exp"), 1)

    def test_doc_id_selects_one_duplicate_page_unambiguously(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        exp_docs = [item for item in manifest["documents"] if item["name"] == "Exp"]
        response = json.dumps(
            {
                "skill": "ascendc-translator",
                "topics": ["vector"],
                "doc_ids": [exp_docs[-1]["doc_id"]],
                "supplements": [],
                "reason": "specific overload family",
            }
        )
        selection = parse_knowledge_selection(
            response, version=self.version, max_api_docs=1, fallback_text=""
        )
        self.assertEqual(selection.doc_ids, [exp_docs[-1]["doc_id"]])
        self.assertEqual(selection.selected_files, [exp_docs[-1]["path"]])

    def test_targeted_candidate_finds_datacopypad_from_compiler_symbols(self) -> None:
        ranked = candidate_doc_ids(
            "error: no member paddingValue in DataCopyPadExtParams; DataCopyPad(dst, src)",
            version=self.version,
            limit=5,
        )
        self.assertIn("atlasascendc_api_07_0265", [doc_id for doc_id, _score in ranked])

    def test_renderer_is_bounded_and_records_only_rendered_ids(self) -> None:
        selection = parse_knowledge_selection(
            json.dumps(
                {
                    "skill": "ascendc-translator",
                    "topics": [],
                    "doc_ids": ["atlasascendc_api_07_0265", "atlasascendc_api_07_0032"],
                    "supplements": [],
                    "reason": "test",
                }
            ),
            version=self.version,
            max_api_docs=2,
            fallback_text="",
        )
        rendered = render_knowledge(
            selection,
            version=self.version,
            max_chars=5000,
            runtime_facts="struct DataCopyPadExtParams { T paddingValue; };",
            conflicts=["runtime header wins"],
        )
        self.assertLessEqual(len(rendered), 5000)
        self.assertLessEqual(len(selection.rendered_doc_ids), 2)
        self.assertIn("paddingValue", rendered)
        for doc_id in selection.rendered_doc_ids:
            self.assertIn(doc_id, rendered)

    def test_legacy_selection_migrates_to_task_state(self) -> None:
        legacy = {
            "selected_files": ["pages/atlasascendc_api_07_0265.md"],
            "supplements": ["dsl2Ascendc_compute_vector.md"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            state = load_knowledge_state(
                Path(temporary) / "knowledge_state.json",
                version=self.version,
                legacy_selection=legacy,
            )
        self.assertTrue(state.initialized)
        self.assertTrue(state.migrated)
        self.assertEqual(state.working_doc_ids, ["atlasascendc_api_07_0265"])


class RuntimeKnowledgeTests(unittest.TestCase):
    def test_installed_header_field_is_injected_as_authoritative_fact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = root / "aarch64-linux/asc/include/basic_api"
            include.mkdir(parents=True)
            (include / "datacopy.h").write_text(
                "template<class T> struct DataCopyPadExtParams {\n"
                "  bool isPad; uint8_t leftPadding; uint8_t rightPadding; T paddingValue;\n"
                "};\n",
                encoding="utf-8",
            )
            with patch("ascendc_multi_turn.runtime_knowledge._cann_root", return_value=root):
                facts = collect_runtime_facts(
                    ["DataCopyPadExtParams"], runtime_version="8.5.2"
                )
        self.assertIn("paddingValue", facts.text)
        self.assertTrue(any("padValue" in item for item in facts.conflicts))

    def test_python_fallback_uses_identifier_boundaries_and_preserves_priority(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = root / "aarch64-linux/asc/include/basic_api"
            include.mkdir(parents=True)
            (include / "addr.h").write_text("void Addr();\n", encoding="utf-8")
            (include / "add_intf.h").write_text("void Add();\n", encoding="utf-8")
            with patch("ascendc_multi_turn.runtime_knowledge._cann_root", return_value=root), patch(
                "ascendc_multi_turn.runtime_knowledge.shutil.which", return_value=None
            ):
                facts = collect_runtime_facts(["Add", "TPipe"], runtime_version="8.5.2")

        self.assertEqual(facts.symbols, ["Add", "TPipe"])
        self.assertIn("void Add()", facts.text)
        self.assertNotIn("void Addr()", facts.text)


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

    def test_deepseek_thinking_request_omits_temperature_and_captures_reasoning(self) -> None:
        provider = OpenAICompatibleProvider(
            model="deepseek-v4-flash",
            base_url="https://api.deepseek.com",
            api_key="test-key",
            provider="deepseek",
        )
        http_response = MagicMock()
        http_response.read.return_value = json.dumps(
            {
                "id": "request-id",
                "model": "deepseek-v4-flash-0731",
                "system_fingerprint": "fingerprint",
                "choices": [
                    {
                        "message": {"content": "", "reasoning_content": "reasoning only"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "total_tokens": 70,
                    "completion_tokens_details": {"reasoning_tokens": 64},
                },
            }
        ).encode("utf-8")
        context = MagicMock()
        context.__enter__.return_value = http_response
        context.__exit__.return_value = False

        with patch("ascendc_multi_turn.llm.urllib.request.urlopen", return_value=context) as urlopen:
            response = provider.generate(
                "test prompt",
                call_config=LLMCallConfig(
                    "generator", 65536, thinking="enabled", reasoning_effort="high"
                ),
            )

        request_body = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(request_body["thinking"], {"type": "enabled"})
        self.assertEqual(request_body["reasoning_effort"], "high")
        self.assertEqual(request_body["max_tokens"], 65536)
        self.assertEqual(request_body["response_format"], {"type": "json_object"})
        self.assertNotIn("temperature", request_body)
        self.assertEqual(response.content, "")
        self.assertEqual(response.reasoning_content, "reasoning only")
        self.assertEqual(response.model, "deepseek-v4-flash-0731")

    def test_run_config_uses_per_call_defaults_and_legacy_blanket_override(self) -> None:
        names = {
            "ASCENDC_LLM_MAX_TOKENS": "",
            "ASCENDC_ROUTER_MAX_TOKENS": "",
            "ASCENDC_GENERATOR_MAX_TOKENS": "",
            "ASCENDC_REPAIR_MAX_TOKENS": "",
        }
        with patch.dict(os.environ, names):
            config = RunConfig("model.py", "output", mock=True)
            legacy = RunConfig("model.py", "output", max_tokens=1234, mock=True)

        self.assertEqual(config.router_max_tokens, 4096)
        self.assertEqual(config.generator_max_tokens, 65536)
        self.assertEqual(config.repair_max_tokens, 65536)
        self.assertEqual(legacy.router_max_tokens, 1234)
        self.assertEqual(legacy.generator_max_tokens, 1234)


if __name__ == "__main__":
    unittest.main()
