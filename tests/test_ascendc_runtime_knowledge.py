from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ascendc_multi_turn.knowledge_probe import (
    ProbeSpec,
    collect_environment_fingerprint,
    load_verified_probe_facts,
    run_compile_probe,
    run_kernel_api_probe,
    write_probe_manifest,
)
from ascendc_multi_turn.runtime_knowledge import collect_runtime_facts


class RuntimeKnowledgeGroundingTests(unittest.TestCase):
    def _header_root(self, root: Path) -> Path:
        include = root / "aarch64-linux/asc/include/basic_api"
        include.mkdir(parents=True)
        return include

    def test_installed_declaration_is_level_two_and_bound_to_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = self._header_root(root)
            (include / "reduce.h").write_text(
                "template <typename T>\n"
                "__aicore__ inline void ReduceSum(const LocalTensor<T>& dst,\n"
                "    const LocalTensor<T>& src, const LocalTensor<uint8_t>& work,\n"
                "    uint32_t count);\n",
                encoding="utf-8",
            )
            with patch("ascendc_multi_turn.runtime_knowledge._cann_root", return_value=root), patch(
                "ascendc_multi_turn.runtime_knowledge.torch_npu_include_roots", return_value=[]
            ):
                facts = collect_runtime_facts(
                    ["ReduceSum"], runtime_version="8.5.2", soc_version="Ascend910B3"
                )

        self.assertIn("LocalTensor<uint8_t>& work", facts.text)
        self.assertIn('#include "kernel_operator.h"', facts.text)
        self.assertEqual(facts.facts[0]["confidence_level"], 2)
        self.assertTrue(facts.environment_fingerprint["fingerprint_id"])

    def test_declaration_ranking_rejects_comment_hits_and_keeps_count_overload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = self._header_root(root)
            (include / "ops.h").write_text(
                "/* Formula: Tanh(x) */\n"
                "__aicore__ inline void Tanh(const LocalTensor<float>& dst, "
                "const LocalTensor<float>& src, const LocalTensor<uint8_t>& work);\n"
                "template <typename T>\n"
                "__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, "
                "const T& scalar, uint64_t mask, uint8_t repeats, const Params& params);\n"
                "template <typename T>\n"
                "__aicore__ inline void Muls(const LocalTensor<T>& dst, const LocalTensor<T>& src, "
                "const T& scalar, const int32_t& count);\n",
                encoding="utf-8",
            )
            with patch("ascendc_multi_turn.runtime_knowledge._cann_root", return_value=root), patch(
                "ascendc_multi_turn.runtime_knowledge.torch_npu_include_roots", return_value=[]
            ):
                facts = collect_runtime_facts(
                    ["Tanh", "Muls"], runtime_version="8.5.2", max_chars=8000
                )

        self.assertNotIn("Formula: Tanh", facts.text)
        self.assertIn("const int32_t& count", facts.text)

    def test_render_budget_does_not_downgrade_later_installed_facts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = self._header_root(root)
            (include / "ops.h").write_text(
                "void Add();\nvoid Mul();\n", encoding="utf-8"
            )
            with patch("ascendc_multi_turn.runtime_knowledge._cann_root", return_value=root), patch(
                "ascendc_multi_turn.runtime_knowledge.torch_npu_include_roots", return_value=[]
            ):
                facts = collect_runtime_facts(
                    ["Add", "Mul"], runtime_version="8.5.2", max_chars=200
                )

        self.assertEqual(
            {item["fact_id"]: item["confidence_level"] for item in facts.facts},
            {"runtime:Add": 2, "runtime:Mul": 2},
        )

    def test_full_selected_runtime_facts_have_no_character_or_symbol_cap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = self._header_root(root)
            declarations = []
            symbols = []
            for index in range(30):
                symbol = f"Api{index}"
                symbols.append(symbol)
                declarations.append(f"void {symbol}(int value_{index});")
            marker = "CompleteTailApi"
            symbols.append(marker)
            declarations.append(f"void {marker}(int complete_tail_parameter);")
            (include / "all_ops.h").write_text("\n".join(declarations), encoding="utf-8")
            with patch("ascendc_multi_turn.runtime_knowledge._cann_root", return_value=root), patch(
                "ascendc_multi_turn.runtime_knowledge.torch_npu_include_roots", return_value=[]
            ):
                facts = collect_runtime_facts(
                    symbols,
                    runtime_version="8.5.2",
                    max_chars=None,
                )

        self.assertEqual(len(facts.symbols), len(symbols))
        self.assertIn("complete_tail_parameter", facts.text)

    def test_matching_probe_promotes_fact_but_stale_probe_does_not(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            include = self._header_root(root)
            (include / "add.h").write_text("void Add();\n", encoding="utf-8")
            with patch("ascendc_multi_turn.runtime_knowledge._cann_root", return_value=root), patch(
                "ascendc_multi_turn.runtime_knowledge.torch_npu_include_roots", return_value=[]
            ):
                initial = collect_runtime_facts(["Add"], runtime_version="8.5.2")
                manifest = root / "manifest.json"
                manifest.write_text(
                    json.dumps(
                        {
                            "fingerprint_id": initial.environment_fingerprint["fingerprint_id"],
                            "probes": [{"probe_id": "add", "success": True, "fact_ids": ["runtime:Add"]}],
                        }
                    ),
                    encoding="utf-8",
                )
                verified = collect_runtime_facts(
                    ["Add"], runtime_version="8.5.2", probe_manifest=manifest
                )
                stale_payload = json.loads(manifest.read_text())
                stale_payload["fingerprint_id"] = "stale"
                manifest.write_text(json.dumps(stale_payload), encoding="utf-8")
                stale = collect_runtime_facts(
                    ["Add"], runtime_version="8.5.2", probe_manifest=manifest
                )

        self.assertEqual(verified.facts[0]["confidence_level"], 3)
        self.assertEqual(stale.facts[0]["confidence_level"], 2)

    def test_runtime_label_does_not_override_resolved_toolkit_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cann-8.5.2"
            include = self._header_root(root)
            (include / "add.h").write_text("void Add();\n", encoding="utf-8")
            with patch("ascendc_multi_turn.runtime_knowledge._cann_root", return_value=root), patch(
                "ascendc_multi_turn.runtime_knowledge.torch_npu_include_roots", return_value=[]
            ):
                from_850 = collect_runtime_facts(
                    ["Add"], runtime_version="8.5.0"
                )
                from_852 = collect_runtime_facts(
                    ["Add"], runtime_version="8.5.2"
                )

        self.assertEqual(
            from_850.environment_fingerprint["fingerprint_id"],
            from_852.environment_fingerprint["fingerprint_id"],
        )
        self.assertEqual(from_850.environment_fingerprint["cann_version"], "8.5.2")

    def test_missing_symbol_is_negative_only_with_direct_compiler_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._header_root(root)
            with patch("ascendc_multi_turn.runtime_knowledge._cann_root", return_value=root), patch(
                "ascendc_multi_turn.runtime_knowledge.torch_npu_include_roots", return_value=[]
            ):
                uncertain = collect_runtime_facts(
                    ["sqrtf"], runtime_version="8.5.2"
                )
                rejected = collect_runtime_facts(
                    ["sqrtf"], runtime_version="8.5.2",
                    failure_evidence="error: 'sqrtf' was not declared in this scope",
                )

        self.assertEqual(uncertain.facts[0]["confidence_level"], 0)
        self.assertNotIn("Do not use it", uncertain.text)
        self.assertEqual(rejected.facts[0]["confidence_level"], 2)
        self.assertIn("Do not use it", rejected.text)

    def test_compile_probe_is_auditable_and_fingerprint_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fingerprint = collect_environment_fingerprint(
                runtime_version="test",
                soc_version="test",
                toolkit_root=None,
                include_roots=[],
            )
            result = run_compile_probe(
                ProbeSpec(
                    probe_id="syntax",
                    source="int main() { return 0; }\n",
                    command=("c++", "{source}", "-o", "{output}"),
                    fact_ids=("runtime:test",),
                ),
                output_dir=root,
                fingerprint=fingerprint,
            )
            manifest = root / "manifest.json"
            write_probe_manifest(manifest, fingerprint=fingerprint, results=[result])

            loaded = load_verified_probe_facts(
                manifest, fingerprint_id=fingerprint.fingerprint_id
            )
            stale = load_verified_probe_facts(manifest, fingerprint_id="other")
            payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertTrue(result.success)
        self.assertIn("runtime:test", loaded)
        self.assertEqual(stale, {})
        self.assertEqual(payload["environment_fingerprint"]["fingerprint_id"], fingerprint.fingerprint_id)

    def test_kernel_api_probe_only_verifies_facts_after_successful_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fingerprint = collect_environment_fingerprint(
                runtime_version="test", soc_version="test", toolkit_root=None,
                include_roots=[],
            )
            with patch(
                "ascendc_multi_turn.knowledge_probe.subprocess.run",
                side_effect=[
                    SimpleNamespace(returncode=0, stdout="configured"),
                    SimpleNamespace(returncode=0, stdout="built"),
                ],
            ):
                result = run_kernel_api_probe(
                    output_dir=root,
                    fingerprint=fingerprint,
                    toolkit_root=root / "cann",
                    soc_version="test",
                )

        self.assertTrue(result.success)
        self.assertIn("runtime:ReduceSum", result.fact_ids)
        self.assertIn("runtime:Tanh", result.fact_ids)


if __name__ == "__main__":
    unittest.main()
