from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ascendc_multi_turn.context_selector import ContextSelector
from ascendc_multi_turn.models import EvalResult
from ascendc_multi_turn.prompts import render_stage_context
from ascendc_multi_turn.skill_adapter import SkillAdapter
from ascendc_multi_turn.structured_knowledge.schema import (
    KnowledgeBundle,
    KnowledgeContext,
)


def _bundle(api: str = "") -> KnowledgeBundle:
    context = KnowledgeContext("op", "debug", "8.5.2", "8.5.2", "test", [])
    if not api:
        return KnowledgeBundle(context=context)
    return KnowledgeBundle(
        context=context,
        api_semantics=[
            {"card_id": f"card:{api}", "api": api, "fact_ids": [f"fact:{api}"], "examples": []},
            {"card_id": "card:Unrelated", "api": "Unrelated", "fact_ids": [], "examples": []},
        ],
        relevant_facts=[
            {
                "fact_id": f"fact:{api}",
                "subject": api,
                "predicate": "signature",
                "value": "installed declaration",
                "applicability": {"api": api},
            }
        ],
    )


class KnowledgeRoutingTrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        capsules = self.root / "knowledge_capsules"
        capsules.mkdir()
        (capsules / "facts.md").write_text(
            "# API\nReduceSum needs the installed workspace overload.\n\n"
            "# Host\nUse NPU tensor and stream facts. Never use CUDA.\n\n"
            "# Permute\nUse output-to-input coordinates and contiguous strides.\n",
            encoding="utf-8",
        )
        mapping = {
            "schema_version": 2,
            "default_source_root": str(self.root / "missing-cannbot"),
            "audience_budgets": {"planner": 6000, "generator": 8000},
            "operator_families": {"reduction": ["layernorm"], "conversion": ["permute"], "elementwise": ["gelu"]},
            "stages": {
                "compile_debug": {"skills": [self._skill("api_compile_debug", "API")]},
                "host_integration_debug": {"skills": [self._skill("host_integration_debug", "Host")]},
                "kernel_design": {"skills": [self._skill("kernel_design", "Permute")]},
                "precision_debug": {"skills": [self._skill("precision_debug", "API")]},
                "code_generation": {"skills": []},
            },
        }
        mapping_path = self.root / "mapping.yaml"
        mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
        self.selector = ContextSelector(SkillAdapter(mapping_path=mapping_path))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _skill(skill_id: str, heading: str) -> dict:
        return {
            "id": skill_id,
            "origin": "adapter",
            "source": ".",
            "purpose": skill_id,
            "expected_artifact": "one grounded repair",
            "knowledge_type": ["regression"],
            "when": {"always": True},
            "provide": [],
            "exclude": [],
            "references": [{"path": "facts.md", "headings": [heading], "max_chars": 1200}],
        }

    @staticmethod
    def _runtime(symbol: str, text: str) -> SimpleNamespace:
        return SimpleNamespace(
            text=text,
            facts=[{"fact_id": f"runtime:{symbol}", "symbol": symbol, "confidence_level": 2}],
            environment_fingerprint={"fingerprint_id": "fixture"},
        )

    def test_reduce_sum_overload_failure_selects_failure_and_source_api_facts(self) -> None:
        previous = EvalResult(
            False,
            False,
            error="build failed",
            failure_stage="ascendc_build",
            compile_output="error: no matching function for call to 'ReduceSum'",
            structured_failure={"related_symbols": ["ReduceSum"]},
        )
        request = self.selector.request(
            audience="generator", workflow_phase="bootstrap", operator="layernorm",
            soc="test", runtime_version="8.5.2", knowledge_version="8.5.2",
            current_exists=True, previous=previous,
            evidence=previous.compile_output + " GlobalTensor(x)",
            source_evidence="ReduceSum(dst, src, work, count); GlobalTensor<float> x;",
            planned_evidence='{"change":"use ReduceSum"}',
        )
        selected, _ = self.selector.select(
            bundle=_bundle("ReduceSum"), request=request,
            runtime_facts=self._runtime("ReduceSum", "Installed ReduceSum declaration requires workspace"),
        )
        rendered = render_stage_context(selected)

        self.assertEqual(request.primary_skill, "api_compile_debug")
        self.assertIn("ReduceSum", request.failure_symbols)
        self.assertIn("GlobalTensor", request.source_symbols)
        self.assertIn("card:ReduceSum", selected.selection_metadata["selected_structured_ids"])
        reduce_item = next(
            item for item in selected.selection_metadata["selected_items"]
            if item["id"] == "card:ReduceSum"
        )
        self.assertIn("failure", reduce_item["symbol_types"])
        self.assertIn("source", reduce_item["symbol_types"])
        self.assertIn("requires workspace", rendered)

    def test_exact_api_cards_follow_failure_source_planned_priority(self) -> None:
        context = KnowledgeContext("op", "debug", "8.5.2", "8.5.2", "test", [])
        bundle = KnowledgeBundle(
            context=context,
            api_semantics=[
                {"card_id": f"card:{api}", "api": api, "fact_ids": [f"fact:{api}"]}
                for api in ("Cast", "Tanh", "ReduceSum")
            ],
            relevant_facts=[
                {
                    "fact_id": f"fact:{api}", "subject": api,
                    "applicability": {"api": api},
                }
                for api in ("Cast", "Tanh", "ReduceSum")
            ],
        )
        previous = EvalResult(
            False, False, error="build failed", failure_stage="ascendc_build",
            compile_output="error: no matching function for call to 'ReduceSum'",
        )
        request = self.selector.request(
            audience="generator", workflow_phase="bootstrap", operator="layernorm",
            soc="test", runtime_version="8.5.2", knowledge_version="8.5.2",
            current_exists=True, previous=previous, evidence=previous.compile_output,
            source_evidence="Tanh(dst, src, work, count);",
            planned_evidence='{"change":"Cast(dst, src, mode, count)"}',
        )
        selected, _ = self.selector.select(bundle=bundle, request=request)

        self.assertEqual(
            selected.selection_metadata["selected_structured_ids"][:3],
            ["fact:ReduceSum", "fact:Tanh", "fact:Cast"],
        )

    def test_cuda_binding_contamination_selects_host_capsule(self) -> None:
        previous = EvalResult(
            False, False, error="Input must be a CUDA tensor",
            failure_stage="ascendc_build",
            compile_output="x.is_cuda(); c10::cuda::getCurrentCUDAStream();",
        )
        request = self.selector.request(
            audience="generator", workflow_phase="bootstrap", operator="gelu",
            soc="test", runtime_version="8.5.2", knowledge_version="8.5.2",
            current_exists=True, previous=previous, evidence=previous.compile_output,
            source_evidence=previous.compile_output,
        )
        selected, _ = self.selector.select(
            bundle=_bundle(), request=request,
            runtime_facts=self._runtime("getCurrentNPUStream", "Installed NPU tensor and stream declarations"),
        )
        rendered = render_stage_context(selected)

        self.assertEqual(request.primary_skill, "host_integration_debug")
        self.assertIn("host_integration_debug", selected.selection_metadata["selected_skill_ids"])
        self.assertIn("Never use CUDA", rendered)
        self.assertNotIn("card:Unrelated", selected.selection_metadata["selected_structured_ids"])

    def test_unrelated_undefined_symbol_is_not_assumed_to_be_host_abi(self) -> None:
        previous = EvalResult(
            False, False, error="undefined symbol: CustomKernelHelper",
            failure_stage="ascendc_build", compile_output="undefined symbol: CustomKernelHelper",
        )
        request = self.selector.request(
            audience="generator", workflow_phase="bootstrap", operator="gelu",
            soc="test", runtime_version="8.5.2", knowledge_version="8.5.2",
            current_exists=True, previous=previous, evidence=previous.compile_output,
            source_evidence="CustomKernelHelper();",
        )

        self.assertEqual(request.primary_skill, "api_compile_debug")

    def test_valid_npu_host_source_does_not_contaminate_kernel_compile_route(self) -> None:
        previous = EvalResult(
            False,
            False,
            error="AscendC build failed",
            failure_stage="ascendc_build",
            compile_output="kernel/add.cpp:91:13: error: no matching function for call to 'Muls'",
        )
        request = self.selector.request(
            audience="generator",
            workflow_phase="bootstrap",
            operator="add",
            soc="test",
            runtime_version="8.5.2",
            knowledge_version="8.5.2",
            current_exists=True,
            previous=previous,
            evidence=(
                previous.compile_output
                + " c10_npu::getCurrentNPUStream(); aclrtStream stream;"
            ),
            source_evidence=(
                "auto stream = c10_npu::getCurrentNPUStream(); "
                "aclrtStream raw = stream.stream(); Muls(dst, src, alpha, count);"
            ),
        )

        self.assertEqual(request.primary_skill, "api_compile_debug")
        self.assertEqual(request.route_evidence_origin, "failure_evidence")
        self.assertIn("Muls", request.failure_symbols)

    def test_permute_broad_mismatch_routes_to_kernel_design_only(self) -> None:
        verification = "Comparison\ncase[0]: max_abs_diff=9.0 passed=False\ncase[148]: mismatch"
        previous = EvalResult(
            True, False, error="correctness verification failed",
            failure_stage="correctness", verify_output=verification,
            structured_failure={"subsystem": "RUNTIME", "related_symbols": []},
        )
        request = self.selector.request(
            audience="generator", workflow_phase="bootstrap", operator="permute",
            soc="test", runtime_version="8.5.2", knowledge_version="8.5.2",
            current_exists=True, previous=previous, evidence=verification,
            source_evidence="GlobalTensor<float> input;",
        )
        selected, _ = self.selector.select(bundle=_bundle(), request=request)
        rendered = render_stage_context(selected)

        self.assertEqual(request.primary_skill, "kernel_design")
        self.assertIsNone(request.secondary_skill)
        self.assertNotIn("precision_debug", request.stages)
        self.assertIn("output-to-input coordinates", rendered)


if __name__ == "__main__":
    unittest.main()
