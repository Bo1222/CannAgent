from __future__ import annotations

import unittest
from types import SimpleNamespace

from ascendc_multi_turn.diagnostics import compact_evaluation, parse_structured_failure
from ascendc_multi_turn.knowledge_v2.router import KnowledgeRouterV2
from ascendc_multi_turn.knowledge_v2.schema import KnowledgeContext
from ascendc_multi_turn.models import EvalResult


class StructuredFailureTests(unittest.TestCase):
    def test_mte_illegal_configuration_is_structured(self) -> None:
        failure = parse_structured_failure(
            stage="correctness",
            output=(
                "AICORE exception: MTE illegal configuration, ACL_ERROR_RT_AICORE_OVER_FLOW\n"
                "core id: 7, block id: 3, sub_error_type: load alignment\n"
                "AscendC::DataCopyPad(dst, src, params)"
            ),
            case_info={"dtype": "float16"},
        )

        self.assertEqual(failure.subsystem, "MTE")
        self.assertEqual(failure.reason, "illegal configuration")
        self.assertEqual(failure.core_id, 7)
        self.assertEqual(failure.block_id, 3)
        self.assertEqual(failure.runtime_code, "ACL_ERROR_RT_AICORE_OVER_FLOW")
        self.assertIn("DataCopyPad", failure.related_symbols)
        self.assertEqual(failure.case_info["dtype"], "float16")

    def test_compile_and_acl_failures_are_classified(self) -> None:
        compile_failure = parse_structured_failure(
            stage="ascendc_build", output="kernel.cpp:7: error: no matching function for DataCopy"
        )
        acl_failure = parse_structured_failure(
            stage="acl_runtime", output="ACL_ERROR_RT_PARAM_INVALID"
        )
        self.assertEqual(compile_failure.subsystem, "COMPILER")
        self.assertIn("DataCopy", compile_failure.related_symbols)
        self.assertEqual(acl_failure.subsystem, "ACL")

    def test_compact_llm_evidence_uses_structure_not_local_details_path(self) -> None:
        structured = parse_structured_failure(
            stage="correctness", output="MTE illegal configuration"
        ).to_dict()
        result = EvalResult(
            False,
            False,
            error="correctness failed",
            failure_stage="correctness",
            error_excerpt="MTE illegal configuration",
            details_path="/private/local/correctness.log",
            structured_failure=structured,
        )

        compact = compact_evaluation(result)

        self.assertEqual(compact["structured_failure"]["subsystem"], "MTE")
        self.assertNotIn("details_path", compact)

    def test_structured_failure_retrieves_matching_failure_card(self) -> None:
        snapshot = SimpleNamespace(
            symbols={},
            api_cards=[],
            facts_by_id={},
            failure_cards=[
                {
                    "card_id": "failure_mte_illegal",
                    "signals": ["MTE", "illegal configuration"],
                    "subsystem": "MTE",
                    "inspection_targets": ["resolved memory-transfer calls"],
                }
            ],
            pattern_cards=[],
            project_contracts=[],
        )
        failure = parse_structured_failure(
            stage="correctness", output="MTE illegal configuration"
        ).to_dict()
        context = KnowledgeContext(
            operator="copy",
            phase="diagnose",
            runtime_version="8.5.0",
            knowledge_version="8.5.0",
            soc="Ascend910B3",
            source_symbols=[],
            failure=failure,
        )

        bundle = KnowledgeRouterV2(snapshot).route(context)

        self.assertEqual(bundle.failure_cards[0]["card_id"], "failure_mte_illegal")


if __name__ == "__main__":
    unittest.main()
