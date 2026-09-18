from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ascendc_multi_turn.bundle import semantic_bundle_hash
from ascendc_multi_turn.case_profiles import build_case_profiles
from ascendc_multi_turn.context_selector import ContextSelector
from ascendc_multi_turn.diagnostics import (
    compact_evaluation,
    evaluation_gates,
    parse_failure_evidence,
)
from ascendc_multi_turn.evaluator import LocalAscendEvaluator, extract_error_excerpt
from ascendc_multi_turn.interface_contract import (
    capture_interface_contract,
    compare_interface_contract,
)
from ascendc_multi_turn.models import EvalResult, FileBundle
from ascendc_multi_turn.repair_state import RepairStateManager
from ascendc_multi_turn.source_validation import validate_source_tree


class ProgressiveEvaluationTests(unittest.TestCase):
    def test_compact_evaluation_exposes_first_case_and_bounded_performance(self) -> None:
        result = EvalResult(
            True,
            False,
            failure_stage="correctness",
            case_results=[
                {"case_index": 0, "status": "passed", "shape": [128]},
                {
                    "case_index": 1,
                    "status": "failed",
                    "shape": [127],
                    "dtype": "float16",
                    "max_abs_diff": 0.5,
                },
            ],
            passed_case_indices=[0],
            performance={
                "overall_speedup": 1.1,
                "per_case_speedup": [
                    {"case_index": index, "speedup": 1.0 + index / 100}
                    for index in range(10)
                ],
            },
        )

        compact = compact_evaluation(result)
        self.assertEqual(compact["first_incomplete_or_failed_case"]["case_index"], 1)
        self.assertEqual(compact["case_result_count"], 2)
        self.assertEqual(len(compact["performance"]["per_case_speedup"]), 8)
        self.assertEqual(compact["performance"]["per_case_speedup_omitted"], 2)

    def test_profiles_keep_explicit_full_gate(self) -> None:
        cases = "\n".join(
            [
                '{"inputs":[{"type":"tensor","shape":[128],"dtype":"float32"},{"type":"tensor","shape":[128],"dtype":"float32"}]}',
                '{"inputs":[{"type":"tensor","shape":[128,128],"dtype":"float32"},{"type":"tensor","shape":[128,1],"dtype":"float32"}]}',
                '{"inputs":[{"type":"tensor","shape":[128],"dtype":"float16"},{"type":"tensor","shape":[128],"dtype":"float16"}]}',
            ]
        )
        profiles = build_case_profiles(cases)
        names = [item.name for item in profiles]
        self.assertEqual(names[0], "smoke")
        self.assertIn("shape", names)
        self.assertIn("dtype", names)
        self.assertEqual(names[-2:], ["full", "benchmark"])

    def test_more_passed_cases_are_monotonic_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = RepairStateManager(Path(temporary))
            before = EvalResult(
                True,
                False,
                error="comparison mismatch",
                failure_stage="correctness",
                active_profile="shape",
                passed_profiles=["smoke"],
                passed_case_indices=[0],
            )
            after = EvalResult(
                True,
                False,
                error="comparison mismatch",
                failure_stage="correctness",
                active_profile="shape",
                passed_profiles=["smoke"],
                passed_case_indices=[0, 1],
            )
            decision = manager.decide(before, after)
            self.assertEqual(decision.outcome, "PARTIAL_KEEP")
            self.assertTrue(decision.accept_candidate)
            self.assertTrue(decision.progress["case_progress"])

    def test_interface_contract_rejects_unapproved_dispatcher_change(self) -> None:
        base = FileBundle(
            files={
                "model_new_ascendc.py": "return torch.ops.cannagent.add(x)\n",
                "op_extension/register.cpp": (
                    "TORCH_LIBRARY(cannagent, m) {}\n"
                    "TORCH_LIBRARY_IMPL(cannagent, PrivateUse1, m) {}\n"
                    "TORCH_LIBRARY_IMPL(cannagent, Meta, m) {}\n"
                ),
            }
        )
        changed = FileBundle(files=dict(base.files))
        changed.files["model_new_ascendc.py"] = changed.files["model_new_ascendc.py"].replace(
            "cannagent.add", "cannagent.add_v2"
        )
        contract = capture_interface_contract(base)
        self.assertTrue(compare_interface_contract(contract, changed))

    def test_comment_only_bundle_change_is_semantic_noop(self) -> None:
        before = FileBundle(files={"kernel/add.cpp": "int value = 1; // old\n"})
        comment = FileBundle(files={"kernel/add.cpp": "int value = 1; // new\n"})
        code = FileBundle(files={"kernel/add.cpp": "int value = 2; // new\n"})
        self.assertEqual(semantic_bundle_hash(before), semantic_bundle_hash(comment))
        self.assertNotEqual(semantic_bundle_hash(before), semantic_bundle_hash(code))

    def test_failure_evidence_separates_compiler_and_device_evidence(self) -> None:
        compile_failure = parse_failure_evidence(
            stage="ascendc_build",
            output="kernel/add.cpp:9: error: no matching function for call to 'Muls' 507035",
        )
        self.assertEqual(compile_failure.subsystem, "COMPILER")
        self.assertIsNone(compile_failure.runtime_code)
        device = parse_failure_evidence(
            stage="correctness",
            output=(
                "507035 MTE exception core id=3 block id=7 "
                "pc start=0x100 pc current=0x108 serial=abc errorStr=illegal address"
            ),
        )
        self.assertEqual(device.faulting_cores, [3])
        self.assertEqual(device.faulting_blocks, [7])
        self.assertEqual(device.pc_current, ["0x108"])

    def test_gate_ledger_distinguishes_runtime_from_comparison(self) -> None:
        runtime = EvalResult(
            True,
            False,
            error="507035 MTE exception",
            failure_stage="correctness",
        )
        comparison = EvalResult(
            True,
            False,
            error="case[2]: comparison mismatch",
            failure_stage="correctness",
            case_results=[{"index": 2, "status": "failed"}],
        )
        self.assertEqual(evaluation_gates(runtime)["comparison_completed"], "not_reached")
        self.assertEqual(evaluation_gates(comparison)["comparison_completed"], "pass")

    def test_signal_termination_is_preserved_and_routes_to_runtime(self) -> None:
        error = "verification process terminated by signal 11 (SIGSEGV); exit_code=-11"
        result = EvalResult(
            True,
            False,
            error=error,
            verify_output=error,
            error_excerpt=extract_error_excerpt(f"warning: harmless\n{error}\n"),
            failure_stage="correctness",
        )
        self.assertIn("SIGSEGV", result.error_excerpt)
        self.assertEqual(evaluation_gates(result)["kernel_started"], "unknown")
        route = ContextSelector.derive_route(
            workflow_phase="bootstrap",
            current_exists=True,
            previous=result,
        )
        self.assertEqual(route.primary_skill, "runtime_debug")


if __name__ == "__main__":
    unittest.main()
