from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ascendc_multi_turn.bundle import semantic_bundle_hash
from ascendc_multi_turn.case_profiles import build_case_profiles
from ascendc_multi_turn.context_selector import ContextSelector
from ascendc_multi_turn.diagnostics import evaluation_gates, parse_structured_failure
from ascendc_multi_turn.evaluator import LocalAscendEvaluator, extract_error_excerpt
from ascendc_multi_turn.interface_contract import (
    capture_interface_contract,
    compare_interface_contract,
)
from ascendc_multi_turn.models import EvalResult, FileBundle
from ascendc_multi_turn.repair_state import RepairStateManager
from ascendc_multi_turn.source_validation import validate_source_tree


class ProgressiveEvaluationTests(unittest.TestCase):
    def test_host_wrapper_cannot_dereference_npu_pointer_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary)
            kernel_dir = task_dir / "kernel"
            kernel_dir.mkdir()
            (kernel_dir / "pybind11.cpp").write_text(
                'extern "C" void add_do(uint32_t blockDim, void *stream, uint8_t *tiling);\n'
                "void run(void *s, uint8_t *t) { add_do(1, s, t); }\n"
                "PYBIND11_MODULE(_add_ext, m) {}\n",
                encoding="utf-8",
            )
            (kernel_dir / "add.cpp").write_text(
                "struct Tiling { int dtype; };\n"
                'extern "C" __global__ __aicore__ void kernel(GM_ADDR tiling) {}\n'
                'extern "C" void add_do(uint32_t blockDim, void *stream, uint8_t *tiling)\n'
                "{\n"
                "  Tiling* value = (Tiling*)tiling;\n"
                "  int dtype = value->dtype;\n"
                "  kernel<<<blockDim, nullptr, stream>>>((GM_ADDR)tiling);\n"
                "}\n",
                encoding="utf-8",
            )
            codes = {item.code for item in validate_source_tree(task_dir)}
            self.assertIn("host_dereferences_device_pointer", codes)

    def test_local_evaluator_stops_at_first_failed_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task_dir = root / "task"
            round_dir = root / "round"
            kernel_dir = task_dir / "kernel"
            kernel_dir.mkdir(parents=True)
            round_dir.mkdir()
            (task_dir / "model_new_ascendc.py").write_text("class ModelNew: pass\n", encoding="utf-8")
            (kernel_dir / "pybind11.cpp").write_text(
                'extern "C" void add_do(uint32_t blockDim, void *stream, uint8_t *x);\n'
                "void run(void *s, uint8_t *x) { add_do(1, s, x); }\n"
                "PYBIND11_MODULE(_add_ext, m) {}\n",
                encoding="utf-8",
            )
            (kernel_dir / "add.cpp").write_text(
                'extern "C" __global__ __aicore__ void add_kernel(GM_ADDR x) {}\n'
                'extern "C" void add_do(uint32_t blockDim, void *stream, uint8_t *x)\n'
                "{\n  add_kernel<<<blockDim, nullptr, stream>>>(x);\n}\n",
                encoding="utf-8",
            )
            (task_dir / "add.json").write_text(
                "\n".join(
                    [
                        '{"inputs":[{"type":"tensor","shape":[128],"dtype":"float32"},{"type":"tensor","shape":[128],"dtype":"float32"}]}',
                        '{"inputs":[{"type":"tensor","shape":[128,128],"dtype":"float32"},{"type":"tensor","shape":[128,1],"dtype":"float32"}]}',
                    ]
                ),
                encoding="utf-8",
            )

            def fake_run(command, *, env, timeout, log_path=None):
                if "verification_ascendc.py" not in " ".join(command):
                    return 0, "passed"
                profile = command[command.index("--profile") + 1]
                report_path = Path(command[command.index("--report-json") + 1])
                if profile == "smoke":
                    report = {
                        "case_results": [{"index": 0, "status": "passed"}],
                        "passed_case_indices": [0],
                        "failed_case_index": None,
                    }
                    report_path.write_text(json.dumps(report), encoding="utf-8")
                    return 0, "case[0]: matched"
                report = {
                    "case_results": [
                        {"index": 0, "status": "passed"},
                        {"index": 1, "status": "failed"},
                    ],
                    "passed_case_indices": [0],
                    "failed_case_index": 1,
                }
                report_path.write_text(json.dumps(report), encoding="utf-8")
                return 1, "case[1]: comparison mismatch"

            evaluator = LocalAscendEvaluator(device=0, soc_version="Ascend910B3")
            with patch("ascendc_multi_turn.evaluator._run", side_effect=fake_run):
                result = evaluator.evaluate(task_dir, round_dir)
            self.assertTrue(result.compiled)
            self.assertFalse(result.correctness)
            self.assertEqual(result.active_profile, "shape")
            self.assertEqual(result.passed_profiles, ["smoke"])
            self.assertEqual(result.passed_case_indices, [0])
            self.assertEqual(result.structured_failure["case_info"]["failed_case_index"], 1)

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

    def test_interface_contract_rejects_unapproved_wrapper_change(self) -> None:
        base = FileBundle(
            files={
                "kernel/pybind11.cpp": (
                    'extern "C" void add_do(uint32_t blockDim, void *stream, uint8_t *x);\n'
                    "PYBIND11_MODULE(_add_ext, m) {}\n"
                ),
                "kernel/add.cpp": (
                    'extern "C" __global__ __aicore__ void add_kernel(GM_ADDR x) {}\n'
                    'extern "C" void add_do(uint32_t blockDim, void *stream, uint8_t *x) {}\n'
                ),
            }
        )
        changed = FileBundle(files=dict(base.files))
        changed.files["kernel/add.cpp"] = changed.files["kernel/add.cpp"].replace(
            "uint8_t *x)", "uint8_t *x, uint64_t descriptor)"
        )
        contract = capture_interface_contract(base)
        self.assertTrue(compare_interface_contract(contract, changed))

    def test_comment_only_bundle_change_is_semantic_noop(self) -> None:
        before = FileBundle(files={"kernel/add.cpp": "int value = 1; // old\n"})
        comment = FileBundle(files={"kernel/add.cpp": "int value = 1; // new\n"})
        code = FileBundle(files={"kernel/add.cpp": "int value = 2; // new\n"})
        self.assertEqual(semantic_bundle_hash(before), semantic_bundle_hash(comment))
        self.assertNotEqual(semantic_bundle_hash(before), semantic_bundle_hash(code))

    def test_structured_failure_separates_compiler_and_device_evidence(self) -> None:
        compile_failure = parse_structured_failure(
            stage="ascendc_build",
            output="kernel/add.cpp:9: error: no matching function for call to 'Muls' 507035",
        )
        self.assertEqual(compile_failure.subsystem, "COMPILER")
        self.assertIsNone(compile_failure.runtime_code)
        device = parse_structured_failure(
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
