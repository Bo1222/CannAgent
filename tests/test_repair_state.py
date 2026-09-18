from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.diagnostics import extract_diagnostic_records
from ascendc_multi_turn.models import EvalResult, FileBundle
from ascendc_multi_turn.prompts import PLANNING_MODE_DIAGNOSE, build_plan_prompt
from ascendc_multi_turn.repair_state import RepairStateManager


def _failure(root: Path, name: str, diagnostics: list[str]) -> EvalResult:
    path = root / name
    path.write_text("\n".join(diagnostics), encoding="utf-8")
    return EvalResult(
        False,
        False,
        error="AscendC build failed",
        failure_stage="ascendc_build",
        failure_code="ascendc_build_failed",
        error_excerpt="\n".join(diagnostics),
        details_path=str(path),
    )


def _bundle(marker: str) -> FileBundle:
    return FileBundle(files={"kernel/add.cpp": marker})


class RepairStateTests(unittest.TestCase):
    ADD = "kernel/add.cpp:184:15: error: no matching constructor for initialization of 'AddTiling'"
    MULS = "kernel/add.cpp:91:13: error: no matching function for call to 'Muls'"

    def _record(
        self,
        manager: RepairStateManager,
        *,
        attempt: int,
        base: FileBundle | None,
        candidate: FileBundle,
        before: EvalResult | None,
        after: EvalResult,
        change: str,
    ):
        path = f"round_{attempt:02d}/candidate.json"
        target = manager.state_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(candidate.to_dict()), encoding="utf-8")
        decision = manager.decide(before, after)
        record = manager.build_attempt(
            attempt_id=attempt,
            evaluation_round=attempt,
            phase="bootstrap",
            base_bundle=base,
            candidate=candidate,
            result=after,
            active_item={"hypothesis": change, "change": change},
            selection={},
            decision=decision,
            frontier_before="source",
            frontier_after="source",
        )
        manager.observe(attempt=record, candidate_path=path, decision=decision)
        return decision, record

    def test_real_add_diagnostics_are_stable_across_line_changes(self) -> None:
        first = extract_diagnostic_records(
            stage="ascendc_build", output=self.MULS
        )[0]
        moved = extract_diagnostic_records(
            stage="ascendc_build", output=self.MULS.replace(":91:13:", ":176:9:")
        )[0]
        self.assertEqual(first.error_id, moved.error_id)
        self.assertEqual(first.symbol, "Muls")
        self.assertEqual(first.category, "kernel_api_overload")

    def test_initial_source_validation_failure_becomes_a_repair_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = RepairStateManager(Path(temporary))
            result = EvalResult(
                False,
                False,
                error="invalid source",
                failure_stage="ascendc_source_validation",
            )
            decision = manager.decide(None, result)
            self.assertEqual(decision.outcome, "INITIAL_KEEP")
            self.assertTrue(decision.accept_candidate)

    def test_source_validation_rules_are_distinct_and_partial_fix_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = RepairStateManager(root)
            before = _failure(
                root,
                "source1.log",
                [
                    "CMakeLists.txt:1: error[missing_asc_package]: project CMake must contain find_package(ASC",
                    "CMakeLists.txt:1: error[missing_asc_language]: project CMake must contain LANGUAGES ASC",
                ],
            )
            before.failure_stage = "ascendc_source_validation"
            after = _failure(
                root,
                "source2.log",
                [
                    "CMakeLists.txt:1: error[missing_asc_language]: project CMake must contain LANGUAGES ASC",
                ],
            )
            after.failure_stage = "ascendc_source_validation"
            records = extract_diagnostic_records(
                stage=before.failure_stage,
                output=before.error_excerpt,
            )
            self.assertEqual(
                [item.category for item in records],
                ["missing_asc_package", "missing_asc_language"],
            )
            self.assertEqual([item.source_file for item in records], ["CMakeLists.txt"] * 2)
            decision = manager.decide(before, after)
            self.assertEqual(decision.outcome, "PARTIAL_KEEP")
            self.assertTrue(decision.accept_candidate)
            self.assertEqual(len(decision.delta["cleared"]), 1)

    def test_diagnose_prompt_projects_raw_attempt_evidence(self) -> None:
        error = (
            "CMakeLists.txt:1: error[missing_asc_language]: "
            "project CMake must contain LANGUAGES ASC"
        )
        prompt = build_plan_prompt(
            reference_code="return x + y",
            cases_text="{}",
            current=None,
            result=None,
            mode="bootstrap",
            history=[
                {
                    "attempt_id": 3,
                    "candidate": "round_03/candidate.json",
                    "evaluation": {
                        "failure_stage": "ascendc_source_validation",
                        "failure_code": "ascendc_source_validation_failed",
                        "error_excerpt": error,
                    },
                    "repair_attempt": {
                        "new_error_ids": ["missing_asc_language:unknown:abc"],
                    },
                    "plan_item": {"hypothesis": "build a project"},
                }
            ],
            planning_mode=PLANNING_MODE_DIAGNOSE,
        )
        self.assertIn(error, prompt)
        self.assertIn('"candidate": "round_03/candidate.json"', prompt)
        self.assertIn("错误 ID/hash 只用于稳定关联", prompt)

    def test_partial_compile_fix_becomes_base_and_regression_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = RepairStateManager(root)
            first_result = _failure(root, "r1.log", [self.ADD, self.MULS])
            first, _ = self._record(
                manager,
                attempt=1,
                base=None,
                candidate=_bundle("round1"),
                before=None,
                after=first_result,
                change="initial implementation",
            )
            self.assertEqual(first.outcome, "INITIAL_KEEP")

            second_result = _failure(root, "r2.log", [self.MULS])
            second, _ = self._record(
                manager,
                attempt=2,
                base=_bundle("round1"),
                candidate=_bundle("round2"),
                before=first_result,
                after=second_result,
                change="copy AddTiling fields locally",
            )
            self.assertEqual(second.outcome, "PARTIAL_KEEP")
            self.assertEqual(manager.accepted_bundle().files["kernel/add.cpp"], "round2")
            self.assertTrue(any("AddTiling" in item for item in manager.state["cleared_error_ids"]))

            regression_result = _failure(root, "r3.log", [self.ADD, self.MULS])
            third, _ = self._record(
                manager,
                attempt=3,
                base=_bundle("round2"),
                candidate=_bundle("round3"),
                before=second_result,
                after=regression_result,
                change="use runtime dtype branch",
            )
            self.assertEqual(third.outcome, "REGRESSION")
            self.assertFalse(third.accept_candidate)
            self.assertEqual(manager.accepted_bundle().files["kernel/add.cpp"], "round2")

    def test_episodic_methods_are_deduplicated_in_prompt_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manager = RepairStateManager(root)
            result = _failure(root, "same.log", [self.MULS])
            self._record(
                manager,
                attempt=1,
                base=None,
                candidate=_bundle("round1"),
                before=None,
                after=result,
                change="initial implementation",
            )
            for attempt in (2, 3, 4):
                self._record(
                    manager,
                    attempt=attempt,
                    base=_bundle("round1"),
                    candidate=_bundle(f"round{attempt}"),
                    before=result,
                    after=result,
                    change="guard Muls with if(sizeof(T))",
                )
            summary = manager.prompt_summary(result)
            methods = next(iter(summary["failed_approaches_by_error"].values()))
            self.assertEqual(len(methods), 1)
            self.assertEqual(methods[0]["occurrences"], 3)
            self.assertEqual(methods[0]["attempt_ids"], [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
