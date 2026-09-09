from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.knowledge_v2.experience import (
    build_incident,
    persist_incident,
    promote_confirmed_experience,
)
from ascendc_multi_turn.knowledge_v2.frontier import FrontierManager, reached_frontier
from ascendc_multi_turn.models import EvalResult, FileBundle


def _bundle(marker: str) -> FileBundle:
    return FileBundle(files={"kernel/op.cpp": marker})


def _failure(stage: str, reason: str = "failed") -> dict:
    return {
        "stage": stage,
        "runtime_code": None,
        "subsystem": "compiler" if stage == "ascendc_build" else "mte",
        "device_exception": None,
        "reason": reason,
        "core_id": None,
        "block_id": None,
        "sub_error_type": None,
        "case_info": {},
        "related_symbols": ["DataCopyPad"],
        "schema_version": 1,
    }


class FrontierTests(unittest.TestCase):
    def test_source_validation_failure_has_no_frontier_and_no_contamination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = FrontierManager(Path(temporary))
            result = EvalResult(
                False,
                False,
                error="invalid source",
                failure_stage="ascendc_source_validation",
            )
            decision = manager.observe(_bundle("bad"), result, 1)

            self.assertIsNone(reached_frontier(result))
            self.assertFalse(decision.advanced)
            self.assertIsNone(decision.rollback)
            self.assertEqual(manager.manifest["entries"], {})

    def test_deeper_candidate_materializes_all_crossed_frontiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = FrontierManager(Path(temporary))
            compile_failure = EvalResult(
                False, False, error="compile", failure_stage="ascendc_build"
            )
            first = manager.observe(_bundle("source-ok"), compile_failure, 1)
            runtime_failure = EvalResult(
                True, False, error="runtime", failure_stage="correctness"
            )
            second = manager.observe(_bundle("compiled"), runtime_failure, 2)

            self.assertEqual(first.highest, "source")
            self.assertEqual(second.highest, "runtime")
            self.assertTrue(second.advanced)
            self.assertEqual(
                set(manager.manifest["entries"]), {"source", "compile", "runtime"}
            )
            self.assertEqual(second.rollback.files["kernel/op.cpp"], "compiled")

    def test_repeated_same_stage_failure_rolls_back_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = FrontierManager(Path(temporary))
            result = EvalResult(False, False, error="compile", failure_stage="ascendc_build")
            manager.observe(_bundle("stable"), result, 1)
            repeated = manager.observe(_bundle("regression"), result, 2)

            self.assertFalse(repeated.advanced)
            self.assertEqual(repeated.rollback.files["kernel/op.cpp"], "stable")
            self.assertEqual(manager.manifest["entries"]["source"]["attempt_id"], 1)

    def test_performance_frontier_only_accepts_a_better_score(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manager = FrontierManager(Path(temporary))
            manager.observe(_bundle("baseline"), EvalResult(True, True, score=2.0), 1)
            slower = manager.observe(_bundle("slower"), EvalResult(True, True, score=1.0), 2)
            faster = manager.observe(_bundle("faster"), EvalResult(True, True, score=3.0), 3)

            self.assertFalse(slower.advanced)
            self.assertEqual(slower.rollback.files["kernel/op.cpp"], "baseline")
            self.assertTrue(faster.advanced)
            self.assertEqual(faster.rollback.files["kernel/op.cpp"], "faster")
            self.assertEqual(manager.manifest["entries"]["performance"]["score"], 3.0)


class ExperienceTests(unittest.TestCase):
    def test_failed_or_persistent_failure_is_not_promoted(self) -> None:
        failure = _failure("correctness")
        failed = build_incident(
            attempt_id=2,
            failure=failure,
            hypothesis={"hypothesis": "fix transfer"},
            before=_bundle("old"),
            after=_bundle("new"),
            resolved_fact_ids=["fact-1"],
            result=EvalResult(
                True,
                False,
                error="still failing",
                failure_stage="correctness",
                structured_failure=failure,
            ),
        )
        self.assertFalse(failed.failure_disappeared)
        self.assertIsNone(promote_confirmed_experience(failed))

    def test_correct_repair_creates_isolated_confirmed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_dir = Path(temporary)
            official = [{"fact_id": "official-1", "authority": "OFFICIAL"}]
            incident = build_incident(
                attempt_id=3,
                failure=_failure("correctness"),
                hypothesis={"hypothesis": "repair contextual API use"},
                before=_bundle("old"),
                after=_bundle("new"),
                resolved_fact_ids=["official-1", "official-1"],
                result=EvalResult(True, True, score=1.5),
            )
            experience = persist_incident(state_dir, incident)

            self.assertIsNotNone(experience)
            self.assertEqual(experience.authority, "CONFIRMED_EXPERIENCE")
            self.assertEqual(experience.resolved_fact_ids, ["official-1"])
            self.assertEqual(official, [{"fact_id": "official-1", "authority": "OFFICIAL"}])
            payload = json.loads(
                next((state_dir / "experience_candidates").glob("*.json")).read_text()
            )
            self.assertEqual(payload["authority"], "CONFIRMED_EXPERIENCE")
            self.assertTrue(next((state_dir / "incidents").glob("*.json")).is_file())


if __name__ == "__main__":
    unittest.main()
