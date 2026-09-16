from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .diagnostics import diagnostic_delta, diagnostics_for_result, evaluation_gates
from .models import EvalResult, FileBundle
from .structured_knowledge.experience import bundle_diff
from .structured_knowledge.schema import AttemptRecord, DiagnosticRecord

_VALIDATION_STAGES = {
    "response_format",
    "bundle_validation",
    "ascendc_source_validation",
    "api_constraint_validation",
    "static_validation",
    "interface_contract_validation",
}
_COMPILE_STAGES = {"ascendc_build", "compile"}


def _stage_rank(result: EvalResult | None) -> int:
    if result is None:
        return -1
    if result.correctness:
        return 4
    if result.compiled:
        return 3
    if (result.failure_stage or "") in _COMPILE_STAGES:
        return 1
    if (result.failure_stage or "") in _VALIDATION_STAGES:
        return 0
    return 2


def _progress_state(result: EvalResult | None) -> dict[str, Any]:
    if result is None:
        return {
            "stage_rank": -1,
            "gates": {},
            "passed_profiles": [],
            "passed_case_indices": [],
        }
    return {
        "stage_rank": _stage_rank(result),
        "gates": evaluation_gates(result),
        "passed_profiles": list(result.passed_profiles),
        "passed_case_indices": sorted(set(result.passed_case_indices)),
    }


@dataclass(frozen=True)
class RepairDecision:
    outcome: str
    accept_candidate: bool
    diagnostics_before: list[DiagnosticRecord]
    diagnostics_after: list[DiagnosticRecord]
    delta: dict[str, list[str]]
    base_attempt_id: int | None
    progress: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome,
            "accept_candidate": self.accept_candidate,
            "diagnostics_before": [item.to_dict() for item in self.diagnostics_before],
            "diagnostics_after": [item.to_dict() for item in self.diagnostics_after],
            "delta": self.delta,
            "base_attempt_id": self.base_attempt_id,
            "progress": self.progress,
        }


class RepairStateManager:
    """Keep run-local partial repairs separate from validated frontier knowledge."""

    def __init__(self, state_dir: Path):
        self.state_dir = state_dir
        self.path = state_dir / "repair_state.json"
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema_version": 1,
                "accepted_attempt_id": None,
                "accepted_bundle": None,
                "accepted_evaluation": None,
                "open_errors": [],
                "cleared_error_ids": [],
                "failed_approaches_by_error": {},
                "latest_rejected_attempt_id": None,
            }
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save(self) -> None:
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.path)

    @property
    def accepted_attempt_id(self) -> int | None:
        value = self.state.get("accepted_attempt_id")
        return int(value) if isinstance(value, int) else None

    def accepted_bundle(self) -> FileBundle | None:
        relative = self.state.get("accepted_bundle")
        if not isinstance(relative, str) or not relative:
            return None
        path = self.state_dir / relative
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FileBundle(
            files=dict(payload.get("files", {})),
            delete=list(payload.get("delete", [])),
            analysis=str(payload.get("analysis", "")),
        )

    def accepted_evaluation(self) -> EvalResult | None:
        payload = self.state.get("accepted_evaluation")
        if not isinstance(payload, dict):
            return None
        try:
            return EvalResult(**payload)
        except TypeError:
            return None

    def decide(self, base_result: EvalResult | None, result: EvalResult) -> RepairDecision:
        before = diagnostics_for_result(base_result)
        after = diagnostics_for_result(result)
        delta = diagnostic_delta(
            before,
            after,
            historically_cleared=self.state.get("cleared_error_ids", []),
        )
        before_progress = _progress_state(base_result)
        after_progress = _progress_state(result)
        before_profiles = set(before_progress["passed_profiles"])
        after_profiles = set(after_progress["passed_profiles"])
        before_cases = set(before_progress["passed_case_indices"])
        after_cases = set(after_progress["passed_case_indices"])
        progress = {
            "before": before_progress,
            "after": after_progress,
            "cleared_count": len(delta["cleared"]),
            "new_count": len(delta["new"]),
            "reintroduced_count": len(delta["reintroduced"]),
            "profile_progress": after_profiles > before_profiles,
            "case_progress": after_cases > before_cases,
        }
        if base_result is None:
            if _stage_rank(result) >= 1:
                outcome, accept = "INITIAL_KEEP", True
            else:
                outcome, accept = "INITIAL_REJECT", False
        elif _stage_rank(result) > _stage_rank(base_result):
            outcome, accept = "FRONTIER_ADVANCED", True
        elif _stage_rank(result) < _stage_rank(base_result):
            outcome, accept = "REGRESSION", False
        elif delta["reintroduced"]:
            outcome, accept = "REGRESSION", False
        elif result.compiled and (
            delta["cleared"]
            or progress["profile_progress"]
            or progress["case_progress"]
        ):
            outcome, accept = "PARTIAL_KEEP", True
        elif (base_result.failure_stage or "") in _COMPILE_STAGES:
            direct_before = [item for item in before if item.category != "stage_error"]
            direct_after = [item for item in after if item.category != "stage_error"]
            if direct_before and direct_after and delta["cleared"]:
                outcome, accept = "PARTIAL_KEEP", True
            elif direct_before and not direct_after and not result.compiled:
                # Losing parseable compiler evidence is not proof of progress.
                outcome, accept = "UNCLASSIFIED", False
            else:
                outcome, accept = "NO_PROGRESS", False
        else:
            outcome, accept = "NO_PROGRESS", False
        return RepairDecision(
            outcome=outcome,
            accept_candidate=accept,
            diagnostics_before=before,
            diagnostics_after=after,
            delta=delta,
            base_attempt_id=self.accepted_attempt_id,
            progress=progress,
        )

    @staticmethod
    def _approach(active_item: dict[str, Any] | None) -> tuple[str, str, str]:
        item = active_item or {}
        hypothesis = str(item.get("hypothesis", "")).strip()
        change = str(item.get("change", "")).strip()
        source = f"{hypothesis}\n{change}".strip() or "unspecified repair"
        approach_id = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        return approach_id, hypothesis, change

    @staticmethod
    def _approach_family(hypothesis: str, change: str) -> str:
        source = f"{hypothesis}\n{change}".lower()
        source = re.sub(r"0x[0-9a-f]+|\b\d+\b", "<n>", source)
        source = re.sub(r"[^a-z_\u4e00-\u9fff]+", " ", source)
        source = re.sub(
            r"\b(?:change|fix|update|modify|replace|use|set|add|remove|ensure)\b",
            " ",
            source,
        )
        normalized = " ".join(source.split()) or "unspecified repair"
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def _selection_ids(selection: Any) -> tuple[list[str], list[str], dict[str, Any]]:
        payload = selection.to_dict() if hasattr(selection, "to_dict") else {}
        metadata = payload.get("selection_metadata", {})
        task = payload.get("task_facts", {})
        skill_ids = list(metadata.get("selected_skill_ids", []))
        fact_ids = list(
            dict.fromkeys(
                [
                    *metadata.get("selected_structured_ids", []),
                    *metadata.get("runtime_fact_ids", []),
                ]
            )
        )
        route = {
            key: task.get(key)
            for key in (
                "primary_skill",
                "route_reason",
                "secondary_skill",
                "secondary_reason",
                "debug_category",
            )
        }
        return skill_ids, fact_ids, route

    def build_attempt(
        self,
        *,
        attempt_id: int,
        evaluation_round: int,
        phase: str,
        base_bundle: FileBundle | None,
        candidate: FileBundle,
        result: EvalResult,
        active_item: dict[str, Any] | None,
        selection: Any,
        decision: RepairDecision,
        frontier_before: str | None,
        frontier_after: str | None,
        outcome_override: str | None = None,
    ) -> AttemptRecord:
        _approach_id, hypothesis, change = self._approach(active_item)
        diff = bundle_diff(base_bundle, candidate)
        old_files = base_bundle.files if base_bundle else {}
        touched = sorted(
            name
            for name in set(old_files) | set(candidate.files)
            if old_files.get(name) != candidate.files.get(name)
        )
        skill_ids, fact_ids, route = self._selection_ids(selection)
        return AttemptRecord(
            attempt_id=attempt_id,
            evaluation_round=evaluation_round,
            base_attempt_id=decision.base_attempt_id,
            phase=phase,
            target_error_ids=decision.delta["before"],
            error_ids_before=decision.delta["before"],
            error_ids_after=decision.delta["after"],
            cleared_error_ids=decision.delta["cleared"],
            new_error_ids=decision.delta["new"],
            reintroduced_error_ids=decision.delta["reintroduced"],
            hypothesis=hypothesis,
            change=change,
            touched_files=touched,
            diff_sha256=hashlib.sha256(diff.encode("utf-8")).hexdigest(),
            selected_skill_ids=skill_ids,
            selected_fact_ids=fact_ids,
            route=route,
            outcome=outcome_override or decision.outcome,
            frontier_before=frontier_before,
            frontier_after=frontier_after,
            result=result.to_dict(),
            progress=decision.progress,
        )

    def observe(
        self,
        *,
        attempt: AttemptRecord,
        candidate_path: str,
        decision: RepairDecision,
    ) -> None:
        cleared = set(self.state.get("cleared_error_ids", []))
        if decision.accept_candidate:
            cleared.update(decision.delta["cleared"])
            cleared.difference_update(decision.delta["after"])
            self.state.update(
                accepted_attempt_id=attempt.attempt_id,
                accepted_bundle=candidate_path,
                accepted_evaluation=attempt.result,
                open_errors=[item.to_dict() for item in decision.diagnostics_after],
                cleared_error_ids=sorted(cleared),
                latest_rejected_attempt_id=None,
            )
        else:
            self.state["latest_rejected_attempt_id"] = attempt.attempt_id
            self._remember_failed_approach(attempt)
        self._save()

    def _remember_failed_approach(self, attempt: AttemptRecord) -> None:
        if attempt.outcome not in {"NO_PROGRESS", "REGRESSION", "UNCLASSIFIED"}:
            return
        approach_id = hashlib.sha256(
            f"{attempt.hypothesis}\n{attempt.change}".encode()
        ).hexdigest()[:12]
        family_id = self._approach_family(attempt.hypothesis, attempt.change)
        summary = (attempt.change or attempt.hypothesis or "unspecified repair").strip()[:600]
        targets = attempt.target_error_ids or attempt.error_ids_after
        mapping = self.state.setdefault("failed_approaches_by_error", {})
        for error_id in targets:
            entries = mapping.setdefault(error_id, [])
            existing = next(
                (item for item in entries if item.get("approach_id") == approach_id), None
            )
            if existing is None:
                entries.append(
                    {
                        "approach_id": approach_id,
                        "approach_family_id": family_id,
                        "summary": summary,
                        "outcome": attempt.outcome,
                        "attempt_ids": [attempt.attempt_id],
                        "occurrences": 1,
                    }
                )
            else:
                existing["attempt_ids"] = [
                    *existing.get("attempt_ids", []),
                    attempt.attempt_id,
                ]
                existing["occurrences"] = int(existing.get("occurrences", 1)) + 1
                existing["outcome"] = attempt.outcome
            family_occurrences = sum(
                int(item.get("occurrences", 1))
                for item in entries
                if item.get("approach_family_id") == family_id
            )
            for item in entries:
                if item.get("approach_family_id") == family_id:
                    item["family_occurrences"] = family_occurrences

    def prompt_summary(self, result: EvalResult | None = None) -> dict[str, Any]:
        open_records = diagnostics_for_result(result) if result is not None else []
        if not open_records:
            open_records = [
                DiagnosticRecord(**item)
                for item in self.state.get("open_errors", [])
                if isinstance(item, dict)
            ]
        open_errors = [item.to_dict() for item in open_records]
        failed = self.state.get("failed_approaches_by_error", {})
        relevant: dict[str, list[dict[str, Any]]] = {}
        for record in open_records:
            exact = list(failed.get(record.error_id, []))
            if not exact and record.symbol:
                for error_id, entries in failed.items():
                    if f":{record.symbol}:" in error_id:
                        exact.extend(entries)
            if exact:
                deduplicated = {item.get("approach_id"): item for item in exact}
                relevant[record.error_id] = list(deduplicated.values())[-3:]
        return {
            "accepted_attempt_id": self.accepted_attempt_id,
            "latest_rejected_attempt_id": self.state.get("latest_rejected_attempt_id"),
            "open_errors": open_errors,
            "cleared_error_ids": list(self.state.get("cleared_error_ids", [])),
            "failed_approaches_by_error": relevant,
            "accepted_progress": _progress_state(self.accepted_evaluation()),
            "escalation": self.escalation_for(result),
        }

    def escalation_for(self, result: EvalResult | None) -> dict[str, Any]:
        records = diagnostics_for_result(result)
        mapping = self.state.get("failed_approaches_by_error", {})
        attempts = 0
        max_family_occurrences = 0
        error_ids: list[str] = []
        for record in records:
            error_ids.append(record.error_id)
            entries = list(mapping.get(record.error_id, []))
            if not entries and record.symbol:
                for error_id, candidates in mapping.items():
                    if f":{record.symbol}:" in error_id:
                        entries.extend(candidates)
            attempts += sum(int(item.get("occurrences", 1)) for item in entries)
            max_family_occurrences = max(
                [max_family_occurrences]
                + [int(item.get("family_occurrences", item.get("occurrences", 1))) for item in entries]
            )
        return {
            "open_error_ids": error_ids,
            "rejected_attempts": attempts,
            "max_approach_family_occurrences": max_family_occurrences,
            "diagnose_required": attempts >= 2 or max_family_occurrences >= 2,
        }
