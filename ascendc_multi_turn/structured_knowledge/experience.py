from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
from typing import Any

from ..models import EvalResult, FileBundle
from .schema import ConfirmedExperience, IncidentRecord


def bundle_diff(before: FileBundle | None, after: FileBundle, *, max_chars: int = 32000) -> str:
    """Return a bounded, auditable unified diff for one generated candidate."""
    old_files = before.files if before else {}
    chunks: list[str] = []
    for name in sorted(set(old_files) | set(after.files)):
        chunks.extend(
            difflib.unified_diff(
                old_files.get(name, "").splitlines(keepends=True),
                after.files.get(name, "").splitlines(keepends=True),
                fromfile=f"before/{name}",
                tofile=f"after/{name}",
            )
        )
    return "".join(chunks)[:max_chars]


def failure_signature(failure: dict[str, Any] | None) -> tuple[Any, ...] | None:
    if not failure:
        return None
    return (
        failure.get("stage"),
        failure.get("runtime_code"),
        failure.get("subsystem"),
        failure.get("device_exception"),
        failure.get("sub_error_type"),
    )


def build_incident(
    *,
    attempt_id: int,
    failure: dict[str, Any] | None,
    hypothesis: dict[str, Any] | None,
    before: FileBundle | None,
    after: FileBundle,
    resolved_fact_ids: list[str],
    result: EvalResult,
    base_attempt_id: int | None = None,
    error_ids_before: list[str] | None = None,
    error_ids_after: list[str] | None = None,
    cleared_error_ids: list[str] | None = None,
) -> IncidentRecord:
    current_failure = result.structured_failure if result.error else None
    before_ids = sorted(set(error_ids_before or []))
    after_ids = sorted(set(error_ids_after or []))
    cleared_ids = sorted(set(cleared_error_ids or []))
    disappeared = (
        bool(before_ids) and not set(before_ids).intersection(after_ids)
        if error_ids_before is not None
        else failure is not None
        and failure_signature(failure) != failure_signature(current_failure)
    )
    digest = hashlib.sha256(
        json.dumps(
            {
                "attempt": attempt_id,
                "failure": failure,
                "hypothesis": hypothesis,
                "result": result.to_dict(),
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:16]
    return IncidentRecord(
        incident_id=f"incident-{attempt_id}-{digest}",
        attempt_id=attempt_id,
        failure=failure,
        hypothesis=hypothesis,
        code_diff=bundle_diff(before, after),
        resolved_fact_ids=sorted(set(resolved_fact_ids)),
        result=result.to_dict(),
        failure_disappeared=disappeared,
        base_attempt_id=base_attempt_id,
        error_ids_before=before_ids,
        error_ids_after=after_ids,
        cleared_error_ids=cleared_ids,
    )


def promote_confirmed_experience(incident: IncidentRecord) -> ConfirmedExperience | None:
    """Promote evidence only after correctness passes and the prior failure disappears."""
    if (
        incident.failure is None
        or not incident.result.get("correctness")
        or not incident.failure_disappeared
    ):
        return None
    return ConfirmedExperience(
        experience_id=f"experience-{incident.incident_id.removeprefix('incident-')}",
        incident_id=incident.incident_id,
        failure=incident.failure,
        hypothesis=incident.hypothesis,
        repair_diff=incident.code_diff,
        resolved_fact_ids=incident.resolved_fact_ids,
        result=incident.result,
    )


def persist_incident(state_dir: Path, incident: IncidentRecord) -> ConfirmedExperience | None:
    incident_dir = state_dir / "incidents"
    incident_dir.mkdir(parents=True, exist_ok=True)
    (incident_dir / f"{incident.incident_id}.json").write_text(
        json.dumps(incident.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    experience = promote_confirmed_experience(incident)
    if experience is not None:
        candidate_dir = state_dir / "experience_candidates"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / f"{experience.experience_id}.json").write_text(
            json.dumps(experience.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return experience
