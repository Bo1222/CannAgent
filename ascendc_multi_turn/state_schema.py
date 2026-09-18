from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FailureEvidence:
    """Run-local failure evidence derived from compiler or evaluator output."""

    stage: str
    runtime_code: str | None
    subsystem: str
    device_exception: str | None
    reason: str
    core_id: int | None
    block_id: int | None
    sub_error_type: str | None
    case_info: dict[str, Any]
    related_symbols: list[str]
    attribution_status: str = "unknown"
    faulting_cores: list[int] = field(default_factory=list)
    faulting_blocks: list[int] = field(default_factory=list)
    pc_start: list[str] = field(default_factory=list)
    pc_current: list[str] = field(default_factory=list)
    serial: list[str] = field(default_factory=list)
    extend_error_str: list[str] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticRecord:
    """One diagnostic backed by direct tool output."""

    error_id: str
    stage: str
    category: str
    symbol: str | None
    source_file: str | None
    normalized_message: str
    evidence_origin: str
    excerpt: str
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AttemptRecord:
    """Run-local episode; it is never promoted to durable API knowledge."""

    attempt_id: int
    evaluation_round: int
    base_attempt_id: int | None
    phase: str
    target_error_ids: list[str]
    error_ids_before: list[str]
    error_ids_after: list[str]
    cleared_error_ids: list[str]
    new_error_ids: list[str]
    reintroduced_error_ids: list[str]
    hypothesis: str
    change: str
    touched_files: list[str]
    diff_sha256: str
    selected_skill_ids: list[str]
    selected_evidence_ids: list[str]
    route: dict[str, Any]
    outcome: str
    frontier_before: str | None
    frontier_after: str | None
    result: dict[str, Any]
    progress: dict[str, Any]
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bundle_diff(before: Any, after: Any, *, max_chars: int = 32000) -> str:
    """Return a bounded unified diff for run-local audit records."""

    old_files = before.files if before else {}
    new_files = after.files
    chunks: list[str] = []
    for name in sorted(set(old_files) | set(new_files)):
        chunks.extend(
            difflib.unified_diff(
                old_files.get(name, "").splitlines(keepends=True),
                new_files.get(name, "").splitlines(keepends=True),
                fromfile=f"before/{name}",
                tofile=f"after/{name}",
            )
        )
    return "".join(chunks)[:max_chars]
