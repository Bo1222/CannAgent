from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from .schema import AtomicFact, NormalizedDocument


@dataclass(frozen=True)
class FactIssue:
    code: str
    subject: str
    message: str


class FactValidator:
    def validate(
        self, fact: AtomicFact, documents: dict[str, NormalizedDocument]
    ) -> list[FactIssue]:
        issues: list[FactIssue] = []
        provenance = fact.provenance
        document = documents.get(provenance.document_id)
        if not provenance.evidence_text.strip():
            issues.append(FactIssue("missing_evidence", fact.subject, "fact has no evidence"))
        if not fact.applicability:
            issues.append(
                FactIssue("missing_applicability", fact.subject, "fact has no applicability")
            )
        if not provenance.source_hash:
            issues.append(FactIssue("missing_source_hash", fact.subject, "fact has no source hash"))
        if document is None:
            issues.append(FactIssue("missing_document", fact.subject, "source document is absent"))
        elif document.source_hash != provenance.source_hash:
            issues.append(FactIssue("source_hash_mismatch", fact.subject, "source hash differs"))
        elif provenance.section_id not in {
            item.section_id for item in [*document.headings, *document.paragraphs, *document.tables]
        }:
            issues.append(FactIssue("missing_section", fact.subject, "source section is absent"))
        return issues


def _context_key(fact: AtomicFact) -> tuple[str, str, str]:
    return (
        fact.subject,
        fact.predicate,
        json.dumps(fact.applicability, ensure_ascii=False, sort_keys=True),
    )


@dataclass(frozen=True)
class FactConflict:
    key: tuple[str, str, str]
    values: list[Any]
    fact_indexes: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConflictResolver:
    """Reports mutually different values only inside an identical context."""

    def find_conflicts(self, facts: list[AtomicFact]) -> list[FactConflict]:
        groups: dict[tuple[str, str, str], list[tuple[int, AtomicFact]]] = {}
        for index, fact in enumerate(facts):
            groups.setdefault(_context_key(fact), []).append((index, fact))
        conflicts: list[FactConflict] = []
        for key, items in groups.items():
            encoded = {
                json.dumps(item.value, ensure_ascii=False, sort_keys=True, default=str)
                for _, item in items
            }
            if len(encoded) > 1:
                conflicts.append(
                    FactConflict(key, [item.value for _, item in items], [index for index, _ in items])
                )
        return conflicts
