from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class HeadingNode:
    section_id: str
    level: int
    title: str
    parent_id: str | None = None


@dataclass(frozen=True)
class Paragraph:
    section_id: str
    text: str


@dataclass(frozen=True)
class TableNode:
    section_id: str
    headers: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class CodeCandidate:
    section_id: str
    text: str
    source_kind: str


@dataclass(frozen=True)
class Provenance:
    document_id: str
    source_path: str
    source_hash: str
    section_id: str
    evidence_text: str


@dataclass(frozen=True)
class AtomicFact:
    subject: str
    predicate: str
    value: Any
    applicability: dict[str, Any]
    provenance: Provenance
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AtomicFact":
        values = dict(payload)
        values["provenance"] = Provenance(**values["provenance"])
        return cls(**values)


@dataclass
class NormalizedDocument:
    document_id: str
    source_path: str
    source_hash: str
    title: str
    source_url: str | None
    headings: list[HeadingNode] = field(default_factory=list)
    paragraphs: list[Paragraph] = field(default_factory=list)
    tables: list[TableNode] = field(default_factory=list)
    code_candidates: list[CodeCandidate] = field(default_factory=list)
    constraints: list[Paragraph] = field(default_factory=list)
    examples: list[Paragraph] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizedDocument":
        values = dict(payload)
        values["headings"] = [HeadingNode(**item) for item in values.get("headings", [])]
        values["paragraphs"] = [Paragraph(**item) for item in values.get("paragraphs", [])]
        values["tables"] = [TableNode(**item) for item in values.get("tables", [])]
        values["code_candidates"] = [
            CodeCandidate(**item) for item in values.get("code_candidates", [])
        ]
        values["constraints"] = [Paragraph(**item) for item in values.get("constraints", [])]
        values["examples"] = [Paragraph(**item) for item in values.get("examples", [])]
        return cls(**values)
