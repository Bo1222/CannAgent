from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .devkit import ASC_DEVKIT_COMMIT, ASC_DEVKIT_VERSION

_TEXT_SUFFIXES = {".md", ".h", ".hpp", ".cpp", ".cc", ".cxx", ".asc", ".py", ".txt"}
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"}


@dataclass(frozen=True)
class EvidenceChunk:
    evidence_id: str
    symbol: str
    source_kind: str
    source_path: str
    version: str
    commit: str
    excerpt: str
    score: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceBundle:
    operator: str
    symbols: list[str]
    evidence: list[EvidenceChunk] = field(default_factory=list)
    retrieval_trace: list[dict[str, Any]] = field(default_factory=list)
    version: str = ASC_DEVKIT_VERSION
    commit: str = ASC_DEVKIT_COMMIT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DevkitRetriever:
    """Deterministic, offline retrieval over the pinned 9.1.0 DevKit tree."""

    ROOTS = (
        ("api_doc", "docs/api"),
        ("example", "examples"),
        ("declaration", "include"),
        ("implementation", "impl"),
    )

    def __init__(self, devkit_root: Path | str, *, max_files_per_root: int = 6, max_excerpt_chars: int = 2400):
        self.root = Path(devkit_root).resolve()
        self.max_files_per_root = max_files_per_root
        self.max_excerpt_chars = max_excerpt_chars

    @staticmethod
    def _tokens(symbols: list[str], text: str) -> list[str]:
        values = [*symbols, *re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text)]
        ignored = {"model", "tensor", "return", "shape", "dtype", "input", "output"}
        return list(dict.fromkeys(value for value in values if value.lower() not in ignored))[:32]

    @staticmethod
    def _excerpt(text: str, tokens: list[str], limit: int) -> str:
        lowered = text.lower()
        positions = [lowered.find(token.lower()) for token in tokens]
        positions = [position for position in positions if position >= 0]
        start = max(0, (min(positions) if positions else 0) - limit // 4)
        end = min(len(text), start + limit)
        return text[start:end].strip()

    def retrieve(self, *, operator: str, symbols: list[str], query_text: str) -> EvidenceBundle:
        tokens = self._tokens(symbols, query_text)
        bundle = EvidenceBundle(operator=operator, symbols=list(dict.fromkeys(symbols)))
        for priority, (source_kind, relative_root) in enumerate(self.ROOTS):
            root = self.root / relative_root
            candidates: list[tuple[int, Path, str]] = []
            if not root.is_dir():
                bundle.retrieval_trace.append({"root": relative_root, "decision": "missing"})
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() in _IMAGE_SUFFIXES or path.suffix.lower() not in _TEXT_SUFFIXES:
                    continue
                name = path.name.lower()
                name_score = sum(40 for token in tokens if token.lower() in name)
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                content_score = sum(min(text.lower().count(token.lower()), 5) * 3 for token in tokens)
                score = name_score + content_score - priority
                if score > 0:
                    candidates.append((score, path, text))
            candidates.sort(key=lambda item: (-item[0], item[1].as_posix()))
            selected = candidates[: self.max_files_per_root]
            bundle.retrieval_trace.append(
                {
                    "root": relative_root,
                    "source_kind": source_kind,
                    "candidates": len(candidates),
                    "selected": [path.relative_to(self.root).as_posix() for _, path, _ in selected],
                }
            )
            for index, (score, path, text) in enumerate(selected):
                relative = path.relative_to(self.root).as_posix()
                matched_symbol = next((item for item in symbols if item.lower() in text.lower() or item.lower() in path.name.lower()), "")
                bundle.evidence.append(
                    EvidenceChunk(
                        evidence_id=f"devkit:{source_kind}:{priority}:{index}:{relative}",
                        symbol=matched_symbol,
                        source_kind=source_kind,
                        source_path=relative,
                        version=ASC_DEVKIT_VERSION,
                        commit=ASC_DEVKIT_COMMIT,
                        excerpt=self._excerpt(text, tokens, self.max_excerpt_chars),
                        score=score,
                    )
                )
        return bundle


def render_evidence(bundle: EvidenceBundle, *, max_chars: int = 18000) -> str:
    sections = [
        f"# Asc DevKit 官方证据\n\nversion={bundle.version}; commit={bundle.commit}",
    ]
    used = len(sections[0])
    for item in bundle.evidence:
        section = (
            f"\n\n## {item.source_kind}: {item.source_path}\n"
            f"evidence_id={item.evidence_id}; symbol={item.symbol or 'n/a'}; score={item.score}\n\n"
            f"{item.excerpt}"
        )
        if used + len(section) > max_chars:
            break
        sections.append(section)
        used += len(section)
    return "".join(sections)
