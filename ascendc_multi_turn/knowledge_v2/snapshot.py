from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .extract import ParameterTableFactExtractor
from .normalize import MarkdownNormalizer
from .schema import ApiCard, AtomicFact, NormalizedDocument, SCHEMA_VERSION
from .validate import ConflictResolver, FactValidator


COMPILER_VERSION = "2"
_API_SYMBOL = re.compile(r"\b(?:AscendC::)?([A-Z][A-Za-z0-9_]{2,})\b")
_SQUASHED_CORE_SYMBOL = re.compile(r"AscendC::(DataCopyPad|DataCopy|TPipe|TQue)")


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fact_id(fact: AtomicFact) -> str:
    return "fact_" + hashlib.sha256(_canonical(fact.to_dict()).encode()).hexdigest()[:20]


def _discover(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*.md") if path.is_file())


def _snapshot_id(source: Path, documents: list[tuple[Path, NormalizedDocument]], version: str) -> str:
    payload = {
        "platform": "cann",
        "version": version,
        "schema_version": SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
        "documents": [(str(path.relative_to(source)), document.source_hash) for path, document in documents],
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:24]


def _card_symbols(document: NormalizedDocument) -> set[str]:
    text = "\n".join(
        [document.title]
        + [item.text for item in document.paragraphs]
        + [cell for table in document.tables for row in table.rows for cell in row]
    )
    return set(_API_SYMBOL.findall(text)) | set(_SQUASHED_CORE_SYMBOL.findall(text))


def build_snapshot(*, source: Path, output: Path, version: str) -> Path:
    source = source.resolve()
    normalizer = MarkdownNormalizer()
    normalized = [(path, normalizer.parse_path(path)) for path in _discover(source)]
    if not normalized:
        raise ValueError(f"no Markdown documents found under {source}")
    snapshot_id = _snapshot_id(source, normalized, version)
    snapshots_root = output.resolve() / "cann" / version / "snapshots"
    destination = snapshots_root / snapshot_id
    if destination.is_dir():
        return destination
    staging = snapshots_root / f".{snapshot_id}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        raw_root = staging / "raw"
        normalized_root = staging / "normalized"
        facts_root = staging / "facts"
        cards_root = staging / "cards"
        indexes_root = staging / "indexes"
        for directory in (raw_root, normalized_root, facts_root, cards_root, indexes_root):
            directory.mkdir(parents=True, exist_ok=True)
        documents_by_id: dict[str, NormalizedDocument] = {}
        for path, document in normalized:
            relative = path.relative_to(source)
            target = raw_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            documents_by_id[document.document_id] = document
            (normalized_root / f"{document.document_id}.json").write_text(
                json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        extractor = ParameterTableFactExtractor()
        facts = [fact for _, document in normalized for fact in extractor.extract(document)]
        validator = FactValidator()
        issues = [
            asdict(issue)
            for fact in facts
            for issue in validator.validate(fact, documents_by_id)
        ]
        conflicts = ConflictResolver().find_conflicts(facts)
        facts_payload = [{"fact_id": _fact_id(fact), **fact.to_dict()} for fact in facts]
        (facts_root / "atomic_facts.json").write_text(
            json.dumps(facts_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        fact_ids_by_api: dict[str, list[str]] = {}
        for item in facts_payload:
            fact_ids_by_api.setdefault(str(item["applicability"].get("api", "")), []).append(
                item["fact_id"]
            )
        document_ids_by_symbol: dict[str, list[str]] = {}
        for _, document in normalized:
            for symbol in _card_symbols(document):
                document_ids_by_symbol.setdefault(symbol, []).append(document.document_id)
        cards = [
            ApiCard(
                card_id=f"api_{symbol}",
                api=symbol,
                fact_ids=fact_ids_by_api.get(symbol, []),
                document_ids=sorted(set(document_ids)),
            )
            for symbol, document_ids in sorted(document_ids_by_symbol.items())
        ]
        (cards_root / "api_cards.json").write_text(
            json.dumps([asdict(card) for card in cards], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (cards_root / "failure_cards.json").write_text("[]\n", encoding="utf-8")
        (cards_root / "pattern_cards.json").write_text("[]\n", encoding="utf-8")
        symbol_index = {card.api: card.card_id for card in cards}
        (indexes_root / "symbols.json").write_text(
            json.dumps(symbol_index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = {
            "valid": not issues and not conflicts,
            "fact_issues": issues,
            "conflicts": [item.to_dict() for item in conflicts],
            "document_count": len(normalized),
            "fact_count": len(facts),
            "api_card_count": len(cards),
        }
        (staging / "validation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if not report["valid"]:
            raise ValueError(f"snapshot validation failed: {len(issues)} issues, {len(conflicts)} conflicts")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "compiler_version": COMPILER_VERSION,
            "platform": "cann",
            "version": version,
            "snapshot_id": snapshot_id,
            "source": str(source),
            "source_documents": [
                {"path": str(path.relative_to(source)), "sha256": document.source_hash}
                for path, document in normalized
            ],
            "counts": {key: report[key] for key in ("document_count", "fact_count", "api_card_count")},
        }
        (staging / "build_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        snapshots_root.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def load_snapshot(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "build_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((path / "validation_report.json").read_text(encoding="utf-8"))
    if not report.get("valid"):
        raise ValueError("knowledge snapshot is not valid")
    if path.name != manifest.get("snapshot_id"):
        raise ValueError("snapshot directory does not match manifest ID")
    return manifest
