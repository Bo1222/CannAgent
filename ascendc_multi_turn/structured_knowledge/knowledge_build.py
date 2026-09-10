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
from .schema import SCHEMA_VERSION, ApiCard, AtomicFact, NormalizedDocument, Provenance
from .validate import ConflictResolver, FactValidator

COMPILER_VERSION = "4"
_API_SYMBOL = re.compile(r"\b(?:AscendC::)?([A-Z][A-Za-z0-9_]{2,})\b")
_SQUASHED_CORE_SYMBOL = re.compile(r"AscendC::(DataCopyPad|DataCopy|TPipe|TQue)")


def _canonical(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fact_id(fact: AtomicFact) -> str:
    return "fact_" + hashlib.sha256(_canonical(fact.to_dict()).encode()).hexdigest()[:20]


def _discover(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*.md") if path.is_file())


def _knowledge_build_id(
    source: Path,
    documents: list[tuple[Path, NormalizedDocument]],
    version: str,
) -> str:
    project_manifest = source / "project_knowledge.json"
    payload = {
        "platform": "cann",
        "version": version,
        "schema_version": SCHEMA_VERSION,
        "compiler_version": COMPILER_VERSION,
        "documents": [(str(path.relative_to(source)), document.source_hash) for path, document in documents],
        "project_manifest": (
            hashlib.sha256(project_manifest.read_bytes()).hexdigest()
            if project_manifest.is_file()
            else None
        ),
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:24]


def _card_symbols(document: NormalizedDocument) -> set[str]:
    text = "\n".join(
        [document.title]
        + [item.text for item in document.paragraphs]
        + [cell for table in document.tables for row in table.rows for cell in row]
    )
    return set(_API_SYMBOL.findall(text)) | set(_SQUASHED_CORE_SYMBOL.findall(text))


def _project_cards(
    source: Path,
    documents: list[tuple[Path, NormalizedDocument]],
) -> dict[str, list[dict[str, Any]]]:
    manifest = source / "project_knowledge.json"
    empty = {"project_contracts": [], "failure_cards": [], "pattern_cards": []}
    if not manifest.is_file():
        return empty
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    by_relative = {path.relative_to(source).as_posix(): document for path, document in documents}
    result = {key: [] for key in empty}
    for key, cards in result.items():
        for raw in payload.get(key, []):
            item = dict(raw)
            guide = item.pop("guide")
            section = item.pop("section")
            evidence = item.pop("evidence")
            document = by_relative.get(guide)
            if document is None or evidence not in Path(document.source_path).read_text(encoding="utf-8"):
                raise ValueError(f"project knowledge evidence is missing: {guide}: {evidence}")
            item["authority"] = "PROJECT_CONTRACT"
            item["provenance"] = asdict(
                Provenance(document.document_id, guide, document.source_hash, section, evidence)
            )
            cards.append(item)
    return result


def _publish_current(version_root: Path, knowledge_build_id: str) -> None:
    current = version_root / "current.json"
    temporary = version_root / f".current-{os.getpid()}.json"
    temporary.write_text(
        json.dumps({"knowledge_build_id": knowledge_build_id}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, current)


def build_knowledge(*, source: Path, output: Path, version: str) -> Path:
    source = source.resolve()
    normalizer = MarkdownNormalizer()
    normalized = [(path, normalizer.parse_path(path)) for path in _discover(source)]
    if not normalized:
        raise ValueError(f"no Markdown documents found under {source}")
    knowledge_build_id = _knowledge_build_id(source, normalized, version)
    version_root = output.resolve() / "cann" / version
    builds_root = version_root / "builds"
    destination = builds_root / knowledge_build_id
    if destination.is_dir():
        _publish_current(version_root, knowledge_build_id)
        return destination
    staging = builds_root / f".{knowledge_build_id}.staging-{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        raw_root = staging / "raw"
        normalized_root = staging / "normalized"
        facts_root = staging / "facts"
        cards_root = staging / "cards"
        indexes_root = staging / "indexes"
        for directory in (raw_root, normalized_root, facts_root, cards_root, indexes_root):
            directory.mkdir(parents=True, exist_ok=True)
        project_manifest = source / "project_knowledge.json"
        if project_manifest.is_file():
            shutil.copy2(project_manifest, raw_root / project_manifest.name)
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
        project_cards = _project_cards(source, normalized)
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
        for path, document in normalized:
            if path.relative_to(source).parts[0] == "project_guides":
                continue
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
        for name in ("failure_cards", "pattern_cards", "project_contracts"):
            (cards_root / f"{name}.json").write_text(
                json.dumps(project_cards[name], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
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
            raise ValueError(
                f"knowledge build validation failed: {len(issues)} issues, "
                f"{len(conflicts)} conflicts"
            )
        source_documents = [
            {"path": str(path.relative_to(source)), "sha256": document.source_hash}
            for path, document in normalized
        ]
        if project_manifest.is_file():
            source_documents.append(
                {
                    "path": "project_knowledge.json",
                    "sha256": hashlib.sha256(project_manifest.read_bytes()).hexdigest(),
                }
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "compiler_version": COMPILER_VERSION,
            "platform": "cann",
            "version": version,
            "knowledge_build_id": knowledge_build_id,
            "source": str(source),
            "source_documents": source_documents,
            "counts": {key: report[key] for key in ("document_count", "fact_count", "api_card_count")},
        }
        (staging / "build_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        builds_root.mkdir(parents=True, exist_ok=True)
        os.replace(staging, destination)
        _publish_current(version_root, knowledge_build_id)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def load_knowledge_build(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "build_manifest.json").read_text(encoding="utf-8"))
    report = json.loads((path / "validation_report.json").read_text(encoding="utf-8"))
    if not report.get("valid"):
        raise ValueError("knowledge build is not valid")
    if path.name != manifest.get("knowledge_build_id"):
        raise ValueError("knowledge build directory does not match manifest ID")
    return manifest
