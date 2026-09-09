from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_config import get_env

from .diagnostics import compact_evaluation, extract_api_symbols
from .models import EvalResult, FileBundle


REPO_ROOT = Path(__file__).resolve().parent.parent
SHARED_ASCENDC_ROOT = REPO_ROOT / "skills/ascendc/ascendc-translator"
REFERENCES_ROOT = SHARED_ASCENDC_ROOT / "references"
KNOWLEDGE_ROOT = REFERENCES_ROOT / "AscendC_knowledge"
CATALOG_PATH = KNOWLEDGE_ROOT / "knowledge_catalog.json"
MANIFEST_PATH = KNOWLEDGE_ROOT / "api_reference/manifest.json"

CORE_DOCUMENTS = (
    REFERENCES_ROOT / "direct_llm_core.md",
)

SUPPLEMENT_DOCUMENTS = {
    path.name: path
    for path in (
        REFERENCES_ROOT / "ascendc_dynamic_quant_kb.md",
        REFERENCES_ROOT / "dequant_kernel_patterns.md",
    )
}


@dataclass(frozen=True)
class KnowledgeVersion:
    runtime_version: str
    knowledge_version: str
    status: str
    warning: str = ""

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class KnowledgeSelection:
    domain: str
    topics: list[str]
    api_names: list[str]
    supplements: list[str]
    reason: str
    selected_files: list[str]
    fallback: bool = False
    mode: str = "initial_full"
    doc_ids: list[str] | None = None
    added_doc_ids: list[str] | None = None
    removed_doc_ids: list[str] | None = None
    rendered_doc_ids: list[str] | None = None
    runtime_fact_symbols: list[str] | None = None
    conflicts: list[str] | None = None

    def __post_init__(self) -> None:
        for name in (
            "doc_ids",
            "added_doc_ids",
            "removed_doc_ids",
            "rendered_doc_ids",
            "runtime_fact_symbols",
            "conflicts",
        ):
            if getattr(self, name) is None:
                setattr(self, name, [])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeState:
    runtime_version: str
    knowledge_version: str
    initialized: bool = False
    working_doc_ids: list[str] | None = None
    supplements: list[str] | None = None
    known_symbols: list[str] | None = None
    failure_fingerprints: list[str] | None = None
    full_route_count: int = 0
    incremental_route_count: int = 0
    last_round: int = 0
    migrated: bool = False

    def __post_init__(self) -> None:
        for name in ("working_doc_ids", "supplements", "known_symbols", "failure_fingerprints"):
            if getattr(self, name) is None:
                setattr(self, name, [])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _parse_version_text(text: str) -> str | None:
    match = re.search(r"(?<!\d)(\d+\.\d+(?:\.\d+)?)(?!\d)", text)
    if not match:
        return None
    parts = match.group(1).split(".")
    return ".".join(parts + ["0"] * (3 - len(parts)))


def detect_cann_version(explicit: str, *, mock: bool = False) -> str:
    if explicit and explicit.lower() != "auto":
        parsed = _parse_version_text(explicit)
        if not parsed:
            raise ValueError(f"invalid CANN version: {explicit!r}")
        return parsed
    if mock:
        return "8.5.0"

    candidates: list[Path] = []
    for name in ("ASCEND_HOME_PATH", "ASCEND_INSTALL_PATH"):
        value = os.getenv(name)
        if value:
            candidates.append(Path(value).expanduser())
    candidates.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/ascend-toolkit"),
            Path("/usr/local/Ascend"),
        ]
    )
    seen: set[Path] = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        for relative in ("compiler/version.info", "opp/version.info", "version.info"):
            path = root / relative
            if path.is_file():
                parsed = _parse_version_text(path.read_text(encoding="utf-8", errors="replace"))
                if parsed:
                    return parsed
    raise RuntimeError(
        "Unable to detect the installed CANN version. Set ASCEND_HOME_PATH "
        "or pass --cann-version X.Y.Z."
    )


def select_knowledge_version(runtime_version: str) -> KnowledgeVersion:
    versions = sorted(_read_catalog()["versions"], key=lambda value: tuple(map(int, value.split("."))))
    if runtime_version in versions:
        return KnowledgeVersion(runtime_version, runtime_version, "exact")
    runtime_major = runtime_version.split(".", 1)[0]
    compatible = [version for version in versions if version.split(".", 1)[0] == runtime_major]
    if not compatible:
        raise RuntimeError(
            f"No bundled AscendC knowledge matches CANN {runtime_version}. "
            "Cross-major knowledge fallback is disabled."
        )
    selected = compatible[-1]
    return KnowledgeVersion(
        runtime_version,
        selected,
        "same-major-fallback",
        f"Runtime CANN {runtime_version} has no exact local documentation; using {selected}. "
        "Compilation and evaluation results take precedence over documentation assumptions.",
    )


def _index_entries(version: KnowledgeVersion) -> list[dict[str, Any]]:
    metadata = _read_catalog()["versions"][version.knowledge_version]
    index_path = KNOWLEDGE_ROOT / metadata["index"]
    if MANIFEST_PATH.is_file():
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        documents = payload.get("documents", [])
        if isinstance(documents, list) and documents:
            return [entry for entry in documents if isinstance(entry, dict)]
    text = index_path.read_text(encoding="utf-8")
    entries: list[dict[str, Any]] = []
    current_title = ""
    for line in text.splitlines():
        if line.startswith("- 标题: "):
            current_title = line.removeprefix("- 标题: ").strip()
        elif current_title and line.startswith("- 文件: "):
            match = re.search(r"\(([^)]+)\)", line)
            if match:
                entries.append(
                    {
                        "doc_id": Path(match.group(1)).stem,
                        "name": current_title.split("-", 1)[0].strip(),
                        "title": current_title,
                        "path": match.group(1),
                        "symbols": [],
                    }
                )
            current_title = ""
    return entries


def build_knowledge_prompt(
    *,
    reference_code: str,
    current: FileBundle | None,
    previous_result: EvalResult | None,
    version: KnowledgeVersion,
    mode: str = "initial_full",
    candidate_doc_ids: list[str] | None = None,
    working_doc_ids: list[str] | None = None,
) -> str:
    entries = _index_entries(version)
    if candidate_doc_ids is not None:
        allowed = set(candidate_doc_ids)
        entries = [entry for entry in entries if entry["doc_id"] in allowed]
    compact_index = "\n".join(
        f"- {entry['doc_id']} | {entry['name']} | {entry.get('family', '')}" for entry in entries
    )
    current_paths = sorted(current.files) if current else []
    previous = compact_evaluation(previous_result)
    supplements = ", ".join(sorted(SUPPLEMENT_DOCUMENTS))
    return f"""Select the direct AscendC knowledge required for the next implementation round.
Return exactly one JSON object with keys: domain, topics, doc_ids, supplements, reason.
domain must be \"ascendc\". doc_ids must use exact IDs from the index below.
supplements may only use these file names: {supplements}.
Choose only directly relevant material; do not return file paths.

Routing mode: {mode}
Runtime CANN: {version.runtime_version}
Knowledge CANN: {version.knowledge_version} ({version.status})
Current source files: {json.dumps(current_paths)}
Current working document IDs: {json.dumps(working_doc_ids or [])}
Previous evaluation: {json.dumps(previous, ensure_ascii=False)}

Reference model:
```python
{reference_code[:16000]}
```

Available API index:
{compact_index}
"""


def _json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("knowledge response does not contain a JSON object")
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("knowledge response must be a JSON object")
    return payload


def _string_list(payload: dict[str, Any], key: str) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"knowledge response {key} must be a string list")
    return value


def parse_knowledge_selection(
    text: str,
    *,
    version: KnowledgeVersion,
    max_api_docs: int,
    fallback_text: str,
) -> KnowledgeSelection:
    entries = _index_entries(version)
    by_name: dict[str, list[dict[str, Any]]] = {}
    by_id = {entry["doc_id"]: entry for entry in entries}
    for entry in entries:
        by_name.setdefault(entry["name"].lower(), []).append(entry)
    try:
        payload = _json_object(text)
        domain = payload.get("domain")
        legacy_skill = payload.get("skill")
        if domain not in {None, "ascendc"}:
            raise ValueError("unsupported knowledge domain")
        if legacy_skill not in {None, "ascendc", "ascendc-translator"}:
            raise ValueError("unsupported legacy skill")
        requested_ids = _string_list(payload, "doc_ids") if "doc_ids" in payload else []
        requested = _string_list(payload, "api_names") if "api_names" in payload else []
        supplements = _string_list(payload, "supplements")[:2]
        if any(name not in SUPPLEMENT_DOCUMENTS for name in supplements):
            raise ValueError("unknown supplement")
        selected_entries: list[dict[str, Any]] = []
        if requested_ids:
            for doc_id in requested_ids:
                entry = by_id.get(doc_id)
                if entry is not None and entry not in selected_entries:
                    selected_entries.append(entry)
                if len(selected_entries) >= max_api_docs:
                    break
        else:
            for name in requested:
                for entry in by_name.get(name.lower(), []):
                    if entry not in selected_entries:
                        selected_entries.append(entry)
                        break
                if len(selected_entries) >= max_api_docs:
                    break
        if (requested_ids or requested) and not selected_entries:
            raise ValueError("no requested API document exists in the selected index")
        return KnowledgeSelection(
            domain="ascendc",
            topics=_string_list(payload, "topics"),
            api_names=[entry["name"] for entry in selected_entries],
            supplements=supplements,
            reason=str(payload.get("reason", "")),
            selected_files=[entry["path"] for entry in selected_entries],
            mode=str(payload.get("mode", "initial_full")),
            doc_ids=[entry["doc_id"] for entry in selected_entries],
        )
    except (ValueError, json.JSONDecodeError, TypeError):
        fallback_ids = [
            doc_id
            for doc_id, _score in candidate_doc_ids(
                fallback_text,
                version=version,
                limit=max_api_docs,
            )
        ]
        selected_entries = [by_id[doc_id] for doc_id in fallback_ids if doc_id in by_id]
        return KnowledgeSelection(
            domain="ascendc",
            topics=["deterministic-fallback"],
            api_names=[entry["name"] for entry in selected_entries],
            supplements=[],
            reason="Router output was invalid or unmatched; selected APIs from task and source text.",
            selected_files=[entry["path"] for entry in selected_entries],
            fallback=True,
            doc_ids=[entry["doc_id"] for entry in selected_entries],
        )


def load_knowledge_state(
    path: Path,
    *,
    version: KnowledgeVersion,
    legacy_selection: dict[str, Any] | None = None,
) -> KnowledgeState:
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        state = KnowledgeState(**payload)
        if (state.runtime_version, state.knowledge_version) != (
            version.runtime_version,
            version.knowledge_version,
        ):
            raise ValueError("knowledge state version does not match the selected CANN knowledge")
        state.supplements = [
            item for item in state.supplements if item in SUPPLEMENT_DOCUMENTS
        ]
        return state
    state = KnowledgeState(version.runtime_version, version.knowledge_version)
    if legacy_selection:
        entries = _index_entries(version)
        by_path = {entry["path"]: entry["doc_id"] for entry in entries}
        state.working_doc_ids = [
            by_path[path]
            for path in legacy_selection.get("selected_files", [])
            if path in by_path
        ]
        state.supplements = [
            item for item in legacy_selection.get("supplements", []) if item in SUPPLEMENT_DOCUMENTS
        ]
        state.initialized = True
        state.migrated = True
    return state


def save_knowledge_state(path: Path, state: KnowledgeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def candidate_doc_ids(
    evidence: str,
    *,
    version: KnowledgeVersion,
    exclude: list[str] | None = None,
    limit: int = 5,
) -> list[tuple[str, int]]:
    excluded = set(exclude or [])
    lowered = evidence.lower()
    symbols = {symbol.lower() for symbol in extract_api_symbols(evidence)}
    ranked: list[tuple[str, int]] = []
    for entry in _index_entries(version):
        if entry["doc_id"] in excluded:
            continue
        name = str(entry["name"])
        primary_name = name.split("(", 1)[0].strip().lower()
        score = 0
        if primary_name in symbols:
            score += 120
        elif len(primary_name) >= 4 and re.search(
            rf"(?<![a-z0-9_]){re.escape(primary_name)}(?![a-z0-9_])", lowered
        ):
            score += 80
        entry_symbols = {str(item).lower() for item in entry.get("symbols", [])}
        score += min(180, 60 * len(symbols & entry_symbols))
        family = str(entry.get("family", "")).lower()
        if family and len(family) >= 4 and family in lowered:
            score += 5
        if score:
            ranked.append((entry["doc_id"], score))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:limit]


def apply_selection_to_state(
    state: KnowledgeState,
    selection: KnowledgeSelection,
    *,
    max_docs: int,
    preferred_doc_ids: list[str] | None = None,
) -> None:
    before = list(state.working_doc_ids)
    preferred = list(dict.fromkeys(preferred_doc_ids or selection.doc_ids or []))
    combined = list(dict.fromkeys([*preferred, *selection.doc_ids, *before]))
    state.working_doc_ids = combined[:max_docs]
    state.supplements = list(dict.fromkeys([*selection.supplements, *state.supplements]))[:2]
    selection.added_doc_ids = [item for item in state.working_doc_ids if item not in before]
    selection.removed_doc_ids = [item for item in before if item not in state.working_doc_ids]
    state.initialized = True


def active_doc_ids(
    state: KnowledgeState,
    evidence: str,
    *,
    version: KnowledgeVersion,
    preferred_doc_ids: list[str] | None = None,
    limit: int = 5,
) -> list[str]:
    preferred = [item for item in preferred_doc_ids or [] if item in state.working_doc_ids]
    ranked = candidate_doc_ids(evidence, version=version, limit=len(_index_entries(version)))
    scores = dict(ranked)
    remaining = sorted(
        (item for item in state.working_doc_ids if item not in preferred),
        key=lambda item: (-scores.get(item, 0), state.working_doc_ids.index(item)),
    )
    return list(dict.fromkeys([*preferred, *remaining]))[:limit]


def selection_from_state(
    state: KnowledgeState,
    *,
    version: KnowledgeVersion,
    mode: str,
    active_ids: list[str] | None = None,
    reason: str = "Reused the task-level knowledge working set.",
) -> KnowledgeSelection:
    by_id = {entry["doc_id"]: entry for entry in _index_entries(version)}
    doc_ids = [item for item in (active_ids or state.working_doc_ids) if item in by_id]
    entries = [by_id[item] for item in doc_ids]
    return KnowledgeSelection(
        domain="ascendc",
        topics=[mode],
        api_names=[entry["name"] for entry in entries],
        supplements=list(state.supplements),
        reason=reason,
        selected_files=[entry["path"] for entry in entries],
        mode=mode,
        doc_ids=doc_ids,
    )


_API_SECTION = re.compile(
    r"^#{1,6}\s+.*(?:功能说明|函数原型|参数说明|约束说明|调用示例|返回值说明).*$",
    re.MULTILINE,
)


def _api_excerpt(content: str, *, max_chars: int = 3000) -> str:
    matches = list(_API_SECTION.finditer(content))
    if not matches:
        excerpt = content[:max_chars]
    else:
        pieces: list[str] = []
        for match in matches:
            next_heading = re.search(r"^#{1,6}\s+", content[match.end() :], re.MULTILINE)
            end = match.end() + next_heading.start() if next_heading else len(content)
            piece = content[match.start() : end].strip()
            if sum(len(item) + 2 for item in pieces) + len(piece) > max_chars:
                continue
            pieces.append(piece)
        excerpt = "\n\n".join(pieces) or content[:max_chars]
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars]
    if len(excerpt) == max_chars and "\n" in excerpt:
        excerpt = excerpt.rsplit("\n", 1)[0]
    return excerpt


def render_knowledge_with_metadata(
    selection: KnowledgeSelection,
    *,
    version: KnowledgeVersion,
    max_chars: int,
    runtime_facts: str = "",
    conflicts: list[str] | None = None,
) -> tuple[str, list[str]]:
    metadata = _read_catalog()["versions"][version.knowledge_version]
    index_dir = (KNOWLEDGE_ROOT / metadata["index"]).parent.resolve()
    by_id = {entry["doc_id"]: entry for entry in _index_entries(version)}
    chunks = [
        "# AscendC task knowledge",
        f"Runtime CANN: {version.runtime_version}",
        f"Fallback documentation: CANN {version.knowledge_version} ({version.status})",
        "Priority: compiler diagnostics and installed public headers override fallback documentation.",
    ]
    if version.warning:
        chunks.append(f"WARNING: {version.warning}")
    if conflicts:
        chunks.append("\n## Runtime/document conflicts\n" + "\n".join(f"- {item}" for item in conflicts))
    if runtime_facts:
        chunks.append("\n## Runtime public-header facts\n" + runtime_facts)
    for path in CORE_DOCUMENTS:
        if path.is_file():
            chunk = f"\n## Core rules: {path.name}\n{path.read_text(encoding='utf-8')}"
            if len("\n".join([*chunks, chunk])) <= max_chars:
                chunks.append(chunk)
    rendered: list[str] = []
    for doc_id in selection.doc_ids:
        entry = by_id.get(doc_id)
        if entry is None:
            continue
        path = (index_dir / entry["path"]).resolve()
        path.relative_to(index_dir)
        if not path.is_file():
            continue
        excerpt = _api_excerpt(path.read_text(encoding="utf-8", errors="replace"))
        chunk = f"\n## API {entry['name']} [{doc_id}]\n{excerpt}"
        if len("\n".join([*chunks, chunk])) > max_chars:
            continue
        chunks.append(chunk)
        rendered.append(doc_id)
    for name in selection.supplements:
        path = SUPPLEMENT_DOCUMENTS.get(name)
        if path and path.is_file():
            chunk = f"\n## Supplement: {name}\n{path.read_text(encoding='utf-8')[:2500]}"
            if len("\n".join([*chunks, chunk])) <= max_chars:
                chunks.append(chunk)
    text = "\n".join(chunks)
    if len(text) > max_chars:
        text = text[:max_chars]
        if "\n" in text:
            text = text.rsplit("\n", 1)[0]
    return text, rendered


def render_knowledge(
    selection: KnowledgeSelection,
    *,
    version: KnowledgeVersion,
    max_chars: int,
    runtime_facts: str = "",
    conflicts: list[str] | None = None,
) -> str:
    text, rendered = render_knowledge_with_metadata(
        selection,
        version=version,
        max_chars=max_chars,
        runtime_facts=runtime_facts,
        conflicts=conflicts,
    )
    selection.rendered_doc_ids = rendered
    return text


def resolve_knowledge_version(config) -> KnowledgeVersion:
    runtime = detect_cann_version(config.cann_version, mock=config.mock)
    return select_knowledge_version(runtime)


def knowledge_limits() -> tuple[int, int]:
    return (
        int(get_env("ASCENDC_KNOWLEDGE_MAX_API_DOCS", "8")),
        int(get_env("ASCENDC_KNOWLEDGE_MAX_CHARS", "24000")),
    )
