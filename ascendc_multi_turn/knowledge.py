from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_config import get_env

from .models import EvalResult, FileBundle


REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSLATOR_ROOT = REPO_ROOT / "skills/ascendc/ascendc-translator"
REFERENCES_ROOT = TRANSLATOR_ROOT / "references"
KNOWLEDGE_ROOT = REFERENCES_ROOT / "AscendC_knowledge"
CATALOG_PATH = KNOWLEDGE_ROOT / "knowledge_catalog.json"

CORE_DOCUMENTS = (
    TRANSLATOR_ROOT / "SKILL.md",
    REFERENCES_ROOT / "dsl2Ascendc.md",
    REFERENCES_ROOT / "TileLang-AscendC-API-Mapping.md",
    REFERENCES_ROOT / "AscendCVerification.md",
)

SUPPLEMENT_DOCUMENTS = {
    path.name: path
    for path in (
        REFERENCES_ROOT / "dsl2Ascendc_compute_vector.md",
        REFERENCES_ROOT / "dsl2Ascendc_compute_cube.md",
        REFERENCES_ROOT / "dsl2Ascendc_compute_cv.md",
        REFERENCES_ROOT / "dsl2Ascendc_cross_core_sync.md",
        REFERENCES_ROOT / "dsl2Ascendc_host.md",
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
    skill: str
    topics: list[str]
    api_names: list[str]
    supplements: list[str]
    reason: str
    selected_files: list[str]
    fallback: bool = False

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


def _index_entries(version: KnowledgeVersion) -> list[dict[str, str]]:
    metadata = _read_catalog()["versions"][version.knowledge_version]
    index_path = KNOWLEDGE_ROOT / metadata["index"]
    text = index_path.read_text(encoding="utf-8")
    entries: list[dict[str, str]] = []
    current_title = ""
    for line in text.splitlines():
        if line.startswith("- 标题: "):
            current_title = line.removeprefix("- 标题: ").strip()
        elif current_title and line.startswith("- 文件: "):
            match = re.search(r"\(([^)]+)\)", line)
            if match:
                entries.append(
                    {
                        "name": current_title.split("-", 1)[0].strip(),
                        "title": current_title,
                        "path": match.group(1),
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
) -> str:
    entries = _index_entries(version)
    compact_index = "\n".join(f"- {entry['name']}: {entry['title']}" for entry in entries)
    current_paths = sorted(current.files) if current else []
    previous = previous_result.to_dict() if previous_result else None
    supplements = ", ".join(sorted(SUPPLEMENT_DOCUMENTS))
    return f"""Select the AscendC skill knowledge required for the next implementation round.
Return exactly one JSON object with keys: skill, topics, api_names, supplements, reason.
skill must be \"ascendc-translator\". api_names must use names from the index below.
supplements may only use these file names: {supplements}.
Choose only directly relevant material; do not return file paths.

Runtime CANN: {version.runtime_version}
Knowledge CANN: {version.knowledge_version} ({version.status})
Current source files: {json.dumps(current_paths)}
Previous evaluation: {json.dumps(previous, ensure_ascii=False)[:12000]}

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
    by_name: dict[str, list[dict[str, str]]] = {}
    for entry in entries:
        by_name.setdefault(entry["name"].lower(), []).append(entry)
    try:
        payload = _json_object(text)
        if payload.get("skill") != "ascendc-translator":
            raise ValueError("unsupported skill")
        requested = _string_list(payload, "api_names")
        supplements = _string_list(payload, "supplements")[:2]
        if any(name not in SUPPLEMENT_DOCUMENTS for name in supplements):
            raise ValueError("unknown supplement")
        selected_entries: list[dict[str, str]] = []
        for name in requested:
            for entry in by_name.get(name.lower(), []):
                if entry not in selected_entries:
                    selected_entries.append(entry)
                    break
            if len(selected_entries) >= max_api_docs:
                break
        if requested and not selected_entries:
            raise ValueError("no requested API name exists in the selected index")
        return KnowledgeSelection(
            skill="ascendc-translator",
            topics=_string_list(payload, "topics"),
            api_names=[entry["name"] for entry in selected_entries],
            supplements=supplements,
            reason=str(payload.get("reason", "")),
            selected_files=[entry["path"] for entry in selected_entries],
        )
    except (ValueError, json.JSONDecodeError, TypeError):
        lowered = fallback_text.lower()
        selected_entries = []
        for entry in entries:
            name = entry["name"]
            if len(name) >= 3 and re.search(rf"\b{re.escape(name.lower())}\b", lowered):
                selected_entries.append(entry)
            if len(selected_entries) >= max_api_docs:
                break
        return KnowledgeSelection(
            skill="ascendc-translator",
            topics=["deterministic-fallback"],
            api_names=[entry["name"] for entry in selected_entries],
            supplements=[],
            reason="Router output was invalid or unmatched; selected APIs from task and source text.",
            selected_files=[entry["path"] for entry in selected_entries],
            fallback=True,
        )


def render_knowledge(
    selection: KnowledgeSelection,
    *,
    version: KnowledgeVersion,
    max_chars: int,
) -> str:
    metadata = _read_catalog()["versions"][version.knowledge_version]
    index_dir = (KNOWLEDGE_ROOT / metadata["index"]).parent
    documents = list(CORE_DOCUMENTS)
    documents.extend(SUPPLEMENT_DOCUMENTS[name] for name in selection.supplements)
    for relative in selection.selected_files:
        candidate = (index_dir / relative).resolve()
        candidate.relative_to(index_dir.resolve())
        documents.append(candidate)

    chunks = [
        "# AscendC skill and versioned knowledge",
        f"Runtime CANN: {version.runtime_version}",
        f"Knowledge CANN: {version.knowledge_version} ({version.status})",
    ]
    if version.warning:
        chunks.append(f"WARNING: {version.warning}")
    used = sum(len(chunk) for chunk in chunks)
    for path in documents:
        if not path.is_file():
            continue
        content = path.read_text(encoding="utf-8")
        header = f"\n\n## {path.name}\n"
        available = max_chars - used - len(header)
        if available <= 0:
            break
        excerpt = content[:available]
        chunks.append(header + excerpt)
        used += len(header) + len(excerpt)
    return "\n".join(chunks)


def resolve_knowledge_version(config) -> KnowledgeVersion:
    runtime = detect_cann_version(config.cann_version, mock=config.mock)
    return select_knowledge_version(runtime)


def knowledge_limits() -> tuple[int, int]:
    return (
        int(get_env("ASCENDC_KNOWLEDGE_MAX_API_DOCS", "8")),
        int(get_env("ASCENDC_KNOWLEDGE_MAX_CHARS", "60000")),
    )
