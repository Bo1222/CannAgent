#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "references" / "AscendC_knowledge" / "api_reference"
INDEX_PATH = API_ROOT / "INDEX.md"
OUTPUT_PATH = API_ROOT / "manifest.json"
IDENTIFIER = re.compile(r"\b[A-Z][A-Za-z0-9_]{2,}\b")
SIGNATURE = re.compile(r"^[^#\n]*\b([A-Z][A-Za-z0-9_]*)\s*(?:<[^>\n]*>)?\s*\([^;{}]*\)\s*;?\s*$")
PARAMETER_TYPE = re.compile(r"(?:AscendC::)?([A-Z][A-Za-z0-9_]*(?:Params|Tiling))")


def index_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    title = ""
    for line in INDEX_PATH.read_text(encoding="utf-8").splitlines():
        if line.startswith("- 标题: "):
            title = line.removeprefix("- 标题: ").strip()
            continue
        if not title or not line.startswith("- 文件: "):
            continue
        match = re.search(r"\(([^)]+)\)", line)
        if not match:
            continue
        relative = match.group(1)
        path = API_ROOT / relative
        content = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        name = title.split("-", 1)[0].strip()
        primary_name = name.split("(", 1)[0].strip()
        signatures: list[str] = []
        signature_symbols: set[str] = set()
        for raw_line in content.splitlines():
            line = raw_line.strip().strip("`")
            match_signature = SIGNATURE.match(line)
            if match_signature and len(line) <= 500:
                signature_symbols.add(match_signature.group(1))
                if primary_name in line and line not in signatures:
                    signatures.append(line)
        struct_symbols = set(
            re.findall(r"\b(?:struct|class)\s+([A-Z][A-Za-z0-9_]*)\b", content)
        )
        symbols = sorted(
            set(IDENTIFIER.findall(primary_name))
            | signature_symbols
            | struct_symbols
            | set(PARAMETER_TYPE.findall(content))
        )
        title_parts = [item.strip() for item in title.split("-") if item.strip()]
        entries.append(
            {
                "doc_id": Path(relative).stem,
                "name": name,
                "title": title,
                "path": relative,
                "family": title_parts[1] if len(title_parts) > 1 else "",
                "symbols": symbols[:80],
                "signatures": signatures[:8],
            }
        )
        title = ""
    return entries


def main() -> None:
    documents = index_entries()
    duplicate_names: dict[str, list[str]] = {}
    for entry in documents:
        duplicate_names.setdefault(str(entry["name"]), []).append(str(entry["doc_id"]))
    payload = {
        "schema_version": 2,
        "source_index": "INDEX.md",
        "duplicate_names": {
            name: ids for name, ids in sorted(duplicate_names.items()) if len(ids) > 1
        },
        "documents": documents,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
