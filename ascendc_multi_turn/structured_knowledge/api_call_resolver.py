from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .router import KnowledgeBuild
from .schema import ResolvedApiCall

_DECLARATION = re.compile(
    r"\b(?:AscendC::)?([A-Z][A-Za-z0-9_]*(?:Params|Tensor|Tiling)(?:<[^;={}]+>)?)\s+([A-Za-z_]\w*)"
)


def _split_arguments(text: str) -> list[str]:
    values: list[str] = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character in "(<[{":
            depth += 1
        elif character in ")>]}":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            values.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        values.append(tail)
    return values


def _calls(text: str, api: str) -> list[tuple[int, list[str]]]:
    pattern = re.compile(rf"\b(?:AscendC::)?{re.escape(api)}\s*(?:<[^;{{}}()]*>)?\s*\(")
    results = []
    for match in pattern.finditer(text):
        depth = 1
        cursor = match.end()
        while cursor < len(text) and depth:
            if text[cursor] == "(":
                depth += 1
            elif text[cursor] == ")":
                depth -= 1
            cursor += 1
        if depth == 0:
            results.append((text[: match.start()].count("\n") + 1, _split_arguments(text[match.end() : cursor - 1])))
    return results


def _base_type(value: str) -> str:
    return value.split("<", 1)[0].split("::")[-1]


def _parameter_name(fact: dict[str, Any]) -> str | None:
    value = fact.get("value")
    if isinstance(value, dict) and isinstance(value.get("parameter"), str):
        return value["parameter"]
    subject = str(fact.get("subject", ""))
    return subject.rsplit(".", 1)[1] if "." in subject else None


def _semantics(fact: dict[str, Any]) -> dict[str, Any]:
    value = fact.get("value")
    result = dict(value) if isinstance(value, dict) else {"value": value}
    description = str(result.get("description", ""))
    unit = re.search(r"单位为(?:：)?\s*(字节|dataBlock|DataBlock|元素个数|元素)", description)
    alignment = re.search(r"(\d+)\s*字节对齐", description)
    if unit and "unit" not in result:
        result["unit"] = {
            "字节": "byte",
            "datablock": "data_block",
            "元素个数": "element",
            "元素": "element",
        }[unit.group(1).lower()]
    if alignment and "alignment" not in result:
        result["alignment"] = int(alignment.group(1))
    return result


class ApiCallResolver:
    def __init__(self, knowledge: KnowledgeBuild):
        self.knowledge = knowledge

    def resolve_text(self, text: str, *, source_path: str) -> list[ResolvedApiCall]:
        variable_types = {name: _base_type(kind) for kind, name in _DECLARATION.findall(text)}
        results: list[ResolvedApiCall] = []
        for api in sorted(self.knowledge.symbols, key=len, reverse=True):
            for line, arguments in _calls(text, api):
                argument_types = [variable_types.get(argument.strip("&* ")) for argument in arguments]
                structures = [kind for kind in argument_types if kind and kind.endswith(("Params", "Tiling"))]
                candidates = [
                    fact
                    for fact in self.knowledge.facts
                    if fact.get("applicability", {}).get("api") == api
                ]
                context_matches = []
                for fact in candidates:
                    structure = str(fact.get("applicability", {}).get("parameter_structure", ""))
                    if structure and not structure.startswith("document_table:") and structure not in structures:
                        continue
                    context_matches.append(fact)
                parameter_semantics: dict[str, dict[str, Any]] = {}
                source_fact_ids = []
                for fact in context_matches:
                    parameter = _parameter_name(fact)
                    if parameter:
                        parameter_semantics[parameter] = _semantics(fact)
                        source_fact_ids.append(str(fact.get("fact_id", "")))
                overloads = {
                    str(fact.get("applicability", {}).get("overload"))
                    for fact in context_matches
                    if fact.get("applicability", {}).get("overload")
                }
                results.append(
                    ResolvedApiCall(
                        api=api,
                        source_path=source_path,
                        line=line,
                        actual_arguments=arguments,
                        argument_types=argument_types,
                        parameter_structures=structures,
                        resolved_overload=next(iter(overloads)) if len(overloads) == 1 else None,
                        parameter_semantics=parameter_semantics,
                        source_fact_ids=[item for item in source_fact_ids if item],
                        status="resolved" if parameter_semantics else "unresolved",
                    )
                )
        return results

    def resolve_tree(self, task_dir: Path) -> list[ResolvedApiCall]:
        results = []
        for path in sorted((task_dir / "kernel").rglob("*")):
            if path.is_file() and path.suffix in {".cpp", ".cc", ".cxx", ".h", ".hpp"}:
                results.extend(
                    self.resolve_text(
                        path.read_text(encoding="utf-8", errors="replace"),
                        source_path=str(path.relative_to(task_dir)),
                    )
                )
        return results
