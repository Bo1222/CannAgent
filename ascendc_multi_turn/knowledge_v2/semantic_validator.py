from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .call_semantics import CallSemanticsResolver
from .router import SnapshotView
from .schema import ResolvedApiCall


@dataclass(frozen=True)
class SemanticIssue:
    path: str
    line: int
    code: str
    message: str
    fact_id: str | None = None

    def render(self) -> str:
        suffix = f" [fact={self.fact_id}]" if self.fact_id else ""
        return f"{self.path}:{self.line}: error[{self.code}]: {self.message}{suffix}"


def _expression_unit(expression: str) -> str | None:
    lowered = expression.lower()
    if "sizeof" in lowered or "byte" in lowered:
        return "byte"
    if "element" in lowered or "numel" in lowered:
        return "element"
    if "block" in lowered:
        return "data_block"
    return None


class SemanticValidator:
    def __init__(self, snapshot: SnapshotView):
        self.snapshot = snapshot
        self.resolver = CallSemanticsResolver(snapshot)

    @staticmethod
    def _assignments(text: str, variable: str, parameter: str) -> list[tuple[int, str]]:
        pattern = re.compile(
            rf"\b{re.escape(variable)}\s*\.\s*{re.escape(parameter)}\s*=\s*([^;]+);"
        )
        return [(text[: match.start()].count("\n") + 1, match.group(1).strip()) for match in pattern.finditer(text)]

    def validate_text(self, text: str, *, source_path: str) -> tuple[list[ResolvedApiCall], list[SemanticIssue]]:
        calls = self.resolver.resolve_text(text, source_path=source_path)
        issues: list[SemanticIssue] = []
        for call in calls:
            fact_id = call.source_fact_ids[0] if call.source_fact_ids else None
            for argument, kind in zip(call.actual_arguments, call.argument_types):
                if not kind or kind not in call.parameter_structures:
                    continue
                variable = argument.strip("&* ")
                for parameter, semantics in call.parameter_semantics.items():
                    for line, expression in self._assignments(text, variable, parameter):
                        expected_unit = semantics.get("unit")
                        actual_unit = _expression_unit(expression)
                        if expected_unit and actual_unit and expected_unit != actual_unit:
                            issues.append(
                                SemanticIssue(
                                    source_path,
                                    line,
                                    "parameter_unit_mismatch",
                                    f"{call.api}.{parameter} requires {expected_unit}, expression appears to use {actual_unit}",
                                    fact_id,
                                )
                            )
                        alignment = semantics.get("alignment")
                        numeric = re.fullmatch(r"\d+", expression)
                        if alignment and numeric and int(expression) % int(alignment):
                            issues.append(
                                SemanticIssue(
                                    source_path,
                                    line,
                                    "parameter_alignment_mismatch",
                                    f"{call.api}.{parameter} value {expression} is not aligned to {alignment}",
                                    fact_id,
                                )
                            )
        for contract in self.snapshot.project_contracts:
            constraint = str(contract.get("constraint", ""))
            if constraint.startswith("required_regex:"):
                pattern = constraint.removeprefix("required_regex:")
                if not re.search(pattern, text, re.MULTILINE):
                    issues.append(SemanticIssue(source_path, 1, "project_contract_missing", contract.get("subject", "required contract"), contract.get("contract_id")))
            elif constraint.startswith("forbidden_regex:"):
                pattern = constraint.removeprefix("forbidden_regex:")
                match = re.search(pattern, text, re.MULTILINE)
                if match:
                    issues.append(SemanticIssue(source_path, text[: match.start()].count("\n") + 1, "project_contract_forbidden", contract.get("subject", "forbidden contract"), contract.get("contract_id")))
        return calls, issues

    def validate_tree(self, task_dir: Path) -> tuple[list[ResolvedApiCall], list[SemanticIssue]]:
        calls: list[ResolvedApiCall] = []
        issues: list[SemanticIssue] = []
        for path in sorted((task_dir / "kernel").rglob("*")):
            if path.is_file() and path.suffix in {".cpp", ".cc", ".cxx", ".h", ".hpp"}:
                current_calls, current_issues = self.validate_text(
                    path.read_text(encoding="utf-8", errors="replace"),
                    source_path=str(path.relative_to(task_dir)),
                )
                calls.extend(current_calls)
                issues.extend(current_issues)
        return calls, issues
