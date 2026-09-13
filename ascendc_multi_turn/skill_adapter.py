from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_MAPPING_PATH = Path(__file__).with_name("skill_mapping.yaml")


@dataclass(frozen=True)
class SkillAdapterContext:
    audience: str
    stages: list[str]
    operator: str
    operator_families: list[str]
    soc: str
    runtime_version: str
    knowledge_version: str
    failure_stage: str | None
    failure_code: str | None
    failure_evidence: str
    failure_symbols: list[str]
    has_correct_baseline: bool
    evidence: str
    source_symbols: list[str] = field(default_factory=list)
    planned_symbols: list[str] = field(default_factory=list)
    symbol_evidence: list[dict[str, Any]] = field(default_factory=list)
    primary_skill: str = "kernel_design"
    route_reason: str = "default_kernel_design"
    secondary_skill: str | None = None
    secondary_reason: str | None = None
    debug_category: str = "kernel_design"
    routing_confidence: str = "fallback"
    route_evidence_origin: str = "workflow"
    matched_route_trigger: str | None = None
    environment_fingerprint: str | None = None


@dataclass
class SkillExcerpt:
    source: str
    headings: list[str]
    text: str
    origin: str = "cannbot"
    confidence_level: int = 1
    provenance: str = "documented_skill"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillCapsule:
    skill_id: str
    stage: str
    purpose: str
    trigger_reason: str
    expected_artifact: str
    knowledge_type: list[str]
    provided_context: list[str]
    exclusions: list[str]
    excerpts: list[SkillExcerpt] = field(default_factory=list)
    role: str = "support"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillAdapterSelection:
    capsules: list[SkillCapsule]
    trace: list[dict[str, Any]]
    source_root: str | None
    source_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower())


def _heading_title(line: str) -> str | None:
    match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
    return match.group(1).strip() if match else None


def _markdown_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    return [item for item in blocks if item]


def _fit_markdown_blocks(text: str, max_chars: int | None) -> str:
    if max_chars is None:
        return "\n\n".join(_markdown_blocks(text))
    selected: list[str] = []
    used = 0
    for block in _markdown_blocks(text):
        addition = len(block) + (2 if selected else 0)
        if used + addition > max_chars:
            break
        selected.append(block)
        used += addition
    return "\n\n".join(selected)


def _extract_markdown_sections(
    text: str, headings: list[str], *, max_chars: int | None
) -> str:
    if not headings:
        return _fit_markdown_blocks(text, max_chars).rstrip()
    wanted = [_normalized_text(item) for item in headings]
    lines = text.splitlines()
    selected: list[str] = []
    index = 0
    while index < len(lines):
        title = _heading_title(lines[index])
        if title is None or not any(item in _normalized_text(title) for item in wanted):
            index += 1
            continue
        level = len(lines[index]) - len(lines[index].lstrip("#"))
        end = index + 1
        while end < len(lines):
            candidate = _heading_title(lines[end])
            if candidate is not None:
                candidate_level = len(lines[end]) - len(lines[end].lstrip("#"))
                if candidate_level <= level:
                    break
            end += 1
        section = "\n".join(lines[index:end]).strip()
        remaining = (
            None if max_chars is None else max_chars - len("\n\n".join(selected))
        )
        bounded = _fit_markdown_blocks(section, remaining)
        if bounded:
            selected.append(bounded)
        index = end
    return "\n\n".join(selected).rstrip()


class SkillAdapter:
    """Deterministically select bounded CANNBot knowledge capsules.

    The adapter reads only explicitly allowlisted references. It neither executes
    skill workflows/scripts nor copies template directories into a task.
    """

    def __init__(
        self,
        *,
        mapping_path: Path | str | None = None,
        source_root: Path | str | None = None,
        full_selected_input: bool = False,
    ):
        self.mapping_path = (
            Path(mapping_path or DEFAULT_MAPPING_PATH).expanduser().resolve()
        )
        self.mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        if int(self.mapping.get("schema_version", 0)) not in {1, 2}:
            raise ValueError(f"unsupported skill mapping schema: {self.mapping_path}")
        configured = os.getenv(str(self.mapping.get("source_root_env", "")), "").strip()
        root_value = (
            source_root or configured or self.mapping.get("default_source_root", "")
        )
        root = Path(root_value).expanduser()
        if not root.is_absolute():
            root = self.mapping_path.parent / root
        self.source_root = root.resolve()
        self.adapter_root = (self.mapping_path.parent / "knowledge_capsules").resolve()
        self.full_selected_input = full_selected_input

    @property
    def audience_budgets(self) -> dict[str, int]:
        return {
            str(key): int(value)
            for key, value in self.mapping.get("audience_budgets", {}).items()
        }

    def detect_operator_families(self, operator: str, evidence: str = "") -> list[str]:
        patterns = self.mapping.get("operator_families", {})
        operator_text = _normalized_text(operator.replace("_", " "))
        direct = [
            family
            for family, terms in patterns.items()
            if any(_normalized_text(str(term)) in operator_text for term in terms)
        ]
        if direct:
            return direct
        evidence_text = _normalized_text(evidence)
        scored = []
        for family, terms in patterns.items():
            score = sum(
                1 for term in terms if _normalized_text(str(term)) in evidence_text
            )
            if score:
                scored.append((score, family))
        if not scored:
            return ["unknown"]
        best = max(item[0] for item in scored)
        return sorted(family for score, family in scored if score == best)

    @staticmethod
    def _condition_matches(
        condition: dict[str, Any], context: SkillAdapterContext
    ) -> tuple[bool, str]:
        reasons: list[str] = []
        if (
            condition.get("requires_correct_baseline")
            and not context.has_correct_baseline
        ):
            return False, "requires a correct baseline"
        stages = [str(item) for item in condition.get("failure_stages", [])]
        if stages:
            if context.failure_stage not in stages:
                return (
                    False,
                    f"failure stage {context.failure_stage or 'none'} not in mapping",
                )
            reasons.append(f"failure_stage={context.failure_stage}")
        terms = [str(item).lower() for item in condition.get("any_terms", [])]
        if terms:
            trigger_evidence = (
                context.failure_evidence
                if context.failure_stage and context.failure_evidence
                else context.evidence
            )
            evidence = _normalized_text(
                " ".join(
                    [
                        trigger_evidence,
                        context.failure_code or "",
                        context.failure_stage or "",
                    ]
                )
            )
            matched = [term for term in terms if term in evidence]
            if not matched:
                return False, "no trigger term matched"
            reasons.append("terms=" + ",".join(matched[:4]))
        families = [str(item) for item in condition.get("operator_families", [])]
        if families and not set(families).intersection(context.operator_families):
            return False, "operator family mismatch"
        if condition.get("always"):
            reasons.append("stage default")
        if condition.get("requires_correct_baseline"):
            reasons.append("correct baseline exists")
        return True, "; ".join(reasons) or "mapping conditions matched"

    @staticmethod
    def _safe_reference(skill_root: Path, allowed_root: Path, relative: str) -> Path | None:
        candidate = (skill_root / relative).resolve()
        try:
            candidate.relative_to(skill_root)
            candidate.relative_to(allowed_root)
        except ValueError:
            return None
        if candidate.suffix.lower() != ".md" or not candidate.is_file():
            return None
        return candidate

    @staticmethod
    def _reference_matches(
        reference: dict[str, Any], context: SkillAdapterContext
    ) -> tuple[bool, str]:
        operators = [str(item).lower() for item in reference.get("operators", [])]
        if operators and context.operator.lower() not in operators:
            return False, "reference operator mismatch"
        families = [str(item) for item in reference.get("operator_families", [])]
        if families and not set(families).intersection(context.operator_families):
            return False, "reference operator family mismatch"
        terms = [str(item).lower() for item in reference.get("when_any", [])]
        if terms:
            evidence = _normalized_text(
                f"{context.failure_evidence} {context.evidence}"
            )
            matched = [term for term in terms if term in evidence]
            if not matched:
                return False, "reference trigger terms did not match"
            return True, "reference terms=" + ",".join(matched[:4])
        return True, "reference mapping matched"

    def select(self, context: SkillAdapterContext) -> SkillAdapterSelection:
        trace: list[dict[str, Any]] = []
        capsules: list[SkillCapsule] = []
        source_available = self.source_root.is_dir()
        if not source_available:
            trace.append(
                {
                    "candidate": str(self.source_root),
                    "decision": "rejected",
                    "reason": "source_unavailable",
                }
            )
        stage_mappings = self.mapping.get("stages", {})
        for stage in context.stages:
            stage_mapping = stage_mappings.get(stage, {})
            for skill in stage_mapping.get("skills", []):
                skill_id = str(skill.get("id", ""))
                role = str(skill.get("role", "support"))
                if role == "primary" and skill_id != context.primary_skill:
                    trace.append(
                        {
                            "candidate": f"{stage}:{skill_id}",
                            "decision": "rejected",
                            "reason": f"primary route is {context.primary_skill}",
                            "role": role,
                        }
                    )
                    continue
                if role == "secondary" and skill_id != context.secondary_skill:
                    trace.append(
                        {
                            "candidate": f"{stage}:{skill_id}",
                            "decision": "rejected",
                            "reason": "secondary route not selected",
                            "role": role,
                        }
                    )
                    continue
                matches, reason = self._condition_matches(
                    skill.get("when", {}), context
                )
                if not matches:
                    trace.append(
                        {
                            "candidate": f"{stage}:{skill_id}",
                            "decision": "rejected",
                            "reason": reason,
                        }
                    )
                    continue
                origin = str(skill.get("origin", "cannbot"))
                allowed_root = self.adapter_root if origin == "adapter" else self.source_root
                if origin == "cannbot" and not source_available:
                    trace.append(
                        {
                            "candidate": f"{stage}:{skill_id}",
                            "decision": "rejected",
                            "reason": "source_unavailable",
                            "role": role,
                            "origin": origin,
                        }
                    )
                    continue
                skill_root = (allowed_root / str(skill.get("source", skill_id))).resolve()
                try:
                    skill_root.relative_to(allowed_root)
                except ValueError:
                    trace.append(
                        {
                            "candidate": f"{stage}:{skill_id}",
                            "decision": "rejected",
                            "reason": f"unsafe {origin} skill source",
                        }
                    )
                    continue
                excerpts: list[SkillExcerpt] = []
                for reference in skill.get("references", []):
                    relative = str(reference.get("path", ""))
                    reference_id = f"{stage}:{skill_id}:{relative}"
                    ref_matches, ref_reason = self._reference_matches(
                        reference, context
                    )
                    if not ref_matches:
                        trace.append(
                            {
                                "candidate": reference_id,
                                "decision": "rejected",
                                "reason": ref_reason,
                            }
                        )
                        continue
                    path = self._safe_reference(skill_root, allowed_root, relative)
                    if path is None:
                        trace.append(
                            {
                                "candidate": reference_id,
                                "decision": "rejected",
                                "reason": "missing or unsafe Markdown reference",
                            }
                        )
                        continue
                    headings = [str(item) for item in reference.get("headings", [])]
                    excerpt = _extract_markdown_sections(
                        path.read_text(encoding="utf-8", errors="replace"),
                        headings,
                        max_chars=(
                            None
                            if self.full_selected_input
                            else int(reference.get("max_chars", 3000))
                        ),
                    )
                    if not excerpt:
                        trace.append(
                            {
                                "candidate": reference_id,
                                "decision": "rejected",
                                "reason": "mapped headings produced no excerpt",
                            }
                        )
                        continue
                    excerpts.append(
                        SkillExcerpt(
                            relative,
                            headings,
                            excerpt,
                            origin=origin,
                            confidence_level=int(reference.get("confidence_level", 1)),
                            provenance=str(reference.get("provenance", "documented_skill")),
                        )
                    )
                    trace.append(
                        {
                            "candidate": reference_id,
                            "decision": "selected",
                            "reason": ref_reason,
                            "role": role,
                            "origin": origin,
                        }
                    )
                capsules.append(
                    SkillCapsule(
                        skill_id=skill_id,
                        stage=stage,
                        purpose=str(skill.get("purpose", "")),
                        trigger_reason=reason,
                        expected_artifact=str(skill.get("expected_artifact", "")),
                        knowledge_type=[
                            str(item) for item in skill.get("knowledge_type", [])
                        ],
                        provided_context=[
                            str(item) for item in skill.get("provide", [])
                        ],
                        exclusions=[str(item) for item in skill.get("exclude", [])],
                        excerpts=excerpts,
                        role=role,
                    )
                )
                trace.append(
                    {
                        "candidate": f"{stage}:{skill_id}",
                        "decision": "selected",
                        "reason": reason,
                        "role": role,
                        "origin": origin,
                    }
                )
        return SkillAdapterSelection(capsules, trace, str(self.source_root), source_available)
