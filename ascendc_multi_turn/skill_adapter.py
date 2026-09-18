from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_MAPPING_PATH = PACKAGE_ROOT / "skill_mapping.yaml"
KNOWLEDGE_MODULE_ROOT = PACKAGE_ROOT / "knowledge_modules"
CANNBOT_KNOWLEDGE_BASE_ROOT = (
    KNOWLEDGE_MODULE_ROOT / "cannbot_a08c4970_knowledge_base"
)
CANNBOT_KNOWLEDGE_BASE_MANIFEST_PATH = (
    CANNBOT_KNOWLEDGE_BASE_ROOT / "knowledge_base_manifest.json"
)
CANNBOT_VENDOR_MANIFEST_PATH = CANNBOT_KNOWLEDGE_BASE_ROOT / "vendor_manifest.json"
EXTERNAL_KNOWLEDGE_MIGRATION_ERROR = (
    "External CANNBot skill paths are no longer supported. Remove "
    "--cannbot-skills-root/--skill-mapping and CANNBOT_SKILLS_ROOT; CannAgent "
    "now validates and loads its embedded CANNBot knowledge base."
)


@dataclass(frozen=True)
class SkillAdapterContext:
    audience: str
    stages: list[str]
    operator: str
    operator_families: list[str]
    shape_regime: dict[str, Any]
    risk_flags: list[str]
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
    symbol_domains: list[str] = field(default_factory=list)
    primary_skill: str = "kernel_design"
    route_reason: str = "default_kernel_design"
    secondary_skill: str | None = None
    secondary_reason: str | None = None
    debug_category: str = "kernel_design"
    routing_confidence: str = "fallback"
    route_evidence_origin: str = "workflow"
    matched_route_trigger: str | None = None
    diagnostic_source_files: list[str] = field(default_factory=list)
    failure_ownership: str = "Kernel"
    active_profile: str | None = None
    profile_case_indices: list[int] = field(default_factory=list)
    profile_features: list[str] = field(default_factory=list)
    environment_fingerprint: str | None = None


@dataclass(frozen=True)
class SkillExcerpt:
    knowledge_module_id: str
    section_id: str
    source: str
    headings: list[str]
    text: str
    origin: str = "cannbot_knowledge_base"
    confidence_level: int = 1
    provenance: str = "documented_skill"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SkillKnowledgeModule:
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
    knowledge_modules: list[SkillKnowledgeModule]
    trace: list[dict[str, Any]]
    source_root: str
    source_available: bool
    knowledge_base_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _KnowledgeModuleDocument:
    knowledge_module_id: str
    path: str
    text: str
    sha256: str
    knowledge_type: tuple[str, ...]
    allowed_stages: tuple[str, ...]
    conflicts_or_exclusions: tuple[str, ...]
    provenance: str
    confidence_level: int
    origin: str


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower())


def _knowledge_content_hash(value: str) -> str:
    normalized = "\n".join(
        line.rstrip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _heading_title(line: str) -> str | None:
    match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
    return match.group(1).strip() if match else None


def _extract_markdown_sections(text: str, headings: list[str]) -> str:
    """Return complete mapped sections; never truncate a selected section."""

    if not headings:
        return text.rstrip()
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
        selected.append("\n".join(lines[index:end]).strip())
        index = end
    return "\n\n".join(selected).rstrip()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


class SkillAdapter:
    """Select only validated, package-local knowledge from an immutable registry."""

    def __init__(
        self,
        *,
        mapping_path: Path | str | None = None,
        source_root: Path | str | None = None,
        full_selected_input: bool = False,
    ):
        if mapping_path is not None or source_root is not None:
            raise ValueError(EXTERNAL_KNOWLEDGE_MIGRATION_ERROR)
        self.mapping_path = DEFAULT_MAPPING_PATH
        self.source_root = CANNBOT_KNOWLEDGE_BASE_ROOT
        self.adapter_root = KNOWLEDGE_MODULE_ROOT
        self.full_selected_input = full_selected_input
        mapping = json.loads(self.mapping_path.read_text(encoding="utf-8"))
        if int(mapping.get("schema_version", 0)) != 3:
            raise ValueError(f"unsupported embedded skill mapping schema: {self.mapping_path}")
        self._validate_vendor_manifest()
        documents, knowledge_base_id = self._load_registry()
        self._validate_mapping(mapping, documents)
        self.mapping: Mapping[str, Any] = _freeze(mapping)
        self.registry: Mapping[str, _KnowledgeModuleDocument] = MappingProxyType(documents)
        self.knowledge_base_id = knowledge_base_id

    @staticmethod
    def _validate_vendor_manifest() -> None:
        payload = json.loads(CANNBOT_VENDOR_MANIFEST_PATH.read_text(encoding="utf-8"))
        if payload.get("upstream_commit") != "a08c49706e35a400d7c77e0875bc7c72a3a79012":
            raise ValueError("embedded CANNBot skill subset has an unexpected upstream commit")
        for item in payload.get("skills", []):
            path = (CANNBOT_KNOWLEDGE_BASE_ROOT / str(item.get("path", ""))).resolve()
            try:
                path.relative_to(CANNBOT_KNOWLEDGE_BASE_ROOT.resolve())
            except ValueError as error:
                raise ValueError(f"unsafe embedded CANNBot skill path: {path}") from error
            if not path.is_file() or path.suffix.lower() != ".md":
                raise ValueError(f"embedded CANNBot skill is missing: {path}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != item.get("sha256"):
                raise ValueError(f"embedded CANNBot skill hash mismatch: {path}")

    @staticmethod
    def _safe_manifest_path(root: Path, relative: str) -> Path:
        path = (root / relative).resolve()
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"unsafe embedded knowledge module path: {relative}") from error
        if path.suffix.lower() != ".md" or not path.is_file():
            raise ValueError(f"embedded knowledge module is missing or not Markdown: {path}")
        return path

    @classmethod
    def _load_manifest_documents(
        cls, manifest_path: Path, *, root: Path, list_key: str, origin: str
    ) -> tuple[dict[str, _KnowledgeModuleDocument], dict[str, Any]]:
        if not manifest_path.is_file():
            raise ValueError(f"embedded knowledge module manifest is missing: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = payload.get(list_key)
        if not isinstance(records, list) or not records:
            raise ValueError(f"embedded knowledge module manifest has no {list_key}: {manifest_path}")
        result: dict[str, _KnowledgeModuleDocument] = {}
        for record in records:
            knowledge_module_id = str(record.get("knowledge_module_id", "")).strip()
            if not knowledge_module_id or knowledge_module_id in result:
                raise ValueError(f"invalid or duplicate knowledge_module_id in {manifest_path}: {knowledge_module_id}")
            path = cls._safe_manifest_path(root, str(record.get("path", "")))
            content = path.read_bytes()
            actual = hashlib.sha256(content).hexdigest()
            expected = str(record.get("sha256", ""))
            if actual != expected:
                raise ValueError(
                    f"embedded knowledge module hash mismatch for {knowledge_module_id}: expected {expected}, got {actual}"
                )
            result[knowledge_module_id] = _KnowledgeModuleDocument(
                knowledge_module_id=knowledge_module_id,
                path=path.relative_to(KNOWLEDGE_MODULE_ROOT).as_posix(),
                text=content.decode("utf-8"),
                sha256=actual,
                knowledge_type=tuple(str(item) for item in record.get("knowledge_type", [])),
                allowed_stages=tuple(str(item) for item in record.get("allowed_stages", [])),
                conflicts_or_exclusions=tuple(
                    str(item) for item in record.get("conflicts_or_exclusions", [])
                ),
                provenance=str(record.get("provenance", "unknown")),
                confidence_level=int(record.get("confidence_level", 1)),
                origin=origin,
            )
        return result, payload

    def _load_registry(self) -> tuple[dict[str, _KnowledgeModuleDocument], str]:
        knowledge_base, payload = self._load_manifest_documents(
            CANNBOT_KNOWLEDGE_BASE_MANIFEST_PATH,
            root=CANNBOT_KNOWLEDGE_BASE_ROOT,
            list_key="documents",
            origin="cannbot_knowledge_base",
        )
        return knowledge_base, str(
            payload.get("knowledge_base_id", "unknown")
        )

    @staticmethod
    def _validate_mapping(mapping: dict[str, Any], documents: dict[str, _KnowledgeModuleDocument]) -> None:
        stage_mappings = [
            *mapping.get("stages", {}).items(),
            ("standing_contract", mapping.get("standing_contracts", {})),
        ]
        for stage, stage_mapping in stage_mappings:
            for skill in stage_mapping.get("skills", []):
                if "source" in skill:
                    raise ValueError(f"embedded mapping exposes a filesystem source in {stage}:{skill.get('id')}")
                for reference in skill.get("references", []):
                    if "path" in reference or "max_chars" in reference:
                        raise ValueError(
                            f"embedded mapping reference must use knowledge_module_id/section_id only: {stage}:{skill.get('id')}"
                        )
                    knowledge_module_id = str(reference.get("knowledge_module_id", ""))
                    section_id = str(reference.get("section_id", ""))
                    if not knowledge_module_id or not section_id:
                        raise ValueError(f"mapping reference lacks knowledge_module_id or section_id: {stage}:{skill.get('id')}")
                    if knowledge_module_id not in documents:
                        continue
                    if stage not in documents[knowledge_module_id].allowed_stages:
                        raise ValueError(f"knowledge module {knowledge_module_id} is not allowed in stage {stage}")

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
            score = sum(1 for term in terms if _normalized_text(str(term)) in evidence_text)
            if score:
                scored.append((score, family))
        if not scored:
            return ["unknown"]
        best = max(item[0] for item in scored)
        return sorted(family for score, family in scored if score == best)

    @staticmethod
    def _condition_matches(condition: Mapping[str, Any], context: SkillAdapterContext) -> tuple[bool, str]:
        reasons: list[str] = []
        if condition.get("requires_correct_baseline") and not context.has_correct_baseline:
            return False, "requires_correct_baseline"
        stages = [str(item) for item in condition.get("failure_stages", ())]
        if stages and context.failure_stage not in stages:
            return False, f"failure_stage={context.failure_stage or 'none'} not allowed"
        if stages:
            reasons.append(f"failure_stage={context.failure_stage}")
        terms = [
            str(item).lower()
            for item in (
                condition.get("any_terms", ()) or condition.get("when_any", ())
            )
        ]
        if terms:
            evidence = _normalized_text(
                f"{context.failure_evidence} {context.failure_code or ''} {context.evidence}"
            )
            matched = [term for term in terms if term in evidence]
            if not matched:
                return False, "no_trigger_term"
            reasons.append("terms=" + ",".join(matched[:4]))
        families = [str(item) for item in condition.get("operator_families", ())]
        if families and not set(families).intersection(context.operator_families):
            return False, "operator_family_mismatch"
        domains = [str(item) for item in condition.get("when_domains", ())]
        if domains and not set(domains).intersection(context.symbol_domains):
            return False, "symbol_domain_mismatch"
        profiles = [str(item) for item in condition.get("profiles", ())]
        if profiles and context.active_profile not in profiles:
            return False, "active_profile_mismatch"
        features = [str(item) for item in condition.get("profile_features", ())]
        if features and not set(features).intersection(context.profile_features):
            return False, "profile_feature_mismatch"
        risks = [str(item) for item in condition.get("risk_flags", ())]
        if risks and not set(risks).intersection(context.risk_flags):
            return False, "risk_flag_mismatch"
        shape_relations = [str(item) for item in condition.get("shape_relations", ())]
        if shape_relations and context.shape_regime.get("relation") not in shape_relations:
            return False, "shape_regime_mismatch"
        if condition.get("always"):
            reasons.append("stage_default")
        if condition.get("requires_correct_baseline"):
            reasons.append("correct_baseline")
        return True, "; ".join(reasons) or "mapping_conditions_matched"

    @staticmethod
    def _reference_matches(reference: Mapping[str, Any], context: SkillAdapterContext) -> tuple[bool, str]:
        operators = [str(item).lower() for item in reference.get("operators", ())]
        if operators and context.operator.lower() not in operators:
            return False, "operator_mismatch"
        return SkillAdapter._condition_matches(reference, context)

    def select(self, context: SkillAdapterContext) -> SkillAdapterSelection:
        trace: list[dict[str, Any]] = []
        knowledge_modules: list[SkillKnowledgeModule] = []
        seen_sections: set[tuple[str, str]] = set()
        seen_content: dict[str, str] = {}
        selected_chars = 0
        audience_budget = self.audience_budgets.get(context.audience, 12000)
        routed_groups = [
            ("standing_contract", self.mapping.get("standing_contracts", {})),
            *(
                (stage, self.mapping.get("stages", {}).get(stage, {}))
                for stage in context.stages
            ),
        ]
        for stage, stage_mapping in routed_groups:
            for skill in stage_mapping.get("skills", ()):
                skill_id = str(skill.get("id", ""))
                role = str(skill.get("role", "support"))
                matches, reason = self._condition_matches(skill.get("when", {}), context)
                group_id = f"{stage}:{skill_id}"
                if not matches:
                    trace.append({"candidate": group_id, "decision": "rejected", "reason": reason})
                    continue
                excerpts: list[SkillExcerpt] = []
                selected_reference_ids: list[tuple[str, str]] = []
                selected_content_hashes: list[tuple[str, str]] = []
                exclusions = [str(item) for item in skill.get("exclude", ())]
                knowledge_types = [str(item) for item in skill.get("knowledge_type", ())]
                for reference in skill.get("references", ()):
                    knowledge_module_id = str(reference.get("knowledge_module_id"))
                    section_id = str(reference.get("section_id"))
                    reference_id = f"{knowledge_module_id}#{section_id}"
                    ref_matches, ref_reason = self._reference_matches(reference, context)
                    if not ref_matches:
                        trace.append({"candidate": reference_id, "decision": "rejected", "reason": ref_reason})
                        continue
                    key = (knowledge_module_id, section_id)
                    if key in seen_sections:
                        trace.append({"candidate": reference_id, "decision": "rejected", "reason": "duplicate_knowledge_module_section"})
                        continue
                    if knowledge_module_id not in self.registry:
                        trace.append({"candidate": reference_id, "decision": "rejected", "reason": "non_cannbot_knowledge_excluded"})
                        continue
                    document = self.registry[knowledge_module_id]
                    headings = [str(item) for item in reference.get("headings", ())]
                    excerpt = _extract_markdown_sections(document.text, headings)
                    if not excerpt:
                        trace.append({"candidate": reference_id, "decision": "rejected", "reason": "mapped_section_not_found"})
                        continue
                    content_hash = _knowledge_content_hash(excerpt)
                    if content_hash in seen_content:
                        trace.append(
                            {
                                "candidate": reference_id,
                                "decision": "rejected",
                                "reason": "duplicate_knowledge_content",
                                "alias_of": seen_content[content_hash],
                                "content_sha256": content_hash,
                            }
                        )
                        continue
                    seen_sections.add(key)
                    seen_content[content_hash] = reference_id
                    knowledge_types.extend(document.knowledge_type)
                    exclusions.extend(document.conflicts_or_exclusions)
                    excerpts.append(
                        SkillExcerpt(
                            knowledge_module_id=knowledge_module_id,
                            section_id=section_id,
                            source=document.path,
                            headings=headings,
                            text=excerpt,
                            origin=document.origin,
                            confidence_level=document.confidence_level,
                            provenance=document.provenance,
                        )
                    )
                    selected_reference_ids.append((reference_id, ref_reason))
                    selected_content_hashes.append((content_hash, reference_id))
                if not excerpts:
                    trace.append(
                        {
                            "candidate": group_id,
                            "decision": "rejected",
                            "reason": "no_reference_selected",
                        }
                    )
                    continue
                knowledge_module_chars = sum(len(item.text) for item in excerpts)
                if selected_chars + knowledge_module_chars > audience_budget:
                    for excerpt in excerpts:
                        seen_sections.discard(
                            (excerpt.knowledge_module_id, excerpt.section_id)
                        )
                    for content_hash, reference_id in selected_content_hashes:
                        if seen_content.get(content_hash) == reference_id:
                            seen_content.pop(content_hash, None)
                    trace.append(
                        {
                            "candidate": group_id,
                            "decision": "rejected",
                            "reason": (
                                "knowledge_module_budget_exceeded: "
                                f"used={selected_chars}, candidate={knowledge_module_chars}, "
                                f"limit={audience_budget}"
                            ),
                        }
                    )
                    trace.extend(
                        {
                            "candidate": reference_id,
                            "decision": "rejected",
                            "reason": "parent_knowledge_module_budget_exceeded",
                        }
                        for reference_id, _ in selected_reference_ids
                    )
                    continue
                knowledge_modules.append(
                    SkillKnowledgeModule(
                        skill_id=skill_id,
                        stage=stage,
                        purpose=str(skill.get("purpose", "")),
                        trigger_reason=reason,
                        expected_artifact=str(skill.get("expected_artifact", "")),
                        knowledge_type=list(dict.fromkeys(knowledge_types)),
                        provided_context=[str(item) for item in skill.get("provide", ())],
                        exclusions=list(dict.fromkeys(exclusions)),
                        excerpts=excerpts,
                        role=role,
                    )
                )
                selected_chars += knowledge_module_chars
                trace.extend(
                    {
                        "candidate": reference_id,
                        "decision": "selected",
                        "reason": ref_reason,
                    }
                    for reference_id, ref_reason in selected_reference_ids
                )
                trace.append({"candidate": group_id, "decision": "selected", "reason": reason})
        return SkillAdapterSelection(
            knowledge_modules=knowledge_modules,
            trace=trace,
            source_root=str(CANNBOT_KNOWLEDGE_BASE_ROOT),
            source_available=True,
            knowledge_base_id=self.knowledge_base_id,
        )
