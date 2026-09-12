from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .models import EvalResult
from .skill_adapter import SkillAdapter, SkillAdapterContext, SkillAdapterSelection
from .structured_knowledge.schema import KnowledgeBundle

_COMPILE_STAGES = {
    "bundle_validation",
    "ascendc_source_validation",
    "api_constraint_validation",
    "static_validation",
    "ascendc_build",
    "compile",
    "response_format",
}
_RUNTIME_MARKERS = (
    "acl_error",
    "aicore exception",
    "mte",
    "507035",
    "161",
    "361",
    "561",
    "kernel not found",
    "plog",
    "device exception",
)
_PRECISION_MARKERS = (
    "mismatch",
    "not close",
    "rtol",
    "atol",
    "nan",
    "inf",
    "all zero",
    "max error",
    "incorrect",
    "precision",
)


@dataclass
class SelectedStageContext:
    schema_version: int
    audience: str
    stages: list[str]
    authority_order: list[str]
    task_facts: dict[str, Any]
    hard_constraints: list[dict[str, Any]]
    api_facts: list[dict[str, Any]]
    design_patterns: list[str]
    failure_guidance: list[dict[str, Any]]
    skill_capsules: list[dict[str, Any]]
    exclusions: list[str]
    provenance: list[dict[str, Any]]
    runtime_facts: str = ""
    budget: dict[str, Any] = field(default_factory=dict)
    selection_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextSelector:
    """Project structured knowledge and CANNBot capsules for one LLM audience."""

    def __init__(self, adapter: SkillAdapter):
        self.adapter = adapter

    @staticmethod
    def derive_stages(
        *,
        audience: str,
        workflow_phase: str,
        current_exists: bool,
        previous: EvalResult | None,
    ) -> list[str]:
        generator = audience == "generator"
        if previous is not None and previous.error:
            failure_stage = previous.failure_stage or ""
            if failure_stage in _COMPILE_STAGES:
                stages = ["compile_debug"]
            elif failure_stage in {"correctness", "runtime", "acl_runtime"}:
                structured = previous.structured_failure or {}
                failure_text = " ".join(
                    [
                        previous.error,
                        previous.error_excerpt,
                        str(previous.failure_code or ""),
                        str(structured.get("runtime_code") or ""),
                        str(structured.get("device_exception") or ""),
                        str(structured.get("subsystem") or ""),
                    ]
                ).lower()
                runtime = bool(
                    structured.get("runtime_code")
                    or structured.get("device_exception")
                    or any(marker in failure_text for marker in _RUNTIME_MARKERS)
                )
                precision = any(marker in failure_text for marker in _PRECISION_MARKERS)
                if runtime and precision:
                    stages = ["runtime_debug", "precision_debug"]
                elif runtime:
                    stages = ["runtime_debug"]
                else:
                    # A correctness failure without a runtime signal is a numerical
                    # mismatch until evidence proves otherwise.
                    stages = ["precision_debug"]
            elif failure_stage == "performance":
                stages = ["optimization"] if workflow_phase == "optimization" else []
            else:
                stages = []
            if generator:
                stages.append("code_generation")
            return list(dict.fromkeys(stages))
        if workflow_phase == "optimization":
            stages = ["optimization"]
            if generator:
                stages.append("code_generation")
            return stages
        if not current_exists:
            return (
                ["kernel_design", "code_generation"]
                if generator
                else [
                    "operator_analysis",
                    "kernel_design",
                ]
            )
        return ["kernel_design", "code_generation"] if generator else ["kernel_design"]

    def request(
        self,
        *,
        audience: str,
        workflow_phase: str,
        operator: str,
        soc: str,
        runtime_version: str,
        knowledge_version: str,
        current_exists: bool,
        previous: EvalResult | None,
        evidence: str,
    ) -> SkillAdapterContext:
        stages = self.derive_stages(
            audience=audience,
            workflow_phase=workflow_phase,
            current_exists=current_exists,
            previous=previous,
        )
        families = self.adapter.detect_operator_families(operator, evidence)
        structured_failure = previous.structured_failure if previous else None
        failure_symbols = (
            [
                str(item)
                for item in (structured_failure or {}).get("related_symbols", [])
            ]
            if isinstance(structured_failure, dict)
            else []
        )
        failure_evidence = (
            " ".join(
                [
                    previous.error,
                    previous.error_excerpt,
                    str(previous.failure_code or ""),
                    str(structured_failure or ""),
                ]
            )
            if previous
            else ""
        )
        return SkillAdapterContext(
            audience=audience,
            stages=stages,
            operator=operator,
            operator_families=families,
            soc=soc,
            runtime_version=runtime_version,
            knowledge_version=knowledge_version,
            failure_stage=previous.failure_stage if previous else None,
            failure_code=previous.failure_code if previous else None,
            failure_evidence=failure_evidence,
            failure_symbols=failure_symbols,
            has_correct_baseline=workflow_phase == "optimization",
            evidence=evidence,
        )

    @staticmethod
    def _compact_contract(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in ("contract_id", "subject", "constraint")
            if key in item
        }

    @staticmethod
    def _compact_fact(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in ("fact_id", "subject", "predicate", "value", "applicability")
            if key in item
        }

    @staticmethod
    def _compact_api_card(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in ("card_id", "api", "fact_ids", "examples")
            if key in item
        }

    @staticmethod
    def _compact_failure(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in ("card_id", "signals", "subsystem", "inspection_targets")
            if key in item
        }

    @staticmethod
    def _compact_provenance(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item[key]
            for key in ("document_id", "source_path", "section_id")
            if key in item
        }

    def select(
        self,
        *,
        bundle: KnowledgeBundle,
        request: SkillAdapterContext,
        runtime_facts: str = "",
    ) -> tuple[SelectedStageContext, SkillAdapterSelection]:
        skills = self.adapter.select(request)
        debug_stage = any(stage.endswith("_debug") for stage in request.stages)
        include_api = request.audience == "generator" or debug_stage
        api_facts = []
        if include_api:
            failure_symbols = set(request.failure_symbols)
            api_cards = bundle.api_semantics
            facts = bundle.relevant_facts
            if debug_stage and failure_symbols:
                api_cards = [
                    item
                    for item in api_cards
                    if str(item.get("api", "")) in failure_symbols
                ]
                facts = [
                    item
                    for item in facts
                    if str(item.get("applicability", {}).get("api", ""))
                    in failure_symbols
                ]
            api_facts.extend(
                {"kind": "atomic_fact", **self._compact_fact(item)} for item in facts
            )
            api_facts.extend(
                {"kind": "api_card", **self._compact_api_card(item)}
                for item in api_cards
            )
        failure_guidance = (
            [self._compact_failure(item) for item in bundle.failure_cards]
            if debug_stage
            else []
        )
        patterns = list(bundle.examples[: (1 if request.audience == "planner" else 2)])
        exclusions = list(
            dict.fromkeys(
                exclusion
                for capsule in skills.capsules
                for exclusion in capsule.exclusions
            )
        )
        selected = SelectedStageContext(
            schema_version=1,
            audience=request.audience,
            stages=request.stages,
            authority_order=[
                "current_evaluation",
                "installed_headers",
                "official_structured_facts",
                "project_contracts",
                "cannbot_practices",
                "confirmed_experience",
                "examples",
            ],
            task_facts={
                "operator": request.operator,
                "operator_families": request.operator_families,
                "soc": request.soc,
                "runtime_cann": request.runtime_version,
                "knowledge_cann": request.knowledge_version,
                "failure_stage": request.failure_stage,
                "failure_code": request.failure_code,
            },
            hard_constraints=[
                self._compact_contract(item) for item in bundle.project_contracts
            ],
            api_facts=api_facts,
            design_patterns=patterns,
            failure_guidance=failure_guidance,
            skill_capsules=[item.to_dict() for item in skills.capsules],
            exclusions=exclusions,
            provenance=[
                self._compact_provenance(item) for item in bundle.provenance[:12]
            ],
            runtime_facts=runtime_facts if include_api else "",
            budget={
                "max_chars": self.adapter.audience_budgets.get(request.audience, 12000),
                "used_chars": 0,
                "truncated_sections": [],
            },
            selection_trace=[
                {
                    "candidate": ",".join(request.stages),
                    "decision": "selected",
                    "reason": f"derived for {request.audience} in the current workflow state",
                },
                *skills.trace,
            ],
        )
        return selected, skills
