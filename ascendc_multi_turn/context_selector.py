from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .diagnostics import extract_symbol_evidence
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
_HOST_MARKERS = (
    "is_cuda",
    "cuda tensor",
    "aten/cuda",
    "c10/cuda",
    "at::cuda",
    "c10::cuda",
    "getcurrentcudastream",
    "getcurrentnpustream",
    "aclrtgetcurrentstream",
)
_PRECISION_PRIMARY_MARKERS = (
    "fp32 pass",
    "float32 pass",
    "accumulation dtype",
    "roundmode",
    "round mode",
    "epsilon placement",
    "overflow",
    "underflow",
    "tolerance boundary",
)
_PRECISION_SECONDARY_MARKERS = (
    "fp16 fail",
    "float16 fail",
    "cast precision",
    "accumulator precision",
    "epsilon",
    "rounding",
)


@dataclass(frozen=True)
class RoutingDecision:
    primary_skill: str
    route_reason: str
    secondary_skill: str | None
    secondary_reason: str | None
    debug_category: str
    routing_confidence: str
    route_evidence_origin: str = "workflow"
    matched_route_trigger: str | None = None


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
    selection_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContextSelector:
    """Project structured knowledge and CANNBot capsules for one LLM audience."""

    def __init__(self, adapter: SkillAdapter):
        self.adapter = adapter

    @staticmethod
    def derive_route(
        *,
        workflow_phase: str,
        current_exists: bool,
        previous: EvalResult | None,
        evidence: str = "",
    ) -> RoutingDecision:
        if previous is None or not previous.error:
            if workflow_phase == "optimization":
                return RoutingDecision("optimization", "correct_baseline_available", None, None, "performance", "direct")
            return RoutingDecision("kernel_design", "initial_or_unresolved_kernel_design", None, None, "kernel_design", "fallback")
        failure_stage = previous.failure_stage or ""
        structured = previous.structured_failure or {}
        failure_text = " ".join(
            [
                previous.error,
                previous.error_excerpt,
                previous.compile_output if failure_stage in _COMPILE_STAGES else "",
                previous.verify_output if failure_stage in {"correctness", "runtime", "acl_runtime"} else "",
                str(previous.failure_code or ""),
                str(structured),
                evidence if failure_stage in _COMPILE_STAGES else "",
            ]
        ).lower()
        if failure_stage in _COMPILE_STAGES:
            host_marker = next(
                (marker for marker in _HOST_MARKERS if marker in failure_text), None
            )
            if host_marker:
                return RoutingDecision(
                    "host_integration_debug",
                    "host_or_cuda_compile_evidence",
                    None,
                    None,
                    "host_abi",
                    "direct",
                    "failure_evidence",
                    host_marker,
                )
            return RoutingDecision(
                "api_compile_debug",
                f"compile_failure_stage={failure_stage}",
                None,
                None,
                "api_contract",
                "direct",
                "failure_evidence",
                failure_stage,
            )
        if failure_stage in {"correctness", "runtime", "acl_runtime"}:
            runtime = bool(
                structured.get("runtime_code")
                or structured.get("device_exception")
                or any(marker in failure_text for marker in _RUNTIME_MARKERS)
            )
            if runtime:
                return RoutingDecision("runtime_debug", "runtime_or_device_exception_evidence", None, None, "runtime", "direct")
            if any(marker in failure_text for marker in _PRECISION_PRIMARY_MARKERS):
                return RoutingDecision("precision_debug", "explicit_precision_failure_evidence", None, None, "precision", "direct")
            secondary = next((marker for marker in _PRECISION_SECONDARY_MARKERS if marker in failure_text), None)
            return RoutingDecision(
                "kernel_design",
                "correctness_failure_after_successful_compilation",
                "precision_debug" if secondary else None,
                f"explicit_precision_hint={secondary}" if secondary else None,
                "algorithmic_correctness",
                "corroborated" if secondary else "direct",
            )
        if failure_stage == "performance":
            return RoutingDecision("optimization", "performance_failure_with_correct_baseline", None, None, "performance", "direct")
        return RoutingDecision("kernel_design", f"unmapped_failure_stage={failure_stage or 'unknown'}", None, None, "kernel_design", "fallback")

    @staticmethod
    def derive_stages(
        *,
        audience: str,
        workflow_phase: str,
        current_exists: bool,
        previous: EvalResult | None,
        evidence: str = "",
    ) -> list[str]:
        generator = audience == "generator"
        route = ContextSelector.derive_route(
            workflow_phase=workflow_phase,
            current_exists=current_exists,
            previous=previous,
            evidence=evidence,
        )
        if previous is not None and previous.error:
            if route.primary_skill in {"host_integration_debug", "api_compile_debug"}:
                stages = ["host_integration_debug" if route.primary_skill == "host_integration_debug" else "compile_debug"]
            elif route.primary_skill == "optimization":
                stages = ["optimization"] if workflow_phase == "optimization" else []
            else:
                stages = [route.primary_skill]
            if route.secondary_skill:
                stages.append(route.secondary_skill)
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
        source_evidence: str = "",
        planned_evidence: str = "",
        environment_fingerprint: str | None = None,
    ) -> SkillAdapterContext:
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
                    previous.compile_output,
                    previous.verify_output,
                    str(previous.failure_code or ""),
                    str(structured_failure or ""),
                ]
            )
            if previous
            else ""
        )
        # Routing is based only on direct failure evidence. The complete current
        # source remains available for symbol retrieval but cannot classify a
        # valid NPU wrapper as a Host ABI failure merely by containing its APIs.
        route = self.derive_route(
            workflow_phase=workflow_phase,
            current_exists=current_exists,
            previous=previous,
            evidence=failure_evidence,
        )
        stages = self.derive_stages(
            audience=audience,
            workflow_phase=workflow_phase,
            current_exists=current_exists,
            previous=previous,
            evidence=failure_evidence,
        )
        families = self.adapter.detect_operator_families(operator, evidence)
        symbols = extract_symbol_evidence(
            failure=failure_evidence,
            source=source_evidence,
            planned=planned_evidence,
        )
        by_kind = {
            kind: [item.symbol for item in symbols if kind in item.kinds]
            for kind in ("failure", "source", "planned")
        }
        failure_symbols = list(dict.fromkeys([*failure_symbols, *by_kind["failure"]]))
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
            source_symbols=by_kind["source"],
            planned_symbols=by_kind["planned"],
            symbol_evidence=[item.to_dict() for item in symbols],
            primary_skill=route.primary_skill,
            route_reason=route.route_reason,
            secondary_skill=route.secondary_skill,
            secondary_reason=route.secondary_reason,
            debug_category=route.debug_category,
            routing_confidence=route.routing_confidence,
            route_evidence_origin=route.route_evidence_origin,
            matched_route_trigger=route.matched_route_trigger,
            environment_fingerprint=environment_fingerprint,
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
        runtime_facts: Any = "",
    ) -> tuple[SelectedStageContext, SkillAdapterSelection]:
        full_selected = self.adapter.full_selected_input
        skills = self.adapter.select(request)
        debug_stage = any(stage.endswith("_debug") for stage in request.stages)
        include_api = request.audience == "generator" or debug_stage
        api_facts = []
        selection_trace: list[dict[str, Any]] = [
            {
                "candidate": ",".join(request.stages),
                "decision": "selected",
                "reason": request.route_reason,
                "primary_skill": request.primary_skill,
                "secondary_skill": request.secondary_skill,
            },
            *skills.trace,
        ]
        if include_api:
            ordered_symbols = list(
                dict.fromkeys(
                    [
                        *request.failure_symbols,
                        *request.source_symbols,
                        *request.planned_symbols,
                    ]
                )
            )
            symbol_lookup = {item.lower(): item for item in ordered_symbols}
            symbol_rank = {item.lower(): index for index, item in enumerate(ordered_symbols)}

            def item_symbol(item: dict[str, Any]) -> str:
                return str(
                    item.get("api")
                    or item.get("applicability", {}).get("api")
                    or item.get("subject")
                    or ""
                )

            api_cards = list(bundle.api_semantics)
            facts = list(bundle.relevant_facts)
            if debug_stage and ordered_symbols:
                matched_cards = sorted(
                    [item for item in api_cards if item_symbol(item).lower() in symbol_lookup],
                    key=lambda item: symbol_rank[item_symbol(item).lower()],
                )
                matched_facts = sorted(
                    [item for item in facts if item_symbol(item).lower() in symbol_lookup],
                    key=lambda item: symbol_rank[item_symbol(item).lower()],
                )
                fallback_cards = [item for item in api_cards if item not in matched_cards][:2]
                fallback_facts = [item for item in facts if item not in matched_facts][:2]
                api_cards = [*matched_cards, *fallback_cards]
                facts = [*matched_facts, *fallback_facts]
                selected_ids = {
                    str(item.get("card_id") or item.get("fact_id"))
                    for item in [*api_cards, *facts]
                }
                for item in [*bundle.api_semantics, *bundle.relevant_facts]:
                    item_id = str(item.get("card_id") or item.get("fact_id"))
                    if item_id and item_id not in selected_ids:
                        selection_trace.append(
                            {
                                "candidate": item_id,
                                "decision": "rejected",
                                "reason": "no failure/source/planned exact symbol match and fallback quota exhausted",
                            }
                        )
            api_facts.extend(
                {
                    "kind": "atomic_fact",
                    **(dict(item) if full_selected else self._compact_fact(item)),
                }
                for item in facts
            )
            api_facts.extend(
                {
                    "kind": "api_card",
                    **(dict(item) if full_selected else self._compact_api_card(item)),
                }
                for item in api_cards
            )
        failure_guidance = (
            [
                dict(item) if full_selected else self._compact_failure(item)
                for item in bundle.failure_cards
            ]
            if debug_stage
            else []
        )
        patterns = list(
            bundle.examples
            if full_selected
            else bundle.examples[: (1 if request.audience == "planner" else 2)]
        )
        exclusions = list(
            dict.fromkeys(
                exclusion
                for capsule in skills.capsules
                for exclusion in capsule.exclusions
            )
        )
        selected = SelectedStageContext(
            schema_version=2,
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
                "failure_symbols": request.failure_symbols,
                "source_symbols": request.source_symbols,
                "planned_symbols": request.planned_symbols,
                "symbol_evidence": request.symbol_evidence,
                "primary_skill": request.primary_skill,
                "route_reason": request.route_reason,
                "secondary_skill": request.secondary_skill,
                "secondary_reason": request.secondary_reason,
                "debug_category": request.debug_category,
                "routing_confidence": request.routing_confidence,
                "route_evidence_origin": request.route_evidence_origin,
                "matched_route_trigger": request.matched_route_trigger,
                "environment_fingerprint": request.environment_fingerprint,
            },
            hard_constraints=[
                dict(item) if full_selected else self._compact_contract(item)
                for item in bundle.project_contracts
            ],
            api_facts=api_facts,
            design_patterns=patterns,
            failure_guidance=failure_guidance,
            skill_capsules=[item.to_dict() for item in skills.capsules],
            exclusions=exclusions,
            provenance=[
                dict(item) if full_selected else self._compact_provenance(item)
                for item in (
                    bundle.provenance if full_selected else bundle.provenance[:12]
                )
            ],
            runtime_facts=(
                str(getattr(runtime_facts, "text", runtime_facts)) if include_api else ""
            ),
            budget={
                "input_mode": "full_selected" if full_selected else "bounded",
                "max_chars": (
                    None
                    if full_selected
                    else self.adapter.audience_budgets.get(request.audience, 12000)
                ),
                "used_chars": 0,
                "truncated_sections": [],
            },
            selection_trace=selection_trace,
            selection_metadata={
                "selected_skill_ids": [item.skill_id for item in skills.capsules],
                "selected_structured_ids": [
                    str(item.get("fact_id") or item.get("card_id")) for item in api_facts
                    if item.get("fact_id") or item.get("card_id")
                ],
                "runtime_fact_ids": [
                    str(item.get("fact_id"))
                    for item in getattr(runtime_facts, "facts", [])
                    if item.get("fact_id")
                ],
                "runtime_facts": list(getattr(runtime_facts, "facts", [])),
                "environment_fingerprint": dict(
                    getattr(runtime_facts, "environment_fingerprint", {})
                ),
                "input_mode": "full_selected" if full_selected else "bounded",
                "input_truncated": False,
            },
        )
        evidence_by_symbol = {
            str(item.get("symbol", "")).lower(): item
            for item in request.symbol_evidence
            if item.get("symbol")
        }
        selected_items: list[dict[str, Any]] = []
        for item in api_facts:
            identifier = str(item.get("fact_id") or item.get("card_id") or "")
            symbol = str(
                item.get("api")
                or item.get("applicability", {}).get("api")
                or item.get("subject")
                or ""
            )
            evidence_item = evidence_by_symbol.get(symbol.lower(), {})
            selected_items.append(
                {
                    "id": identifier,
                    "source": "structured",
                    "evidence_symbol": symbol or None,
                    "symbol_types": list(evidence_item.get("kinds", [])) or ["fallback"],
                    "confidence_level": 1,
                    "provenance": "structured_knowledge",
                }
            )
        for capsule in skills.capsules:
            selected_items.append(
                {
                    "id": capsule.skill_id,
                    "source": next(
                        (excerpt.origin for excerpt in capsule.excerpts), "skill_mapping"
                    ),
                    "evidence_symbol": None,
                    "symbol_types": [],
                    "confidence_level": max(
                        (excerpt.confidence_level for excerpt in capsule.excerpts),
                        default=1,
                    ),
                    "provenance": sorted(
                        {excerpt.provenance for excerpt in capsule.excerpts}
                    ) or ["skill_mapping"],
                    "role": capsule.role,
                }
            )
        for item in getattr(runtime_facts, "facts", []):
            symbol = str(item.get("symbol", ""))
            evidence_item = evidence_by_symbol.get(symbol.lower(), {})
            selected_items.append(
                {
                    "id": str(item.get("fact_id", "")),
                    "source": "runtime",
                    "evidence_symbol": symbol or None,
                    "symbol_types": list(evidence_item.get("kinds", [])) or ["fallback"],
                    "confidence_level": item.get("confidence_level", 0),
                    "provenance": item.get("provenance", "unknown"),
                    "probe_id": item.get("probe_id"),
                }
            )
        selected.selection_metadata["selected_items"] = selected_items
        return selected, skills
