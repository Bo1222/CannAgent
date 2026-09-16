from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

from .bundle import (
    capture_bundle,
    parse_file_bundle,
    restore_bundle,
    semantic_bundle_hash,
    validate_initial_bundle,
)
from .context_selector import ContextSelector
from .diagnostics import (
    diagnostic_fingerprint,
    evaluation_gates,
    extract_api_symbols,
    observe_evaluation_stages,
    parse_structured_failure,
    read_result_log,
)
from .evaluator import Evaluator, extract_error_excerpt
from .interface_contract import (
    capture_interface_contract,
    compare_interface_contract,
    load_interface_contract,
    save_interface_contract,
)
from .knowledge import (
    active_doc_ids,
    apply_selection_to_state,
    build_knowledge_prompt,
    candidate_doc_ids,
    knowledge_limits,
    load_knowledge_state,
    parse_knowledge_selection,
    render_knowledge,
    resolve_knowledge_version,
    save_knowledge_state,
    selection_from_state,
)
from .llm import LLMProvider, system_prompt_for, system_prompt_id_for
from .logging import TrajectoryLogger
from .models import EvalResult, FileBundle, LLMCallConfig, LLMResponse, RunConfig
from .progress import ProgressReporter, token_detail
from .prompts import build_plan_prompt, build_prompt, parse_plan, render_stage_context
from .repair_policy import build_repair_policy
from .repair_state import RepairStateManager
from .runtime_knowledge import collect_runtime_facts
from .skill_adapter import SkillAdapter
from .structured_knowledge import (
    FrontierManager,
    KnowledgeBuild,
    KnowledgeBundle,
    KnowledgeContext,
    RetrievalTraceEntry,
    StructuredKnowledgeRouter,
    build_incident,
    locate_knowledge_build,
    persist_incident,
    render_bundle,
)


class LLMCallFailure(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class CandidatePreparationFailure(RuntimeError):
    def __init__(self, result: EvalResult):
        super().__init__(result.error)
        self.result = result


class MultiTurnRunner:
    """Evidence-driven AscendC bootstrap and optimization loop."""

    CONSECUTIVE_FAIL_THRESHOLD = 3
    REPO_ROOT = Path(__file__).resolve().parent.parent

    def _probe_manifest_path(self) -> Path:
        configured = os.getenv("CANNAGENT_PROBE_MANIFEST", "").strip()
        return Path(configured).expanduser().resolve() if configured else self.state_dir / "knowledge_probes/manifest.json"

    def __init__(
        self,
        config: RunConfig,
        provider: LLMProvider,
        evaluator: Evaluator,
        progress: ProgressReporter | None = None,
    ):
        self.config = config
        self.provider = provider
        self.evaluator = evaluator
        self.task_dir = Path(config.output_dir).expanduser().resolve()
        self.state_dir = self.task_dir / ".llm_state"
        self.progress = progress or ProgressReporter()
        self.skill_adapter = (
            SkillAdapter(
                full_selected_input=config.uses_full_selected_input,
            )
            if config.uses_skills
            else None
        )
        self.context_selector = (
            ContextSelector(self.skill_adapter) if self.skill_adapter is not None else None
        )

    def _prepare_task(self) -> None:
        source = Path(self.config.op_file).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"reference model does not exist: {source}")
        if self.config.resume:
            if not (self.state_dir / "trajectory.json").is_file():
                raise ValueError(f"cannot resume: no trajectory found in {self.state_dir}")
            return
        if self.task_dir.exists() and any(self.task_dir.iterdir()):
            raise ValueError(
                f"output directory is not empty: {self.task_dir}; use --resume or choose another directory"
            )
        self.task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, self.task_dir / "model.py")
        cases = source.with_suffix(".json")
        if cases.is_file():
            shutil.copy2(cases, self.task_dir / cases.name)

    def _inputs(self) -> tuple[str, str]:
        reference = (self.task_dir / "model.py").read_text(encoding="utf-8")
        cases_name = Path(self.config.op_file).expanduser().resolve().with_suffix(".json").name
        cases_path = self.task_dir / cases_name
        cases = cases_path.read_text(encoding="utf-8") if cases_path.is_file() else "(no JSON case file)"
        return reference, cases

    @staticmethod
    def _write_bundle(path: Path, bundle: FileBundle) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_bundle(path: Path) -> FileBundle:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FileBundle(
            files=payload["files"],
            delete=payload.get("delete", []),
            analysis=payload.get("analysis", ""),
        )

    @staticmethod
    def _is_valid_result(result: EvalResult | None) -> bool:
        return bool(
            result is not None
            and not result.error
            and result.compiled
            and result.correctness
            and isinstance(result.score, (int, float))
            and math.isfinite(float(result.score))
            and float(result.score) > 0
        )

    @classmethod
    def _is_better(cls, candidate: EvalResult, best: EvalResult | None) -> bool:
        if not cls._is_valid_result(candidate):
            return False
        return best is None or not cls._is_valid_result(best) or float(candidate.score) > float(best.score)

    @staticmethod
    def _failure_result(
        *,
        stage: str,
        code: str,
        message: str,
        details_path: Path | None = None,
        failure_kind: str = "orchestration",
    ) -> EvalResult:
        return EvalResult(
            False,
            False,
            error=message,
            failure_stage=stage,
            failure_code=code,
            error_excerpt=message[:4000],
            details_path=str(details_path.resolve()) if details_path else None,
            failure_kind=failure_kind,
        )

    @staticmethod
    def _write_error(path: Path, error: Exception | str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{error}\n", encoding="utf-8")

    def _provider_generate(self, prompt: str, call_config: LLMCallConfig) -> LLMResponse:
        parameters = inspect.signature(self.provider.generate).parameters
        supports_config = "call_config" in parameters or any(
            item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
        )
        return (
            self.provider.generate(prompt, call_config=call_config)
            if supports_config
            else self.provider.generate(prompt)
        )

    @staticmethod
    def _prompt_metadata(prompt: str) -> dict[str, Any]:
        import hashlib

        def section(titles: tuple[str, ...], end_titles: tuple[str, ...]) -> str:
            for title in titles:
                start_match = re.search(
                    rf"^## {re.escape(title)}\s*$\n", prompt, re.MULTILINE
                )
                if not start_match:
                    continue
                start = start_match.end()
                ends = [
                    match.start()
                    for end_title in end_titles
                    if (
                        match := re.search(
                            rf"^## {re.escape(end_title)}\s*$",
                            prompt[start:],
                            re.MULTILINE,
                        )
                    )
                ]
                end = start + min(ends) if ends else len(prompt)
                return prompt[start:end]
            return ""

        knowledge = section(
            (
                "精确 installed/verified 事实和相关知识模块",
                "按阶段选择的 AscendC 参考",
            ),
            ("明确禁止模式", "Reference PyTorch model 与 benchmark 语义（只读）", "输出契约"),
        )
        source = section(
            ("当前实现",),
            ("上一轮评测", "最新评测证据"),
        )
        evidence = section(
            ("上一轮评测", "最新评测证据"),
            ("按阶段选择的 AscendC 参考", "输出契约"),
        )
        repair_state = section(
            ("开放错误、已清除错误和相关失败方案", "当前轨迹修复状态"),
            ("当前阶段契约", "必须满足的语义和 ABI 契约", "Reference PyTorch model 与 benchmark 语义（只读）", "输出契约"),
        )

        def measure(value: str) -> dict[str, int]:
            return {
                "chars": len(value),
                "bytes": len(value.encode("utf-8")),
                "estimated_tokens": (len(value) + 3) // 4,
            }

        return {
            "prompt": measure(prompt),
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "knowledge_reference": measure(knowledge),
            "source_code": measure(source),
            "compiler_evaluator_evidence": measure(evidence),
            "trajectory_repair_state": measure(repair_state),
        }

    def _call_llm(
        self,
        *,
        prompt: str,
        call_type: str,
        label: str,
        logger: TrajectoryLogger,
        attempt_id: int,
        evaluation_round: int,
        response_path: Path,
        knowledge_selection: Any = None,
    ) -> LLMResponse:
        call_config = self.config.call_config(call_type)
        prompt_metadata = self._prompt_metadata(prompt)
        system_prompt = system_prompt_for(call_type)
        system_path = response_path.parent / f"system_prompt_{call_type}.txt"
        system_path.parent.mkdir(parents=True, exist_ok=True)
        system_path.write_text(system_prompt, encoding="utf-8")
        prompt_metadata.update(
            {
                "system_prompt_id": system_prompt_id_for(call_type),
                "system_prompt_sha256": hashlib.sha256(
                    system_prompt.encode("utf-8")
                ).hexdigest(),
                "system_prompt_path": str(system_path.relative_to(self.state_dir)),
            }
        )
        if knowledge_selection is not None:
            selection_payload = (
                knowledge_selection.to_dict()
                if hasattr(knowledge_selection, "to_dict")
                else dict(knowledge_selection)
            )
            prompt_metadata["knowledge_selection"] = {
                "task_facts": selection_payload.get("task_facts", {}),
                "selection_metadata": selection_payload.get("selection_metadata", {}),
                "selection_trace": selection_payload.get("selection_trace", []),
            }
            selection_metadata = selection_payload.get("selection_metadata", {})
            skill_text = "\n".join(
                str(excerpt.get("text", ""))
                for knowledge_module in selection_payload.get(
                    "skill_knowledge_modules", []
                )
                for excerpt in knowledge_module.get("excerpts", [])
            )
            structured_text = json.dumps(
                {
                    "hard_constraints": selection_payload.get("hard_constraints", []),
                    "api_facts": selection_payload.get("api_facts", []),
                    "design_patterns": selection_payload.get("design_patterns", []),
                    "failure_guidance": selection_payload.get("failure_guidance", []),
                    "provenance": selection_payload.get("provenance", []),
                },
                ensure_ascii=False,
            )
            prompt_metadata["knowledge_input"] = {
                "mode": selection_payload.get("budget", {}).get("input_mode", "bounded"),
                "input_truncated": bool(selection_metadata.get("input_truncated", False)),
                "truncated_sections": list(selection_metadata.get("truncated_sections", [])),
                "selected_skill_ids": list(selection_metadata.get("selected_skill_ids", [])),
                "rendered_skill_ids": list(selection_metadata.get("rendered_skill_ids", [])),
                "selected_structured_ids": list(selection_metadata.get("selected_structured_ids", [])),
                "rendered_structured_ids": list(selection_metadata.get("rendered_structured_ids", [])),
                "runtime_fact_ids": list(selection_metadata.get("runtime_fact_ids", [])),
                "rendered_runtime_fact_ids": list(selection_metadata.get("rendered_runtime_fact_ids", [])),
                "skill_chars": len(skill_text),
                "structured_chars": len(structured_text),
                "runtime_chars": len(str(selection_payload.get("runtime_facts", ""))),
            }
        last_error = ""
        for retry in range(self.config.llm_transient_retries + 1):
            task = self.progress.start(label if retry == 0 else f"{label} retry {retry}")
            try:
                response = self._provider_generate(prompt, call_config)
            except Exception as error:
                last_error = str(error)
                task.finish(status="failed", detail=type(error).__name__)
                logger.save_call_failure(
                    attempt_id,
                    call_type=call_type,
                    evaluation_round=evaluation_round,
                    retry=retry,
                    error=last_error,
                    request_options=call_config.__dict__,
                    prompt_metadata=prompt_metadata,
                )
                if retry < self.config.llm_transient_retries:
                    continue
                raise LLMCallFailure("llm_transport_failed", last_error) from error
            logger.save_call(
                attempt_id,
                response.to_dict(),
                call_type=call_type,
                evaluation_round=evaluation_round,
                retry=retry,
                prompt_metadata=prompt_metadata,
            )
            target = response_path if retry == 0 else response_path.with_name(
                f"{response_path.stem}_retry_{retry:02d}{response_path.suffix}"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(response.content, encoding="utf-8")
            if response.content.strip():
                task.finish(status="completed", detail=f"{token_detail(response.usage)}, model={response.model}")
                return response
            last_error = f"LLM returned no final content (finish_reason={response.finish_reason or 'unknown'})"
            task.finish(status="failed", detail=last_error)
            if retry >= self.config.llm_transient_retries:
                code = "llm_output_exhausted" if response.finish_reason == "length" else "llm_empty_content"
                raise LLMCallFailure(code, last_error)
        raise LLMCallFailure("llm_call_failed", last_error or "LLM call failed")

    def _normalize_evaluation(self, result: EvalResult, round_dir: Path) -> EvalResult:
        if result.error:
            if result.structured_failure is None:
                result.structured_failure = parse_structured_failure(
                    stage=result.failure_stage or "unknown", output=read_result_log(result)
                ).to_dict()
            return result
        if self._is_valid_result(result):
            return result
        if not result.compiled:
            stage, code, message = "ascendc_build", "evaluation_incomplete", "candidate was not compiled"
        elif not result.correctness:
            stage, code, message = "correctness", "evaluation_incomplete", "candidate correctness was not verified"
        else:
            stage, code, message = "performance", "performance_score_missing", "candidate has no valid performance score"
        path = round_dir / "evaluation_incomplete.log"
        self._write_error(path, message)
        result.error = message
        result.failure_stage = stage
        result.failure_code = code
        result.error_excerpt = message
        result.details_path = str(path.resolve())
        result.structured_failure = parse_structured_failure(stage=stage, output=message).to_dict()
        return result

    def _report_result(self, number: int, decision: str, result: EvalResult) -> None:
        score = f", score={result.score:.4f}" if result.score is not None else ""
        self.progress.emit(
            f"Evaluation {number} · {decision} (compiled={result.compiled}, correctness={result.correctness}{score})"
        )
        if result.error:
            self.progress.emit(
                f"Evaluation {number} failure · stage={result.failure_stage or 'unknown'} "
                f"code={result.failure_code or 'unknown'} · {result.error}"
            )
            for line in result.error_excerpt.splitlines():
                self.progress.emit(f"  {line}")

    def _failure_summary(self, round_num: int, evaluation: dict[str, Any]) -> dict[str, Any] | None:
        message = str(evaluation.get("error") or "")
        if not message:
            return None
        stage = evaluation.get("failure_stage") or "unknown"
        code = evaluation.get("failure_code") or "unknown"
        if stage == "unknown":
            lowered = message.lower()
            for marker, inferred_stage, inferred_code in (
                ("static validation", "static_validation", "static_validation_failed"),
                ("ascendc source validation", "ascendc_source_validation", "ascendc_source_validation_failed"),
                ("ascendc build", "ascendc_build", "ascendc_build_failed"),
                ("correctness", "correctness", "correctness_failed"),
                ("performance", "performance", "performance_failed"),
                ("invalid llm file bundle", "response_format", "response_format_failed"),
            ):
                if marker in lowered:
                    stage, code = inferred_stage, inferred_code
                    break
        excerpt = evaluation.get("error_excerpt") or ""
        if not excerpt:
            raw = evaluation.get("verify_output") if stage == "correctness" else evaluation.get("compile_output")
            excerpt = extract_error_excerpt(str(raw or message))
        return {
            "round": round_num,
            "stage": stage,
            "code": code,
            "message": message,
            "excerpt": excerpt,
            "details_path": evaluation.get("details_path"),
            "kind": evaluation.get("failure_kind", "candidate"),
            "structured_failure": evaluation.get("structured_failure"),
        }

    @staticmethod
    def _truncation_retry_prompt(prompt: str) -> str:
        return f"""{prompt}

## 输出截断后的重试
从头只返回一个紧凑 JSON 对象，不要使用 Markdown。仅包含最小但完整的文件 delta，
并确保响应不超过输出 token 上限。
"""

    @staticmethod
    def _round_record(
        *,
        attempt_id: int,
        evaluation_round: int,
        budget_phase: str,
        budget_round: int,
        decision: str,
        result: EvalResult,
        selection: Any,
        response: LLMResponse,
        generation_attempts: int,
        candidate_path: str,
        plan_item: dict[str, Any] | None,
        fingerprint: str | None,
        frontier: dict[str, Any],
        incident_id: str,
        confirmed_experience_id: str | None,
        stage_observation: dict[str, Any],
        stage_timings: list[dict[str, Any]],
        repair_attempt: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selection_payload = selection.to_dict()
        input_route = selection_payload.get("task_facts", {}).get("input_route")
        result_route = None
        if result.error:
            route = ContextSelector.derive_route(
                workflow_phase=budget_phase,
                current_exists=True,
                previous=result,
                evidence=read_result_log(result),
            )
            result_route = {
                "primary": route.primary_skill,
                "reason": route.route_reason,
                "ownership": route.failure_ownership,
                "debug_category": route.debug_category,
                "matched_trigger": route.matched_route_trigger,
            }
        return {
            "round": attempt_id,
            "attempt_id": attempt_id,
            "evaluation_round": evaluation_round,
            "budget_phase": budget_phase,
            "budget_round": budget_round,
            "counts_toward_budget": True,
            "decision": decision,
            "evaluation": result.to_dict(),
            "response_model": response.model,
            "usage": response.usage,
            "latency_seconds": response.latency_seconds,
            "generation_attempts": generation_attempts,
            "compile_repair_attempts": 0,
            "evaluation_attempts": [result.to_dict()],
            "repair_error": None,
            "response_finish_reason": response.finish_reason,
            "knowledge": selection_payload,
            "input_route": input_route,
            "result_failure_route": result_route,
            "candidate": candidate_path,
            "plan_item": plan_item,
            "failure_fingerprint": fingerprint,
            "structured_failure": result.structured_failure,
            "frontier": frontier,
            "incident_id": incident_id,
            "confirmed_experience_id": confirmed_experience_id,
            "stage_observation": stage_observation,
            "evaluation_gates": evaluation_gates(result),
            "stage_timings": stage_timings,
            "repair_attempt": repair_attempt,
        }

    @staticmethod
    def _resolved_fact_ids(selection: Any) -> list[str]:
        facts = getattr(selection, "relevant_facts", [])
        if not facts:
            facts = getattr(selection, "api_facts", [])
        return sorted(
            {
                str(fact["fact_id"])
                for fact in facts
                if isinstance(fact, dict) and fact.get("fact_id")
            }
        )

    def _migrate_trajectory(self, logger: TrajectoryLogger) -> None:
        records = [item for item in logger.data.get("rounds", []) if isinstance(item, dict)]
        needs_migration = logger.data.get("schema_version", 0) < 3 or any(
            logger.counts_toward_budget(item) and not item.get("budget_phase") for item in records
        )
        if not needs_migration:
            return
        found_baseline = False
        baseline_round = None
        baseline_score = None
        bootstrap = optimization = 0
        for item in records:
            if not logger.counts_toward_budget(item):
                continue
            raw = item.get("evaluation") or {}
            try:
                valid = self._is_valid_result(EvalResult(**raw))
            except TypeError:
                valid = False
            if not found_baseline:
                bootstrap += 1
                item.update(budget_phase="bootstrap", budget_round=bootstrap)
                if valid:
                    found_baseline = True
                    baseline_round = item.get("round")
                    baseline_score = raw.get("score")
                    item["decision"] = "BASELINE_KEEP"
                elif item.get("decision") == "DISCARD":
                    item["decision"] = "FAIL"
            else:
                optimization += 1
                item.update(budget_phase="optimization", budget_round=optimization)
        logger.data["schema_version"] = 3
        logger.data.setdefault("workflow", {}).update(
            {
                "phase": "PLAN" if records else "INITIAL_GENERATE",
                "baseline_round": baseline_round,
                "baseline_score": baseline_score,
                "bootstrap_attempts_completed": bootstrap,
                "optimization_rounds_completed": optimization,
                "consecutive_failures": 0,
                "plan_version": 0,
                "migrated_from_schema": 2,
            }
        )
        logger._save()

    @staticmethod
    def _budget_counts(logger: TrajectoryLogger) -> tuple[int, int]:
        bootstrap = optimization = 0
        for item in logger.data.get("rounds", []):
            if not isinstance(item, dict) or not logger.counts_toward_budget(item):
                continue
            if item.get("budget_phase") == "optimization":
                optimization += 1
            else:
                bootstrap += 1
        return bootstrap, optimization

    def _resume_state(
        self, logger: TrajectoryLogger
    ) -> tuple[FileBundle | None, FileBundle | None, EvalResult | None, EvalResult | None, int | None]:
        working = capture_bundle(self.task_dir)
        current = working if working.files else None
        counted = [
            item
            for item in logger.data.get("rounds", [])
            if isinstance(item, dict) and logger.counts_toward_budget(item)
        ]
        previous = EvalResult(**counted[-1]["evaluation"]) if counted and counted[-1].get("evaluation") else None
        best_bundle = best_result = None
        best_round = logger.data.get("best_round")
        best_path = self.state_dir / "best.json"
        if best_path.is_file():
            best_bundle = self._read_bundle(best_path)
            current = best_bundle
            restore_bundle(self.task_dir, best_bundle)
            record = next((item for item in counted if item.get("round") == best_round), None)
            if record and record.get("evaluation"):
                best_result = EvalResult(**record["evaluation"])
        return current, best_bundle, previous, best_result, best_round

    def _load_plan(self) -> dict[str, Any] | None:
        path = self.state_dir / "plan.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None

    def _save_plan(self, parsed: dict[str, Any], *, version: int, mode: str, origin: str) -> dict[str, Any]:
        plan = {
            "version": version,
            "mode": mode,
            "origin": origin,
            "evidence_status": parsed.get("evidence_status", "sufficient"),
            "observations": parsed.get("observations", []),
            "ruled_out": parsed.get("ruled_out", []),
            "unknowns": parsed.get("unknowns", []),
            "diagnosis": parsed.get("diagnosis", ""),
            "items": [{**item, "status": "PENDING", "decision": None} for item in parsed["items"]],
        }
        (self.state_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._write_plan_markdown(plan)
        return plan

    def _write_plan_markdown(self, plan: dict[str, Any]) -> None:
        lines = [
            f"# AscendC plan v{plan['version']}",
            "",
            f"Mode: {plan['mode']}",
            f"Base attempt: {RepairStateManager(self.state_dir).accepted_attempt_id or 'none'}",
            f"Evidence status: {plan.get('evidence_status', 'unknown')}",
            "",
            plan.get("diagnosis", ""),
            "",
        ]
        if plan.get("unknowns"):
            lines.extend(["## Unknowns", ""])
            for item in plan["unknowns"]:
                lines.append(
                    f"- {item.get('question')}: 需要 {item.get('required_evidence')}"
                )
            lines.append("")
        for item in plan["items"]:
            checked = "x" if item.get("status") == "SETTLED" else " "
            outcome = f" [{item['decision']}]" if item.get("decision") else ""
            lines.extend(
                [
                    f"- [{checked}] {item['id']}: {item['change']}{outcome}",
                    f"  - Hypothesis: {item['hypothesis']}",
                    f"  - Expected: {item['expected_signal']}",
                    f"  - Target files: {', '.join(item.get('target_files', [])) or 'unspecified'}",
                    f"  - Allow interface change: {bool(item.get('allow_interface_change', False))}",
                    f"  - Falsifies: {', '.join(item.get('falsifies', [])) or 'none'}",
                ]
            )
        (self.state_dir / "plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _save_plan_progress(self, plan: dict[str, Any]) -> None:
        (self.state_dir / "plan.json").write_text(
            json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._write_plan_markdown(plan)

    @staticmethod
    def _active_plan_item(plan: dict[str, Any] | None) -> dict[str, Any] | None:
        if not plan:
            return None
        return next((item for item in plan.get("items", []) if item.get("status") == "PENDING"), None)

    def _create_plan(
        self,
        *,
        logger: TrajectoryLogger,
        current: FileBundle | None,
        previous: EvalResult | None,
        reference: str,
        cases: str,
        attempt_id: int,
        evaluation_round: int,
        mode: str,
        diagnosis_required: bool,
        knowledge_context: str,
        knowledge_selection: Any = None,
        repair_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        version = int(logger.data.get("workflow", {}).get("plan_version", 0)) + 1
        initial = current is None and previous is None
        call_type = "diagnose" if diagnosis_required else "planner"
        plan_dir = self.state_dir / f"{call_type}_v{version:02d}"
        prompt = build_plan_prompt(
            reference_code=reference,
            cases_text=cases,
            current=current,
            result=previous,
            mode=mode,
            history=logger.data.get("rounds", []),
            diagnosis_required=diagnosis_required,
            knowledge_context=knowledge_context,
            initial=initial,
            repair_state=(
                repair_state if mode == "bootstrap" and not initial else None
            ),
        )
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        response_path = plan_dir / "response.txt"
        parsed = None
        if self.config.resume and response_path.is_file():
            try:
                parsed = parse_plan(
                    response_path.read_text(encoding="utf-8"),
                    min_items=1,
                    max_items=1,
                    require_evidence=diagnosis_required,
                    allow_insufficient=diagnosis_required,
                )
                self.progress.emit(
                    f"{call_type.upper()} v{version} · reused persisted response"
                )
            except (ValueError, json.JSONDecodeError):
                parsed = None
        if parsed is None:
            response = self._call_llm(
                prompt=prompt,
                call_type=call_type,
                label=f"{call_type.upper()} v{version}",
                logger=logger,
                attempt_id=attempt_id,
                evaluation_round=evaluation_round,
                response_path=response_path,
                knowledge_selection=knowledge_selection,
            )
            parsed = parse_plan(
                response.content,
                min_items=1,
                max_items=1,
                require_evidence=diagnosis_required,
                allow_insufficient=diagnosis_required,
            )
        (plan_dir / "result.json").write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        plan = self._save_plan(
            parsed,
            version=version,
            mode=mode,
            origin=(
                "diagnose"
                if diagnosis_required
                else "initial_plan" if initial else "plan"
            ),
        )
        if plan["items"]:
            logger.update_workflow(
                phase="EDIT",
                plan_version=version,
                active_plan_item=plan["items"][0]["id"],
                consecutive_failures=(
                    0
                    if diagnosis_required
                    else logger.data.get("workflow", {}).get("consecutive_failures", 0)
                ),
            )
        else:
            logger.update_pending_phase("DIAGNOSE")
            logger.update_workflow(
                phase="DIAGNOSE",
                plan_version=version,
                active_plan_item=None,
                diagnose_pending=True,
                diagnosis_block={
                    "evidence_status": plan.get("evidence_status"),
                    "diagnosis": plan.get("diagnosis"),
                    "unknowns": plan.get("unknowns", []),
                    "plan_version": version,
                },
            )
        return plan

    def _knowledge_context(
        self,
        *,
        logger: TrajectoryLogger,
        knowledge_state: Any,
        knowledge_version: Any,
        reference: str,
        cases: str,
        current: FileBundle | None,
        previous: EvalResult | None,
        round_dir: Path,
        attempt_id: int,
        evaluation_round: int,
        max_api_docs: int,
        max_knowledge_chars: int,
        knowledge_state_path: Path,
        active_plan: dict[str, Any] | None = None,
        audience: str = "generator",
        workflow_phase: str = "bootstrap",
    ) -> tuple[Any, str]:
        source_evidence = "\n".join(current.files.values()) if current else ""
        failure_evidence = read_result_log(previous) if previous else ""
        planned_evidence = json.dumps(active_plan or {}, ensure_ascii=False)
        evidence_parts = [reference, cases, source_evidence, failure_evidence, planned_evidence]
        evidence = "\n".join(evidence_parts)
        if self.config.knowledge_mode == "structured":
            failure_case_info = (
                (previous.structured_failure or {}).get("case_info", {})
                if previous and isinstance(previous.structured_failure, dict)
                else {}
            )
            profile_match = re.search(
                r"\b(?:active_)?profile\s*[:=]\s*(smoke|shape|dtype|full|benchmark)\b",
                failure_evidence,
                re.IGNORECASE,
            )
            active_profile = str(
                (previous.active_profile if previous else None)
                or failure_case_info.get("profile")
                or failure_case_info.get("active_profile")
                or (profile_match.group(1).lower() if profile_match else "")
            ) or None
            raw_indices = failure_case_info.get(
                "profile_case_indices", failure_case_info.get("case_indices", [])
            )
            profile_indices = (
                [int(item) for item in raw_indices]
                if isinstance(raw_indices, (list, tuple))
                else []
            )
            raw_features = failure_case_info.get(
                "profile_features", failure_case_info.get("features", [])
            )
            profile_features = (
                [str(item) for item in raw_features]
                if isinstance(raw_features, (list, tuple))
                else []
            )
            stage_request = (
                self.context_selector.request(
                    audience=audience,
                    workflow_phase=workflow_phase,
                    operator=Path(self.config.op_file).stem,
                    soc=self.config.soc_version,
                    runtime_version=knowledge_version.runtime_version,
                    knowledge_version=knowledge_version.knowledge_version,
                    current_exists=bool(current and current.files),
                    previous=previous,
                    evidence=evidence,
                    source_evidence=source_evidence,
                    planned_evidence=planned_evidence,
                    active_profile=active_profile,
                    profile_case_indices=profile_indices,
                    profile_features=profile_features,
                )
                if self.context_selector is not None
                else None
            )
            context = KnowledgeContext(
                operator=Path(self.config.op_file).stem,
                phase=(
                    ",".join(stage_request.stages)
                    if stage_request is not None
                    else "diagnose" if previous and previous.error else "plan_generate"
                ),
                runtime_version=knowledge_version.runtime_version,
                knowledge_version=knowledge_version.knowledge_version,
                soc=self.config.soc_version,
                source_symbols=extract_api_symbols(evidence),
                failure=previous.structured_failure if previous and previous.error else None,
                active_plan=active_plan,
            )
            if self.config.uses_structured_prompt:
                knowledge_build_path = locate_knowledge_build(
                    Path(self.config.knowledge_store),
                    knowledge_version.knowledge_version,
                    self.config.knowledge_build_id,
                )
                bundle = StructuredKnowledgeRouter(
                    KnowledgeBuild(knowledge_build_path)
                ).route(context)
            else:
                bundle = KnowledgeBundle(
                    context=context,
                    retrieval_trace=[
                        RetrievalTraceEntry(
                            "source_policy",
                            "structured",
                            "rejected",
                            "structured prompt source disabled by knowledge_source=skills",
                        )
                    ],
                )
            payload = bundle.to_dict()
            bundle_path = round_dir / (
                f"knowledge_bundle_{audience}.json"
                if self.context_selector is not None
                else "knowledge_bundle.json"
            )
            bundle_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            trace_path = round_dir / (
                f"retrieval_trace_{audience}.json"
                if self.context_selector is not None
                else "retrieval_trace.json"
            )
            trace_path.write_text(
                json.dumps(payload["retrieval_trace"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            selected_result: Any = bundle
            if self.context_selector is not None and stage_request is not None:
                runtime_facts: Any = ""
                if audience == "generator" or any(
                    stage.endswith("_debug") for stage in stage_request.stages
                ):
                    symbols = list(
                        dict.fromkeys(
                            [
                                *stage_request.failure_symbols,
                                *stage_request.source_symbols,
                                *stage_request.planned_symbols,
                            ]
                        )
                    ) or extract_api_symbols(evidence)
                    runtime_facts = collect_runtime_facts(
                        symbols,
                        runtime_version=knowledge_version.runtime_version,
                        cache_path=round_dir
                        / f"runtime_header_facts_{audience}.json",
                        max_chars=(
                            4000
                        ),
                        soc_version=self.config.soc_version,
                        project_root=self.REPO_ROOT,
                        probe_manifest=self._probe_manifest_path(),
                        failure_evidence=failure_evidence,
                    )
                selected_context, skill_selection = self.context_selector.select(
                    bundle=bundle,
                    request=stage_request,
                    runtime_facts=runtime_facts,
                )
                selected_context.task_facts["knowledge_source"] = (
                    self.config.knowledge_source
                )
                if getattr(runtime_facts, "environment_fingerprint", None):
                    selected_context.task_facts["environment_fingerprint"] = runtime_facts.environment_fingerprint
                selected_context.budget["max_chars"] = min(
                    int(selected_context.budget["max_chars"]),
                    max_knowledge_chars,
                )
                rendered = render_stage_context(selected_context)
                selected_result = selected_context
                context_payload = selected_context.to_dict()
                (round_dir / f"{audience}_context.json").write_text(
                    json.dumps(context_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                (round_dir / f"skill_selection_{audience}.json").write_text(
                    json.dumps(skill_selection.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                references_name = (
                    "planner_references.md" if audience == "planner" else "references.md"
                )
                (round_dir / references_name).write_text(rendered, encoding="utf-8")
                # Preserve the established generic artifacts for downstream tools.
                if audience == "generator":
                    (round_dir / "knowledge_bundle.json").write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    (round_dir / "retrieval_trace.json").write_text(
                        json.dumps(payload["retrieval_trace"], ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    (round_dir / "selected_knowledge.json").write_text(
                        json.dumps(context_payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            else:
                rendered = render_bundle(bundle, max_chars=max_knowledge_chars)
                (round_dir / "references.md").write_text(rendered, encoding="utf-8")
                (round_dir / "selected_knowledge.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            return selected_result, rendered
        candidates = candidate_doc_ids(
            evidence,
            version=knowledge_version,
            exclude=knowledge_state.working_doc_ids,
            limit=5,
        )
        route_mode = "reuse"
        route_candidates: list[str] | None = []
        if not knowledge_state.initialized:
            route_mode, route_candidates = "initial_full", None
        elif candidates and candidates[0][1] >= 40:
            route_mode = "incremental"
            route_candidates = [doc_id for doc_id, _ in candidates]
        response = None
        if route_mode in {"initial_full", "incremental"}:
            prompt = build_knowledge_prompt(
                reference_code=reference,
                current=current,
                previous_result=previous,
                version=knowledge_version,
                mode=route_mode,
                candidate_doc_ids=route_candidates,
                working_doc_ids=knowledge_state.working_doc_ids,
            )
            (round_dir / "knowledge_prompt.txt").write_text(prompt, encoding="utf-8")
            response = self._call_llm(
                prompt=prompt,
                call_type="knowledge_router",
                label=f"Evaluation {evaluation_round} · knowledge router ({route_mode})",
                logger=logger,
                attempt_id=attempt_id,
                evaluation_round=evaluation_round,
                response_path=round_dir / "knowledge_response.txt",
            )
            if route_mode == "initial_full":
                knowledge_state.full_route_count += 1
            else:
                knowledge_state.incremental_route_count += 1
        else:
            (round_dir / "knowledge_reuse.json").write_text(
                json.dumps({"mode": route_mode, "working_doc_ids": knowledge_state.working_doc_ids}, indent=2),
                encoding="utf-8",
            )
        if response is not None:
            selection = parse_knowledge_selection(
                response.content,
                version=knowledge_version,
                max_api_docs=max_api_docs if route_mode == "initial_full" else 2,
                fallback_text=evidence,
            )
            selection.mode = route_mode
            apply_selection_to_state(
                knowledge_state,
                selection,
                max_docs=max_api_docs,
                preferred_doc_ids=selection.doc_ids,
            )
        else:
            selection = selection_from_state(knowledge_state, version=knowledge_version, mode=route_mode)
        active_ids = active_doc_ids(
            knowledge_state,
            evidence,
            version=knowledge_version,
            preferred_doc_ids=selection.doc_ids,
            limit=min(5, max_api_docs),
        )
        selection = selection_from_state(
            knowledge_state,
            version=knowledge_version,
            mode=route_mode,
            active_ids=active_ids,
            reason=selection.reason,
        )
        symbols = extract_api_symbols(evidence)
        facts = collect_runtime_facts(
            symbols,
            runtime_version=knowledge_version.runtime_version,
            cache_path=round_dir / "runtime_header_facts.json",
            soc_version=self.config.soc_version,
            project_root=self.REPO_ROOT,
            probe_manifest=self._probe_manifest_path(),
            failure_evidence=failure_evidence,
        )
        selection.runtime_fact_symbols = facts.symbols
        selection.conflicts = facts.conflicts
        context = render_knowledge(
            selection,
            version=knowledge_version,
            max_chars=max_knowledge_chars,
            runtime_facts=facts.text,
            conflicts=facts.conflicts,
        )
        knowledge_state.known_symbols = list(dict.fromkeys([*knowledge_state.known_symbols, *symbols]))[:128]
        knowledge_state.last_round = attempt_id
        save_knowledge_state(knowledge_state_path, knowledge_state)
        (round_dir / "selected_knowledge.json").write_text(
            json.dumps(selection.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (round_dir / "references.md").write_text(context, encoding="utf-8")
        return selection, context

    def _record_orchestration_failure(
        self,
        logger: TrajectoryLogger,
        *,
        attempt_id: int,
        evaluation_round: int,
        result: EvalResult,
    ) -> None:
        logger.save_orchestration_attempt(
            {
                "attempt_id": attempt_id,
                "evaluation_round": evaluation_round,
                "stage": result.failure_stage,
                "code": result.failure_code,
                "error": result.error,
                "details_path": result.details_path,
                "counts_toward_budget": False,
            }
        )

    def _prepare_candidate(
        self,
        *,
        logger: TrajectoryLogger,
        knowledge_state: Any,
        knowledge_version: Any,
        knowledge_state_path: Path,
        reference: str,
        cases: str,
        current: FileBundle | None,
        previous: EvalResult | None,
        active_item: dict[str, Any] | None,
        budget_phase: str,
        budget_round: int,
        evaluation_round: int,
        attempt_id: int,
        max_api_docs: int,
        max_knowledge_chars: int,
        repair_state_manager: RepairStateManager,
        prepared_knowledge: tuple[Any, str] | None = None,
    ) -> tuple[FileBundle, FileBundle | None, Any, LLMResponse, int, str]:
        round_dir = self.state_dir / f"round_{attempt_id:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = f"round_{attempt_id:02d}/candidate.json"
        checkpoint = round_dir / "candidate.json"
        base_checkpoint = round_dir / "base.json"
        base_bundle = FileBundle(files=dict(current.files)) if current else None
        if logger.pending_state().get("pending_phase") == "EVAL" and checkpoint.is_file():
            if base_checkpoint.is_file():
                base_bundle = self._read_bundle(base_checkpoint)
            candidate = self._read_bundle(checkpoint)
            restore_bundle(self.task_dir, candidate)
            if self.config.knowledge_mode == "structured":
                selection, _ = self._knowledge_context(
                    logger=logger,
                    knowledge_state=knowledge_state,
                    knowledge_version=knowledge_version,
                    reference=reference,
                    cases=cases,
                    current=current,
                    previous=previous,
                    round_dir=round_dir,
                    attempt_id=attempt_id,
                    evaluation_round=evaluation_round,
                    max_api_docs=max_api_docs,
                    max_knowledge_chars=max_knowledge_chars,
                    knowledge_state_path=knowledge_state_path,
                    active_plan=active_item,
                    audience="generator",
                    workflow_phase=budget_phase,
                )
            else:
                selection = selection_from_state(
                    knowledge_state,
                    version=knowledge_version,
                    mode="resume_eval",
                )
            response = LLMResponse(content="", model="checkpoint", usage={})
            return candidate, base_bundle, selection, response, 1, candidate_path

        self._write_bundle(base_checkpoint, base_bundle or FileBundle(files={}))

        if prepared_knowledge is None:
            selection, knowledge_context = self._knowledge_context(
                logger=logger,
                knowledge_state=knowledge_state,
                knowledge_version=knowledge_version,
                reference=reference,
                cases=cases,
                current=current,
                previous=previous,
                round_dir=round_dir,
                attempt_id=attempt_id,
                evaluation_round=evaluation_round,
                max_api_docs=max_api_docs,
                max_knowledge_chars=max_knowledge_chars,
                knowledge_state_path=knowledge_state_path,
                active_plan=active_item,
                audience="generator",
                workflow_phase=budget_phase,
            )
        else:
            selection, knowledge_context = prepared_knowledge
        protected_regions = build_repair_policy(
            current,
            previous,
            active_item,
            workflow_phase=budget_phase,
        ).to_dict()
        interface_contract = load_interface_contract(
            self.state_dir / "interface_contract.json"
        )
        if interface_contract is not None:
            protected_regions["interface_contract"] = interface_contract
        prompt = build_prompt(
            reference_code=reference,
            cases_text=cases,
            current=current,
            previous_result=previous,
            round_num=attempt_id,
            knowledge_context=knowledge_context,
            phase=budget_phase.upper(),
            plan_item=active_item,
            repair_state=repair_state_manager.prompt_summary(previous),
            knowledge_selection=selection,
            protected_regions=protected_regions,
        )
        (round_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        response = self._call_llm(
            prompt=prompt,
            call_type="generator",
            label=f"{budget_phase.title()} {budget_round} · generator",
            logger=logger,
            attempt_id=attempt_id,
            evaluation_round=evaluation_round,
            response_path=round_dir / "response.txt",
            knowledge_selection=selection,
        )
        generation_attempts = 1
        if response.finish_reason == "length":
            generation_attempts += 1
            retry_prompt = self._truncation_retry_prompt(prompt)
            (round_dir / "retry_prompt.txt").write_text(retry_prompt, encoding="utf-8")
            response = self._call_llm(
                prompt=retry_prompt,
                call_type="generator_retry",
                label=f"{budget_phase.title()} {budget_round} · truncated retry",
                logger=logger,
                attempt_id=attempt_id,
                evaluation_round=evaluation_round,
                response_path=round_dir / "response_retry_01.txt",
                knowledge_selection=selection,
            )
        try:
            delta = parse_file_bundle(response.content)
            candidate = FileBundle(files=dict(current.files) if current else {})
            for path in delta.delete:
                candidate.files.pop(path, None)
            candidate.files.update(delta.files)
            validate_initial_bundle(candidate)
            restore_bundle(self.task_dir, candidate)
            candidate = capture_bundle(self.task_dir)
            self._write_bundle(checkpoint, candidate)
        except (ValueError, json.JSONDecodeError) as error:
            if base_bundle is not None:
                restore_bundle(self.task_dir, base_bundle)
            path = round_dir / "response_format.log"
            self._write_error(path, f"invalid LLM file bundle: {error}")
            raise CandidatePreparationFailure(
                self._failure_result(
                    stage="response_format",
                    code="response_format_failed",
                    message=f"invalid LLM file bundle: {error}",
                    details_path=path,
                )
            ) from error
        logger.update_pending_phase("EVAL")
        return candidate, base_bundle, selection, response, generation_attempts, candidate_path

    def run(self) -> dict[str, Any]:
        self.progress.emit("Preparing task and resume state")
        self._prepare_task()
        logger = TrajectoryLogger(self.state_dir, self.config.to_dict())
        recorded_config = logger.data.get("config", {})
        if self.config.resume and isinstance(recorded_config, dict):
            recorded_source = str(recorded_config.get("knowledge_source") or "").lower()
            if not recorded_source:
                if str(recorded_config.get("knowledge_mode", "structured")).lower() == "document":
                    recorded_source = "document"
                elif recorded_config.get("skill_adapter"):
                    recorded_source = "hybrid"
                else:
                    recorded_source = "structured"
            if recorded_source != self.config.knowledge_source:
                raise ValueError(
                    "cannot resume with different knowledge source: "
                    f"recorded={recorded_source}, current={self.config.knowledge_source}"
                )
        self._migrate_trajectory(logger)
        logger.mark_running()
        logger.save_invocation(self.config.to_dict())
        if self.config.deprecated_repair_options_active:
            self.progress.emit("WARNING: --repair-* options are deprecated and no longer trigger repair calls")
        self.progress.emit(
            "LLM configuration: "
            f"provider={self.config.provider}, requested_model={self.config.model}, "
            f"knowledge_source={self.config.knowledge_source}, "
            f"knowledge_input_mode={self.config.knowledge_input_mode}, "
            f"bootstrap={self.config.max_bootstrap_rounds}, optimization={self.config.max_rounds}, "
            f"total={self.config.max_total_rounds or 'unlimited'}, "
            f"generator={self.config.generator_max_tokens}/{self.config.generator_thinking}/"
            f"{self.config.generator_reasoning_effort}, "
            f"planner={self.config.planner_max_tokens}/{self.config.planner_thinking}/"
            f"{self.config.planner_reasoning_effort}"
        )
        if self.config.blanket_max_tokens_override_active:
            self.progress.emit("WARNING: deprecated --max-tokens blanket override is active")

        reference, cases = self._inputs()
        current, best_bundle, previous, best_result, best_round = self._resume_state(logger)
        frontier_manager = FrontierManager(self.state_dir)
        repair_state_manager = RepairStateManager(self.state_dir)
        if frontier_manager.highest_bundle() is None and best_bundle is not None and best_result is not None:
            frontier_manager.observe(best_bundle, best_result, int(best_round or 0))
        workflow = logger.data.setdefault("workflow", {})
        baseline_round = workflow.get("baseline_round")
        baseline_score = workflow.get("baseline_score")
        if baseline_round is None and self._is_valid_result(best_result):
            baseline_round, baseline_score = best_round, best_result.score
        if baseline_round is None:
            repair_bundle = repair_state_manager.accepted_bundle()
            repair_evaluation = repair_state_manager.accepted_evaluation()
            if repair_bundle is not None and repair_evaluation is not None:
                current = repair_bundle
                previous = repair_evaluation
                restore_bundle(self.task_dir, repair_bundle)
        plan = self._load_plan()

        task = self.progress.start("Resolve CANN knowledge")
        try:
            knowledge_version = resolve_knowledge_version(self.config)
        except Exception:
            task.finish(status="failed")
            raise
        task.finish(status="completed", detail=f"version={knowledge_version.knowledge_version}")
        recorded_knowledge = logger.data.get("knowledge")
        if recorded_knowledge and recorded_knowledge != knowledge_version.to_dict():
            raise ValueError(
                "cannot resume with different CANN knowledge: "
                f"recorded={recorded_knowledge}, current={knowledge_version.to_dict()}"
            )
        logger.save_knowledge(knowledge_version.to_dict())
        max_api_docs, max_knowledge_chars = knowledge_limits()
        historical_selection = next(
            (
                item.get("knowledge")
                for item in reversed(logger.data.get("rounds", []))
                if isinstance(item, dict) and item.get("knowledge")
            ),
            None,
        )
        knowledge_state_path = self.state_dir / "knowledge_state.json"
        knowledge_state = load_knowledge_state(
            knowledge_state_path,
            version=knowledge_version,
            historical_selection=historical_selection,
        )

        preflight = getattr(self.evaluator, "preflight", None)
        if callable(preflight):
            logger.update_workflow(phase="ENVIRONMENT_PREFLIGHT")
            preflight_failure = preflight(self.task_dir, self.state_dir)
            if preflight_failure is not None:
                self._record_orchestration_failure(
                    logger,
                    attempt_id=0,
                    evaluation_round=0,
                    result=preflight_failure,
                )
                return self._finish(
                    logger=logger,
                    final_status="paused",
                    stop_failure=preflight_failure,
                    stop_attempt_id=0,
                    baseline_round=baseline_round,
                    baseline_score=baseline_score,
                    best_bundle=best_bundle,
                    best_result=best_result,
                    best_round=best_round,
                    knowledge_version=knowledge_version,
                )

        stop_failure: EvalResult | None = None
        stop_attempt_id: int | None = None
        final_status = "running"
        stop_reason: str | None = None
        while True:
            bootstrap_count, optimization_count = self._budget_counts(logger)
            evaluations_completed = bootstrap_count + optimization_count
            if (
                self.config.max_total_rounds is not None
                and evaluations_completed >= self.config.max_total_rounds
            ):
                final_status = "completed" if baseline_round is not None else "blocked"
                stop_reason = "max_total_rounds"
                break
            if baseline_round is None and bootstrap_count >= self.config.max_bootstrap_rounds:
                final_status = "blocked"
                stop_reason = "max_bootstrap_rounds"
                break
            if baseline_round is not None and optimization_count >= self.config.max_rounds:
                final_status = "completed"
                stop_reason = "max_rounds"
                break

            budget_phase = "bootstrap" if baseline_round is None else "optimization"
            budget_round = (bootstrap_count if budget_phase == "bootstrap" else optimization_count) + 1
            evaluation_round = bootstrap_count + optimization_count + 1
            diagnose_required = bool(workflow.get("diagnose_pending"))
            active_item = self._active_plan_item(plan)
            need_plan = active_item is None or diagnose_required
            if diagnose_required:
                plan = None
                active_item = None
            pending_phase = "DIAGNOSE" if diagnose_required else ("PLAN" if need_plan else "EDIT")
            saved_pending = logger.pending_state()
            if (
                saved_pending.get("pending_evaluation_round") == evaluation_round
                and saved_pending.get("pending_phase") == "EVAL"
            ):
                pending_phase = "EVAL"
            attempt_id = logger.begin_pending(evaluation_round, phase=pending_phase)
            stop_attempt_id = attempt_id

            prepared_knowledge: tuple[Any, str] | None = None
            if need_plan:
                round_dir = self.state_dir / f"round_{attempt_id:02d}"
                round_dir.mkdir(parents=True, exist_ok=True)
                if pending_phase != "EVAL":
                    try:
                        prepared_knowledge = self._knowledge_context(
                            logger=logger,
                            knowledge_state=knowledge_state,
                            knowledge_version=knowledge_version,
                            reference=reference,
                            cases=cases,
                            current=current,
                            previous=previous,
                            round_dir=round_dir,
                            attempt_id=attempt_id,
                            evaluation_round=evaluation_round,
                            max_api_docs=max_api_docs,
                            max_knowledge_chars=max_knowledge_chars,
                            knowledge_state_path=knowledge_state_path,
                            active_plan=active_item,
                            audience="planner",
                            workflow_phase=budget_phase,
                        )
                    except LLMCallFailure as error:
                        path = round_dir / "llm_error.log"
                        self._write_error(path, error)
                        stop_failure = self._failure_result(
                            stage="llm_knowledge_router",
                            code=error.code,
                            message=str(error),
                            details_path=path,
                        )
                        self._record_orchestration_failure(
                            logger,
                            attempt_id=attempt_id,
                            evaluation_round=evaluation_round,
                            result=stop_failure,
                        )
                        final_status = "paused"
                        break
                if pending_phase == "EVAL":
                    active_item = self._active_plan_item(plan)
                else:
                    assert prepared_knowledge is not None
                    planning_selection, planning_knowledge = prepared_knowledge
                    try:
                        plan = self._create_plan(
                            logger=logger,
                            current=current,
                            previous=previous,
                            reference=reference,
                            cases=cases,
                            attempt_id=attempt_id,
                            evaluation_round=evaluation_round,
                            mode=budget_phase,
                            diagnosis_required=diagnose_required,
                            knowledge_context=planning_knowledge,
                            knowledge_selection=planning_selection,
                            repair_state=repair_state_manager.prompt_summary(previous),
                        )
                    except (LLMCallFailure, ValueError, json.JSONDecodeError) as error:
                        path = self.state_dir / f"planning_error_{attempt_id:02d}.log"
                        self._write_error(path, error)
                        stop_failure = self._failure_result(
                            stage="llm_planning",
                            code=getattr(error, "code", "planning_response_invalid"),
                            message=f"planning failed: {error}",
                            details_path=path,
                        )
                        self._record_orchestration_failure(
                            logger,
                            attempt_id=attempt_id,
                            evaluation_round=evaluation_round,
                            result=stop_failure,
                        )
                        final_status = "paused"
                        break
                    workflow = logger.data.setdefault("workflow", {})
                    active_item = self._active_plan_item(plan)
                    if (
                        diagnose_required
                        and plan.get("evidence_status") == "insufficient"
                        and active_item is None
                    ):
                        logger.update_workflow(
                            phase="DIAGNOSE",
                            diagnose_pending=True,
                            active_plan_item=None,
                        )
                        final_status = "blocked"
                        stop_reason = "diagnosis_evidence_insufficient"
                        break
                    workflow["diagnose_pending"] = False
                    workflow.pop("diagnosis_block", None)
                    logger.update_workflow(**workflow)
                    logger.update_pending_phase("EDIT")
            if active_item is not None:
                logger.update_workflow(phase="EDIT", active_plan_item=active_item["id"])
                workflow = logger.data["workflow"]

            round_dir = self.state_dir / f"round_{attempt_id:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            base_result = previous
            bootstrap_attempt = baseline_round is None
            generator_knowledge = (
                None if self.context_selector is not None else prepared_knowledge
            )
            try:
                candidate, base_bundle, selection, response, generation_attempts, candidate_path = (
                    self._prepare_candidate(
                    logger=logger,
                    knowledge_state=knowledge_state,
                    knowledge_version=knowledge_version,
                    knowledge_state_path=knowledge_state_path,
                    reference=reference,
                    cases=cases,
                    current=current,
                    previous=previous,
                    active_item=active_item,
                    budget_phase=budget_phase,
                    budget_round=budget_round,
                    evaluation_round=evaluation_round,
                    attempt_id=attempt_id,
                    max_api_docs=max_api_docs,
                    max_knowledge_chars=max_knowledge_chars,
                    repair_state_manager=repair_state_manager,
                    prepared_knowledge=generator_knowledge,
                    )
                )
            except LLMCallFailure as error:
                path = round_dir / "llm_error.log"
                self._write_error(path, error)
                stage = "llm_knowledge_router" if not knowledge_state.initialized else "llm_generator"
                stop_failure = self._failure_result(
                    stage=stage,
                    code=error.code,
                    message=str(error),
                    details_path=path,
                )
                self._record_orchestration_failure(
                    logger,
                    attempt_id=attempt_id,
                    evaluation_round=evaluation_round,
                    result=stop_failure,
                )
                final_status = "paused"
                break
            except CandidatePreparationFailure as error:
                stop_failure = error.result
                self._record_orchestration_failure(
                    logger,
                    attempt_id=attempt_id,
                    evaluation_round=evaluation_round,
                    result=stop_failure,
                )
                final_status = "paused"
                break

            if (
                not self.config.mock
                and
                base_bundle is not None
                and semantic_bundle_hash(candidate) == semantic_bundle_hash(base_bundle)
            ):
                restore_bundle(self.task_dir, base_bundle)
                no_op_path = round_dir / "no_op_edit.log"
                message = "candidate changes only comments/whitespace or makes no source change"
                self._write_error(no_op_path, message)
                no_op_result = self._failure_result(
                    stage="no_op_edit",
                    code="no_op_edit_detected",
                    message=message,
                    details_path=no_op_path,
                )
                self._record_orchestration_failure(
                    logger,
                    attempt_id=attempt_id,
                    evaluation_round=evaluation_round,
                    result=no_op_result,
                )
                if active_item is not None:
                    active_item["status"] = "SETTLED"
                    active_item["decision"] = "NO_OP"
                    active_item["attempt_id"] = attempt_id
                    self._save_plan_progress(plan)
                no_op_streak = int(workflow.get("no_op_streak", 0)) + 1
                logger.complete_pending()
                logger.update_workflow(
                    phase="DIAGNOSE",
                    active_plan_item=None,
                    diagnose_pending=True,
                    no_op_streak=no_op_streak,
                )
                workflow = logger.data["workflow"]
                plan = None
                if no_op_streak >= 3:
                    stop_failure = no_op_result
                    final_status = "paused"
                    stop_reason = "repeated_no_op_edit"
                    break
                continue

            progress_cursor = self.progress.event_cursor()
            interface_path = self.state_dir / "interface_contract.json"
            interface_contract = load_interface_contract(interface_path)
            allow_interface_change = bool(
                (active_item or {}).get(
                    "allow_interface_change",
                    (active_item or {}).get("allow_abi_change", False),
                )
            )
            interface_issues = (
                []
                if interface_contract is None or allow_interface_change
                else compare_interface_contract(interface_contract, candidate)
            )
            if interface_issues:
                path = round_dir / "interface_contract_validation.log"
                message = "interface contract violation:\n" + "\n".join(
                    f"- {issue}" for issue in interface_issues
                )
                self._write_error(path, message)
                result = self._failure_result(
                    stage="interface_contract_validation",
                    code="interface_contract_changed_without_authorization",
                    message=message,
                    details_path=path,
                    failure_kind="candidate",
                )
            else:
                try:
                    result = self._normalize_evaluation(
                        self.evaluator.evaluate(self.task_dir, round_dir), round_dir
                    )
                except Exception as error:
                    path = round_dir / "evaluation_error.log"
                    self._write_error(path, error)
                    result = self._failure_result(
                        stage="evaluation",
                        code="evaluator_exception",
                        message=f"local evaluator failed unexpectedly: {error}",
                        details_path=path,
                        failure_kind="infrastructure",
                    )
            stage_timings = self.progress.events_since(progress_cursor)
            stage_observation = observe_evaluation_stages(
                result, workflow_phase=budget_phase
            )
            if result.failure_kind in {"infrastructure", "orchestration"}:
                if base_bundle is not None:
                    restore_bundle(self.task_dir, base_bundle)
                stop_failure = result
                self._record_orchestration_failure(
                    logger,
                    attempt_id=attempt_id,
                    evaluation_round=evaluation_round,
                    result=result,
                )
                final_status = "paused"
                break

            frontier_before = frontier_manager.manifest.get("highest")
            repair_decision = repair_state_manager.decide(base_result, result)
            frontier = frontier_manager.observe(candidate, result, attempt_id)
            valid = self._is_valid_result(result)
            if bootstrap_attempt:
                if valid:
                    decision = "BASELINE_KEEP"
                    baseline_round, baseline_score = attempt_id, result.score
                    best_bundle, best_result, best_round = candidate, result, attempt_id
                    current = candidate
                    previous = result
                    self._write_bundle(self.state_dir / "baseline.json", candidate)
                    self._write_bundle(self.state_dir / "best.json", candidate)
                elif repair_decision.accept_candidate:
                    decision = (
                        "PARTIAL_KEEP"
                        if repair_decision.outcome != "INITIAL_KEEP"
                        else "FAIL"
                    )
                    current = candidate
                    previous = result
                    restore_bundle(self.task_dir, candidate)
                else:
                    decision = "FAIL"
                    current = base_bundle
                    previous = base_result
                    restore_bundle(self.task_dir, current or FileBundle(files={}))
            elif valid and self._is_better(result, best_result):
                decision = "KEEP"
                best_bundle, best_result, best_round = candidate, result, attempt_id
                current = candidate
                previous = result
                self._write_bundle(self.state_dir / "best.json", candidate)
            elif valid:
                decision = "DISCARD"
                if best_bundle:
                    restore_bundle(self.task_dir, best_bundle)
                    current = best_bundle
                    previous = best_result
            else:
                decision = "FAIL"
                if best_bundle:
                    restore_bundle(self.task_dir, best_bundle)
                    current = best_bundle
                    previous = best_result

            accepted_candidate = (
                bootstrap_attempt and (valid or repair_decision.accept_candidate)
            ) or (not bootstrap_attempt and decision == "KEEP")
            if result.compiled and accepted_candidate and (
                interface_contract is None or allow_interface_change
            ):
                save_interface_contract(
                    interface_path, capture_interface_contract(candidate)
                )

            attempt_record = repair_state_manager.build_attempt(
                attempt_id=attempt_id,
                evaluation_round=evaluation_round,
                phase=budget_phase,
                base_bundle=base_bundle,
                candidate=candidate,
                result=result,
                active_item=active_item,
                selection=selection,
                decision=repair_decision,
                frontier_before=frontier_before,
                frontier_after=frontier.highest,
                outcome_override=(
                    None if bootstrap_attempt else decision
                ),
            )
            logger.save_attempt_record(attempt_record)
            if bootstrap_attempt:
                repair_state_manager.observe(
                    attempt=attempt_record,
                    candidate_path=candidate_path,
                    decision=repair_decision,
                )

            addressed_failure = (
                base_result.structured_failure
                if base_result and base_result.error
                else None
            )
            incident = build_incident(
                attempt_id=attempt_id,
                failure=addressed_failure or result.structured_failure,
                hypothesis=dict(active_item) if active_item else None,
                before=base_bundle,
                after=candidate,
                resolved_fact_ids=self._resolved_fact_ids(selection),
                result=result,
                base_attempt_id=attempt_record.base_attempt_id,
                error_ids_before=attempt_record.error_ids_before,
                error_ids_after=attempt_record.error_ids_after,
                cleared_error_ids=attempt_record.cleared_error_ids,
            )
            confirmed_experience = persist_incident(self.state_dir, incident)

            fingerprint = None
            if result.error:
                fingerprint = diagnostic_fingerprint(read_result_log(result))
            if active_item is not None:
                active_item["status"] = "SETTLED"
                active_item["decision"] = decision
                active_item["attempt_id"] = attempt_id
                self._save_plan_progress(plan)

            logger.save_round(
                self._round_record(
                    attempt_id=attempt_id,
                    evaluation_round=evaluation_round,
                    budget_phase=budget_phase,
                    budget_round=budget_round,
                    decision=decision,
                    result=result,
                    selection=selection,
                    response=response,
                    generation_attempts=generation_attempts,
                    candidate_path=candidate_path,
                    plan_item=dict(active_item) if active_item else None,
                    fingerprint=fingerprint,
                    frontier={
                        "reached": frontier.reached,
                        "advanced": frontier.advanced,
                        "highest": frontier.highest,
                    },
                    incident_id=incident.incident_id,
                    confirmed_experience_id=(
                        confirmed_experience.experience_id if confirmed_experience else None
                    ),
                    stage_observation=stage_observation,
                    stage_timings=stage_timings,
                    repair_attempt={
                        "base_attempt_id": attempt_record.base_attempt_id,
                        "outcome": attempt_record.outcome,
                        "target_error_ids": attempt_record.target_error_ids,
                        "cleared_error_ids": attempt_record.cleared_error_ids,
                        "new_error_ids": attempt_record.new_error_ids,
                        "reintroduced_error_ids": attempt_record.reintroduced_error_ids,
                        "progress": repair_decision.progress,
                    },
                ),
                best_round=best_round,
            )
            logger.complete_pending()
            workflow = logger.data.setdefault("workflow", {})
            consecutive = int(workflow.get("consecutive_failures", 0)) + 1 if decision == "FAIL" else 0
            escalation = repair_state_manager.escalation_for(previous)
            bootstrap_count, optimization_count = self._budget_counts(logger)
            logger.update_workflow(
                phase="PLAN" if decision == "BASELINE_KEEP" else "EDIT",
                baseline_round=baseline_round,
                baseline_score=baseline_score,
                bootstrap_attempts_completed=bootstrap_count,
                optimization_rounds_completed=optimization_count,
                consecutive_failures=consecutive,
                active_plan_item=None,
                diagnose_pending=(
                    consecutive >= self.CONSECUTIVE_FAIL_THRESHOLD
                    or bool(escalation.get("diagnose_required"))
                ),
                no_op_streak=0,
            )
            workflow = logger.data["workflow"]
            self._report_result(evaluation_round, decision, result)
            if decision == "BASELINE_KEEP":
                plan = None
                try:
                    (self.state_dir / "plan.json").unlink()
                except FileNotFoundError:
                    pass
            elif self._active_plan_item(plan) is None:
                plan = None

        if best_bundle is not None:
            restore_bundle(self.task_dir, best_bundle)
        return self._finish(
            logger=logger,
            final_status=final_status,
            stop_failure=stop_failure,
            stop_attempt_id=stop_attempt_id,
            baseline_round=baseline_round,
            baseline_score=baseline_score,
            best_bundle=best_bundle,
            best_result=best_result,
            best_round=best_round,
            knowledge_version=knowledge_version,
            stop_reason=stop_reason,
        )

    def _finish(
        self,
        *,
        logger: TrajectoryLogger,
        final_status: str,
        stop_failure: EvalResult | None,
        stop_attempt_id: int | None,
        baseline_round: int | None,
        baseline_score: float | None,
        best_bundle: FileBundle | None,
        best_result: EvalResult | None,
        best_round: int | None,
        knowledge_version: Any,
        stop_reason: str | None = None,
    ) -> dict[str, Any]:
        bootstrap_count, optimization_count = self._budget_counts(logger)
        counted = [
            item
            for item in logger.data.get("rounds", [])
            if isinstance(item, dict) and logger.counts_toward_budget(item)
        ]
        last_record = counted[-1] if counted else None
        last_eval = last_record.get("evaluation", {}) if last_record else {}
        last_round = None
        if last_record:
            last_round = {
                "round": last_record.get("evaluation_round"),
                "attempt_id": last_record.get("attempt_id", last_record.get("round")),
                "phase": last_record.get("budget_phase"),
                "decision": last_record.get("decision"),
                "compiled": last_eval.get("compiled", False),
                "correctness": last_eval.get("correctness", False),
                "score": last_eval.get("score"),
            }
        last_evaluation_failure = (
            self._failure_summary(int(last_record.get("round")), last_eval) if last_record else None
        )
        orchestration_failure = (
            self._failure_summary(int(stop_attempt_id or 0), stop_failure.to_dict())
            if stop_failure is not None
            else None
        )
        success = baseline_round is not None and best_bundle is not None
        completed = final_status == "completed"
        if stop_reason is None:
            stop_reason = (
                stop_failure.failure_code
                if final_status == "paused" and stop_failure is not None
                else final_status
            )
        pending_phase = None
        if final_status == "blocked":
            pending_phase = (
                "DIAGNOSE"
                if stop_reason == "diagnosis_evidence_insufficient"
                else "BOOTSTRAP"
            )
        elif final_status == "paused":
            pending_phase = logger.pending_state().get("pending_phase")
        phase = "FINISH" if completed else final_status.upper()
        logger.update_workflow(
            phase=phase,
            baseline_round=baseline_round,
            baseline_score=baseline_score,
            bootstrap_attempts_completed=bootstrap_count,
            optimization_rounds_completed=optimization_count,
        )
        totals = logger.token_totals()
        totals_by_type = logger.token_totals_by_call_type()
        normalized_metrics = logger.normalized_metrics()
        repair_state = RepairStateManager(self.state_dir).state
        accepted_attempt_id = repair_state.get("accepted_attempt_id")
        rejected_outcomes = {"NO_PROGRESS", "REGRESSION", "UNCLASSIFIED", "INITIAL_REJECT"}
        discarded_attempts = [
            item.get("attempt_id", item.get("round"))
            for item in counted
            if item.get("repair_attempt", {}).get("outcome") in rejected_outcomes
        ]
        summary = {
            "success": success,
            "status": final_status,
            "completed": completed,
            "phase": phase,
            "pending_phase": pending_phase,
            "rounds_completed": len(counted),
            "evaluations_completed": len(counted),
            "bootstrap_attempts_completed": bootstrap_count,
            "bootstrap_attempts_limit": self.config.max_bootstrap_rounds,
            "optimization_rounds_completed": optimization_count,
            "optimization_rounds_limit": self.config.max_rounds,
            "total_rounds_limit": self.config.max_total_rounds,
            "stop_reason": stop_reason,
            "historical_records": logger.historical_records,
            "attempts_completed": logger.last_attempt_id,
            "accepted_attempt_id": accepted_attempt_id,
            "latest_attempt_id": logger.last_attempt_id,
            "accepted_candidate": repair_state.get("accepted_bundle"),
            "latest_candidate": last_record.get("candidate") if last_record else None,
            "discarded_attempts": discarded_attempts,
            "orchestration_failures": logger.orchestration_attempts,
            "pending_round": None if completed else len(counted) + 1,
            "baseline_round": baseline_round,
            "baseline_score": baseline_score,
            "best_round": best_round,
            "best_score": best_result.score if best_result else None,
            "last_round": last_round,
            "last_evaluation": last_round,
            "last_evaluation_failure": last_evaluation_failure,
            "last_orchestration_failure": orchestration_failure,
            "failure": orchestration_failure or last_evaluation_failure,
            "blocked_detail": (
                logger.data.get("workflow", {}).get("diagnosis_block")
                if stop_reason == "diagnosis_evidence_insufficient"
                else None
            ),
            "token_usage": totals,
            "token_usage_by_call_type": totals_by_type,
            "stage_normalized_metrics": normalized_metrics,
            "knowledge_source": {
                "mode": self.config.knowledge_source,
                "input_mode": self.config.knowledge_input_mode,
                "structured_prompt_enabled": self.config.uses_structured_prompt,
                "skills_enabled": self.config.uses_skills,
            },
            "knowledge": knowledge_version.to_dict(),
            "task_dir": str(self.task_dir),
        }
        (self.state_dir / "token_usage.json").write_text(
            json.dumps({"total": totals, "by_call_type": totals_by_type}, indent=2), encoding="utf-8"
        )
        (self.state_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if completed:
            logger.mark_done(success)
        elif final_status == "blocked":
            logger.mark_blocked(success)
        else:
            logger.mark_paused(success)
        self.progress.emit(
            f"Run {final_status}: success={success}, bootstrap={bootstrap_count}/"
            f"{self.config.max_bootstrap_rounds}, optimization={optimization_count}/{self.config.max_rounds}, "
            f"total={len(counted)}/{self.config.max_total_rounds or 'unlimited'}, "
            f"stop_reason={stop_reason}, best_round={best_round}, pending_phase={pending_phase}"
        )
        return summary
