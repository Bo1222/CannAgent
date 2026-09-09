from __future__ import annotations

import inspect
import json
import math
import shutil
from pathlib import Path
from typing import Any

from .bundle import capture_bundle, parse_file_bundle, restore_bundle, validate_initial_bundle
from .diagnostics import (
    diagnostic_fingerprint,
    extract_api_symbols,
    parse_structured_failure,
    read_result_log,
)
from .evaluator import Evaluator, extract_error_excerpt
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
from .llm import LLMProvider
from .knowledge_v2 import (
    FrontierManager,
    KnowledgeContext,
    KnowledgeRouterV2,
    SnapshotView,
    build_incident,
    locate_snapshot,
    persist_incident,
    render_bundle,
)
from .logging import TrajectoryLogger
from .models import EvalResult, FileBundle, LLMCallConfig, LLMResponse, RunConfig
from .progress import ProgressReporter, token_detail
from .prompts import build_plan_prompt, build_prompt, parse_plan
from .runtime_knowledge import collect_runtime_facts


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
    ) -> LLMResponse:
        call_config = self.config.call_config(call_type)
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

## Retry after truncated output
Return exactly one compact JSON object from scratch. Include only the smallest complete
file delta and no Markdown. The response must fit within the output-token limit.
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
    ) -> dict[str, Any]:
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
            "knowledge": selection.to_dict(),
            "candidate": candidate_path,
            "plan_item": plan_item,
            "failure_fingerprint": fingerprint,
            "structured_failure": result.structured_failure,
            "frontier": frontier,
            "incident_id": incident_id,
            "confirmed_experience_id": confirmed_experience_id,
        }

    @staticmethod
    def _resolved_fact_ids(selection: Any) -> list[str]:
        facts = getattr(selection, "relevant_facts", [])
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
            "",
            plan.get("diagnosis", ""),
            "",
        ]
        for item in plan["items"]:
            checked = "x" if item.get("status") == "SETTLED" else " "
            outcome = f" [{item['decision']}]" if item.get("decision") else ""
            lines.extend(
                [
                    f"- [{checked}] {item['id']}: {item['change']}{outcome}",
                    f"  - Hypothesis: {item['hypothesis']}",
                    f"  - Expected: {item['expected_signal']}",
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
        )
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        response = self._call_llm(
            prompt=prompt,
            call_type=call_type,
            label=f"{call_type.upper()} v{version}",
            logger=logger,
            attempt_id=attempt_id,
            evaluation_round=evaluation_round,
            response_path=plan_dir / "response.txt",
        )
        parsed = parse_plan(
            response.content,
            min_items=1 if initial else 3,
            max_items=1 if initial else 5,
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
        logger.update_workflow(
            phase="EDIT",
            plan_version=version,
            active_plan_item=plan["items"][0]["id"],
            consecutive_failures=0 if diagnosis_required else logger.data.get("workflow", {}).get("consecutive_failures", 0),
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
    ) -> tuple[Any, str]:
        evidence_parts = [reference, cases]
        if current:
            evidence_parts.extend(current.files.values())
        if previous:
            evidence_parts.append(read_result_log(previous))
        evidence = "\n".join(evidence_parts)
        if self.config.knowledge_mode == "semantic":
            snapshot_path = locate_snapshot(
                Path(self.config.knowledge_store),
                knowledge_version.knowledge_version,
                self.config.knowledge_snapshot,
            )
            context = KnowledgeContext(
                operator=Path(self.config.op_file).stem,
                phase="diagnose" if previous and previous.error else "plan_generate",
                runtime_version=knowledge_version.runtime_version,
                knowledge_version=knowledge_version.knowledge_version,
                soc=self.config.soc_version,
                source_symbols=extract_api_symbols(evidence),
                failure=previous.structured_failure if previous and previous.error else None,
                active_plan=active_plan,
            )
            bundle = KnowledgeRouterV2(SnapshotView(snapshot_path)).route(context)
            payload = bundle.to_dict()
            (round_dir / "knowledge_bundle.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (round_dir / "retrieval_trace.json").write_text(
                json.dumps(payload["retrieval_trace"], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            rendered = render_bundle(bundle, max_chars=max_knowledge_chars)
            (round_dir / "references.md").write_text(rendered, encoding="utf-8")
            (round_dir / "selected_knowledge.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            return bundle, rendered
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
        prepared_knowledge: tuple[Any, str] | None = None,
    ) -> tuple[FileBundle, FileBundle | None, Any, LLMResponse, int, str]:
        round_dir = self.state_dir / f"round_{attempt_id:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = f"round_{attempt_id:02d}/candidate.json"
        checkpoint = round_dir / "candidate.json"
        base_bundle = FileBundle(files=dict(current.files)) if current else None
        if logger.pending_state().get("pending_phase") == "EVAL" and checkpoint.is_file():
            candidate = self._read_bundle(checkpoint)
            restore_bundle(self.task_dir, candidate)
            if self.config.knowledge_mode == "semantic":
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
                )
            else:
                selection = selection_from_state(
                    knowledge_state,
                    version=knowledge_version,
                    mode="resume_eval",
                )
            response = LLMResponse(content="", model="checkpoint", usage={})
            return candidate, base_bundle, selection, response, 1, candidate_path

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
            )
        else:
            selection, knowledge_context = prepared_knowledge
        prompt = build_prompt(
            reference_code=reference,
            cases_text=cases,
            current=current,
            previous_result=previous,
            round_num=attempt_id,
            knowledge_context=knowledge_context,
            phase=budget_phase.upper(),
            plan_item=active_item,
            failure_fingerprints=knowledge_state.failure_fingerprints,
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
        self._migrate_trajectory(logger)
        logger.mark_running()
        logger.save_invocation(self.config.to_dict())
        if self.config.deprecated_repair_options_active:
            self.progress.emit("WARNING: --repair-* options are deprecated and no longer trigger repair calls")
        self.progress.emit(
            "LLM configuration: "
            f"provider={self.config.provider}, requested_model={self.config.model}, "
            f"bootstrap={self.config.max_bootstrap_rounds}, optimization={self.config.max_rounds}, "
            f"total={self.config.max_total_rounds or 'unlimited'}, "
            f"generator={self.config.generator_max_tokens}/{self.config.generator_thinking}/"
            f"{self.config.generator_reasoning_effort}, "
            f"planner={self.config.planner_max_tokens}/{self.config.planner_thinking}/"
            f"{self.config.planner_reasoning_effort}"
        )
        if self.config.legacy_max_tokens_active:
            self.progress.emit("WARNING: legacy --max-tokens blanket override is active")

        reference, cases = self._inputs()
        current, best_bundle, previous, best_result, best_round = self._resume_state(logger)
        frontier_manager = FrontierManager(self.state_dir)
        if frontier_manager.highest_bundle() is None and best_bundle is not None and best_result is not None:
            frontier_manager.observe(best_bundle, best_result, int(best_round or 0))
        workflow = logger.data.setdefault("workflow", {})
        baseline_round = workflow.get("baseline_round")
        baseline_score = workflow.get("baseline_score")
        if baseline_round is None and self._is_valid_result(best_result):
            baseline_round, baseline_score = best_round, best_result.score
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
        legacy_selection = next(
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
            legacy_selection=legacy_selection,
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
                    _, planning_knowledge = prepared_knowledge
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
                    workflow["diagnose_pending"] = False
                    logger.update_workflow(**workflow)
                    active_item = self._active_plan_item(plan)
                    logger.update_pending_phase("EDIT")
            if active_item is not None:
                logger.update_workflow(phase="EDIT", active_plan_item=active_item["id"])
                workflow = logger.data["workflow"]

            round_dir = self.state_dir / f"round_{attempt_id:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
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
                    prepared_knowledge=prepared_knowledge,
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

            frontier = frontier_manager.observe(candidate, result, attempt_id)
            valid = self._is_valid_result(result)
            if baseline_round is None:
                if valid:
                    decision = "BASELINE_KEEP"
                    baseline_round, baseline_score = attempt_id, result.score
                    best_bundle, best_result, best_round = candidate, result, attempt_id
                    current = candidate
                    self._write_bundle(self.state_dir / "baseline.json", candidate)
                    self._write_bundle(self.state_dir / "best.json", candidate)
                else:
                    decision = "FAIL"
                    current = frontier.rollback
                    restore_bundle(self.task_dir, current or FileBundle(files={}))
            elif valid and self._is_better(result, best_result):
                decision = "KEEP"
                best_bundle, best_result, best_round = candidate, result, attempt_id
                current = candidate
                self._write_bundle(self.state_dir / "best.json", candidate)
            elif valid:
                decision = "DISCARD"
                if best_bundle:
                    restore_bundle(self.task_dir, best_bundle)
                    current = best_bundle
            else:
                decision = "FAIL"
                if best_bundle:
                    restore_bundle(self.task_dir, best_bundle)
                    current = best_bundle

            addressed_failure = previous.structured_failure if previous and previous.error else None
            incident = build_incident(
                attempt_id=attempt_id,
                failure=addressed_failure or result.structured_failure,
                hypothesis=dict(active_item) if active_item else None,
                before=base_bundle,
                after=candidate,
                resolved_fact_ids=self._resolved_fact_ids(selection),
                result=result,
            )
            confirmed_experience = persist_incident(self.state_dir, incident)

            fingerprint = None
            if result.error:
                fingerprint = diagnostic_fingerprint(read_result_log(result))
                knowledge_state.failure_fingerprints = [
                    *knowledge_state.failure_fingerprints,
                    fingerprint,
                ][-16:]
                save_knowledge_state(knowledge_state_path, knowledge_state)
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
                ),
                best_round=best_round,
            )
            logger.complete_pending()
            previous = result
            workflow = logger.data.setdefault("workflow", {})
            consecutive = int(workflow.get("consecutive_failures", 0)) + 1 if decision == "FAIL" else 0
            bootstrap_count, optimization_count = self._budget_counts(logger)
            logger.update_workflow(
                phase="PLAN" if decision == "BASELINE_KEEP" else "EDIT",
                baseline_round=baseline_round,
                baseline_score=baseline_score,
                bootstrap_attempts_completed=bootstrap_count,
                optimization_rounds_completed=optimization_count,
                consecutive_failures=consecutive,
                active_plan_item=None,
                diagnose_pending=consecutive >= self.CONSECUTIVE_FAIL_THRESHOLD,
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
            pending_phase = "BOOTSTRAP"
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
            "token_usage": totals,
            "token_usage_by_call_type": totals_by_type,
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
