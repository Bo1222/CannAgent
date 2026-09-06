from __future__ import annotations

import inspect
import json
import shutil
from pathlib import Path
from typing import Any

from .bundle import capture_bundle, parse_file_bundle, restore_bundle, validate_initial_bundle
from .diagnostics import diagnostic_fingerprint, diagnostic_lines, extract_api_symbols, read_result_log
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
from .logging import TrajectoryLogger
from .models import EvalResult, FileBundle, LLMCallConfig, LLMResponse, RunConfig
from .progress import ProgressReporter, token_detail
from .prompts import build_compile_repair_prompt, build_prompt
from .runtime_knowledge import collect_runtime_facts


class LLMCallFailure(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class MultiTurnRunner:
    """Explicit generate/evaluate/select loop with no dependency on an agent runtime."""

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
            raise ValueError(f"output directory is not empty: {self.task_dir}; use --resume or choose another directory")
        self.task_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, self.task_dir / "model.py")
        cases = source.with_suffix(".json")
        if cases.is_file():
            shutil.copy2(cases, self.task_dir / cases.name)

    def _inputs(self) -> tuple[str, str]:
        model_path = self.task_dir / "model.py"
        reference = model_path.read_text(encoding="utf-8")
        configured_cases = Path(self.config.op_file).expanduser().resolve().with_suffix(".json").name
        cases_path = self.task_dir / configured_cases
        cases = cases_path.read_text(encoding="utf-8") if cases_path.is_file() else "(no separate JSON case file found)"
        return reference, cases

    @staticmethod
    def _write_bundle(path: Path, bundle: FileBundle) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_bundle(path: Path) -> FileBundle:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FileBundle(files=payload["files"], delete=payload.get("delete", []), analysis=payload.get("analysis", ""))

    @staticmethod
    def _is_better(candidate: EvalResult, best: EvalResult | None) -> bool:
        if candidate.error or not candidate.correctness or candidate.score is None:
            return False
        if best is None:
            return True
        return best.score is None or candidate.score > best.score

    @staticmethod
    def _failure_result(
        *,
        stage: str,
        code: str,
        message: str,
        details_path: Path | None = None,
    ) -> EvalResult:
        return EvalResult(
            False,
            False,
            error=message,
            failure_stage=stage,
            failure_code=code,
            error_excerpt=message[:4000],
            details_path=str(details_path.resolve()) if details_path is not None else None,
        )

    @staticmethod
    def _write_error(path: Path, error: Exception | str) -> None:
        path.write_text(f"{error}\n", encoding="utf-8")

    def _provider_generate(self, prompt: str, call_config: LLMCallConfig) -> LLMResponse:
        """Pass call metadata when supported while retaining simple test providers."""
        parameters = inspect.signature(self.provider.generate).parameters
        supports_config = "call_config" in parameters or any(
            item.kind == inspect.Parameter.VAR_KEYWORD for item in parameters.values()
        )
        if supports_config:
            return self.provider.generate(prompt, call_config=call_config)
        return self.provider.generate(prompt)

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
            suffix = "" if retry == 0 else f" retry {retry}/{self.config.llm_transient_retries}"
            task = self.progress.start(f"{label}{suffix}")
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
                    self.progress.emit(
                        f"Evaluation {evaluation_round} · {call_type} transient failure; retrying"
                    )
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
            target.write_text(response.content, encoding="utf-8")
            if response.content.strip():
                details = response.usage.get("completion_tokens_details", {})
                reasoning_tokens = details.get("reasoning_tokens") if isinstance(details, dict) else None
                reasoning_detail = (
                    f", reasoning_tokens={reasoning_tokens}"
                    if isinstance(reasoning_tokens, int)
                    else f", reasoning_present={bool(response.reasoning_content)}"
                )
                task.finish(
                    status="completed",
                    detail=(
                        f"{token_detail(response.usage)}, served_model={response.model}, "
                        f"thinking={response.request_options.get('thinking_requested', call_config.thinking)}, "
                        f"effort={response.request_options.get('reasoning_effort_requested', call_config.reasoning_effort)}"
                        f"{reasoning_detail}"
                    ),
                )
                return response

            finish_reason = response.finish_reason or "unknown"
            reasoning_present = bool(response.reasoning_content.strip())
            last_error = (
                "LLM returned no final content "
                f"(finish_reason={finish_reason}, reasoning_content_present={reasoning_present})"
            )
            task.finish(
                status="failed",
                detail=(
                    f"{last_error}, {token_detail(response.usage)}, "
                    f"served_model={response.model}, "
                    f"thinking={response.request_options.get('thinking_requested', call_config.thinking)}, "
                    f"effort={response.request_options.get('reasoning_effort_requested', call_config.reasoning_effort)}"
                ),
            )
            if retry < self.config.llm_transient_retries:
                self.progress.emit(
                    f"Evaluation {evaluation_round} · {call_type} empty final content; retrying"
                )
                continue
            code = "llm_output_exhausted" if finish_reason == "length" else "llm_empty_content"
            raise LLMCallFailure(code, last_error)
        raise LLMCallFailure("llm_call_failed", last_error or "LLM call failed")

    def _normalize_evaluation(self, result: EvalResult, round_dir: Path) -> EvalResult:
        if result.error or (result.compiled and result.correctness and result.score is not None):
            return result
        if not result.compiled:
            stage, code, message = "ascendc_build", "evaluation_incomplete", "candidate was not compiled"
        elif not result.correctness:
            stage, code, message = "correctness", "evaluation_incomplete", "candidate correctness was not verified"
        else:
            stage, code, message = "performance", "performance_score_missing", "candidate has no valid performance score"
        error_path = round_dir / "evaluation_incomplete.log"
        self._write_error(error_path, message)
        result.error = message
        result.failure_stage = stage
        result.failure_code = code
        result.error_excerpt = message
        result.details_path = str(error_path.resolve())
        return result

    def _report_result(self, round_num: int, decision: str, result: EvalResult) -> None:
        score = f", score={result.score:.4f}" if result.score is not None else ""
        self.progress.emit(f"Round {round_num} · {decision} (compiled={result.compiled}, correctness={result.correctness}{score})")
        if not result.error:
            return
        self.progress.emit(
            f"Round {round_num} failure · stage={result.failure_stage or 'unknown'} "
            f"code={result.failure_code or 'unknown'} · {result.error}"
        )
        for line in result.error_excerpt.splitlines():
            self.progress.emit(f"  {line}")
        if result.details_path:
            self.progress.emit(f"Full log: {result.details_path}")

    def _failure_summary(self, round_num: int, evaluation: dict[str, Any]) -> dict[str, Any] | None:
        message = str(evaluation.get("error") or "")
        if not message:
            return None
        stage = evaluation.get("failure_stage")
        code = evaluation.get("failure_code")
        if not stage:
            legacy_errors = (
                ("static validation", "static_validation", "static_validation_failed"),
                (
                    "ascendc source validation",
                    "ascendc_source_validation",
                    "ascendc_source_validation_failed",
                ),
                ("ascendc build", "ascendc_build", "ascendc_build_failed"),
                ("correctness", "correctness", "correctness_failed"),
                ("performance", "performance", "performance_failed"),
                ("invalid llm file bundle", "response_format", "response_format_failed"),
            )
            lowered = message.lower()
            for marker, inferred_stage, inferred_code in legacy_errors:
                if marker in lowered:
                    stage, code = inferred_stage, inferred_code
                    break
        stage = stage or "unknown"
        code = code or "unknown"
        excerpt = evaluation.get("error_excerpt") or ""
        if not excerpt:
            diagnostic_output = (
                evaluation.get("verify_output")
                if stage == "correctness"
                else evaluation.get("compile_output")
            )
            excerpt = extract_error_excerpt(str(diagnostic_output or message))
        details_path = evaluation.get("details_path")
        if not details_path and stage != "unknown":
            filename = {
                "static_validation": "static_validation.log",
                "ascendc_source_validation": "source_validation.log",
                "ascendc_build": "build.log",
                "correctness": "correctness.log",
                "performance": "performance.log",
                "response_format": "response_format.log",
            }.get(stage)
            possible_path = self.state_dir / f"round_{round_num:02d}" / filename if filename else None
            if possible_path is not None and possible_path.is_file():
                details_path = str(possible_path.resolve())
        return {
            "round": round_num,
            "stage": stage,
            "code": code,
            "message": message,
            "excerpt": excerpt,
            "details_path": details_path,
        }

    @staticmethod
    def _round_record(
        *,
        round_num: int,
        evaluation_round: int | None = None,
        decision: str,
        result: EvalResult,
        selection: Any = None,
        response: Any = None,
        generation_attempts: int = 0,
        compile_repair_attempts: int = 0,
        evaluation_attempts: list[dict[str, Any]] | None = None,
        repair_error: str | None = None,
        candidate_path: str | None = None,
    ) -> dict[str, Any]:
        return {
            "round": round_num,
            "attempt_id": round_num,
            "evaluation_round": evaluation_round or round_num,
            "counts_toward_budget": True,
            "decision": decision,
            "evaluation": result.to_dict(),
            "response_model": response.model if response is not None else None,
            "usage": response.usage if response is not None else {},
            "latency_seconds": response.latency_seconds if response is not None else None,
            "generation_attempts": generation_attempts,
            "compile_repair_attempts": compile_repair_attempts,
            "evaluation_attempts": evaluation_attempts or [result.to_dict()],
            "repair_error": repair_error,
            "response_finish_reason": response.finish_reason if response is not None else None,
            "knowledge": selection.to_dict() if selection is not None else None,
            "candidate": candidate_path,
        }

    @staticmethod
    def _truncation_retry_prompt(prompt: str) -> str:
        return f"""{prompt}

## Retry after truncated output
The previous response reached the model output-token limit. Regenerate the answer
from scratch as exactly one compact JSON object. Do not use Markdown or prose
outside the JSON. Remove nonessential comments and explanations. When a current
implementation exists, return only the smallest complete file delta needed for
this round. Every file included must still contain its complete, syntactically
valid content. The response must fit within the configured output-token limit.
"""

    def _resume_state(self, logger: TrajectoryLogger) -> tuple[FileBundle | None, FileBundle | None, EvalResult | None, EvalResult | None, int | None]:
        working = capture_bundle(self.task_dir)
        current = working if working.files else None
        previous = None
        best_bundle = None
        best_result = None
        best_round = logger.data.get("best_round")
        counted_records = [
            record
            for record in logger.data.get("rounds", [])
            if isinstance(record, dict) and logger.counts_toward_budget(record)
        ]
        if counted_records:
            raw = counted_records[-1].get("evaluation")
            if raw:
                previous = EvalResult(**raw)
        best_path = self.state_dir / "best.json"
        if best_path.is_file():
            best_bundle = self._read_bundle(best_path)
            restore_bundle(self.task_dir, best_bundle)
            current = best_bundle
            if best_round:
                best_record = next(
                    (
                        record
                        for record in logger.data.get("rounds", [])
                        if record.get("round") == best_round
                    ),
                    None,
                )
                raw_best = best_record.get("evaluation") if best_record else None
                if raw_best:
                    best_result = EvalResult(**raw_best)
        return current, best_bundle, previous, best_result, best_round

    def run(self) -> dict:
        self.progress.emit("Preparing task and resume state")
        self._prepare_task()
        logger = TrajectoryLogger(self.state_dir, self.config.to_dict())
        logger.mark_running()
        logger.save_invocation(self.config.to_dict())
        self.progress.emit(
            "LLM configuration: "
            f"provider={self.config.provider}, requested_model={self.config.model}, "
            f"router={self.config.router_max_tokens}/{self.config.router_thinking}, "
            f"generator={self.config.generator_max_tokens}/{self.config.generator_thinking}/"
            f"{self.config.generator_reasoning_effort}, "
            f"repair={self.config.repair_max_tokens}/{self.config.repair_thinking}/"
            f"{self.config.repair_reasoning_effort}"
        )
        if self.config.legacy_max_tokens_active:
            self.progress.emit(
                "WARNING: legacy ASCENDC_LLM_MAX_TOKENS/--max-tokens blanket override is active; "
                "it replaces all per-call token budgets"
            )
        if self.config.provider == "deepseek" and self.config.model in {
            "deepseek-chat",
            "deepseek-reasoner",
        }:
            self.progress.emit(
                f"WARNING: {self.config.model} is a legacy alias; use deepseek-v4-flash explicitly"
            )
        reference, cases = self._inputs()
        current, best_bundle, previous, best_result, best_round = self._resume_state(logger)
        completed_before_run = logger.completed_rounds
        if self.config.resume:
            remaining = max(0, self.config.max_rounds - completed_before_run)
            self.progress.emit(
                f"Resuming task: historical_records={logger.historical_records}, "
                f"evaluated={completed_before_run}, target={self.config.max_rounds}, remaining={remaining}"
            )
        else:
            self.progress.emit(f"Starting task: target_rounds={self.config.max_rounds}")
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
                record.get("knowledge")
                for record in reversed(logger.data.get("rounds", []))
                if isinstance(record, dict) and record.get("knowledge")
            ),
            None,
        )
        knowledge_state_path = self.state_dir / "knowledge_state.json"
        knowledge_state = load_knowledge_state(
            knowledge_state_path,
            version=knowledge_version,
            legacy_selection=legacy_selection,
        )

        stop_failure: EvalResult | None = None
        stop_attempt_id: int | None = None
        for evaluation_round in range(logger.completed_rounds + 1, self.config.max_rounds + 1):
            round_num = logger.begin_pending(evaluation_round)
            stop_attempt_id = round_num
            self.progress.emit(
                f"Evaluation {evaluation_round}/{self.config.max_rounds} · attempt {round_num} · started"
            )
            round_dir = self.state_dir / f"round_{round_num:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            fallback_parts = [reference, cases]
            if current:
                fallback_parts.extend(current.files.values())
            if previous:
                fallback_parts.append(read_result_log(previous))
            evidence = "\n".join(fallback_parts)
            candidates = candidate_doc_ids(
                evidence,
                version=knowledge_version,
                exclude=knowledge_state.working_doc_ids,
                limit=5,
            )
            route_mode = "reuse"
            route_candidates: list[str] | None = []
            if not knowledge_state.initialized:
                route_mode = "initial_full"
                route_candidates = None
            elif candidates and len([item for item in candidates if item[1] == candidates[0][1]]) == 1 and candidates[0][1] >= 100:
                deterministic_ids = [candidates[0][0]]
                by_doc = {item[0] for item in candidates}
                selection = parse_knowledge_selection(
                    json.dumps(
                        {
                            "skill": "ascendc-translator",
                            "topics": ["deterministic-symbol-match"],
                            "doc_ids": [item for item in deterministic_ids if item in by_doc],
                            "supplements": [],
                            "reason": "Compiler/source symbols matched unique API documents deterministically.",
                            "mode": "deterministic_escape",
                        }
                    ),
                    version=knowledge_version,
                    max_api_docs=2,
                    fallback_text=evidence,
                )
                selection.mode = "deterministic_escape"
                apply_selection_to_state(
                    knowledge_state,
                    selection,
                    max_docs=max_api_docs,
                    preferred_doc_ids=deterministic_ids,
                )
                route_mode = "deterministic_escape"
            elif candidates and candidates[0][1] >= 40:
                route_mode = "incremental"
                route_candidates = [doc_id for doc_id, _score in candidates]

            knowledge_response = None
            if route_mode in {"initial_full", "incremental"}:
                knowledge_prompt = build_knowledge_prompt(
                    reference_code=reference,
                    current=current,
                    previous_result=previous,
                    version=knowledge_version,
                    mode=route_mode,
                    candidate_doc_ids=route_candidates,
                    working_doc_ids=knowledge_state.working_doc_ids,
                )
                (round_dir / "knowledge_prompt.txt").write_text(knowledge_prompt, encoding="utf-8")
                try:
                    knowledge_response = self._call_llm(
                        prompt=knowledge_prompt,
                        call_type="knowledge_router",
                        label=(
                            f"Evaluation {evaluation_round}/{self.config.max_rounds} · "
                            f"knowledge router LLM ({route_mode})"
                        ),
                        logger=logger,
                        attempt_id=round_num,
                        evaluation_round=evaluation_round,
                        response_path=round_dir / "knowledge_response.txt",
                    )
                except LLMCallFailure as error:
                    error_path = round_dir / "knowledge_router_error.log"
                    self._write_error(error_path, error)
                    result = self._failure_result(
                        stage="llm_knowledge_router",
                        code=error.code,
                        message=f"knowledge router LLM call failed: {error}",
                        details_path=error_path,
                    )
                    stop_failure = result
                    logger.save_orchestration_attempt(
                        {
                            "attempt_id": round_num,
                            "evaluation_round": evaluation_round,
                            "stage": result.failure_stage,
                            "code": result.failure_code,
                            "error": result.error,
                            "details_path": result.details_path,
                            "counts_toward_budget": False,
                        }
                    )
                    self._report_result(evaluation_round, "PAUSED", result)
                    break
                if route_mode == "initial_full":
                    knowledge_state.full_route_count += 1
                else:
                    knowledge_state.incremental_route_count += 1
            else:
                self.progress.emit(
                    f"Evaluation {evaluation_round}/{self.config.max_rounds} · knowledge {route_mode} (no router LLM call)"
                )
                (round_dir / "knowledge_reuse.json").write_text(
                    json.dumps(
                        {"mode": route_mode, "working_doc_ids": knowledge_state.working_doc_ids},
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            task = self.progress.start(
                f"Evaluation {evaluation_round}/{self.config.max_rounds} · local knowledge selection"
            )
            try:
                if knowledge_response is not None:
                    selection = parse_knowledge_selection(
                        knowledge_response.content,
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
                elif route_mode == "reuse":
                    selection = selection_from_state(
                        knowledge_state,
                        version=knowledge_version,
                        mode=route_mode,
                    )
                added_doc_ids = list(selection.added_doc_ids)
                removed_doc_ids = list(selection.removed_doc_ids)
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
                selection.added_doc_ids = added_doc_ids
                selection.removed_doc_ids = removed_doc_ids
                diagnostic_evidence = read_result_log(previous) if previous else ""
                current_sources = list(current.files.values()) if current else []
                symbols = extract_api_symbols(
                    diagnostic_evidence,
                    *current_sources,
                    reference,
                    cases,
                )
                runtime_facts = collect_runtime_facts(
                    symbols,
                    runtime_version=knowledge_version.runtime_version,
                    cache_path=round_dir / "runtime_header_facts.json",
                )
                selection.runtime_fact_symbols = runtime_facts.symbols
                selection.conflicts = runtime_facts.conflicts
                knowledge_context = render_knowledge(
                    selection,
                    version=knowledge_version,
                    max_chars=max_knowledge_chars,
                    runtime_facts=runtime_facts.text,
                    conflicts=runtime_facts.conflicts,
                )
                knowledge_state.known_symbols = list(
                    dict.fromkeys([*knowledge_state.known_symbols, *symbols])
                )[:128]
                knowledge_state.last_round = round_num
                save_knowledge_state(knowledge_state_path, knowledge_state)
            except Exception as error:
                task.finish(status="failed", detail=type(error).__name__)
                error_path = round_dir / "knowledge_selection_error.log"
                self._write_error(error_path, error)
                result = self._failure_result(
                    stage="knowledge_selection",
                    code="knowledge_selection_failed",
                    message=f"local knowledge selection failed: {error}",
                    details_path=error_path,
                )
                stop_failure = result
                logger.save_orchestration_attempt(
                    {
                        "attempt_id": round_num,
                        "evaluation_round": evaluation_round,
                        "stage": result.failure_stage,
                        "code": result.failure_code,
                        "error": result.error,
                        "details_path": result.details_path,
                        "counts_toward_budget": False,
                    }
                )
                self._report_result(evaluation_round, "PAUSED", result)
                break
            task.finish(
                status="completed",
                detail=f"mode={route_mode}, working={len(knowledge_state.working_doc_ids)}, rendered={len(selection.rendered_doc_ids)}",
            )
            (round_dir / "selected_knowledge.json").write_text(
                json.dumps(selection.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (round_dir / "references.md").write_text(knowledge_context, encoding="utf-8")
            prompt = build_prompt(
                reference_code=reference,
                cases_text=cases,
                current=current,
                previous_result=previous,
                round_num=round_num,
                knowledge_context=knowledge_context,
            )
            (round_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            try:
                response = self._call_llm(
                    prompt=prompt,
                    call_type="generator",
                    label=f"Evaluation {evaluation_round}/{self.config.max_rounds} · generator LLM",
                    logger=logger,
                    attempt_id=round_num,
                    evaluation_round=evaluation_round,
                    response_path=round_dir / "response.txt",
                )
            except LLMCallFailure as error:
                error_path = round_dir / "generator_error.log"
                self._write_error(error_path, error)
                result = self._failure_result(
                    stage="llm_generator",
                    code=error.code,
                    message=f"generator LLM call failed: {error}",
                    details_path=error_path,
                )
                stop_failure = result
                logger.save_orchestration_attempt(
                    {
                        "attempt_id": round_num,
                        "evaluation_round": evaluation_round,
                        "stage": result.failure_stage,
                        "code": result.failure_code,
                        "error": result.error,
                        "details_path": result.details_path,
                        "counts_toward_budget": False,
                    }
                )
                self._report_result(evaluation_round, "PAUSED", result)
                break
            generation_attempts = 1
            truncation_seen = response.finish_reason == "length"
            if truncation_seen:
                retry_prompt = self._truncation_retry_prompt(prompt)
                (round_dir / "retry_prompt.txt").write_text(retry_prompt, encoding="utf-8")
                generation_attempts += 1
                try:
                    response = self._call_llm(
                        prompt=retry_prompt,
                        call_type="generator_retry",
                        label=(
                            f"Evaluation {evaluation_round}/{self.config.max_rounds} · "
                            "generator truncated-output retry"
                        ),
                        logger=logger,
                        attempt_id=round_num,
                        evaluation_round=evaluation_round,
                        response_path=round_dir / "response_retry_01.txt",
                    )
                except LLMCallFailure as error:
                    error_path = round_dir / "generator_retry_error.log"
                    self._write_error(error_path, error)
                    result = self._failure_result(
                        stage="llm_generator",
                        code=error.code,
                        message=f"generator LLM retry failed: {error}",
                        details_path=error_path,
                    )
                    stop_failure = result
                    logger.save_orchestration_attempt(
                        {
                            "attempt_id": round_num,
                            "evaluation_round": evaluation_round,
                            "stage": result.failure_stage,
                            "code": result.failure_code,
                            "error": result.error,
                            "details_path": result.details_path,
                            "counts_toward_budget": False,
                        }
                    )
                    self._report_result(evaluation_round, "PAUSED", result)
                    break
                truncation_seen = truncation_seen or response.finish_reason == "length"

            task = self.progress.start(
                f"Evaluation {evaluation_round}/{self.config.max_rounds} · parse and apply candidate"
            )
            failure_stage = "response_format"
            evaluation_attempts: list[dict[str, Any]] = []
            compile_repair_attempts = 0
            repair_error = None
            try:
                delta = parse_file_bundle(response.content)
                failure_stage = "bundle_validation"
                candidate = FileBundle(files=dict(current.files) if current else {})
                for path in delta.delete:
                    candidate.files.pop(path, None)
                candidate.files.update(delta.files)
                validate_initial_bundle(candidate)
                restore_bundle(self.task_dir, candidate)
                candidate = capture_bundle(self.task_dir)
                self._write_bundle(round_dir / "candidate.json", candidate)
                task.finish(status="completed", detail=f"files={len(candidate.files)}")
            except (ValueError, json.JSONDecodeError) as error:
                task.finish(status="failed", detail=type(error).__name__)
                if failure_stage == "response_format":
                    detail = f"invalid LLM file bundle: {error}"
                    code = "response_format_failed"
                else:
                    detail = f"invalid candidate bundle: {error}"
                    code = "bundle_validation_failed"
                if truncation_seen:
                    detail += "; response reached the output-token limit and one same-round retry was attempted"
                error_path = round_dir / f"{failure_stage}.log"
                self._write_error(error_path, detail)
                result = self._failure_result(
                    stage=failure_stage,
                    code=code,
                    message=detail,
                    details_path=error_path,
                )
                stop_failure = result
                logger.save_orchestration_attempt(
                    {
                        "attempt_id": round_num,
                        "evaluation_round": evaluation_round,
                        "stage": result.failure_stage,
                        "code": result.failure_code,
                        "error": result.error,
                        "details_path": result.details_path,
                        "counts_toward_budget": False,
                    }
                )
                self._report_result(evaluation_round, "PAUSED", result)
                break
            else:
                try:
                    result = self.evaluator.evaluate(self.task_dir, round_dir)
                except Exception as error:
                    error_path = round_dir / "evaluation_error.log"
                    self._write_error(error_path, error)
                    result = self._failure_result(
                        stage="evaluation",
                        code="evaluator_exception",
                        message=f"local evaluator failed unexpectedly: {error}",
                        details_path=error_path,
                    )
                result = self._normalize_evaluation(result, round_dir)
                evaluation_attempts.append(result.to_dict())

                if result.failure_stage in {"ascendc_build", "ascendc_source_validation"}:
                    original_result = result
                    compile_repair_attempts = 1
                    self.progress.emit(
                        f"Evaluation {evaluation_round} · {result.error}; "
                        "starting one targeted same-round repair"
                    )
                    for line in result.error_excerpt.splitlines():
                        self.progress.emit(f"  {line}")
                    if result.details_path:
                        self.progress.emit(f"Full compiler log: {result.details_path}")
                    repair_dir = round_dir / "repair_01"
                    repair_dir.mkdir(parents=True, exist_ok=True)
                    compiler_output = read_result_log(result)
                    repair_symbols = extract_api_symbols(compiler_output, *candidate.files.values())
                    repair_candidates = candidate_doc_ids(
                        compiler_output,
                        version=knowledge_version,
                        exclude=knowledge_state.working_doc_ids,
                        limit=5,
                    )
                    repair_doc_ids = [doc_id for doc_id, _score in repair_candidates[:2]]
                    if repair_doc_ids:
                        repair_selection_delta = parse_knowledge_selection(
                            json.dumps(
                                {
                                    "skill": "ascendc-translator",
                                    "topics": ["compiler-repair"],
                                    "doc_ids": repair_doc_ids,
                                    "supplements": [],
                                    "reason": "Selected from concrete compiler API diagnostics.",
                                    "mode": "compiler_escape",
                                }
                            ),
                            version=knowledge_version,
                            max_api_docs=2,
                            fallback_text=compiler_output,
                        )
                        apply_selection_to_state(
                            knowledge_state,
                            repair_selection_delta,
                            max_docs=max_api_docs,
                            preferred_doc_ids=repair_doc_ids,
                        )
                    repair_active_ids = active_doc_ids(
                        knowledge_state,
                        compiler_output,
                        version=knowledge_version,
                        preferred_doc_ids=repair_doc_ids,
                        limit=min(5, max_api_docs),
                    )
                    repair_selection = selection_from_state(
                        knowledge_state,
                        version=knowledge_version,
                        mode="compiler_repair",
                        active_ids=repair_active_ids,
                        reason="Targeted same-round repair from real compiler diagnostics.",
                    )
                    repair_facts = collect_runtime_facts(
                        repair_symbols,
                        runtime_version=knowledge_version.runtime_version,
                        cache_path=repair_dir / "runtime_header_facts.json",
                    )
                    repair_selection.runtime_fact_symbols = repair_facts.symbols
                    repair_selection.conflicts = repair_facts.conflicts
                    repair_knowledge = render_knowledge(
                        repair_selection,
                        version=knowledge_version,
                        max_chars=max_knowledge_chars,
                        runtime_facts=repair_facts.text,
                        conflicts=repair_facts.conflicts,
                    )
                    (repair_dir / "selected_knowledge.json").write_text(
                        json.dumps(repair_selection.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    (repair_dir / "references.md").write_text(repair_knowledge, encoding="utf-8")
                    repair_prompt = build_compile_repair_prompt(
                        reference_code=reference,
                        cases_text=cases,
                        candidate=candidate,
                        result=result,
                        round_num=round_num,
                        knowledge_context=repair_knowledge,
                    )
                    (repair_dir / "prompt.txt").write_text(repair_prompt, encoding="utf-8")
                    try:
                        repair_response = self._call_llm(
                            prompt=repair_prompt,
                            call_type="compile_repair",
                            label=(
                                f"Evaluation {evaluation_round}/{self.config.max_rounds} · "
                                "compiler repair LLM"
                            ),
                            logger=logger,
                            attempt_id=round_num,
                            evaluation_round=evaluation_round,
                            response_path=repair_dir / "response.txt",
                        )
                        repair_delta = parse_file_bundle(repair_response.content)
                        repaired_candidate = FileBundle(files=dict(candidate.files))
                        for path in repair_delta.delete:
                            repaired_candidate.files.pop(path, None)
                        repaired_candidate.files.update(repair_delta.files)
                        validate_initial_bundle(repaired_candidate)
                        restore_bundle(self.task_dir, repaired_candidate)
                        repaired_candidate = capture_bundle(self.task_dir)
                        self._write_bundle(repair_dir / "candidate.json", repaired_candidate)
                        try:
                            repaired_result = self.evaluator.evaluate(self.task_dir, repair_dir)
                        except Exception as error:
                            error_path = repair_dir / "evaluation_error.log"
                            self._write_error(error_path, error)
                            repaired_result = self._failure_result(
                                stage="evaluation",
                                code="evaluator_exception",
                                message=f"local evaluator failed unexpectedly after compiler repair: {error}",
                                details_path=error_path,
                            )
                        repaired_result = self._normalize_evaluation(repaired_result, repair_dir)
                        evaluation_attempts.append(repaired_result.to_dict())
                        original_error_count = len(diagnostic_lines(read_result_log(original_result)))
                        repaired_error_count = len(diagnostic_lines(read_result_log(repaired_result)))
                        repair_regressed = bool(
                            repaired_result.error
                            and (
                                repaired_error_count > original_error_count
                                or (
                                    original_result.failure_stage == "ascendc_build"
                                    and repaired_result.failure_stage == "ascendc_source_validation"
                                )
                            )
                        )
                        if repair_regressed:
                            repair_error = (
                                "same-round compiler repair introduced additional deterministic/compiler "
                                f"errors ({original_error_count} -> {repaired_error_count}); restored pre-repair candidate"
                            )
                            self._write_error(repair_dir / "repair_regression.log", repair_error)
                            restore_bundle(self.task_dir, candidate)
                            result = original_result
                        else:
                            candidate, result = repaired_candidate, repaired_result
                            self._write_bundle(round_dir / "candidate_repair_01.json", candidate)
                    except Exception as error:
                        repair_error = f"same-round compiler repair failed: {error}"
                        self._write_error(repair_dir / "repair_error.log", repair_error)
                    knowledge_state.known_symbols = list(
                        dict.fromkeys([*knowledge_state.known_symbols, *repair_symbols])
                    )[:128]
                    fingerprint = diagnostic_fingerprint(compiler_output)
                    knowledge_state.failure_fingerprints = [
                        *knowledge_state.failure_fingerprints,
                        fingerprint,
                    ][-16:]
                    save_knowledge_state(knowledge_state_path, knowledge_state)
                keep = self._is_better(result, best_result)
                decision = "KEEP" if keep else "DISCARD"
                if keep:
                    best_bundle, best_result, best_round = candidate, result, round_num
                    current = candidate
                    self._write_bundle(self.state_dir / "best.json", candidate)
                elif best_bundle is not None:
                    restore_bundle(self.task_dir, best_bundle)
                    current = best_bundle
                else:
                    # Keep a failing first draft as repair context until a valid best exists.
                    current = candidate

            previous = result
            logger.save_round(
                self._round_record(
                    round_num=round_num,
                    evaluation_round=evaluation_round,
                    decision=decision,
                    result=result,
                    selection=selection,
                    response=response,
                    generation_attempts=generation_attempts,
                    compile_repair_attempts=compile_repair_attempts,
                    evaluation_attempts=evaluation_attempts,
                    repair_error=repair_error,
                    candidate_path=f"round_{round_num:02d}/candidate.json" if (round_dir / "candidate.json").is_file() else None,
                ),
                best_round=best_round,
            )
            logger.complete_pending()
            self._report_result(evaluation_round, decision, result)

        if best_bundle is not None:
            restore_bundle(self.task_dir, best_bundle)
        totals = logger.token_totals()
        totals_by_call_type = logger.token_totals_by_call_type()
        counted_records = [
            record
            for record in logger.data.get("rounds", [])
            if isinstance(record, dict) and logger.counts_toward_budget(record)
        ]
        last_record = counted_records[-1] if counted_records else None
        last_evaluation = last_record.get("evaluation", {}) if last_record else {}
        last_round = None
        last_evaluation_failure = None
        if last_record:
            last_round = {
                "round": last_record.get("evaluation_round", last_record.get("round")),
                "attempt_id": last_record.get("attempt_id", last_record.get("round")),
                "decision": last_record.get("decision"),
                "compiled": last_evaluation.get("compiled", False),
                "correctness": last_evaluation.get("correctness", False),
                "score": last_evaluation.get("score"),
            }
            last_evaluation_failure = self._failure_summary(
                int(last_record.get("round")), last_evaluation
            )
        orchestration_failure = None
        if stop_failure is not None:
            orchestration_failure = self._failure_summary(
                int(stop_attempt_id or 0), stop_failure.to_dict()
            )
            if orchestration_failure is not None:
                orchestration_failure["evaluation_round"] = logger.completed_rounds + 1
        completed = stop_failure is None and logger.completed_rounds >= self.config.max_rounds
        status = "completed" if completed else "paused"
        summary = {
            "success": best_bundle is not None,
            "status": status,
            "completed": completed,
            "rounds_completed": logger.completed_rounds,
            "evaluations_completed": logger.completed_rounds,
            "historical_records": logger.historical_records,
            "attempts_completed": logger.last_attempt_id,
            "orchestration_failures": logger.orchestration_attempts,
            "pending_round": None if completed else logger.completed_rounds + 1,
            "best_round": best_round,
            "best_score": best_result.score if best_result else None,
            "last_round": last_round,
            "last_evaluation": last_round,
            "last_evaluation_failure": last_evaluation_failure,
            "last_orchestration_failure": orchestration_failure,
            "failure": orchestration_failure or last_evaluation_failure,
            "token_usage": totals,
            "token_usage_by_call_type": totals_by_call_type,
            "knowledge": knowledge_version.to_dict(),
            "task_dir": str(self.task_dir),
        }
        usage_report = {"total": totals, "by_call_type": totals_by_call_type}
        (self.state_dir / "token_usage.json").write_text(json.dumps(usage_report, indent=2), encoding="utf-8")
        (self.state_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if completed:
            logger.mark_done(summary["success"])
        else:
            logger.mark_paused(summary["success"])
        self.progress.emit(
            f"Run {status}: success={summary['success']}, "
            f"evaluations_completed={summary['evaluations_completed']}, "
            f"best_round={summary['best_round']}, pending_round={summary['pending_round']}"
        )
        return summary
