from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    def __init__(self, state_dir: Path, config: dict[str, Any]):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = state_dir / "trajectory.json"
        self.calls_path = state_dir / "calls.jsonl"
        self.attempt_ledger_path = state_dir / "attempt_ledger.jsonl"
        self.attempt_records_dir = state_dir / "attempts"
        self.attempts_path = state_dir / "orchestration_attempts.jsonl"
        self.invocations_path = state_dir / "invocations.jsonl"
        self.run_state_path = state_dir / "run_state.json"
        self.done_path = state_dir / "DONE"
        self.data = self._load()
        self.data.setdefault("config", config)
        self.data.setdefault("rounds", [])
        self.data.setdefault("best_round", None)
        self.data.setdefault("schema_version", 3)
        self.data.setdefault("workflow", {})
        self._save()

    @staticmethod
    def counts_toward_budget(record: dict[str, Any]) -> bool:
        explicit = record.get("counts_toward_budget")
        if isinstance(explicit, bool):
            return explicit
        decision = record.get("decision")
        if decision in {"KEEP", "DISCARD"}:
            return True
        # Historical orchestration failures sometimes stored a synthetic
        # EvalResult in evaluation_attempts even though no evaluator ran.
        if decision in {"LLM_FAIL", "LOCAL_FAIL", "FORMAT_FAIL", "PAUSED"}:
            return False
        attempts = record.get("evaluation_attempts")
        if isinstance(attempts, list) and attempts:
            return True
        return False

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load(self) -> dict[str, Any]:
        if not self.log_path.is_file():
            return {}
        return json.loads(self.log_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        temporary = self.log_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.log_path)

    @staticmethod
    def _write_private_text(path: Path, text: str) -> None:
        """Atomically persist one model artifact with owner-only permissions."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)
        path.chmod(0o600)

    @staticmethod
    def _reasoning_path(response_path: Path) -> Path:
        name = response_path.name
        if name.startswith("response"):
            name = "reasoning" + name[len("response") :]
        else:
            name = f"{response_path.stem}_reasoning{response_path.suffix}"
        return response_path.with_name(name)

    def persist_response_artifacts(
        self,
        response_path: Path,
        *,
        content: str,
        reasoning_content: str,
        reasoning_log_mode: str,
    ) -> tuple[Path, Path | None]:
        """Persist final content and, in full mode, the raw reasoning text."""

        self._write_private_text(response_path, content)
        reasoning_path = None
        if reasoning_log_mode == "full" and reasoning_content:
            reasoning_path = self._reasoning_path(response_path)
            self._write_private_text(reasoning_path, reasoning_content)
        return response_path, reasoning_path

    def _artifact_reference(self, path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return path.resolve().relative_to(self.state_dir.resolve()).as_posix()
        except ValueError:
            return str(path.resolve())

    def save_call(
        self,
        round_num: int,
        response: dict[str, Any],
        *,
        call_type: str = "generator",
        evaluation_round: int | None = None,
        retry: int = 0,
        prompt_metadata: dict[str, Any] | None = None,
        response_content_path: Path | None = None,
        reasoning_content_path: Path | None = None,
    ) -> None:
        # Raw model output is stored once in round_N/response.txt.  Keep this
        # append-only file compact so it can be aggregated across many ops.
        record = {
            "timestamp": self._timestamp(),
            "round": round_num,
            "evaluation_round": evaluation_round,
            "call_type": call_type,
            "retry": retry,
            **{
                key: value
                for key, value in response.items()
                if key not in {"content", "reasoning_content"}
            },
            "prompt_metadata": prompt_metadata or {},
        }
        record["response_content_path"] = self._artifact_reference(response_content_path)
        record["reasoning_content_path"] = self._artifact_reference(reasoning_content_path)
        requested_model = response.get("requested_model")
        returned_model = response.get("model")
        record["model_route_mismatch"] = bool(
            requested_model and returned_model and requested_model != returned_model
        )
        reasoning = response.get("reasoning_content")
        if isinstance(reasoning, str):
            record["reasoning_content_present"] = bool(reasoning)
            record["reasoning_content_chars"] = len(reasoning)
            record["reasoning_content_bytes"] = len(reasoning.encode("utf-8"))
            record["reasoning_content_sha256"] = (
                hashlib.sha256(reasoning.encode("utf-8")).hexdigest() if reasoning else None
            )
        with self.calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_call_failure(
        self,
        round_num: int,
        *,
        call_type: str,
        evaluation_round: int,
        retry: int,
        error: str,
        request_options: dict[str, Any],
        prompt_metadata: dict[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp": self._timestamp(),
            "round": round_num,
            "evaluation_round": evaluation_round,
            "call_type": call_type,
            "retry": retry,
            "error": error,
            "request_options": request_options,
            "prompt_metadata": prompt_metadata or {},
        }
        with self.calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_orchestration_attempt(self, record: dict[str, Any]) -> None:
        record = {"timestamp": self._timestamp(), **record}
        with self.attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_attempt_record(self, record: Any) -> None:
        payload = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        self.attempt_records_dir.mkdir(parents=True, exist_ok=True)
        attempt_id = int(payload.get("attempt_id", 0))
        record_path = self.attempt_records_dir / f"attempt_{attempt_id:04d}.json"
        record_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        summary = {
            key: payload.get(key)
            for key in (
                "attempt_id",
                "evaluation_round",
                "base_attempt_id",
                "phase",
                "target_error_ids",
                "cleared_error_ids",
                "new_error_ids",
                "reintroduced_error_ids",
                "touched_files",
                "selected_skill_ids",
                "selected_fact_ids",
                "route",
                "outcome",
                "frontier_before",
                "frontier_after",
                "progress",
            )
        }
        with self.attempt_ledger_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary, ensure_ascii=False) + "\n")

    def save_invocation(self, config: dict[str, Any]) -> None:
        record = {"timestamp": self._timestamp(), "config": config}
        with self.invocations_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_run_state(self) -> dict[str, Any]:
        if not self.run_state_path.is_file():
            return {}
        try:
            payload = json.loads(self.run_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_run_state(self, state: dict[str, Any]) -> None:
        temporary = self.run_state_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.run_state_path)

    def begin_pending(self, evaluation_round: int, *, phase: str = "EDIT") -> int:
        state = self._load_run_state()
        if state.get("pending_evaluation_round") == evaluation_round and isinstance(
            state.get("attempt_id"), int
        ):
            if state.get("pending_phase") != phase:
                state["pending_phase"] = phase
                self._save_run_state(state)
            return int(state["attempt_id"])
        historical = [
            int(item.get("round", 0))
            for item in self.data.get("rounds", [])
            if isinstance(item, dict)
        ]
        if self.attempts_path.is_file():
            for line in self.attempts_path.read_text(encoding="utf-8").splitlines():
                try:
                    historical.append(int(json.loads(line).get("attempt_id", 0)))
                except (ValueError, TypeError, json.JSONDecodeError):
                    continue
        attempt_id = max(historical, default=0) + 1
        self._save_run_state(
            {
                "schema_version": 3,
                "pending_evaluation_round": evaluation_round,
                "attempt_id": attempt_id,
                "pending_phase": phase,
            }
        )
        return attempt_id

    def update_pending_phase(self, phase: str) -> None:
        state = self._load_run_state()
        if state:
            state["pending_phase"] = phase
            self._save_run_state(state)

    def complete_pending(self) -> None:
        self._save_run_state({"schema_version": 3})

    def pending_state(self) -> dict[str, Any]:
        return self._load_run_state()

    def save_round(self, record: dict[str, Any], *, best_round: int | None) -> None:
        record.setdefault("counts_toward_budget", True)
        self.data["rounds"].append(record)
        self.data["best_round"] = best_round
        self._save()

    def mark_done(self, success: bool) -> None:
        self.data["success"] = success
        self.data["status"] = "completed"
        self._save()
        self.done_path.touch()

    def mark_running(self) -> None:
        self.data["status"] = "running"
        self._save()
        try:
            self.done_path.unlink()
        except FileNotFoundError:
            pass

    def mark_paused(self, success: bool) -> None:
        self.data["success"] = success
        self.data["status"] = "paused"
        self._save()

    def mark_blocked(self, success: bool) -> None:
        self.data["success"] = success
        self.data["status"] = "blocked"
        self._save()
        try:
            self.done_path.unlink()
        except FileNotFoundError:
            pass

    def update_workflow(self, **values: Any) -> None:
        workflow = self.data.setdefault("workflow", {})
        workflow.update(values)
        self.data["schema_version"] = 3
        self._save()

    def save_knowledge(self, knowledge: dict[str, Any]) -> None:
        self.data["knowledge"] = knowledge
        self._save()

    @property
    def completed_rounds(self) -> int:
        return sum(
            self.counts_toward_budget(item)
            for item in self.data.get("rounds", [])
            if isinstance(item, dict)
        )

    @property
    def historical_records(self) -> int:
        return len(self.data.get("rounds", []))

    @property
    def orchestration_attempts(self) -> int:
        if not self.attempts_path.is_file():
            return 0
        return sum(1 for line in self.attempts_path.read_text(encoding="utf-8").splitlines() if line.strip())

    @property
    def last_attempt_id(self) -> int:
        values = [
            int(item.get("round", 0))
            for item in self.data.get("rounds", [])
            if isinstance(item, dict)
        ]
        pending = self._load_run_state().get("attempt_id")
        if isinstance(pending, int):
            values.append(pending)
        return max(values, default=0)

    def token_totals(self) -> dict[str, int]:
        totals: dict[str, int] = {}
        if not self.calls_path.is_file():
            return totals
        for line in self.calls_path.read_text(encoding="utf-8").splitlines():
            try:
                usage = json.loads(line).get("usage", {})
            except json.JSONDecodeError:
                continue
            for key, value in usage.items():
                if isinstance(value, int) and ("token" in key):
                    totals[key] = totals.get(key, 0) + value
            details = usage.get("completion_tokens_details", {})
            if isinstance(details, dict):
                reasoning = details.get("reasoning_tokens")
                if isinstance(reasoning, int):
                    totals["reasoning_tokens"] = totals.get("reasoning_tokens", 0) + reasoning
        return totals

    def token_totals_by_call_type(self) -> dict[str, dict[str, int]]:
        totals: dict[str, dict[str, int]] = {}
        if not self.calls_path.is_file():
            return totals
        for line in self.calls_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            call_type = str(record.get("call_type", "generator"))
            bucket = totals.setdefault(call_type, {})
            for key, value in record.get("usage", {}).items():
                if isinstance(value, int) and "token" in key:
                    bucket[key] = bucket.get(key, 0) + value
            details = record.get("usage", {}).get("completion_tokens_details", {})
            if isinstance(details, dict):
                reasoning = details.get("reasoning_tokens")
                if isinstance(reasoning, int):
                    bucket["reasoning_tokens"] = bucket.get("reasoning_tokens", 0) + reasoning
        return totals

    def normalized_metrics(self) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        if self.calls_path.is_file():
            for line in self.calls_path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    calls.append(record)
        token_values = [
            int(item.get("usage", {}).get("total_tokens", 0))
            for item in calls
            if isinstance(item.get("usage"), dict)
        ]
        latencies = [
            float(item.get("latency_seconds", 0.0) or 0.0) for item in calls
        ]
        milestones: dict[str, Any] = {}
        stage_names = (
            "A_source_valid",
            "B_compile",
            "C_load",
            "D_execute",
            "E_correct",
            "F_benchmark",
            "G_optimization",
            "H_optimized_correct",
        )
        rounds = [item for item in self.data.get("rounds", []) if isinstance(item, dict)]
        for stage_name in stage_names:
            reached = next(
                (
                    item for item in rounds
                    if item.get("stage_observation", {}).get(stage_name, {}).get("status") == "pass"
                ),
                None,
            )
            if reached is None:
                milestones[stage_name] = {
                    "status": "censored",
                    "evaluation_round": None,
                    "tokens": None,
                    "time_seconds": None,
                }
                continue
            evaluation_round = int(reached.get("evaluation_round", 0))
            call_indexes = [
                index for index, call in enumerate(calls)
                if int(call.get("evaluation_round") or 0) <= evaluation_round
            ]
            last_index = max(call_indexes, default=-1)
            eligible_rounds = [
                item for item in rounds
                if int(item.get("evaluation_round", 0)) <= evaluation_round
            ]
            stage_seconds = sum(
                float(event.get("elapsed_seconds", 0.0) or 0.0)
                for item in eligible_rounds
                for event in item.get("stage_timings", [])
                if isinstance(event, dict)
            )
            milestones[stage_name] = {
                "status": "reached",
                "evaluation_round": evaluation_round,
                "tokens": sum(token_values[: last_index + 1]),
                "time_seconds": sum(latencies[: last_index + 1]) + stage_seconds,
            }
        component_totals: dict[str, dict[str, int]] = {}
        for call in calls:
            metadata = call.get("prompt_metadata", {})
            if not isinstance(metadata, dict):
                continue
            for name in (
                "prompt",
                "knowledge_reference",
                "source_code",
                "compiler_evaluator_evidence",
            ):
                measurement = metadata.get(name, {})
                if not isinstance(measurement, dict):
                    continue
                bucket = component_totals.setdefault(name, {"chars": 0, "estimated_tokens": 0})
                for unit in ("chars", "estimated_tokens"):
                    value = measurement.get(unit)
                    if isinstance(value, int):
                        bucket[unit] += value

        stage_latency: dict[str, float] = {}
        for item in rounds:
            for event in item.get("stage_timings", []):
                if not isinstance(event, dict):
                    continue
                label = str(event.get("label", "")).lower()
                category = next(
                    (
                        name
                        for marker, name in (
                            ("source validation", "source_validation"),
                            ("api constraint validation", "api_constraint_validation"),
                            ("static validation", "static_validation"),
                            ("ascendc build", "compile"),
                            ("correctness", "npu_execution_and_correctness"),
                            ("performance", "benchmark"),
                        )
                        if marker in label
                    ),
                    "other_evaluation",
                )
                stage_latency[category] = stage_latency.get(category, 0.0) + float(
                    event.get("elapsed_seconds", 0.0) or 0.0
                )

        transition_pairs = (
            ("A_source_valid", "B_compile", "source_valid_to_compile"),
            ("B_compile", "C_load", "compile_to_load"),
            ("C_load", "D_execute", "load_to_execute"),
            ("D_execute", "E_correct", "execute_to_correct"),
            ("E_correct", "F_benchmark", "correct_to_benchmark"),
            ("G_optimization", "H_optimized_correct", "optimization_to_retained_correctness"),
        )
        transitions: dict[str, dict[str, Any]] = {}
        for source, target, name in transition_pairs:
            eligible = [
                item for item in rounds
                if item.get("stage_observation", {}).get(source, {}).get("status") == "pass"
            ]
            passed = sum(
                item.get("stage_observation", {}).get(target, {}).get("status") == "pass"
                for item in eligible
            )
            transitions[name] = {
                "eligible": len(eligible),
                "passed": passed,
                "rate": passed / len(eligible) if eligible else None,
            }

        short_names = {
            "source_valid": "A_source_valid",
            "compile": "B_compile",
            "load": "C_load",
            "npu_execution": "D_execute",
            "correct": "E_correct",
            "benchmark": "F_benchmark",
            "optimization": "G_optimization",
            "optimized_correct": "H_optimized_correct",
        }
        return {
            "llm_calls": len(calls),
            "tokens_per_llm_call": (sum(token_values) / len(token_values)) if token_values else None,
            "llm_latency_seconds": sum(latencies),
            "evaluation_latency_seconds": sum(stage_latency.values()),
            "evaluation_latency_by_stage_seconds": stage_latency,
            "prompt_component_totals": component_totals,
            "milestones": milestones,
            "tokens_to_first": {
                name: milestones[stage]["tokens"] for name, stage in short_names.items()
            },
            "latency_to_first_seconds": {
                name: milestones[stage]["time_seconds"] for name, stage in short_names.items()
            },
            "transition_rates": transitions,
        }
