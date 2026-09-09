from __future__ import annotations

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

    def save_call(
        self,
        round_num: int,
        response: dict[str, Any],
        *,
        call_type: str = "generator",
        evaluation_round: int | None = None,
        retry: int = 0,
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
        }
        reasoning = response.get("reasoning_content")
        if isinstance(reasoning, str):
            import hashlib

            record["reasoning_content_present"] = bool(reasoning)
            record["reasoning_content_chars"] = len(reasoning)
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
    ) -> None:
        record = {
            "timestamp": self._timestamp(),
            "round": round_num,
            "evaluation_round": evaluation_round,
            "call_type": call_type,
            "retry": retry,
            "error": error,
            "request_options": request_options,
        }
        with self.calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_orchestration_attempt(self, record: dict[str, Any]) -> None:
        record = {"timestamp": self._timestamp(), **record}
        with self.attempts_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

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
