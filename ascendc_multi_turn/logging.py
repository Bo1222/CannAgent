from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class TrajectoryLogger:
    def __init__(self, state_dir: Path, config: dict[str, Any]):
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = state_dir / "trajectory.json"
        self.calls_path = state_dir / "calls.jsonl"
        self.done_path = state_dir / "DONE"
        self.data = self._load()
        self.data.setdefault("config", config)
        self.data.setdefault("rounds", [])
        self.data.setdefault("best_round", None)
        self._save()

    def _load(self) -> dict[str, Any]:
        if not self.log_path.is_file():
            return {}
        return json.loads(self.log_path.read_text(encoding="utf-8"))

    def _save(self) -> None:
        temporary = self.log_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.log_path)

    def save_call(self, round_num: int, response: dict[str, Any], *, call_type: str = "generator") -> None:
        # Raw model output is stored once in round_N/response.txt.  Keep this
        # append-only file compact so it can be aggregated across many ops.
        record = {
            "round": round_num,
            "call_type": call_type,
            **{key: value for key, value in response.items() if key != "content"},
        }
        with self.calls_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def save_round(self, record: dict[str, Any], *, best_round: int | None) -> None:
        self.data["rounds"].append(record)
        self.data["best_round"] = best_round
        self._save()

    def mark_done(self, success: bool) -> None:
        self.data["success"] = success
        self._save()
        self.done_path.touch()

    def save_knowledge(self, knowledge: dict[str, Any]) -> None:
        self.data["knowledge"] = knowledge
        self._save()

    @property
    def completed_rounds(self) -> int:
        return len(self.data.get("rounds", []))

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
        return totals
