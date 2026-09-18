from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import EvalResult, FileBundle

FRONTIERS = ("source", "compile", "runtime", "correctness", "performance")


@dataclass(frozen=True)
class FrontierDecision:
    reached: str | None
    advanced: bool
    rollback: FileBundle | None
    highest: str | None


def reached_frontier(result: EvalResult) -> str | None:
    from .diagnostics import evaluation_gates

    if (result.failure_stage or "") in {
        "response_format", "static_validation", "ascendc_source_validation",
        "bundle_validation", "interface_contract_validation",
    }:
        return None
    if not result.compiled:
        return "source"
    gates = evaluation_gates(result)
    if gates["loaded"] != "pass" or gates["kernel_started"] != "pass":
        return "compile"
    if gates["comparison_completed"] != "pass":
        return "runtime"
    if isinstance(result.score, (int, float)) and math.isfinite(float(result.score)) and float(result.score) > 0:
        return "performance"
    return "correctness"


class FrontierManager:
    """Persist only verified run-local frontiers."""

    def __init__(self, state_dir: Path):
        self.root = state_dir / "frontiers"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"schema_version": 2, "highest": None, "entries": {}}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        payload.setdefault("entries", {})
        return payload

    @staticmethod
    def _rank(name: str | None) -> int:
        return FRONTIERS.index(name) if name in FRONTIERS else -1

    def highest_bundle(self) -> FileBundle | None:
        highest = self.manifest.get("highest")
        entry = self.manifest.get("entries", {}).get(highest, {})
        path = self.root / entry.get("bundle", "") if entry else None
        if not path or not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FileBundle(files=payload["files"], delete=payload.get("delete", []), analysis=payload.get("analysis", ""))

    def observe(self, candidate: FileBundle, result: EvalResult, attempt_id: int) -> FrontierDecision:
        reached = reached_frontier(result)
        highest = self.manifest.get("highest")
        advanced = self._rank(reached) > self._rank(highest)
        if reached == highest == "performance":
            old_score = self.manifest["entries"]["performance"].get("score")
            advanced = old_score is None or float(result.score) > float(old_score)
        if advanced and reached:
            for frontier in FRONTIERS[self._rank(highest) + 1 : self._rank(reached) + 1]:
                name = f"{frontier}.json"
                (self.root / name).write_text(json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                self.manifest["entries"][frontier] = {"attempt_id": attempt_id, "bundle": name, "score": result.score if frontier == "performance" else None}
            self.manifest["highest"] = reached
            self.manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            highest = reached
        return FrontierDecision(reached, advanced, self.highest_bundle(), highest)
