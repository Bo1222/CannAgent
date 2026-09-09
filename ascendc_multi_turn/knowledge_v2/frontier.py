from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import EvalResult, FileBundle


FRONTIERS = ("source", "compile", "runtime", "correctness", "performance")


@dataclass(frozen=True)
class FrontierDecision:
    reached: str | None
    advanced: bool
    rollback: FileBundle | None
    highest: str | None


def reached_frontier(result: EvalResult) -> str | None:
    stage = result.failure_stage or ""
    if stage in {
        "response_format",
        "static_validation",
        "ascendc_source_validation",
        "semantic_validation",
        "bundle_validation",
    }:
        return None
    if not result.compiled:
        return "source"
    if not result.correctness:
        return "runtime"
    if (
        isinstance(result.score, (int, float))
        and math.isfinite(float(result.score))
        and float(result.score) > 0
    ):
        return "performance"
    return "correctness"


class FrontierManager:
    """Persist the deepest evaluated candidate and select safe rollback sources."""

    def __init__(self, state_dir: Path):
        self.root = state_dir / "frontiers"
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest: dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {"schema_version": 1, "highest": None, "entries": {}}
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        payload.setdefault("entries", {})
        return payload

    @staticmethod
    def _rank(name: str | None) -> int:
        return FRONTIERS.index(name) if name in FRONTIERS else -1

    @staticmethod
    def _bundle_from(path: Path) -> FileBundle:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return FileBundle(
            files=payload["files"],
            delete=payload.get("delete", []),
            analysis=payload.get("analysis", ""),
        )

    def highest_bundle(self) -> FileBundle | None:
        highest = self.manifest.get("highest")
        entry = self.manifest.get("entries", {}).get(highest, {})
        path = self.root / entry.get("bundle", "") if entry else None
        return self._bundle_from(path) if path and path.is_file() else None

    def _save(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def observe(self, candidate: FileBundle, result: EvalResult, attempt_id: int) -> FrontierDecision:
        reached = reached_frontier(result)
        highest = self.manifest.get("highest")
        advanced = self._rank(reached) > self._rank(highest)
        if reached == "performance" and highest == "performance":
            old_score = self.manifest["entries"]["performance"].get("score")
            advanced = old_score is None or float(result.score) > float(old_score)
        if advanced and reached is not None:
            crossed = list(FRONTIERS[self._rank(highest) + 1 : self._rank(reached) + 1])
            if not crossed:
                crossed = [reached]
            for frontier in crossed:
                bundle_name = f"{frontier}.json"
                (self.root / bundle_name).write_text(
                    json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
                )
                self.manifest["entries"][frontier] = {
                    "attempt_id": attempt_id,
                    "bundle": bundle_name,
                    "score": result.score if frontier == "performance" else None,
                }
            self.manifest["highest"] = reached
            self._save()
            highest = reached
        return FrontierDecision(
            reached=reached,
            advanced=advanced,
            rollback=self.highest_bundle(),
            highest=highest,
        )
