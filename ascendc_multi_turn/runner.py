from __future__ import annotations

import json
import shutil
from pathlib import Path

from .bundle import capture_bundle, parse_file_bundle, restore_bundle, validate_initial_bundle
from .evaluator import Evaluator
from .llm import LLMProvider
from .logging import TrajectoryLogger
from .models import EvalResult, FileBundle, RunConfig
from .prompts import build_prompt


class MultiTurnRunner:
    """Explicit generate/evaluate/select loop with no dependency on an agent runtime."""

    def __init__(self, config: RunConfig, provider: LLMProvider, evaluator: Evaluator):
        self.config = config
        self.provider = provider
        self.evaluator = evaluator
        self.task_dir = Path(config.output_dir).expanduser().resolve()
        self.state_dir = self.task_dir / ".llm_state"

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
        if not candidate.correctness:
            return False
        if best is None:
            return True
        if candidate.score is None:
            return False
        return best.score is None or candidate.score > best.score

    def _resume_state(self, logger: TrajectoryLogger) -> tuple[FileBundle | None, FileBundle | None, EvalResult | None, EvalResult | None, int | None]:
        working = capture_bundle(self.task_dir)
        current = working if working.files else None
        previous = None
        best_bundle = None
        best_result = None
        best_round = logger.data.get("best_round")
        if logger.data.get("rounds"):
            raw = logger.data["rounds"][-1].get("evaluation")
            if raw:
                previous = EvalResult(**raw)
        best_path = self.state_dir / "best.json"
        if best_path.is_file():
            best_bundle = self._read_bundle(best_path)
            restore_bundle(self.task_dir, best_bundle)
            current = best_bundle
            if best_round:
                raw_best = logger.data["rounds"][best_round - 1].get("evaluation")
                if raw_best:
                    best_result = EvalResult(**raw_best)
        return current, best_bundle, previous, best_result, best_round

    def run(self) -> dict:
        self._prepare_task()
        logger = TrajectoryLogger(self.state_dir, self.config.to_dict())
        reference, cases = self._inputs()
        current, best_bundle, previous, best_result, best_round = self._resume_state(logger)

        for round_num in range(logger.completed_rounds + 1, self.config.max_rounds + 1):
            round_dir = self.state_dir / f"round_{round_num:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            prompt = build_prompt(
                reference_code=reference,
                cases_text=cases,
                current=current,
                previous_result=previous,
                round_num=round_num,
            )
            (round_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
            response = self.provider.generate(prompt)
            (round_dir / "response.txt").write_text(response.content, encoding="utf-8")
            logger.save_call(round_num, response.to_dict())

            try:
                delta = parse_file_bundle(response.content)
                candidate = FileBundle(files=dict(current.files) if current else {})
                for path in delta.delete:
                    candidate.files.pop(path, None)
                candidate.files.update(delta.files)
                validate_initial_bundle(candidate)
                restore_bundle(self.task_dir, candidate)
                candidate = capture_bundle(self.task_dir)
                self._write_bundle(round_dir / "candidate.json", candidate)
                result = self.evaluator.evaluate(self.task_dir, round_dir)
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
            except (ValueError, json.JSONDecodeError) as error:
                result = EvalResult(False, False, error=f"invalid LLM file bundle: {error}")
                decision = "FORMAT_FAIL"

            previous = result
            logger.save_round(
                {
                    "round": round_num,
                    "decision": decision,
                    "evaluation": result.to_dict(),
                    "response_model": response.model,
                    "usage": response.usage,
                    "latency_seconds": response.latency_seconds,
                    "candidate": f"round_{round_num:02d}/candidate.json" if (round_dir / "candidate.json").is_file() else None,
                },
                best_round=best_round,
            )

        if best_bundle is not None:
            restore_bundle(self.task_dir, best_bundle)
        totals = logger.token_totals()
        summary = {
            "success": best_bundle is not None,
            "rounds_completed": logger.completed_rounds,
            "best_round": best_round,
            "best_score": best_result.score if best_result else None,
            "token_usage": totals,
            "task_dir": str(self.task_dir),
        }
        (self.state_dir / "token_usage.json").write_text(json.dumps(totals, indent=2), encoding="utf-8")
        (self.state_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.mark_done(summary["success"])
        return summary
