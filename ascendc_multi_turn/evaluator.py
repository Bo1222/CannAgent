from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from .bundle import capture_bundle, validate_initial_bundle
from .diagnostics import parse_structured_failure
from .models import EvalResult
from .progress import ProgressReporter
from .source_validation import render_issues, validate_source_tree
from .knowledge_v2.semantic_validator import SemanticValidator


REPO_ROOT = Path(__file__).resolve().parent.parent


class Evaluator(Protocol):
    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult: ...


def _write_log(path: Path, output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _full_log(path: Path, fallback: str) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return fallback


def _run(command: list[str], *, env: dict[str, str], timeout: int, log_path: Path | None = None) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        output = completed.stdout or ""
        if log_path is not None:
            _write_log(log_path, output)
        return completed.returncode, output[-20000:]
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        output = f"{output}\nTimed out after {timeout}s".lstrip("\n")
        if log_path is not None:
            _write_log(log_path, output)
        return 124, output[-20000:]
    except OSError as error:
        output = f"Failed to start command: {error}\n"
        if log_path is not None:
            _write_log(log_path, output)
        return 127, output


_ERROR_LINE = re.compile(r"error:|fatal(?: error)?:|traceback|\[fail(?:ed)?\]|timed out", re.IGNORECASE)


def extract_error_excerpt(output: str, *, max_lines: int = 20, max_chars: int = 4000) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    preferred = [line for line in lines if _ERROR_LINE.search(line)]
    candidates = preferred if preferred else lines[-max_lines:]
    selected: list[str] = []
    seen: set[str] = set()
    for line in candidates:
        if line in seen:
            continue
        seen.add(line)
        selected.append(line)
        if len(selected) == max_lines:
            break
    excerpt = "\n".join(selected)
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 3].rstrip() + "..."
    return excerpt


def _failure(
    *,
    compiled: bool,
    correctness: bool,
    stage: str,
    code: str,
    message: str,
    output: str,
    details_path: Path | None,
    compile_output: str = "",
    verify_output: str = "",
    failure_kind: str = "candidate",
) -> EvalResult:
    structured = parse_structured_failure(stage=stage, output=output)
    return EvalResult(
        compiled=compiled,
        correctness=correctness,
        compile_output=compile_output,
        verify_output=verify_output,
        error=message,
        failure_stage=stage,
        failure_code=code,
        error_excerpt=extract_error_excerpt(output),
        details_path=str(details_path.resolve()) if details_path is not None else None,
        failure_kind=failure_kind,
        structured_failure=structured.to_dict(),
    )


def _return_code_kind(return_code: int) -> str:
    return "infrastructure" if return_code in {124, 126, 127} else "candidate"


class LocalAscendEvaluator:
    def __init__(
        self,
        *,
        device: int,
        soc_version: str,
        timeout: int = 600,
        progress: ProgressReporter | None = None,
        semantic_validator: SemanticValidator | None = None,
    ):
        self.device = device
        self.soc_version = soc_version
        self.timeout = timeout
        self.progress = progress or ProgressReporter()
        self.semantic_validator = semantic_validator

    def preflight(self, task_dir: Path, state_dir: Path) -> EvalResult | None:
        log_path = state_dir / "environment_preflight.log"
        failures: list[str] = []
        if shutil.which("cmake") is None:
            failures.append("cmake is not available on PATH")
        for path in (
            REPO_ROOT / "utils/build_ascendc.py",
            REPO_ROOT / "utils/verification_ascendc.py",
            REPO_ROOT / "skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py",
        ):
            if not path.is_file():
                failures.append(f"required tool does not exist: {path}")
        if not failures:
            env = os.environ.copy()
            env["ASCEND_RT_VISIBLE_DEVICES"] = str(self.device)
            rc, output = _run(
                [
                    sys.executable,
                    "-c",
                    "import torch, torch_npu; assert torch.npu.is_available(), 'NPU is unavailable'",
                ],
                env=env,
                timeout=min(self.timeout, 60),
                log_path=log_path,
            )
            if rc == 0:
                return None
            failures.append(output or f"NPU runtime preflight exited with {rc}")
        message = "environment preflight failed"
        output = "\n".join(failures)
        _write_log(log_path, output + "\n")
        return _failure(
            compiled=False,
            correctness=False,
            stage="environment_preflight",
            code="environment_preflight_failed",
            message=message,
            output=output,
            details_path=log_path,
            failure_kind="infrastructure",
        )

    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult:
        try:
            validate_initial_bundle(capture_bundle(task_dir))
        except ValueError as error:
            message = str(error)
            bundle_log = round_dir / "bundle_validation.log"
            _write_log(bundle_log, f"{message}\n")
            return _failure(
                compiled=False,
                correctness=False,
                stage="bundle_validation",
                code="bundle_validation_failed",
                message=message,
                output=message,
                details_path=bundle_log,
            )

        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(self.device)
        python = sys.executable
        source_log = round_dir / "source_validation.log"
        task = self.progress.start(f"{round_dir.name} · AscendC source validation")
        source_issues = validate_source_tree(task_dir)
        source_output = render_issues(source_issues)
        _write_log(source_log, source_output + ("\n" if source_output else ""))
        task.finish(status="failed" if source_issues else "passed", detail=f"issues={len(source_issues)}")
        if source_issues:
            return _failure(
                compiled=False,
                correctness=False,
                stage="ascendc_source_validation",
                code="ascendc_source_validation_failed",
                message="AscendC source validation failed",
                output=source_output,
                details_path=source_log,
                compile_output=source_output,
            )

        if self.semantic_validator is not None:
            semantic_log = round_dir / "semantic_validation.log"
            task = self.progress.start(f"{round_dir.name} · semantic validation")
            resolved_calls, semantic_issues = self.semantic_validator.validate_tree(task_dir)
            semantic_output = "\n".join(issue.render() for issue in semantic_issues)
            _write_log(semantic_log, semantic_output + ("\n" if semantic_output else ""))
            (round_dir / "resolved_api_calls.json").write_text(
                json.dumps([call.to_dict() for call in resolved_calls], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            task.finish(
                status="failed" if semantic_issues else "passed",
                detail=f"calls={len(resolved_calls)}, issues={len(semantic_issues)}",
            )
            if semantic_issues:
                return _failure(
                    compiled=False,
                    correctness=False,
                    stage="semantic_validation",
                    code="semantic_validation_failed",
                    message="AscendC semantic validation failed",
                    output=semantic_output,
                    details_path=semantic_log,
                    compile_output=semantic_output,
                )

        validator = REPO_ROOT / "skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py"
        static_log = round_dir / "static_validation.log"
        task = self.progress.start(f"{round_dir.name} · static validation")
        rc, static_output = _run(
            [
                python,
                str(validator),
                str(task_dir / "model_new_ascendc.py"),
                "--pybind-file",
                str(task_dir / "kernel" / "pybind11.cpp"),
            ],
            env=env,
            timeout=min(self.timeout, 120),
            log_path=static_log,
        )
        task.finish(status="passed" if rc == 0 else "failed", detail=f"exit_code={rc}")
        if rc != 0:
            static_full_output = _full_log(static_log, static_output)
            return _failure(
                compiled=False,
                correctness=False,
                stage="static_validation",
                code="static_validation_failed",
                message="static validation failed",
                output=static_full_output,
                details_path=static_log,
                compile_output=static_full_output,
                failure_kind=_return_code_kind(rc),
            )

        build_log = round_dir / "build.log"
        task = self.progress.start(f"{round_dir.name} · AscendC build")
        rc, build_output = _run(
            [python, str(REPO_ROOT / "utils/build_ascendc.py"), str(task_dir), "-v", self.soc_version, "--clean"],
            env=env,
            timeout=self.timeout,
            log_path=build_log,
        )
        task.finish(status="passed" if rc == 0 else "failed", detail=f"exit_code={rc}")
        compile_output = f"{static_output}\n{build_output}".strip()
        if rc != 0:
            build_full_output = _full_log(build_log, build_output)
            return _failure(
                compiled=False,
                correctness=False,
                stage="ascendc_build",
                code="ascendc_build_failed",
                message="AscendC build failed",
                output=build_full_output,
                details_path=build_log,
                compile_output=f"{static_output}\n{build_full_output}".strip(),
                failure_kind=_return_code_kind(rc),
            )

        correctness_log = round_dir / "correctness.log"
        task = self.progress.start(f"{round_dir.name} · correctness")
        rc, verify_output = _run(
            [python, str(REPO_ROOT / "utils/verification_ascendc.py"), str(task_dir)],
            env=env,
            timeout=self.timeout,
            log_path=correctness_log,
        )
        task.finish(status="passed" if rc == 0 else "failed", detail=f"exit_code={rc}")
        if rc != 0:
            verify_full_output = _full_log(correctness_log, verify_output)
            return _failure(
                compiled=True,
                correctness=False,
                stage="correctness",
                code="correctness_failed",
                message="correctness verification failed",
                output=verify_full_output,
                details_path=correctness_log,
                compile_output=compile_output,
                verify_output=verify_full_output,
                failure_kind=_return_code_kind(rc),
            )

        perf_path = round_dir / "performance.json"
        performance_log = round_dir / "performance.log"
        task = self.progress.start(f"{round_dir.name} · performance")
        rc, perf_output = _run(
            [
                python,
                str(REPO_ROOT / "skills/ascendc/performance-analyzer/references/performance.py"),
                "--output_dir",
                str(task_dir),
                "--warmup",
                "10",
                "--repeats",
                "50",
                "--output",
                str(perf_path),
            ],
            env=env,
            timeout=self.timeout,
            log_path=performance_log,
        )
        perf_ok = rc == 0 and perf_path.is_file()
        task.finish(status="passed" if perf_ok else "failed", detail=f"exit_code={rc}")
        if rc != 0 or not perf_path.is_file():
            failure_output = perf_output or "performance.json was not produced"
            if not perf_path.is_file():
                _append_log(performance_log, "\nperformance.json was not produced\n")
            failure_output = _full_log(performance_log, failure_output)
            return _failure(
                compiled=True,
                correctness=True,
                stage="performance",
                code="performance_failed",
                message="performance evaluation failed",
                output=failure_output,
                details_path=performance_log,
                compile_output=compile_output,
                verify_output=verify_output,
                failure_kind=_return_code_kind(rc),
            )
        try:
            performance = json.loads(perf_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            message = f"invalid performance result: {error}"
            _append_log(performance_log, f"\n{message}\n")
            return _failure(
                compiled=True,
                correctness=True,
                stage="performance",
                code="performance_result_invalid",
                message="performance evaluation failed",
                output=message,
                details_path=performance_log,
                compile_output=compile_output,
                verify_output=verify_output,
            )
        speedups = [
            item.get("speedup")
            for item in performance.get("per_case_speedup", [])
            if isinstance(item.get("speedup"), (int, float)) and item.get("speedup") > 0
        ]
        if not speedups:
            message = "performance result contains no positive numeric per-case speedup"
            _append_log(performance_log, f"\n{message}\n")
            return _failure(
                compiled=True,
                correctness=True,
                stage="performance",
                code="performance_score_missing",
                message="performance evaluation produced no valid score",
                output=message,
                details_path=performance_log,
                compile_output=compile_output,
                verify_output=verify_output,
            )
        score = math.exp(sum(math.log(value) for value in speedups) / len(speedups))
        return EvalResult(True, True, score=score, compile_output=compile_output, verify_output=verify_output, performance=performance)


class MockEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult:
        self.calls += 1
        try:
            validate_initial_bundle(capture_bundle(task_dir))
        except ValueError as error:
            message = str(error)
            return EvalResult(
                False,
                False,
                error=message,
                failure_stage="bundle_validation",
                failure_code="bundle_validation_failed",
                error_excerpt=message,
            )
        score = 1.0 + self.calls * 0.1
        return EvalResult(
            compiled=True,
            correctness=True,
            score=score,
            compile_output="mock compile passed",
            verify_output="mock verification passed",
            performance={"overall_speedup": score, "mock": True},
        )
