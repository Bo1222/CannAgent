from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from .bundle import capture_bundle, validate_initial_bundle
from .case_profiles import build_case_profiles
from .diagnostics import parse_failure_evidence
from .models import EvalResult
from .progress import ProgressReporter
from .source_validation import render_issues, validate_source_tree

REPO_ROOT = Path(__file__).resolve().parent.parent


class Evaluator(Protocol):
    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult: ...


def benchmark_command(
    *,
    python: str,
    task_dir: Path,
    output_path: Path,
    warmup: int = 5,
    repeats: int = 20,
    profile_top_k: int = 3,
) -> list[str]:
    """Build the package-local benchmark command without consulting repository skills."""

    return [
        python,
        "-m",
        "ascendc_multi_turn.benchmark",
        "--task-dir",
        str(task_dir),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--profile-top-k",
        str(profile_top_k),
        "--output",
        str(output_path),
    ]


def validate_performance_report(
    payload: object,
    *,
    expected_case_indices: set[int] | None = None,
) -> float:
    if not isinstance(payload, dict):
        raise TypeError("performance report must be a JSON object")
    if payload.get("schema_version") != 2:
        raise ValueError(f"unsupported performance schema: {payload.get('schema_version')!r}")
    if payload.get("status") != "ok":
        raise ValueError(f"benchmark status is not ok: {payload.get('error') or payload.get('status')}")
    measurement = payload.get("measurement")
    if not isinstance(measurement, dict) or measurement.get("method") != "torch.npu.Event":
        raise ValueError("benchmark did not use torch.npu.Event")
    cases = payload.get("per_case_speedup")
    if not isinstance(cases, list) or not cases:
        raise ValueError("performance report contains no measured cases")
    indices: set[int] = set()
    speedups: list[float] = []
    for item in cases:
        if not isinstance(item, dict) or not isinstance(item.get("index"), int):
            raise TypeError("performance case lacks an integer index")
        index = int(item["index"])
        if index in indices:
            raise ValueError(f"duplicate performance case index: {index}")
        indices.add(index)
        for field in ("reference_ms", "ascendc_ms", "speedup"):
            value = item.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"case {index} has invalid {field}: {value!r}")
        speedups.append(float(item["speedup"]))
    if expected_case_indices is not None and indices != expected_case_indices:
        raise ValueError(
            "benchmark case set differs from full correctness: "
            f"expected={sorted(expected_case_indices)}, measured={sorted(indices)}"
        )
    computed = math.exp(math.fsum(math.log(value) for value in speedups) / len(speedups))
    for field in ("score", "overall_speedup"):
        value = payload.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"performance report has invalid {field}: {value!r}")
        if not math.isclose(float(value), computed, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f"performance report {field} does not match per-case geometric mean")
    return computed


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


_ERROR_LINE = re.compile(
    r"error:|fatal(?: error)?:|traceback|\[fail(?:ed)?\]|timed out|"
    r"507001|507035|acl_error|mte|aicore exception|device exception|"
    r"sigsegv|terminated by signal",
    re.IGNORECASE,
)
_LOW_INFORMATION_LINE = re.compile(r"^\s*(traceback|during handling of)", re.IGNORECASE)
_VOLATILE_LOG_TOKEN = re.compile(
    r"(?:0x[0-9a-f]+|\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?|"
    r"\b(?:pid|tid|task_id)\s*[:=]\s*\d+)",
    re.IGNORECASE,
)


def collapse_repeated_diagnostics(lines: list[str]) -> list[str]:
    """Collapse repeated device/compiler lines while retaining their first occurrence."""

    ordered: list[str] = []
    positions: dict[str, int] = {}
    counts: dict[str, int] = {}
    for line in lines:
        key = re.sub(r"\s+", " ", _VOLATILE_LOG_TOKEN.sub("<volatile>", line)).strip()
        if key in positions:
            counts[key] += 1
            continue
        positions[key] = len(ordered)
        counts[key] = 1
        ordered.append(line)
    for key, count in counts.items():
        if count > 1:
            index = positions[key]
            ordered[index] = f"{ordered[index]} [重复 {count} 次]"
    return ordered


def extract_error_excerpt(output: str, *, max_lines: int = 20, max_chars: int = 4000) -> str:
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    preferred = [line for line in lines if _ERROR_LINE.search(line)]
    candidates = preferred if preferred else lines[-max_lines:]
    candidates = sorted(
        collapse_repeated_diagnostics(candidates),
        key=lambda line: (
            bool(_LOW_INFORMATION_LINE.search(line)),
            not bool(re.search(r"(?:[^:\n]+):\d+(?::\d+)?:\s*(?:fatal\s+)?error:|507001|507035|ACL_ERROR|MTE", line, re.IGNORECASE)),
        ),
    )
    selected: list[str] = []
    for line in candidates:
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
    case_info: dict | None = None,
    active_profile: str | None = None,
    passed_profiles: list[str] | None = None,
    case_results: list[dict] | None = None,
    passed_case_indices: list[int] | None = None,
) -> EvalResult:
    failure_evidence = parse_failure_evidence(stage=stage, output=output, case_info=case_info)
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
        failure_evidence=failure_evidence.to_dict(),
        active_profile=active_profile,
        passed_profiles=list(passed_profiles or []),
        case_results=list(case_results or []),
        passed_case_indices=sorted(set(passed_case_indices or [])),
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
    ):
        self.device = device
        self.soc_version = soc_version
        self.timeout = timeout
        self.progress = progress or ProgressReporter()

    @staticmethod
    def _case_text(task_dir: Path) -> str:
        candidates = [
            path
            for path in sorted(task_dir.glob("*.json"))
            if not path.name.endswith("_all_case.json")
            and path.name not in {"performance.json"}
        ]
        if not candidates:
            return "(no JSON case file)"
        return candidates[0].read_text(encoding="utf-8")

    @staticmethod
    def _read_case_report(path: Path, *, profile: str) -> dict:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {
                "active_profile": profile,
                "case_results": [],
                "passed_case_indices": [],
                "failed_case_index": None,
            }
        return payload if isinstance(payload, dict) else {}

    def preflight(self, task_dir: Path, state_dir: Path) -> EvalResult | None:
        log_path = state_dir / "environment_preflight.log"
        failures: list[str] = []
        if shutil.which("cmake") is None:
            failures.append("cmake is not available on PATH")
        for path in (
            REPO_ROOT / "utils/verification_ascendc.py",
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
        except (TypeError, ValueError) as error:
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

        build_log = round_dir / "build.log"
        task = self.progress.start(f"{round_dir.name} · AscendC build")
        build_dir = task_dir / "build"
        configure_rc, configure_output = _run(
            ["cmake", "-S", str(task_dir), "-B", str(build_dir), f"-DSOC_VERSION={self.soc_version}"],
            env=env,
            timeout=self.timeout,
            log_path=build_log,
        )
        if configure_rc == 0:
            rc, build_output = _run(
                ["cmake", "--build", str(build_dir), "--parallel"],
                env=env,
                timeout=self.timeout,
                log_path=build_log,
            )
            build_output = f"{configure_output}\n{build_output}".strip()
        else:
            rc, build_output = configure_rc, configure_output
        task.finish(status="passed" if rc == 0 else "failed", detail=f"exit_code={rc}")
        compile_output = build_output.strip()
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
                compile_output=build_full_output.strip(),
                failure_kind=_return_code_kind(rc),
            )

        correctness_log = round_dir / "correctness.log"
        passed_profiles: list[str] = []
        passed_case_indices: set[int] = set()
        all_case_results: list[dict] = []
        verify_outputs: list[str] = []
        try:
            profiles = build_case_profiles(self._case_text(task_dir))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            profiles = []
            profile_error = f"invalid evaluation cases: {error}"
            _write_log(correctness_log, profile_error + "\n")
            return _failure(
                compiled=True,
                correctness=False,
                stage="correctness_profile",
                code="correctness_profile_invalid",
                message="correctness profile construction failed",
                output=profile_error,
                details_path=correctness_log,
                compile_output=compile_output,
            )

        for profile in (item for item in profiles if not item.run_performance):
            profile_log = round_dir / f"correctness_{profile.name}.log"
            report_path = round_dir / f"correctness_{profile.name}.json"
            command = [
                python,
                str(REPO_ROOT / "utils/verification_ascendc.py"),
                str(task_dir),
                "--profile",
                profile.name,
                "--report-json",
                str(report_path),
            ]
            if profile.case_indices:
                command.extend(
                    ["--case-indices", ",".join(str(index) for index in profile.case_indices)]
                )
            task = self.progress.start(
                f"{round_dir.name} · correctness/{profile.name}"
            )
            rc, verify_output = _run(
                command,
                env=env,
                timeout=self.timeout,
                log_path=profile_log,
            )
            task.finish(status="passed" if rc == 0 else "failed", detail=f"exit_code={rc}")
            full_output = _full_log(profile_log, verify_output)
            if rc < 0:
                signal_number = -rc
                try:
                    signal_name = signal.Signals(signal_number).name
                except ValueError:
                    signal_name = f"SIGNAL_{signal_number}"
                signal_line = (
                    "verification process terminated by signal "
                    f"{signal_number} ({signal_name}); exit_code={rc}"
                )
                _append_log(profile_log, f"\n{signal_line}\n")
                full_output = f"{full_output.rstrip()}\n{signal_line}\n"
            verify_outputs.append(f"## profile={profile.name}\n{full_output}".rstrip())
            _write_log(correctness_log, "\n\n".join(verify_outputs) + "\n")
            report = self._read_case_report(report_path, profile=profile.name)
            report_results = report.get("case_results", [])
            if isinstance(report_results, list):
                all_case_results.extend(
                    item for item in report_results if isinstance(item, dict)
                )
            report_passed = report.get("passed_case_indices", [])
            if isinstance(report_passed, list):
                passed_case_indices.update(
                    int(index) for index in report_passed if isinstance(index, int)
                )
            if rc != 0:
                case_info = {
                    "active_profile": profile.name,
                    "profile_features": list(profile.features),
                    "profile_case_indices": list(profile.case_indices),
                    "failed_case_index": report.get("failed_case_index"),
                    "passed_case_indices": sorted(passed_case_indices),
                    "case_results": report_results,
                }
                verify_full_output = "\n\n".join(verify_outputs)
                return _failure(
                    compiled=True,
                    correctness=False,
                    stage="correctness",
                    code="correctness_failed",
                    message=f"correctness verification failed in {profile.name} profile",
                    output=full_output,
                    details_path=profile_log,
                    compile_output=compile_output,
                    verify_output=verify_full_output,
                    failure_kind=_return_code_kind(rc),
                    case_info=case_info,
                    active_profile=profile.name,
                    passed_profiles=passed_profiles,
                    case_results=all_case_results,
                    passed_case_indices=sorted(passed_case_indices),
                )
            passed_profiles.append(profile.name)

        verify_output = "\n\n".join(verify_outputs)

        perf_path = round_dir / "performance.json"
        performance_log = round_dir / "performance.log"
        task = self.progress.start(f"{round_dir.name} · performance")
        rc, perf_output = _run(
            benchmark_command(
                python=python,
                task_dir=task_dir,
                output_path=perf_path,
            ),
            env=env,
            timeout=self.timeout,
            log_path=performance_log,
        )
        if not perf_path.is_file():
            task.finish(status="failed", detail=f"exit_code={rc}")
            failure_output = perf_output or "performance.json was not produced"
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
                active_profile="benchmark",
                passed_profiles=passed_profiles,
                case_results=all_case_results,
                passed_case_indices=sorted(passed_case_indices),
            )
        performance: object
        try:
            performance = json.loads(perf_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            task.finish(status="failed", detail=f"exit_code={rc}")
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
                active_profile="benchmark",
                passed_profiles=passed_profiles,
                case_results=all_case_results,
                passed_case_indices=sorted(passed_case_indices),
            )
        try:
            score = validate_performance_report(
                performance,
                expected_case_indices=set(passed_case_indices),
            )
        except (TypeError, ValueError) as error:
            task.finish(status="failed", detail=f"exit_code={rc}")
            message = f"invalid performance result: {error}"
            _append_log(performance_log, f"\n{message}\n")
            return _failure(
                compiled=True,
                correctness=True,
                stage="performance",
                code="performance_result_invalid",
                message="performance evaluation produced an invalid report",
                output=message,
                details_path=performance_log,
                compile_output=compile_output,
                verify_output=verify_output,
                active_profile="benchmark",
                passed_profiles=passed_profiles,
                case_results=all_case_results,
                passed_case_indices=sorted(passed_case_indices),
            )
        assert isinstance(performance, dict)
        if rc != 0:
            diagnostics = performance.setdefault("diagnostics", {})
            if isinstance(diagnostics, dict):
                diagnostics["benchmark_process_exit_code"] = rc
                diagnostics["benchmark_process_note"] = (
                    "timing report was complete; supplementary profiling process did not exit cleanly"
                )
            _append_log(
                performance_log,
                "\nAccepted complete Event timing report despite supplementary profiler "
                f"exit_code={rc}.\n",
            )
            temporary = perf_path.with_suffix(perf_path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(performance, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(perf_path)
        task.finish(status="passed", detail=f"exit_code={rc}; score={score:.6g}")
        return EvalResult(
            True,
            True,
            score=score,
            compile_output=compile_output,
            verify_output=verify_output,
            performance=performance,
            active_profile="benchmark",
            passed_profiles=[*passed_profiles, "benchmark"],
            case_results=all_case_results,
            passed_case_indices=sorted(passed_case_indices),
        )


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
