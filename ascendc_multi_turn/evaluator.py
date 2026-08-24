from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol

from .bundle import capture_bundle, validate_initial_bundle
from .models import EvalResult


REPO_ROOT = Path(__file__).resolve().parent.parent


class Evaluator(Protocol):
    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult: ...


def _run(command: list[str], *, env: dict[str, str], timeout: int) -> tuple[int, str]:
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
        return completed.returncode, completed.stdout[-20000:]
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return 124, f"{output[-18000:]}\nTimed out after {timeout}s"


class LocalAscendEvaluator:
    def __init__(self, *, device: int, soc_version: str, timeout: int = 600):
        self.device = device
        self.soc_version = soc_version
        self.timeout = timeout

    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult:
        try:
            validate_initial_bundle(capture_bundle(task_dir))
        except ValueError as error:
            return EvalResult(False, False, error=str(error))

        env = os.environ.copy()
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(self.device)
        python = sys.executable
        validator = REPO_ROOT / "skills/ascendc/ascendc-translator/scripts/validate_ascendc_impl.py"
        rc, static_output = _run(
            [python, str(validator), str(task_dir / "model_new_ascendc.py")],
            env=env,
            timeout=min(self.timeout, 120),
        )
        if rc != 0:
            return EvalResult(False, False, compile_output=static_output, error="static validation failed")

        rc, build_output = _run(
            [python, str(REPO_ROOT / "utils/build_ascendc.py"), str(task_dir), "-v", self.soc_version, "--clean"],
            env=env,
            timeout=self.timeout,
        )
        compile_output = f"{static_output}\n{build_output}".strip()
        if rc != 0:
            return EvalResult(False, False, compile_output=compile_output, error="AscendC build failed")

        rc, verify_output = _run(
            [python, str(REPO_ROOT / "utils/verification_ascendc.py"), str(task_dir)],
            env=env,
            timeout=self.timeout,
        )
        if rc != 0:
            return EvalResult(True, False, compile_output=compile_output, verify_output=verify_output, error="correctness verification failed")

        perf_path = round_dir / "performance.json"
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
        )
        if rc != 0 or not perf_path.is_file():
            return EvalResult(True, True, compile_output=compile_output, verify_output=verify_output, error=f"performance evaluation failed\n{perf_output[-4000:]}")
        performance = json.loads(perf_path.read_text(encoding="utf-8"))
        speedups = [
            item.get("speedup")
            for item in performance.get("per_case_speedup", [])
            if isinstance(item.get("speedup"), (int, float)) and item.get("speedup") > 0
        ]
        score = math.exp(sum(math.log(value) for value in speedups) / len(speedups)) if speedups else None
        return EvalResult(True, True, score=score, compile_output=compile_output, verify_output=verify_output, performance=performance)


class MockEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult:
        self.calls += 1
        try:
            validate_initial_bundle(capture_bundle(task_dir))
        except ValueError as error:
            return EvalResult(False, False, error=str(error))
        score = 1.0 + self.calls * 0.1
        return EvalResult(
            compiled=True,
            correctness=True,
            score=score,
            compile_output="mock compile passed",
            verify_output="mock verification passed",
            performance={"overall_speedup": score, "mock": True},
        )
