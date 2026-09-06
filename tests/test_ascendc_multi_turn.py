from __future__ import annotations

import io
import json
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ascendc_multi_turn.bundle import parse_file_bundle
from ascendc_multi_turn.__main__ import main
from ascendc_multi_turn.evaluator import LocalAscendEvaluator, MockEvaluator, _run, extract_error_excerpt
from ascendc_multi_turn.llm import MockProvider
from ascendc_multi_turn.models import EvalResult, LLMResponse, RunConfig
from ascendc_multi_turn.progress import ProgressReporter
from ascendc_multi_turn.runner import MultiTurnRunner


class TruncatingProvider:
    def __init__(self, retry_content: str, *, retry_finish_reason: str | None = "stop"):
        self.calls = 0
        self.retry_content = retry_content
        self.retry_finish_reason = retry_finish_reason

    def generate(self, prompt: str) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            content = json.dumps(
                {
                    "skill": "ascendc-translator",
                    "topics": [],
                    "api_names": [],
                    "supplements": [],
                    "reason": "test",
                }
            )
            finish_reason = "stop"
        elif self.calls == 2:
            content = '{"analysis":"truncated"'
            finish_reason = "length"
        else:
            content = self.retry_content
            finish_reason = self.retry_finish_reason
        return LLMResponse(
            content=content,
            model="test-model",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            finish_reason=finish_reason,
        )


class FailingProvider:
    def generate(self, prompt: str) -> LLMResponse:
        raise RuntimeError("simulated API outage")


class EmptyRouterThenSuccessProvider:
    def __init__(self):
        self.calls = 0
        self.call_configs = []

    def generate(self, prompt: str, *, call_config=None) -> LLMResponse:
        self.calls += 1
        self.call_configs.append(call_config)
        if self.calls == 1:
            return LLMResponse(
                content="",
                reasoning_content="router reasoning exhausted the response",
                model="served-v4-flash",
                requested_model="deepseek-v4-flash",
                finish_reason="length",
                usage={
                    "completion_tokens": 4096,
                    "total_tokens": 4100,
                    "completion_tokens_details": {"reasoning_tokens": 4096},
                },
                request_options={"thinking_requested": "disabled", "max_tokens": 4096},
            )
        if self.calls == 2:
            content = json.dumps(
                {
                    "skill": "ascendc-translator",
                    "topics": [],
                    "doc_ids": [],
                    "supplements": [],
                    "reason": "retry succeeded",
                }
            )
        else:
            content = _valid_bundle_response()
        return LLMResponse(
            content=content,
            model="served-v4-flash",
            requested_model="deepseek-v4-flash",
            finish_reason="stop",
            usage={"total_tokens": 2},
        )


class SuccessThenFailureEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult:
        self.calls += 1
        if self.calls == 1:
            return EvalResult(True, True, score=1.5)
        error_log = round_dir / "build.log"
        error_log.write_text("kernel.cpp:10: error: simulated build error\n", encoding="utf-8")
        return EvalResult(
            False,
            False,
            error="AscendC build failed",
            failure_stage="ascendc_build",
            failure_code="ascendc_build_failed",
            error_excerpt="kernel.cpp:10: error: simulated build error",
            details_path=str(error_log.resolve()),
        )


class UnscoredEvaluator:
    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult:
        return EvalResult(True, True, score=None)


class CompilerRepairProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, prompt: str) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            content = json.dumps(
                {
                    "skill": "ascendc-translator",
                    "topics": ["vector"],
                    "doc_ids": [],
                    "supplements": [],
                    "reason": "test",
                }
            )
        elif self.calls == 2:
            content = _valid_bundle_response()
        else:
            content = json.dumps(
                {
                    "analysis": "use the runtime field spelling",
                    "files": [
                        {
                            "path": "kernel/retry.cpp",
                            "content": "// repaired: DataCopyPadExtParams::paddingValue\n",
                        }
                    ],
                    "delete": [],
                }
            )
        return LLMResponse(
            content=content,
            model="test-model",
            usage={"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            finish_reason="stop",
        )


class FailThenSuccessEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult:
        self.calls += 1
        if self.calls == 1:
            log = round_dir / "build.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(
                "kernel/retry.cpp:9: error: no member named 'padValue' in "
                "'AscendC::DataCopyPadExtParams<float>'\n",
                encoding="utf-8",
            )
            return EvalResult(
                False,
                False,
                error="AscendC build failed",
                failure_stage="ascendc_build",
                failure_code="ascendc_build_failed",
                details_path=str(log.resolve()),
            )
        return EvalResult(True, True, score=1.25)


class RepairRegressionEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult:
        self.calls += 1
        log = round_dir / "build.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        diagnostics = ["kernel/retry.cpp:9: error: original API failure"]
        if self.calls == 2:
            diagnostics.extend(
                [
                    "kernel/retry.cpp:10: error: new duplicate declaration",
                    "kernel/retry.cpp:11: error: new invalid API owner",
                ]
            )
        log.write_text("\n".join(diagnostics), encoding="utf-8")
        return EvalResult(
            False,
            False,
            error="AscendC build failed",
            failure_stage="ascendc_build",
            failure_code="ascendc_build_failed",
            error_excerpt="\n".join(diagnostics),
            details_path=str(log.resolve()),
        )


class RepeatingBuildFailureEvaluator:
    def evaluate(self, task_dir: Path, round_dir: Path) -> EvalResult:
        log = round_dir / "build.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        diagnostic = "kernel/retry.cpp:9: error: repeated unknown compiler failure"
        log.write_text(diagnostic, encoding="utf-8")
        return EvalResult(
            False,
            False,
            error="AscendC build failed",
            failure_stage="ascendc_build",
            failure_code="ascendc_build_failed",
            error_excerpt=diagnostic,
            details_path=str(log.resolve()),
        )


def _valid_bundle_response() -> str:
    return json.dumps(
        {
            "analysis": "compact retry",
            "files": [
                {"path": "model_new_ascendc.py", "content": "class ModelNew: pass\n"},
                {
                    "path": "kernel/pybind11.cpp",
                    "content": "#include <pybind11/pybind11.h>\nPYBIND11_MODULE(_retry_ext, m) {}\n",
                },
                {"path": "kernel/retry.cpp", "content": "// AscendC retry kernel\n"},
            ],
            "delete": [],
        }
    )


class BundleTests(unittest.TestCase):
    def test_rejects_path_traversal(self) -> None:
        payload = {"files": [{"path": "../escape.cpp", "content": "bad"}]}
        with self.assertRaisesRegex(ValueError, "unsafe file path"):
            parse_file_bundle(json.dumps(payload))


class EvaluatorTests(unittest.TestCase):
    def test_static_validator_receives_matching_pybind_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary)
            kernel_dir = task_dir / "kernel"
            kernel_dir.mkdir()
            (task_dir / "model_new_ascendc.py").write_text("class ModelNew: pass\n", encoding="utf-8")
            (kernel_dir / "pybind11.cpp").write_text(
                "PYBIND11_MODULE(_test_ext, m) {}\n", encoding="utf-8"
            )
            (kernel_dir / "test.cpp").write_text("// kernel\n", encoding="utf-8")
            evaluator = LocalAscendEvaluator(device=0, soc_version="Ascend910B3")

            with patch("ascendc_multi_turn.evaluator._run", return_value=(1, "expected stop")) as run:
                result = evaluator.evaluate(task_dir, task_dir / "round")

            command = run.call_args.args[0]
            self.assertEqual(result.error, "static validation failed")
            self.assertIn("--pybind-file", command)
            self.assertEqual(
                command[command.index("--pybind-file") + 1],
                str(kernel_dir / "pybind11.cpp"),
            )

    def test_build_failure_has_structured_diagnostics_and_full_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary) / "task"
            round_dir = task_dir / ".llm_state" / "round_07"
            kernel_dir = task_dir / "kernel"
            kernel_dir.mkdir(parents=True)
            round_dir.mkdir(parents=True)
            (task_dir / "model_new_ascendc.py").write_text("class ModelNew: pass\n", encoding="utf-8")
            (kernel_dir / "pybind11.cpp").write_text("PYBIND11_MODULE(_gelu, m) {}\n", encoding="utf-8")
            (kernel_dir / "gelu.cpp").write_text("// kernel\n", encoding="utf-8")
            build_output = "\n".join(
                [
                    "gelu_kernel.cpp:25:9: error: use of undeclared identifier 'CopyTiling'",
                    "gelu_kernel.cpp:67:18: error: no member named 'Get' in 'GlobalTensor<unsigned char>'",
                    "gelu_kernel.cpp:68:14: error: no member named 'Barrier' in 'TPipe'",
                    "gelu_kernel.cpp:101:5: error: too few arguments to function call",
                ]
            )

            def fake_run(command, *, env, timeout, log_path=None):
                output = "static passed" if "validate_ascendc_impl.py" in " ".join(command) else build_output
                if log_path is not None:
                    log_path.write_text(output, encoding="utf-8")
                return (0, output) if output == "static passed" else (1, output)

            evaluator = LocalAscendEvaluator(device=0, soc_version="Ascend910B3")
            with patch("ascendc_multi_turn.evaluator._run", side_effect=fake_run):
                result = evaluator.evaluate(task_dir, round_dir)

            self.assertEqual(result.failure_stage, "ascendc_build")
            self.assertEqual(result.failure_code, "ascendc_build_failed")
            self.assertIn("CopyTiling", result.error_excerpt)
            self.assertIn("Barrier", result.error_excerpt)
            self.assertEqual(result.details_path, str((round_dir / "build.log").resolve()))
            self.assertEqual((round_dir / "build.log").read_text(encoding="utf-8"), build_output)

    def test_generated_source_guard_stops_before_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task_dir = Path(temporary) / "task"
            round_dir = task_dir / ".llm_state" / "round_01"
            kernel_dir = task_dir / "kernel"
            kernel_dir.mkdir(parents=True)
            round_dir.mkdir(parents=True)
            (task_dir / "model_new_ascendc.py").write_text("class ModelNew: pass\n", encoding="utf-8")
            (kernel_dir / "pybind11.cpp").write_text("PYBIND11_MODULE(_test, m) {}\n", encoding="utf-8")
            (kernel_dir / "test.cpp").write_text(
                "AscendC::TPipe pipe_;\nvoid Run() { pipe_.EnQue(1); }\n",
                encoding="utf-8",
            )
            evaluator = LocalAscendEvaluator(device=0, soc_version="Ascend910B3")

            with patch("ascendc_multi_turn.evaluator._run", return_value=(0, "static passed")) as run:
                result = evaluator.evaluate(task_dir, round_dir)

            self.assertEqual(run.call_count, 1)
            self.assertEqual(result.failure_stage, "ascendc_source_validation")
            self.assertIn("TPipe does not own EnQue", result.error_excerpt)

    def test_command_log_keeps_full_output_while_result_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "stage.log"
            full_output = "prefix\n" + ("x" * 25000) + "\nerror: final failure\n"
            completed = subprocess.CompletedProcess(["tool"], 1, stdout=full_output)
            with patch("ascendc_multi_turn.evaluator.subprocess.run", return_value=completed):
                rc, output = _run(["tool"], env={}, timeout=1, log_path=log_path)

            self.assertEqual(rc, 1)
            self.assertLessEqual(len(output), 20000)
            self.assertEqual(log_path.read_text(encoding="utf-8"), full_output)

    def test_timeout_log_keeps_partial_output_and_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            log_path = Path(temporary) / "timeout.log"
            timeout = subprocess.TimeoutExpired(["tool"], 3, output=b"partial output\n")
            with patch("ascendc_multi_turn.evaluator.subprocess.run", side_effect=timeout):
                rc, output = _run(["tool"], env={}, timeout=3, log_path=log_path)

            self.assertEqual(rc, 124)
            self.assertIn("partial output", output)
            self.assertIn("Timed out after 3s", output)
            self.assertEqual(log_path.read_text(encoding="utf-8"), output)

    def test_static_correctness_and_performance_failures_are_classified(self) -> None:
        scenarios = {
            "static_validation": [1],
            "correctness": [0, 0, 1],
            "performance": [0, 0, 0, 1],
        }
        for expected_stage, return_codes in scenarios.items():
            with self.subTest(stage=expected_stage), tempfile.TemporaryDirectory() as temporary:
                task_dir = Path(temporary) / "task"
                round_dir = task_dir / ".llm_state" / "round_01"
                kernel_dir = task_dir / "kernel"
                kernel_dir.mkdir(parents=True)
                round_dir.mkdir(parents=True)
                (task_dir / "model_new_ascendc.py").write_text("class ModelNew: pass\n", encoding="utf-8")
                (kernel_dir / "pybind11.cpp").write_text("PYBIND11_MODULE(_test, m) {}\n", encoding="utf-8")
                (kernel_dir / "test.cpp").write_text("// kernel\n", encoding="utf-8")
                codes = iter(return_codes)

                def fake_run(command, *, env, timeout, log_path=None):
                    rc = next(codes)
                    output = "stage passed" if rc == 0 else f"error: simulated {expected_stage} failure"
                    if log_path is not None:
                        log_path.write_text(output, encoding="utf-8")
                    return rc, output

                evaluator = LocalAscendEvaluator(device=0, soc_version="Ascend910B3")
                with patch("ascendc_multi_turn.evaluator._run", side_effect=fake_run):
                    result = evaluator.evaluate(task_dir, round_dir)

                self.assertEqual(result.failure_stage, expected_stage)
                self.assertTrue(result.failure_code.endswith("failed"))
                self.assertTrue(Path(result.details_path).is_file())

    def test_error_excerpt_prefers_compiler_errors(self) -> None:
        output = "\n".join(
            ["configure noise"] * 50
            + [
                "gelu.cpp:25: error: CopyTiling was not declared",
                "gelu.cpp:67: error: GlobalTensor<uint8_t> has no member Get<T>",
                "gelu.cpp:68: error: TPipe has no member Barrier",
                "gelu.cpp:101: error: too few arguments",
            ]
            + ["build noise"] * 50
        )

        excerpt = extract_error_excerpt(output)

        self.assertIn("CopyTiling", excerpt)
        self.assertIn("Get<T>", excerpt)
        self.assertIn("Barrier", excerpt)
        self.assertIn("too few arguments", excerpt)
        self.assertNotIn("configure noise", excerpt)
        self.assertEqual(excerpt.count("CopyTiling"), 1)


class ProgressTests(unittest.TestCase):
    def test_heartbeat_is_emitted_and_thread_stops(self) -> None:
        stream = io.StringIO()
        reporter = ProgressReporter(enabled=True, stream=stream, heartbeat_interval=0.01)
        task = reporter.start("slow stage")
        time.sleep(0.035)
        task.finish(detail="ok")

        output = stream.getvalue()
        self.assertIn("slow stage · started", output)
        self.assertIn("still running", output)
        self.assertIn("slow stage · completed", output)
        self.assertIsNotNone(task._thread)
        self.assertFalse(task._thread.is_alive())


class RunnerTests(unittest.TestCase):
    def test_mock_end_to_end_keeps_best_and_counts_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "1_identity.py"
            source.write_text(
                "import torch\nimport torch.nn as nn\n"
                "class Model(nn.Module):\n    def forward(self, x): return x\n"
                "def get_inputs(): return [torch.randn(4)]\n"
                "def get_init_inputs(): return []\n",
                encoding="utf-8",
            )
            source.with_suffix(".json").write_text('{"shape": [4]}\n', encoding="utf-8")
            output = root / "generated"
            config = RunConfig(str(source), str(output), max_rounds=3, evaluator="mock", mock=True)

            summary = MultiTurnRunner(config, MockProvider(), MockEvaluator()).run()

            self.assertTrue(summary["success"])
            self.assertEqual(summary["rounds_completed"], 3)
            self.assertEqual(summary["best_round"], 3)
            self.assertAlmostEqual(summary["best_score"], 1.3)
            self.assertGreater(summary["token_usage"]["total_tokens"], 0)
            self.assertTrue((output / "model_new_ascendc.py").is_file())
            self.assertTrue((output / "kernel" / "mock.cpp").is_file())
            self.assertTrue((output / ".llm_state" / "summary.json").is_file())
            self.assertTrue((output / ".llm_state" / "round_01" / "selected_knowledge.json").is_file())
            self.assertTrue((output / ".llm_state" / "round_01" / "references.md").is_file())
            self.assertTrue((output / ".llm_state" / "round_01" / "runtime_header_facts.json").is_file())
            calls = (output / ".llm_state" / "calls.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"content"', calls)
            self.assertIn('"call_type": "knowledge_router"', calls)
            self.assertIn('"call_type": "generator"', calls)
            call_records = [json.loads(line) for line in calls.splitlines()]
            self.assertEqual(
                sum(item["call_type"] == "knowledge_router" for item in call_records),
                1,
            )
            self.assertTrue((output / ".llm_state" / "knowledge_state.json").is_file())
            self.assertTrue((output / ".llm_state" / "round_02" / "knowledge_reuse.json").is_file())
            self.assertEqual(summary["last_round"]["decision"], "KEEP")
            self.assertIsNone(summary["failure"])

    def test_build_failure_gets_one_same_round_compiler_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            provider = CompilerRepairProvider()
            evaluator = FailThenSuccessEvaluator()
            config = RunConfig(str(source), str(output), max_rounds=1, evaluator="mock", mock=True)

            summary = MultiTurnRunner(config, provider, evaluator).run()

            self.assertTrue(summary["success"])
            self.assertEqual(provider.calls, 3)
            self.assertEqual(evaluator.calls, 2)
            repair_dir = output / ".llm_state" / "round_01" / "repair_01"
            self.assertTrue((repair_dir / "prompt.txt").is_file())
            self.assertTrue((repair_dir / "response.txt").is_file())
            self.assertTrue((repair_dir / "candidate.json").is_file())
            trajectory = json.loads(
                (output / ".llm_state" / "trajectory.json").read_text(encoding="utf-8")
            )
            record = trajectory["rounds"][0]
            self.assertEqual(record["compile_repair_attempts"], 1)
            self.assertEqual(len(record["evaluation_attempts"]), 2)
            calls = [
                json.loads(line)["call_type"]
                for line in (output / ".llm_state" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(calls, ["knowledge_router", "generator", "compile_repair"])

    def test_regressing_compiler_repair_restores_the_pre_repair_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            config = RunConfig(str(source), str(output), max_rounds=1, evaluator="mock", mock=True)

            summary = MultiTurnRunner(
                config, CompilerRepairProvider(), RepairRegressionEvaluator()
            ).run()

            self.assertFalse(summary["success"])
            self.assertEqual(summary["rounds_completed"], 1)
            self.assertIn(
                "original API failure",
                summary["last_evaluation_failure"]["excerpt"],
            )
            self.assertEqual(
                (output / "kernel" / "retry.cpp").read_text(encoding="utf-8"),
                "// AscendC retry kernel\n",
            )
            trajectory = json.loads(
                (output / ".llm_state" / "trajectory.json").read_text(encoding="utf-8")
            )
            self.assertIn("restored pre-repair", trajectory["rounds"][0]["repair_error"])
            self.assertTrue(
                (output / ".llm_state" / "round_01" / "repair_01" / "repair_regression.log").is_file()
            )

    def test_repeated_failure_never_reopens_the_full_knowledge_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            config = RunConfig(str(source), str(output), max_rounds=3, evaluator="mock", mock=True)

            summary = MultiTurnRunner(
                config, MockProvider(), RepeatingBuildFailureEvaluator()
            ).run()

            self.assertEqual(summary["rounds_completed"], 3)
            calls = [
                json.loads(line)
                for line in (output / ".llm_state" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(sum(item["call_type"] == "knowledge_router" for item in calls), 1)
            state = json.loads(
                (output / ".llm_state" / "knowledge_state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["full_route_count"], 1)

    def test_refuses_nonempty_output_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            output.mkdir()
            (output / "owned.txt").write_text("do not overwrite", encoding="utf-8")
            config = RunConfig(str(source), str(output), max_rounds=1, evaluator="mock", mock=True)
            with self.assertRaisesRegex(ValueError, "not empty"):
                MultiTurnRunner(config, MockProvider(), MockEvaluator()).run()

    def test_resume_continues_from_next_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            first = RunConfig(str(source), str(output), max_rounds=1, evaluator="mock", mock=True)
            MultiTurnRunner(first, MockProvider(), MockEvaluator()).run()

            resumed = RunConfig(str(source), str(output), max_rounds=2, evaluator="mock", mock=True, resume=True)
            summary = MultiTurnRunner(resumed, MockProvider(), MockEvaluator()).run()

            self.assertEqual(summary["rounds_completed"], 2)
            trajectory = json.loads((output / ".llm_state" / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual([item["round"] for item in trajectory["rounds"]], [1, 2])

    def test_resume_migrates_legacy_non_evaluation_failures_without_id_collision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            first = RunConfig(str(source), str(output), max_rounds=1, evaluator="mock", mock=True)
            MultiTurnRunner(first, MockProvider(), MockEvaluator()).run()

            trajectory_path = output / ".llm_state" / "trajectory.json"
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            trajectory["rounds"].append(
                {
                    "round": 2,
                    "decision": "LLM_FAIL",
                    # Historical orchestration failures could contain a
                    # synthetic evaluation_attempts entry without ever
                    # reaching the evaluator.
                    "evaluation_attempts": [
                        {
                            "compiled": False,
                            "correctness": False,
                            "error": "legacy API outage",
                        }
                    ],
                    "evaluation": {
                        "compiled": False,
                        "correctness": False,
                        "error": "legacy API outage",
                        "failure_stage": "llm_generator",
                    },
                }
            )
            trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
            (output / ".llm_state" / "knowledge_state.json").unlink()

            resumed = RunConfig(
                str(source), str(output), max_rounds=2, evaluator="mock", mock=True, resume=True
            )
            provider = MockProvider()
            summary = MultiTurnRunner(resumed, provider, MockEvaluator()).run()

            self.assertEqual(summary["rounds_completed"], 2)
            self.assertEqual(summary["historical_records"], 3)
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
            self.assertEqual([item["round"] for item in trajectory["rounds"]], [1, 2, 3])
            self.assertEqual(trajectory["rounds"][-1]["evaluation_round"], 2)
            self.assertTrue((output / ".llm_state" / "round_03" / "candidate.json").is_file())
            self.assertEqual(provider.calls, 1)
            self.assertEqual(trajectory["status"], "completed")

    def test_truncated_generation_retries_once_in_same_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            provider = TruncatingProvider(_valid_bundle_response())
            config = RunConfig(str(source), str(output), max_rounds=1, evaluator="mock", mock=True)

            summary = MultiTurnRunner(config, provider, MockEvaluator()).run()

            self.assertTrue(summary["success"])
            self.assertEqual(provider.calls, 3)
            self.assertTrue((output / ".llm_state" / "round_01" / "retry_prompt.txt").is_file())
            self.assertTrue((output / ".llm_state" / "round_01" / "response_retry_01.txt").is_file())
            calls = [
                json.loads(line)
                for line in (output / ".llm_state" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [call["call_type"] for call in calls],
                ["knowledge_router", "generator", "generator_retry"],
            )
            self.assertEqual(summary["token_usage"]["total_tokens"], 6)
            trajectory = json.loads((output / ".llm_state" / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(trajectory["rounds"][0]["generation_attempts"], 2)
            self.assertEqual(trajectory["rounds"][0]["response_finish_reason"], "stop")

    def test_empty_reasoning_only_response_is_logged_before_same_checkpoint_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            provider = EmptyRouterThenSuccessProvider()
            config = RunConfig(
                str(source),
                str(output),
                max_rounds=1,
                evaluator="mock",
                mock=True,
                llm_transient_retries=1,
            )

            summary = MultiTurnRunner(config, provider, MockEvaluator()).run()

            self.assertTrue(summary["completed"])
            self.assertEqual(summary["rounds_completed"], 1)
            calls = [
                json.loads(line)
                for line in (output / ".llm_state" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([item["retry"] for item in calls[:2]], [0, 1])
            self.assertTrue(calls[0]["reasoning_content_present"])
            self.assertEqual(calls[0]["model"], "served-v4-flash")
            self.assertNotIn("reasoning_content", calls[0])
            self.assertEqual(provider.call_configs[0].call_type, "knowledge_router")

    def test_truncated_retry_failure_does_not_retry_forever(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            provider = TruncatingProvider("{", retry_finish_reason="length")
            config = RunConfig(str(source), str(output), max_rounds=1, evaluator="mock", mock=True)

            summary = MultiTurnRunner(config, provider, MockEvaluator()).run()

            self.assertFalse(summary["success"])
            self.assertEqual(provider.calls, 3)
            trajectory = json.loads((output / ".llm_state" / "trajectory.json").read_text(encoding="utf-8"))
            self.assertEqual(trajectory["rounds"], [])
            self.assertEqual(summary["rounds_completed"], 0)
            self.assertEqual(summary["status"], "paused")
            self.assertEqual(summary["failure"]["stage"], "response_format")
            self.assertTrue(Path(summary["failure"]["details_path"]).is_file())

    def test_failed_or_unscored_candidate_cannot_be_best(self) -> None:
        self.assertFalse(
            MultiTurnRunner._is_better(
                EvalResult(True, True, score=2.0, error="performance failed"),
                None,
            )
        )
        self.assertFalse(MultiTurnRunner._is_better(EvalResult(True, True, score=None), None))

    def test_unscored_evaluator_result_produces_failure_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            config = RunConfig(str(source), str(output), max_rounds=1, evaluator="mock", mock=True)

            summary = MultiTurnRunner(config, MockProvider(), UnscoredEvaluator()).run()

            self.assertFalse(summary["success"])
            self.assertEqual(summary["failure"]["stage"], "performance")
            self.assertEqual(summary["failure"]["code"], "performance_score_missing")

    def test_quiet_cli_keeps_stdout_as_pure_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "ascendc_multi_turn",
                "--op-file",
                str(source),
                "--output-dir",
                str(output),
                "--max-rounds",
                "1",
                "--mock",
                "--quiet",
            ]

            with patch("sys.argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["success"])
            self.assertEqual(stderr.getvalue(), "")

    def test_default_cli_writes_progress_only_to_stderr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            stdout = io.StringIO()
            stderr = io.StringIO()
            argv = [
                "ascendc_multi_turn",
                "--op-file",
                str(source),
                "--output-dir",
                str(output),
                "--max-rounds",
                "1",
                "--mock",
            ]

            with patch("sys.argv", argv), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main()

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["success"])
            self.assertIn("Evaluation 1/1", stderr.getvalue())
            self.assertNotIn("started", stdout.getvalue())

    def test_llm_exception_pauses_without_consuming_an_evaluation_round(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            config = RunConfig(str(source), str(output), max_rounds=1, evaluator="mock", mock=True)

            summary = MultiTurnRunner(config, FailingProvider(), MockEvaluator()).run()

            self.assertFalse(summary["success"])
            self.assertEqual(summary["rounds_completed"], 0)
            self.assertEqual(summary["status"], "paused")
            self.assertIsNone(summary["last_round"])
            self.assertEqual(summary["failure"]["stage"], "llm_knowledge_router")
            self.assertEqual(summary["orchestration_failures"], 1)
            self.assertTrue(Path(summary["failure"]["details_path"]).is_file())
            calls = [
                json.loads(line)
                for line in (output / ".llm_state" / "calls.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(calls), config.llm_transient_retries + 1)
            self.assertTrue(all(item["error"] == "simulated API outage" for item in calls))

    def test_resume_reuses_the_paused_checkpoint_attempt_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            failed = RunConfig(
                str(source),
                str(output),
                max_rounds=1,
                evaluator="mock",
                mock=True,
                llm_transient_retries=0,
            )
            first_summary = MultiTurnRunner(failed, FailingProvider(), MockEvaluator()).run()

            resumed = RunConfig(
                str(source),
                str(output),
                max_rounds=1,
                evaluator="mock",
                mock=True,
                resume=True,
            )
            second_summary = MultiTurnRunner(resumed, MockProvider(), MockEvaluator()).run()

            self.assertEqual(first_summary["pending_round"], 1)
            self.assertTrue(second_summary["completed"])
            self.assertEqual(second_summary["rounds_completed"], 1)
            self.assertEqual(second_summary["last_round"]["attempt_id"], 1)
            self.assertEqual(second_summary["orchestration_failures"], 1)

    def test_later_failure_is_visible_without_losing_an_existing_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            config = RunConfig(str(source), str(output), max_rounds=2, evaluator="mock", mock=True)

            summary = MultiTurnRunner(config, MockProvider(), SuccessThenFailureEvaluator()).run()

            self.assertTrue(summary["success"])
            self.assertEqual(summary["best_round"], 1)
            self.assertEqual(summary["last_round"]["decision"], "DISCARD")
            self.assertEqual(summary["failure"]["stage"], "ascendc_build")

    def test_legacy_failure_is_inferred_without_inventing_a_log_path(self) -> None:
        runner = object.__new__(MultiTurnRunner)
        runner.state_dir = Path("/path/that/does/not/exist")
        failure = runner._failure_summary(
            7,
            {
                "compiled": False,
                "correctness": False,
                "error": "AscendC build failed",
                "compile_output": "gelu.cpp:25: error: CopyTiling was not declared",
            },
        )

        self.assertEqual(failure["stage"], "ascendc_build")
        self.assertEqual(failure["code"], "ascendc_build_failed")
        self.assertIn("CopyTiling", failure["excerpt"])
        self.assertIsNone(failure["details_path"])


if __name__ == "__main__":
    unittest.main()
