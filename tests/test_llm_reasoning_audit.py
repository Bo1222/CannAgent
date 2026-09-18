from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ascendc_multi_turn.evaluator import MockEvaluator
from ascendc_multi_turn.llm import MockProvider
from ascendc_multi_turn.llm_diagnostic import run_diagnostics
from ascendc_multi_turn.logging import TrajectoryLogger
from ascendc_multi_turn.models import LLMResponse, RunConfig
from ascendc_multi_turn.runner import LLMCallFailure, MultiTurnRunner


def _project_bundle_text() -> str:
    files = {
        "model_new_ascendc.py": "class ModelNew: pass\n",
        "op_kernel/add_tiling.h": "#pragma once\n",
        "op_kernel/add_kernel.asc": "// kernel\n",
        "op_host/add.asc": "// host\n",
        "op_host/data_utils.h": "#pragma once\n",
        "op_extension/add_torch.cpp": "// torch\n",
        "op_extension/register.cpp": "// registration\n",
        "op_extension/ops.h": "#pragma once\n",
        "scripts/golden.py": "# golden\n",
        "scripts/test_torch.py": "# test\n",
        "CMakeLists.txt": "project(add LANGUAGES ASC CXX)\n",
    }
    return json.dumps(
        {
            "analysis": "diagnostic fixture",
            "files": [{"path": path, "content": content} for path, content in files.items()],
            "delete": [],
        }
    )


class _LengthProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, *, call_config=None):
        self.calls += 1
        return LLMResponse(
            content="",
            reasoning_content="反复推理但没有最终答案",
            model="deepseek-flash",
            requested_model="deepseek-flash",
            finish_reason="length",
            usage={
                "completion_tokens": 65536,
                "completion_tokens_details": {"reasoning_tokens": 65536},
            },
        )


class _TransportThenSuccessProvider:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, prompt, *, call_config=None):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("temporary transport failure")
        return LLMResponse(content="{}", model="mock", requested_model="mock")


class _PlannerThenLengthProvider:
    def __init__(self) -> None:
        self.mock = MockProvider()
        self.generator_calls = 0

    def generate(self, prompt, *, call_config=None):
        if call_config and call_config.call_type == "planner":
            return self.mock.generate(prompt, call_config=call_config)
        self.generator_calls += 1
        return LLMResponse(
            content="",
            reasoning_content="生成器推理耗尽",
            model="deepseek-flash",
            requested_model="deepseek-flash",
            finish_reason="length",
            usage={"completion_tokens": 32, "completion_tokens_details": {"reasoning_tokens": 32}},
        )


class _PlannerThenPartialLengthProvider:
    def __init__(self) -> None:
        self.mock = MockProvider()
        self.generator_calls = 0

    def generate(self, prompt, *, call_config=None):
        if call_config and call_config.call_type == "planner":
            return self.mock.generate(prompt, call_config=call_config)
        self.generator_calls += 1
        return LLMResponse(
            content='{"analysis":"truncated"',
            reasoning_content=f"第 {self.generator_calls} 次截断推理",
            model="deepseek-flash",
            requested_model="deepseek-flash",
            finish_reason="length",
            usage={"completion_tokens": 32, "completion_tokens_details": {"reasoning_tokens": 16}},
        )


class _DiagnosticProvider:
    def __init__(self) -> None:
        self.call_types: list[str] = []

    def list_models(self):
        return {"object": "list", "data": [{"id": "deepseek-flash", "object": "model"}]}

    def generate(self, prompt, *, call_config=None):
        self.call_types.append(call_config.call_type)
        is_real = prompt == "REAL PROMPT"
        if is_real and call_config.thinking == "disabled":
            content = _project_bundle_text()
            reasoning = ""
        elif is_real:
            content = ""
            reasoning = "检查真实工程约束并持续推理"
        else:
            content = '{"ok":true,"purpose":"deepseek api diagnostic"}'
            reasoning = "先确认 JSON 结构" if call_config.thinking == "enabled" else ""
        return LLMResponse(
            content=content,
            reasoning_content=reasoning,
            model="deepseek-flash",
            requested_model="deepseek-flash",
            finish_reason="length" if is_real and call_config.thinking == "enabled" else "stop",
            latency_seconds=0.1,
            usage={"completion_tokens": 10},
        )


class ReasoningAuditTests(unittest.TestCase):
    def _config(self, root: Path, **overrides) -> RunConfig:
        values = {
            "op_file": str(root / "model.py"),
            "output_dir": str(root / "task"),
            "mock": True,
            "model": "deepseek-flash",
            "generator_thinking": "disabled",
            "reasoning_log_mode": "full",
            "llm_transient_retries": 3,
            "max_bootstrap_rounds": 1,
            "max_rounds": 1,
            "max_total_rounds": 1,
        }
        values.update(overrides)
        return RunConfig(**values)

    def test_full_reasoning_is_private_and_referenced_from_call_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = TrajectoryLogger(root, {"test": True})
            response_path, reasoning_path = logger.persist_response_artifacts(
                root / "round_01/response_retry_01.txt",
                content="最终内容",
                reasoning_content="完整思考链",
                reasoning_log_mode="full",
            )
            self.assertEqual(reasoning_path, root / "round_01/reasoning_retry_01.txt")
            self.assertEqual(reasoning_path.read_text(encoding="utf-8"), "完整思考链")
            self.assertEqual(stat.S_IMODE(reasoning_path.stat().st_mode), 0o600)
            logger.save_call(
                1,
                LLMResponse(content="最终内容", reasoning_content="完整思考链", model="m").to_dict(),
                response_content_path=response_path,
                reasoning_content_path=reasoning_path,
            )
            record = json.loads(logger.calls_path.read_text(encoding="utf-8"))
            self.assertEqual(record["reasoning_content_path"], "round_01/reasoning_retry_01.txt")
            self.assertEqual(record["reasoning_content_chars"], 5)
            self.assertNotIn("reasoning_content", record)

    def test_metadata_mode_does_not_write_raw_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logger = TrajectoryLogger(root, {"test": True})
            _, reasoning_path = logger.persist_response_artifacts(
                root / "response.txt",
                content="",
                reasoning_content="不落盘",
                reasoning_log_mode="metadata",
            )
            self.assertIsNone(reasoning_path)
            self.assertFalse((root / "reasoning.txt").exists())

    def test_length_is_not_retried_and_reasoning_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.py").write_text("class Model: pass\n", encoding="utf-8")
            provider = _LengthProvider()
            runner = MultiTurnRunner(self._config(root), provider, MockEvaluator())
            logger = TrajectoryLogger(runner.state_dir, runner.config.to_dict())
            with self.assertRaises(LLMCallFailure) as caught:
                runner._call_llm(
                    prompt="probe",
                    call_type="generator",
                    label="probe",
                    logger=logger,
                    attempt_id=1,
                    evaluation_round=1,
                    response_path=runner.state_dir / "round_01/response.txt",
                )
            self.assertEqual(caught.exception.code, "llm_output_exhausted")
            self.assertEqual(provider.calls, 1)
            self.assertEqual(
                (runner.state_dir / "round_01/reasoning.txt").read_text(encoding="utf-8"),
                "反复推理但没有最终答案",
            )

    def test_transport_failures_still_use_transient_retry_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.py").write_text("class Model: pass\n", encoding="utf-8")
            provider = _TransportThenSuccessProvider()
            runner = MultiTurnRunner(
                self._config(root, llm_transient_retries=2), provider, MockEvaluator()
            )
            logger = TrajectoryLogger(runner.state_dir, runner.config.to_dict())
            response = runner._call_llm(
                prompt="probe",
                call_type="generator",
                label="probe",
                logger=logger,
                attempt_id=1,
                evaluation_round=1,
                response_path=runner.state_dir / "response.txt",
            )
            self.assertEqual(response.content, "{}")
            self.assertEqual(provider.calls, 3)

    def test_persistence_failure_stops_the_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.py").write_text("class Model: pass\n", encoding="utf-8")
            runner = MultiTurnRunner(
                self._config(root), _TransportThenSuccessProvider(), MockEvaluator()
            )
            logger = TrajectoryLogger(runner.state_dir, runner.config.to_dict())
            with (
                mock.patch.object(
                    logger,
                    "persist_response_artifacts",
                    side_effect=OSError("read-only filesystem"),
                ),
                self.assertRaises(LLMCallFailure) as caught,
            ):
                runner._call_llm(
                    prompt="probe",
                    call_type="generator",
                    label="probe",
                    logger=logger,
                    attempt_id=1,
                    evaluation_round=1,
                    response_path=runner.state_dir / "response.txt",
                )
            self.assertEqual(caught.exception.code, "reasoning_persistence_failed")

    def test_generator_exhaustion_finishes_with_auditable_pause(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.py").write_text("class Model: pass\n", encoding="utf-8")
            provider = _PlannerThenLengthProvider()
            runner = MultiTurnRunner(self._config(root), provider, MockEvaluator())
            summary = runner.run()
            self.assertEqual(summary["status"], "paused")
            self.assertEqual(summary["evaluations_completed"], 0)
            self.assertEqual(summary["failure"]["stage"], "llm_generator")
            self.assertEqual(summary["failure"]["code"], "llm_output_exhausted")
            self.assertEqual(provider.generator_calls, 1)
            self.assertTrue((runner.state_dir / "summary.json").is_file())
            self.assertTrue((runner.state_dir / "round_01/reasoning.txt").is_file())

    def test_partial_length_gets_one_semantic_retry_then_pauses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model.py").write_text("class Model: pass\n", encoding="utf-8")
            provider = _PlannerThenPartialLengthProvider()
            runner = MultiTurnRunner(self._config(root), provider, MockEvaluator())
            summary = runner.run()
            self.assertEqual(summary["status"], "paused")
            self.assertEqual(summary["failure"]["stage"], "llm_generator")
            self.assertEqual(summary["failure"]["code"], "llm_output_exhausted")
            self.assertEqual(provider.generator_calls, 2)
            self.assertTrue((runner.state_dir / "round_01/reasoning.txt").is_file())
            self.assertTrue((runner.state_dir / "round_01/reasoning_retry_01.txt").is_file())

    def test_diagnostic_manifest_preserves_reasoning_and_validates_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "diagnostic"
            provider = _DiagnosticProvider()
            manifest = run_diagnostics(
                provider,
                output_dir=root,
                real_prompt="REAL PROMPT",
                model="deepseek-flash",
                reasoning_probe_max_tokens=128,
                generation_max_tokens=512,
            )
            self.assertTrue(manifest["success"])
            thinking = next(
                item for item in manifest["steps"] if item["label"] == "real_prompt_thinking_high"
            )
            self.assertEqual(thinking["finish_reason"], "length")
            self.assertTrue((root / thinking["reasoning_path"]).is_file())
            nonthinking = next(
                item for item in manifest["steps"] if item["label"] == "real_prompt_nonthinking"
            )
            self.assertTrue(nonthinking["valid"])
            self.assertEqual(nonthinking["bundle_file_count"], 11)
            self.assertEqual(
                provider.call_types,
                ["diagnostic", "diagnostic", "generator", "generator"],
            )

    def test_fresh_defaults_use_canonical_model_and_disable_generator_thinking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {
                "DEEPSEEK_MODEL": "",
                "ASCENDC_GENERATOR_THINKING": "",
                "ASCENDC_REASONING_LOG_MODE": "",
            },
        ):
            config = RunConfig(
                op_file=str(Path(temporary) / "model.py"),
                output_dir=str(Path(temporary) / "task"),
                mock=True,
            )
            self.assertEqual(config.model, "deepseek-flash")
            self.assertEqual(config.generator_thinking, "disabled")
            self.assertEqual(config.reasoning_log_mode, "full")


if __name__ == "__main__":
    unittest.main()
