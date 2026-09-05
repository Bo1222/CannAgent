from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.bundle import parse_file_bundle
from ascendc_multi_turn.evaluator import MockEvaluator
from ascendc_multi_turn.llm import MockProvider
from ascendc_multi_turn.models import LLMResponse, RunConfig
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
            calls = (output / ".llm_state" / "calls.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"content"', calls)
            self.assertIn('"call_type": "knowledge_router"', calls)
            self.assertIn('"call_type": "generator"', calls)

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
            round_record = trajectory["rounds"][0]
            self.assertEqual(round_record["decision"], "FORMAT_FAIL")
            self.assertEqual(round_record["generation_attempts"], 2)
            self.assertIn("output-token limit", round_record["evaluation"]["error"])


if __name__ == "__main__":
    unittest.main()
