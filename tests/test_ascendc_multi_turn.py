from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.bundle import parse_file_bundle
from ascendc_multi_turn.evaluator import MockEvaluator
from ascendc_multi_turn.llm import MockProvider
from ascendc_multi_turn.models import RunConfig
from ascendc_multi_turn.runner import MultiTurnRunner


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


if __name__ == "__main__":
    unittest.main()
