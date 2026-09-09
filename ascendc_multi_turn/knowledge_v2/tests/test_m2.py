from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.evaluator import MockEvaluator
from ascendc_multi_turn.knowledge_v2.router import KnowledgeRouterV2, SnapshotView, locate_snapshot
from ascendc_multi_turn.knowledge_v2.schema import KnowledgeContext
from ascendc_multi_turn.knowledge_v2.snapshot import build_snapshot
from ascendc_multi_turn.llm import MockProvider
from ascendc_multi_turn.models import RunConfig
from ascendc_multi_turn.runner import MultiTurnRunner


def _document(api: str, parameter: str) -> str:
    return f"""# {api}-API-CANN\n**页面ID:** {api.lower()}\n**来源:** https://example.test/{api}\n\n#### 参数说明\n\n| 参数名称 | 含义 |\n| --- | --- |\n| {parameter} | {api} parameter semantics |\n"""


class SemanticRouterTests(unittest.TestCase):
    def _snapshot(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        for api in ("DataCopy", "DataCopyPad", "DataCopyExt"):
            (source / f"{api}.md").write_text(_document(api, "blockLen"), encoding="utf-8")
        return build_snapshot(source=source, output=root / "store", version="8.5.0")

    def test_exact_api_identity_rejects_similar_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = self._snapshot(Path(temporary))
            context = KnowledgeContext(
                operator="copy",
                phase="generate",
                runtime_version="8.5.0",
                knowledge_version="8.5.0",
                soc="Ascend910B3",
                source_symbols=["DataCopy"],
            )

            bundle = KnowledgeRouterV2(SnapshotView(snapshot)).route(context)

            self.assertEqual([item["api"] for item in bundle.api_semantics], ["DataCopy"])
            self.assertTrue(all(item["applicability"]["api"] == "DataCopy" for item in bundle.relevant_facts))
            rejected = {
                item.candidate for item in bundle.retrieval_trace if item.decision == "rejected"
            }
            self.assertTrue({"DataCopyPad", "DataCopyExt"} <= rejected)

    def test_snapshot_selection_requires_disambiguation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._snapshot(root)
            self.assertEqual(locate_snapshot(root / "store", "8.5.0"), first)
            second = first.parent / "another"
            second.mkdir()
            with self.assertRaisesRegex(ValueError, "exactly one published"):
                locate_snapshot(root / "store", "8.5.0")

    def test_semantic_runner_uses_bundle_without_router_llm_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = self._snapshot(root)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            config = RunConfig(
                str(source),
                str(output),
                max_rounds=1,
                evaluator="mock",
                mock=True,
                knowledge_mode="semantic",
                knowledge_store=str(root / "store"),
                knowledge_snapshot=snapshot.name,
            )

            summary = MultiTurnRunner(config, MockProvider(), MockEvaluator()).run()

            self.assertTrue(summary["success"])
            calls = [
                json.loads(line)["call_type"]
                for line in (output / ".llm_state/calls.jsonl").read_text().splitlines()
            ]
            self.assertNotIn("knowledge_router", calls)
            self.assertTrue((output / ".llm_state/round_01/knowledge_bundle.json").is_file())
            self.assertTrue((output / ".llm_state/round_01/retrieval_trace.json").is_file())


if __name__ == "__main__":
    unittest.main()
