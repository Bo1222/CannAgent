from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.evaluator import MockEvaluator
from ascendc_multi_turn.llm import MockProvider
from ascendc_multi_turn.models import RunConfig
from ascendc_multi_turn.runner import MultiTurnRunner
from ascendc_multi_turn.structured_knowledge.knowledge_build import build_knowledge
from ascendc_multi_turn.structured_knowledge.router import (
    KnowledgeBuild,
    StructuredKnowledgeRouter,
    locate_knowledge_build,
)
from ascendc_multi_turn.structured_knowledge.schema import KnowledgeContext


def _document(api: str, parameter: str) -> str:
    return f"""# {api}-API-CANN\n**页面ID:** {api.lower()}\n**来源:** https://example.test/{api}\n\n#### 参数说明\n\n| 参数名称 | 含义 |\n| --- | --- |\n| {parameter} | {api} parameter semantics |\n"""


class StructuredRouterTests(unittest.TestCase):
    def _knowledge_build(self, root: Path) -> Path:
        source = root / "source"
        source.mkdir()
        for api in ("DataCopy", "DataCopyPad", "DataCopyExt"):
            (source / f"{api}.md").write_text(_document(api, "blockLen"), encoding="utf-8")
        return build_knowledge(source=source, output=root / "store", version="8.5.0")

    def test_exact_api_identity_rejects_similar_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            knowledge_build = self._knowledge_build(Path(temporary))
            context = KnowledgeContext(
                operator="copy",
                phase="generate",
                runtime_version="8.5.0",
                knowledge_version="8.5.0",
                soc="Ascend910B3",
                source_symbols=["DataCopy"],
            )

            bundle = StructuredKnowledgeRouter(KnowledgeBuild(knowledge_build)).route(context)

            self.assertEqual([item["api"] for item in bundle.api_semantics], ["DataCopy"])
            self.assertTrue(all(item["applicability"]["api"] == "DataCopy" for item in bundle.relevant_facts))
            rejected = {
                item.candidate for item in bundle.retrieval_trace if item.decision == "rejected"
            }
            self.assertTrue({"DataCopyPad", "DataCopyExt"} <= rejected)

    def test_current_published_build_is_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._knowledge_build(root)
            self.assertEqual(locate_knowledge_build(root / "store", "8.5.0"), first)
            second = first.parent / "another"
            second.mkdir()
            self.assertEqual(locate_knowledge_build(root / "store", "8.5.0"), first)
            self.assertEqual(
                locate_knowledge_build(root / "store", "8.5.0", first.name),
                first,
            )

    def test_missing_installation_names_the_build_command(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(
            ValueError,
            "python -m ascendc_multi_turn.knowledge.build",
        ):
            locate_knowledge_build(Path(temporary), "8.5.0")

    def test_structured_runner_uses_bundle_without_router_llm_call(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            knowledge_build = self._knowledge_build(root)
            source = root / "model.py"
            source.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "generated"
            config = RunConfig(
                str(source),
                str(output),
                max_rounds=1,
                evaluator="mock",
                mock=True,
                knowledge_mode="structured",
                knowledge_store=str(root / "store"),
                knowledge_build_id=knowledge_build.name,
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
