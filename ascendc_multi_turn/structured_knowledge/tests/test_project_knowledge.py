from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.structured_knowledge.knowledge_build import build_knowledge
from ascendc_multi_turn.structured_knowledge.router import (
    KnowledgeBuild,
    StructuredKnowledgeRouter,
)
from ascendc_multi_turn.structured_knowledge.schema import KnowledgeContext


class ProjectKnowledgeTests(unittest.TestCase):
    def test_project_cards_have_evidence_and_route_host_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            (source / "project_guides").mkdir(parents=True)
            (source / "project_guides/host.md").write_text(
                "# Host\n## Binding\nUse a literal `PYBIND11_MODULE(module_name, m)` declaration.\n",
                encoding="utf-8",
            )
            (source / "project_knowledge.json").write_text(
                json.dumps({
                    "project_contracts": [{"contract_id":"host","subject":"literal import","constraint":"guidance","guide":"project_guides/host.md","section":"Binding","evidence":"Use a literal `PYBIND11_MODULE(module_name, m)` declaration."}],
                    "failure_cards": [{"card_id":"failure_host","signals":["ascendc_ext_imported","static_validation"],"subsystem":"HOST_BINDING","inspection_targets":["literal import"],"guide":"project_guides/host.md","section":"Binding","evidence":"Use a literal `PYBIND11_MODULE(module_name, m)` declaration."}],
                    "pattern_cards": [{"card_id":"pattern_host","name":"host binding","description":"literal PYBIND11_MODULE binding","keywords":["host","binding"],"guide":"project_guides/host.md","section":"Binding","evidence":"Use a literal `PYBIND11_MODULE(module_name, m)` declaration."}]
                }), encoding="utf-8"
            )
            knowledge_build = build_knowledge(
                source=source,
                output=root / "store",
                version="8.5.0",
            )
            view = KnowledgeBuild(knowledge_build)
            bundle = StructuredKnowledgeRouter(view).route(KnowledgeContext(
                operator="gelu", phase="diagnose", runtime_version="8.5.0",
                knowledge_version="8.5.0", soc="Ascend910B3", source_symbols=[],
                failure={"stage":"static_validation", "reason":"ascendc_ext_imported"},
            ))
            self.assertEqual(view.project_contracts[0]["authority"], "PROJECT_CONTRACT")
            self.assertTrue(view.project_contracts[0]["provenance"]["source_hash"])
            self.assertEqual(bundle.failure_cards[0]["card_id"], "failure_host")
            self.assertTrue(any(item.candidate == "failure_host" for item in bundle.retrieval_trace))
            self.assertTrue(any(item["document_id"] == "host" for item in bundle.provenance))
            self.assertTrue((knowledge_build / "raw/project_knowledge.json").is_file())
            self.assertNotIn("Host", view.symbols)

    def test_missing_project_evidence_rejects_knowledge_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            (source / "guide.md").write_text("# Guide\nreal evidence\n", encoding="utf-8")
            (source / "project_knowledge.json").write_text(json.dumps({
                "project_contracts": [{"contract_id":"bad","subject":"bad","constraint":"guidance","guide":"guide.md","section":"Guide","evidence":"invented evidence"}]
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence is missing"):
                build_knowledge(
                    source=source,
                    output=Path(temporary) / "store",
                    version="8.5.0",
                )


if __name__ == "__main__":
    unittest.main()
