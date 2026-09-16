from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ascendc_multi_turn.context_selector import SelectedStageContext
from ascendc_multi_turn.evaluator import MockEvaluator
from ascendc_multi_turn.llm import MockProvider
from ascendc_multi_turn.models import RunConfig
from ascendc_multi_turn.prompts import render_stage_context
from ascendc_multi_turn.runner import MultiTurnRunner
from ascendc_multi_turn.skill_adapter import (
    CANNBOT_KNOWLEDGE_BASE_MANIFEST_PATH,
    DEFAULT_MAPPING_PATH,
    KNOWLEDGE_MODULE_ROOT,
    SkillAdapter,
    SkillAdapterContext,
)
from ascendc_multi_turn.structured_knowledge.knowledge_build import build_knowledge


def _context(*, stages: list[str], operator: str = "add", domains: list[str] | None = None) -> SkillAdapterContext:
    return SkillAdapterContext(
        audience="generator",
        stages=stages,
        operator=operator,
        operator_families=["broadcast"] if operator == "add" else ["elementwise"],
        soc="Ascend910B3",
        runtime_version="8.5.2",
        knowledge_version="8.5.2",
        failure_stage=None,
        failure_code=None,
        failure_evidence="broadcast stride=0 [128,128] + [128,1] descriptor",
        failure_symbols=[],
        has_correct_baseline=False,
        evidence="DataCopy TQue broadcast stride=0 [128,128] + [128,1] descriptor",
        symbol_domains=domains or ["operator_semantic", "launch_abi"],
        active_profile="shape",
        profile_features=["broadcast"],
    )


class EmbeddedSkillAdapterTests(unittest.TestCase):
    def test_knowledge_source_normalization_and_legacy_alias(self) -> None:
        structured = RunConfig(op_file="model.py", output_dir="out", mock=True)
        hybrid = RunConfig(op_file="model.py", output_dir="out", mock=True, skill_adapter=True)
        skills = RunConfig(op_file="model.py", output_dir="out", mock=True, knowledge_source="skills")

        self.assertEqual(structured.knowledge_source, "structured")
        self.assertFalse(structured.uses_skills)
        self.assertEqual(hybrid.knowledge_source, "hybrid")
        self.assertTrue(hybrid.uses_structured_prompt)
        self.assertEqual(skills.knowledge_source, "skills")
        self.assertFalse(skills.uses_structured_prompt)

    def test_external_root_mapping_and_environment_fail_with_migration_error(self) -> None:
        for kwargs in (
            {"cannbot_skills_root": "/tmp/old-skills"},
            {"skill_mapping": "/tmp/old-mapping.yaml"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "embedded CANNBot knowledge base"):
                RunConfig(op_file="model.py", output_dir="out", mock=True, **kwargs)
        with patch.dict(os.environ, {"CANNBOT_SKILLS_ROOT": "/tmp/old-skills"}), self.assertRaisesRegex(
            ValueError, "embedded CANNBot knowledge base"
        ):
            RunConfig(op_file="model.py", output_dir="out", mock=True)
        with self.assertRaisesRegex(ValueError, "External CANNBot"):
            SkillAdapter(source_root="/tmp/old-skills")
        with self.assertRaisesRegex(ValueError, "External CANNBot"):
            SkillAdapter(mapping_path="/tmp/old-mapping.yaml")

    def test_all_manifest_files_exist_match_hashes_and_mapping_uses_ids(self) -> None:
        adapter = SkillAdapter()
        self.assertTrue(adapter.source_root.is_dir())
        self.assertGreaterEqual(len(adapter.registry), 67)
        mapping = json.loads(DEFAULT_MAPPING_PATH.read_text(encoding="utf-8"))
        for stage in mapping["stages"].values():
            for skill in stage["skills"]:
                self.assertNotIn("source", skill)
                for reference in skill["references"]:
                    self.assertIn(reference["knowledge_module_id"], adapter.registry)
                    self.assertTrue(reference["section_id"])
                    self.assertNotIn("path", reference)
                    self.assertNotIn("max_chars", reference)
        for document in adapter.registry.values():
            path = KNOWLEDGE_MODULE_ROOT / document.path
            self.assertTrue(path.is_file())
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), document.sha256)

    def test_hash_mismatch_fails_fast(self) -> None:
        original = json.loads(CANNBOT_KNOWLEDGE_BASE_MANIFEST_PATH.read_text(encoding="utf-8"))
        original["documents"][0]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            manifest = Path(temporary) / "manifest.json"
            manifest.write_text(json.dumps(original), encoding="utf-8")
            with patch(
                "ascendc_multi_turn.skill_adapter.CANNBOT_KNOWLEDGE_BASE_MANIFEST_PATH", manifest
            ), self.assertRaisesRegex(ValueError, "hash mismatch"):
                SkillAdapter()

    def test_leaf_sections_are_explicit_complete_and_deduplicated(self) -> None:
        adapter = SkillAdapter()
        selected = adapter.select(_context(stages=["kernel_design"]))
        excerpts = [
            excerpt
            for knowledge_module in selected.knowledge_modules
            for excerpt in knowledge_module.excerpts
        ]
        keys = [(item.knowledge_module_id, item.section_id) for item in excerpts]
        self.assertEqual(len(keys), len(set(keys)))
        broadcast = next(item for item in excerpts if item.knowledge_module_id == "cannagent.broadcast-semantics")
        self.assertIn("[128,128] + [128,1]", broadcast.text)
        self.assertIn("Do not turn `stride == 0`", broadcast.text)
        self.assertTrue(any("broadcast.onedim" in item.knowledge_module_id for item in excerpts))

        duplicate_stages = adapter.select(
            _context(stages=["code_generation", "host_integration_debug"])
        )
        duplicate_keys = [
            (item.knowledge_module_id, item.section_id)
            for knowledge_module in duplicate_stages.knowledge_modules
            for item in knowledge_module.excerpts
        ]
        self.assertEqual(len(duplicate_keys), len(set(duplicate_keys)))
        self.assertTrue(
            any(item["reason"] == "duplicate_knowledge_module_section" for item in duplicate_stages.trace)
        )

    def test_renderer_never_truncates_selected_knowledge_module(self) -> None:
        marker = "selected-knowledge-module-tail-marker"
        selection = SelectedStageContext(
            schema_version=2,
            audience="generator",
            stages=["kernel_design"],
            authority_order=["installed", "documented"],
            task_facts={"operator": "add"},
            hard_constraints=[],
            api_facts=[],
            design_patterns=[],
            failure_guidance=[],
            skill_knowledge_modules=[{
                "skill_id": "broadcast",
                "stage": "kernel_design",
                "purpose": "semantics",
                "trigger_reason": "shape",
                "expected_artifact": "correct map",
                "excerpts": [{
                    "knowledge_module_id": "fixture.broadcast",
                    "section_id": "all",
                    "source": "fixture.md",
                    "text": "x" * 6000 + marker,
                }],
            }],
            exclusions=[],
            provenance=[],
            budget={"input_mode": "bounded", "max_chars": 500, "used_chars": 0, "truncated_sections": []},
        )
        rendered = render_stage_context(selection)
        self.assertIn(marker, rendered)
        self.assertIn("fixture.broadcast#all", selection.selection_metadata["rendered_knowledge_module_sections"])

    def test_missing_sibling_repository_is_irrelevant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, patch(
            "pathlib.Path.cwd", return_value=Path(temporary)
        ):
            adapter = SkillAdapter()
        self.assertTrue(adapter.registry)
        self.assertTrue(adapter.select(_context(stages=["kernel_design"])).source_available)


class EmbeddedSkillRunnerTests(unittest.TestCase):
    @staticmethod
    def _knowledge_store(root: Path) -> str:
        docs = root / "source_docs"
        docs.mkdir()
        (docs / "DataCopy.md").write_text(
            "# DataCopy-API-CANN\n**页面ID:** datacopy\n**来源:** https://example.test\n",
            encoding="utf-8",
        )
        build_knowledge(source=docs, output=root / "store", version="8.5.0")
        return str(root / "store")

    def test_hybrid_runner_persists_distinct_embedded_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            op = root / "gelu.py"
            op.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "output"
            config = RunConfig(
                op_file=str(op), output_dir=str(output), max_bootstrap_rounds=1,
                max_rounds=1, max_total_rounds=1, mock=True,
                knowledge_source="hybrid", knowledge_store=self._knowledge_store(root),
            )
            summary = MultiTurnRunner(config, MockProvider(), MockEvaluator()).run()
            round_dir = output / ".llm_state/round_01"
            planner = json.loads((round_dir / "planner_context.json").read_text())
            generator = json.loads((round_dir / "generator_context.json").read_text())
            self.assertTrue(summary["success"])
            self.assertEqual(planner["stages"], ["operator_analysis", "kernel_design"])
            self.assertEqual(generator["stages"], ["kernel_design", "code_generation"])
            self.assertTrue(generator["skill_knowledge_modules"])

    def test_skills_only_does_not_fallback_to_structured(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            op = root / "add.py"
            op.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "output"
            config = RunConfig(
                op_file=str(op), output_dir=str(output), max_bootstrap_rounds=1,
                max_rounds=1, max_total_rounds=1, mock=True,
                knowledge_source="skills", knowledge_store=str(root / "missing"),
            )
            summary = MultiTurnRunner(config, MockProvider(), MockEvaluator()).run()
            bundle = json.loads(
                (output / ".llm_state/round_01/knowledge_bundle_generator.json").read_text()
            )
            context = json.loads(
                (output / ".llm_state/round_01/generator_context.json").read_text()
            )
            self.assertTrue(summary["success"])
            self.assertFalse(summary["knowledge_source"]["structured_prompt_enabled"])
            self.assertEqual(bundle["api_semantics"], [])
            self.assertEqual(bundle["project_contracts"], [])
            self.assertTrue(context["skill_knowledge_modules"])

    def test_adapter_rejects_document_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires structured"):
            RunConfig(op_file="model.py", output_dir="out", mock=True, knowledge_mode="document", skill_adapter=True)


if __name__ == "__main__":
    unittest.main()
