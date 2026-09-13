from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.context_selector import ContextSelector, SelectedStageContext
from ascendc_multi_turn.evaluator import MockEvaluator
from ascendc_multi_turn.llm import MockProvider
from ascendc_multi_turn.models import EvalResult, RunConfig
from ascendc_multi_turn.prompts import render_stage_context
from ascendc_multi_turn.runner import MultiTurnRunner
from ascendc_multi_turn.skill_adapter import SkillAdapter, SkillAdapterContext
from ascendc_multi_turn.structured_knowledge.knowledge_build import build_knowledge


def _mapping(root: Path) -> Path:
    payload = {
        "schema_version": 1,
        "default_source_root": str(root),
        "audience_budgets": {"planner": 2000, "generator": 3000},
        "operator_families": {"elementwise": ["gelu"]},
        "stages": {
            "operator_analysis": {
                "skills": [
                    {
                        "id": "arch",
                        "source": "arch",
                        "purpose": "architecture",
                        "expected_artifact": "hardware constraints",
                        "knowledge_type": ["hardware_fact"],
                        "when": {"always": True},
                        "provide": ["soc"],
                        "exclude": ["unrelated hardware"],
                        "references": [
                            {
                                "path": "references/guide.md",
                                "headings": ["Selected"],
                                "max_chars": 500,
                            }
                        ],
                    }
                ]
            },
            "kernel_design": {
                "skills": [
                    {
                        "id": "tiling",
                        "source": "tiling",
                        "purpose": "tiling",
                        "expected_artifact": "tile formula",
                        "knowledge_type": ["design_rule"],
                        "when": {"always": True},
                        "provide": ["shape"],
                        "exclude": ["other families"],
                        "references": [],
                    }
                ]
            },
            "code_generation": {
                "skills": [
                    {
                        "id": "template",
                        "source": "template",
                        "purpose": "code structure",
                        "expected_artifact": "kernel skeleton",
                        "knowledge_type": ["code_pattern"],
                        "when": {"always": True},
                        "provide": ["host ABI"],
                        "exclude": ["full scaffold"],
                        "references": [],
                    }
                ]
            },
            "runtime_debug": {
                "skills": [
                    {
                        "id": "runtime",
                        "source": "runtime",
                        "purpose": "runtime diagnosis",
                        "expected_artifact": "ranked checks",
                        "knowledge_type": ["diagnostic"],
                        "when": {
                            "failure_stages": ["correctness"],
                            "any_terms": ["aicore"],
                        },
                        "provide": ["error"],
                        "exclude": [],
                        "references": [],
                    }
                ]
            },
            "precision_debug": {
                "skills": [
                    {
                        "id": "precision",
                        "source": "precision",
                        "purpose": "precision diagnosis",
                        "expected_artifact": "root cause",
                        "knowledge_type": ["diagnostic"],
                        "when": {
                            "failure_stages": ["correctness"],
                            "any_terms": ["mismatch"],
                        },
                        "provide": ["error"],
                        "exclude": [],
                        "references": [],
                    }
                ]
            },
        },
    }
    path = root / "mapping.yaml"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _skill_root(root: Path) -> Path:
    for name in ("arch", "tiling", "template", "runtime", "precision"):
        (root / name / "references").mkdir(parents=True)
    (root / "arch/references/guide.md").write_text(
        "# Ignore\nnot selected\n\n## Selected\nkeep this architecture fact\n\n## Other\nomit this\n",
        encoding="utf-8",
    )
    return root


class SkillAdapterTests(unittest.TestCase):
    def test_knowledge_source_normalization_and_legacy_alias(self) -> None:
        structured = RunConfig(op_file="model.py", output_dir="out", mock=True)
        legacy_hybrid = RunConfig(
            op_file="model.py", output_dir="out", mock=True, skill_adapter=True
        )
        skills = RunConfig(
            op_file="model.py",
            output_dir="out",
            mock=True,
            knowledge_source="skills",
        )

        self.assertEqual(structured.knowledge_source, "structured")
        self.assertTrue(structured.uses_structured_prompt)
        self.assertFalse(structured.uses_skills)
        self.assertEqual(legacy_hybrid.knowledge_source, "hybrid")
        self.assertTrue(legacy_hybrid.uses_structured_prompt)
        self.assertTrue(legacy_hybrid.uses_skills)
        self.assertEqual(skills.knowledge_source, "skills")
        self.assertFalse(skills.uses_structured_prompt)
        self.assertTrue(skills.uses_skills)

        full_selected = RunConfig(
            op_file="model.py",
            output_dir="out",
            mock=True,
            knowledge_source="hybrid",
            knowledge_input_mode="full-selected",
        )
        self.assertEqual(full_selected.knowledge_input_mode, "full_selected")
        self.assertTrue(full_selected.uses_full_selected_input)

    def test_legacy_skill_flag_rejects_conflicting_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy alias"):
            RunConfig(
                op_file="model.py",
                output_dir="out",
                mock=True,
                skill_adapter=True,
                knowledge_source="structured",
            )

    def test_selects_only_allowlisted_heading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _skill_root(Path(temporary))
            adapter = SkillAdapter(mapping_path=_mapping(root), source_root=root)
            context = SkillAdapterContext(
                audience="planner",
                stages=["operator_analysis"],
                operator="gelu",
                operator_families=["elementwise"],
                soc="test",
                runtime_version="8.5.0",
                knowledge_version="8.5.0",
                failure_stage=None,
                failure_code=None,
                failure_evidence="",
                failure_symbols=[],
                has_correct_baseline=False,
                evidence="gelu",
            )

            selected = adapter.select(context)

            self.assertEqual([item.skill_id for item in selected.capsules], ["arch"])
            excerpt = selected.capsules[0].excerpts[0].text
            self.assertIn("keep this architecture fact", excerpt)
            self.assertNotIn("not selected", excerpt)
            self.assertNotIn("omit this", excerpt)

    def test_full_selected_keeps_complete_mapped_heading(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = _skill_root(Path(temporary))
            marker = "full-selected-tail-marker"
            guide = root / "arch/references/guide.md"
            guide.write_text(
                "## Selected\n" + ("complete knowledge line\n" * 80) + marker + "\n\n## Other\nomit this\n",
                encoding="utf-8",
            )
            adapter = SkillAdapter(
                mapping_path=_mapping(root),
                source_root=root,
                full_selected_input=True,
            )
            context = SkillAdapterContext(
                audience="planner",
                stages=["operator_analysis"],
                operator="gelu",
                operator_families=["elementwise"],
                soc="test",
                runtime_version="8.5.0",
                knowledge_version="8.5.0",
                failure_stage=None,
                failure_code=None,
                failure_evidence="",
                failure_symbols=[],
                has_correct_baseline=False,
                evidence="gelu",
            )

            selected = adapter.select(context)
            excerpt = selected.capsules[0].excerpts[0].text

            self.assertGreater(len(excerpt), 500)
            self.assertIn(marker, excerpt)
            self.assertNotIn("omit this", excerpt)

    def test_missing_source_fails_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter = SkillAdapter(
                mapping_path=_mapping(root),
                source_root=root / "missing",
            )
            selected = adapter.select(
                SkillAdapterContext(
                    audience="planner",
                    stages=["operator_analysis"],
                    operator="gelu",
                    operator_families=["elementwise"],
                    soc="test",
                    runtime_version="8.5.0",
                    knowledge_version="8.5.0",
                    failure_stage=None,
                    failure_code=None,
                    failure_evidence="",
                    failure_symbols=[],
                    has_correct_baseline=False,
                    evidence="gelu",
                )
            )

            self.assertFalse(selected.source_available)
            self.assertEqual(selected.capsules, [])
            self.assertEqual(selected.trace[0]["reason"], "source_unavailable")

    def test_stage_derivation_separates_planner_generator_and_debug(self) -> None:
        self.assertEqual(
            ContextSelector.derive_stages(
                audience="planner",
                workflow_phase="bootstrap",
                current_exists=False,
                previous=None,
            ),
            ["operator_analysis", "kernel_design"],
        )
        self.assertEqual(
            ContextSelector.derive_stages(
                audience="generator",
                workflow_phase="bootstrap",
                current_exists=False,
                previous=None,
            ),
            ["kernel_design", "code_generation"],
        )
        runtime = EvalResult(
            compiled=True,
            correctness=False,
            error="AICORE exception",
            failure_stage="correctness",
            structured_failure={
                "subsystem": "AICORE",
                "device_exception": "AICORE exception",
            },
        )
        mismatch = EvalResult(
            compiled=True,
            correctness=False,
            error="outputs mismatch",
            failure_stage="correctness",
            error_excerpt="max error exceeds rtol",
            structured_failure={"subsystem": "CORRECTNESS"},
        )
        self.assertEqual(
            ContextSelector.derive_stages(
                audience="planner",
                workflow_phase="bootstrap",
                current_exists=True,
                previous=runtime,
            ),
            ["runtime_debug"],
        )
        self.assertEqual(
            ContextSelector.derive_stages(
                audience="planner",
                workflow_phase="bootstrap",
                current_exists=True,
                previous=mismatch,
            ),
            ["kernel_design"],
        )

    def test_prompt_renderer_respects_budget_and_records_omissions(self) -> None:
        selection = SelectedStageContext(
            schema_version=1,
            audience="planner",
            stages=["kernel_design"],
            authority_order=["official", "examples"],
            task_facts={"operator": "gelu"},
            hard_constraints=[{"constraint": "fixed ABI"}],
            api_facts=[],
            design_patterns=["x" * 3000],
            failure_guidance=[],
            skill_capsules=[],
            exclusions=["do not copy workflows"],
            provenance=[],
            budget={"max_chars": 1200, "used_chars": 0, "truncated_sections": []},
        )

        rendered = render_stage_context(selection)

        self.assertLessEqual(len(rendered), 1200)
        self.assertTrue(selection.budget["truncated_sections"])
        self.assertNotIn("x" * 100, rendered)

    def test_prompt_renderer_full_selected_ignores_all_render_budgets(self) -> None:
        marker = "full-selected-skill-tail"
        selection = SelectedStageContext(
            schema_version=2,
            audience="generator",
            stages=["kernel_design", "code_generation"],
            authority_order=["runtime", "skills"],
            task_facts={"operator": "add"},
            hard_constraints=[{"contract_id": "host", "constraint": "h" * 4000}],
            api_facts=[{"fact_id": "runtime:Add", "value": "a" * 7000}],
            design_patterns=["p" * 3000],
            failure_guidance=[],
            skill_capsules=[
                {
                    "skill_id": "ascendc-api-best-practices",
                    "stage": "kernel_design",
                    "purpose": "api",
                    "trigger_reason": "selected",
                    "expected_artifact": "code",
                    "excerpts": [
                        {
                            "source": "references/api-arithmetic.md",
                            "headings": ["API"],
                            "text": "s" * 6000 + marker,
                        }
                    ],
                }
            ],
            exclusions=[],
            provenance=[],
            runtime_facts="r" * 7000,
            budget={
                "input_mode": "full_selected",
                "max_chars": None,
                "used_chars": 0,
                "truncated_sections": [],
            },
            selection_metadata={
                "selected_skill_ids": ["ascendc-api-best-practices"],
                "selected_structured_ids": ["runtime:Add"],
                "runtime_fact_ids": ["runtime:Add"],
            },
        )

        rendered = render_stage_context(selection)

        self.assertGreater(len(rendered), 20000)
        self.assertIn(marker, rendered)
        self.assertEqual(selection.budget["truncated_sections"], [])
        self.assertFalse(selection.selection_metadata["input_truncated"])
        self.assertEqual(
            selection.selection_metadata["rendered_skill_ids"],
            selection.selection_metadata["selected_skill_ids"],
        )


class SkillAdapterRunnerTests(unittest.TestCase):
    def test_mock_runner_persists_distinct_audience_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skills = _skill_root(root / "skills")
            mapping = _mapping(skills)
            source_docs = root / "source_docs"
            source_docs.mkdir()
            (source_docs / "DataCopy.md").write_text(
                "# DataCopy-API-CANN\n**页面ID:** datacopy\n**来源:** https://example.test\n",
                encoding="utf-8",
            )
            knowledge_build = build_knowledge(
                source=source_docs,
                output=root / "store",
                version="8.5.0",
            )
            op = root / "gelu.py"
            op.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "output"
            config = RunConfig(
                op_file=str(op),
                output_dir=str(output),
                max_bootstrap_rounds=1,
                max_rounds=1,
                max_total_rounds=1,
                mock=True,
                skill_adapter=True,
                cannbot_skills_root=str(skills),
                skill_mapping=str(mapping),
                knowledge_store=str(root / "store"),
                knowledge_build_id=knowledge_build.name,
            )

            summary = MultiTurnRunner(config, MockProvider(), MockEvaluator()).run()

            round_dir = output / ".llm_state/round_01"
            self.assertTrue(summary["success"])
            self.assertEqual(summary["knowledge_source"]["mode"], "hybrid")
            planner = json.loads((round_dir / "planner_context.json").read_text())
            generator = json.loads((round_dir / "generator_context.json").read_text())
            self.assertEqual(planner["stages"], ["operator_analysis", "kernel_design"])
            self.assertEqual(generator["stages"], ["kernel_design", "code_generation"])
            self.assertNotEqual(
                (round_dir / "planner_references.md").read_text(),
                (round_dir / "references.md").read_text(),
            )

    def test_skills_only_uses_empty_official_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skills = _skill_root(root / "skills")
            mapping = _mapping(skills)
            op = root / "gelu.py"
            op.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "output"
            config = RunConfig(
                op_file=str(op),
                output_dir=str(output),
                max_bootstrap_rounds=1,
                max_rounds=1,
                max_total_rounds=1,
                mock=True,
                knowledge_source="skills",
                cannbot_skills_root=str(skills),
                skill_mapping=str(mapping),
                knowledge_store=str(root / "unused-store"),
            )

            summary = MultiTurnRunner(config, MockProvider(), MockEvaluator()).run()

            round_dir = output / ".llm_state/round_01"
            planner_bundle = json.loads(
                (round_dir / "knowledge_bundle_planner.json").read_text()
            )
            generator_bundle = json.loads(
                (round_dir / "knowledge_bundle_generator.json").read_text()
            )
            planner_context = json.loads(
                (round_dir / "planner_context.json").read_text()
            )
            self.assertTrue(summary["success"])
            self.assertEqual(summary["knowledge_source"]["mode"], "skills")
            self.assertFalse(
                summary["knowledge_source"]["structured_prompt_enabled"]
            )
            for bundle in (planner_bundle, generator_bundle):
                self.assertEqual(bundle["api_semantics"], [])
                self.assertEqual(bundle["relevant_facts"], [])
                self.assertEqual(bundle["examples"], [])
                self.assertEqual(bundle["failure_cards"], [])
                self.assertEqual(bundle["project_contracts"], [])
                self.assertEqual(
                    bundle["retrieval_trace"][0]["reason"],
                    "structured prompt source disabled by knowledge_source=skills",
                )
            self.assertEqual(
                planner_context["task_facts"]["knowledge_source"], "skills"
            )
            self.assertTrue(planner_context["skill_capsules"])
            self.assertIn(
                "keep this architecture fact",
                (round_dir / "planner_references.md").read_text(),
            )

    def test_resume_rejects_a_different_knowledge_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            skills = _skill_root(root / "skills")
            mapping = _mapping(skills)
            op = root / "gelu.py"
            op.write_text("class Model: pass\n", encoding="utf-8")
            output = root / "output"
            initial = RunConfig(
                op_file=str(op),
                output_dir=str(output),
                max_bootstrap_rounds=1,
                max_rounds=1,
                max_total_rounds=1,
                mock=True,
                knowledge_source="skills",
                cannbot_skills_root=str(skills),
                skill_mapping=str(mapping),
            )
            MultiTurnRunner(initial, MockProvider(), MockEvaluator()).run()
            resumed = RunConfig(
                op_file=str(op),
                output_dir=str(output),
                max_bootstrap_rounds=1,
                max_rounds=1,
                max_total_rounds=1,
                mock=True,
                resume=True,
                knowledge_source="hybrid",
                cannbot_skills_root=str(skills),
                skill_mapping=str(mapping),
            )

            with self.assertRaisesRegex(ValueError, "different knowledge source"):
                MultiTurnRunner(resumed, MockProvider(), MockEvaluator()).run()

    def test_adapter_rejects_document_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires structured"):
            RunConfig(
                op_file="model.py",
                output_dir="out",
                mock=True,
                knowledge_mode="document",
                skill_adapter=True,
            )


if __name__ == "__main__":
    unittest.main()
