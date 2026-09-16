"""Rebuild metadata for the already-curated embedded CANNBot Markdown knowledge base."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CANNBOT_KNOWLEDGE_BASE_ROOT = (
    REPO_ROOT
    / "ascendc_multi_turn/knowledge_modules/cannbot_a08c4970_knowledge_base"
)
UPSTREAM_COMMIT = "a08c49706e35a400d7c77e0875bc7c72a3a79012"
KNOWLEDGE_BASE_ID = "cannbot-a08c4970-cannagent-routing-v2"


def _metadata(relative: str) -> tuple[list[str], list[str], list[str]]:
    skill = relative.split("/", 1)[0]
    if skill == "npu-arch":
        return ["hardware_fact", "compatibility_rule"], ["operator_analysis"], []
    if skill == "ascendc-tiling-design":
        return ["design_rule", "algorithm", "formula"], ["kernel_design", "optimization"], []
    if skill == "ascendc-api-best-practices":
        return ["api_constraint", "best_practice", "anti_pattern"], ["kernel_design", "compile_debug", "precision_debug"], []
    if skill == "ascendc-direct-invoke-template":
        return ["launch_pattern", "dataflow"], ["code_generation", "host_integration_debug"], ["project_scaffold", "workflow_commands"]
    if skill == "ascendc-registry-invoke-to-direct-invoke":
        return ["launch_abi", "migration_constraint"], ["code_generation", "host_integration_debug", "compile_debug"], ["registry_workflow", "copy_commands"]
    if skill == "torch-ascendc-op-extension":
        return ["host_abi", "stream_integration"], ["code_generation", "host_integration_debug"], ["stream(true)", "stream(false)", "foreign_project_layout"]
    if skill in {"ascendc-runtime-debug", "ascendc-crash-debug"}:
        return ["runtime_diagnostic", "memory_safety"], ["runtime_debug"], ["workflow_commands"]
    if skill == "ascendc-sync-audit":
        return ["synchronization_rule", "pipeline_pattern"], ["kernel_design", "runtime_debug"], ["audit_workflow", "scripts"]
    if skill == "ascendc-precision-debug":
        return ["precision_rule", "diagnostic_pattern"], ["precision_debug"], ["workflow_commands"]
    if skill == "ascendc-performance-best-practices":
        stages = ["optimization"]
        if "/broadcast/" in relative:
            stages.insert(0, "kernel_design")
        return ["performance_pattern", "design_rule"], stages, []
    raise ValueError(f"unclassified knowledge base document: {relative}")


def main() -> int:
    documents = []
    for path in sorted((CANNBOT_KNOWLEDGE_BASE_ROOT / "docs").rglob("*.md")):
        relative = path.relative_to(CANNBOT_KNOWLEDGE_BASE_ROOT / "docs").as_posix()
        knowledge_type, allowed_stages, conflicts = _metadata(relative)
        identifier = "cannbot." + relative.removesuffix(".md").replace("/", ".")
        documents.append(
            {
                "knowledge_module_id": identifier,
                "path": f"docs/{relative}",
                "upstream_skill": relative.split("/", 1)[0],
                "upstream_path": f"ops/{relative}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "knowledge_type": knowledge_type,
                "allowed_stages": allowed_stages,
                "conflicts_or_exclusions": conflicts,
                "provenance": "cannbot_upstream_documentation",
                "confidence_level": 1,
            }
        )
    payload = {
        "schema_version": 1,
        "knowledge_base_id": KNOWLEDGE_BASE_ID,
        "upstream_commit": UPSTREAM_COMMIT,
        "license": "CANN Open Software License Agreement 2.0",
        "confidence_policy": "Static knowledge base text is Level 1 Documented. Installed headers, project source, or local compile/correctness evidence may supersede it.",
        "documents": documents,
    }
    (CANNBOT_KNOWLEDGE_BASE_ROOT / "knowledge_base_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
