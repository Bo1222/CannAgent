from __future__ import annotations

import json
import re
from typing import Any

from .diagnostics import compact_evaluation
from .models import EvalResult, FileBundle

OUTPUT_CONTRACT = r"""
Return exactly one JSON object and no Markdown fence:
{
  "analysis": "short explanation",
  "files": [
    {"path": "model_new_ascendc.py", "content": "complete file content"},
    {"path": "kernel/pybind11.cpp", "content": "complete file content"},
    {"path": "kernel/<name>.cpp", "content": "complete file content"}
  ],
  "delete": ["kernel/obsolete_file.cpp"]
}
Paths omitted from files/delete remain unchanged. On the first round, provide a complete
self-contained implementation including model_new_ascendc.py, kernel/pybind11.cpp, at
least one non-pybind kernel .cpp, and every required header. Never write build artifacts.
""".strip()


RULES = """
- Implement the complete reference semantics with AscendC; do not use torch/ATen to perform core computation.
- model_new_ascendc.py may create/reshape tensors and call the compiled extension, but may not use torch operators as a fallback.
- pybind11.cpp handles validation, output/workspace allocation, tiling parameters and kernel launch only.
- pybind11.cpp must declare and call extern "C" *_do host wrappers. Define each wrapper
  beside its __aicore__ kernel and launch with kernel<<<blockDim, nullptr, stream>>>.
  Never include acl/acl_rt_launch.h or use ACLRT_LAUNCH_KERNEL in this project.
- Keep PYBIND11_MODULE's literal module name consistent with the module imported by model_new_ascendc.py.
- Cover every dtype, shape and attribute represented by the supplied cases.
- Prefer vectorized AscendC operations, aligned transfers and bounded UB usage.
- Before returning, verify every API owner (for example TQue rather than TPipe for EnQue)
  and check every local helper call's arity.
- Do not invent AscendC fields or overloads. When runtime header declarations are supplied,
  they override examples and fallback-version documentation.
""".strip()


PLAN_OUTPUT_CONTRACT = r"""
Return exactly one JSON object and no Markdown fence:
{
  "diagnosis": "evidence-based explanation of the current bottleneck or failure",
  "items": [
    {
      "id": "p1",
      "kind": "correctness or performance",
      "hypothesis": "one testable hypothesis",
      "change": "one concrete source change",
      "expected_signal": "evaluation result that would validate the hypothesis",
      "target_files": ["kernel/<name>.cpp"],
      "edit_scope": "file or region",
      "allow_interface_change": false,
      "evidence_refs": [{"source": "evaluation or source", "line_excerpt": "exact excerpt"}],
      "falsifies": ["previous hypothesis or approach id"],
      "order": 1
    }
  ]
}
Return exactly one next item grounded in the current accepted implementation and evidence.
Do not return source files in this response.
""".strip()


INITIAL_PLAN_OUTPUT_CONTRACT = r"""
Return exactly one JSON object and no Markdown fence:
{
  "diagnosis": "direct summary of the reference semantics and implementation constraints",
  "items": [
    {
      "id": "bootstrap-1",
      "kind": "correctness",
      "hypothesis": "one testable end-to-end AscendC implementation hypothesis",
      "change": "a complete implementation blueprint covering the algorithm, block/tiling strategy, memory and data movement, dtype/shape/tail handling, Host ABI, and source-file layout",
      "expected_signal": "static validation, compilation, all correctness cases, and a valid benchmark score succeed",
      "target_files": ["model_new_ascendc.py", "kernel/pybind11.cpp", "kernel/<name>.cpp"],
      "edit_scope": "file",
      "allow_interface_change": true,
      "evidence_refs": [],
      "falsifies": [],
      "order": 1
    }
  ]
}
Return exactly one complete baseline item. Do not return source files in this response.
Do not use or propose TileLang, another DSL, an intermediate implementation, or source-to-source conversion.
""".strip()


COMPILATION_REPAIR_CONTRACT = """
- The current implementation is the accepted repair base. Apply repairs cumulatively to it.
- Resolve the listed open compiler errors and preserve fixes represented by cleared error IDs.
- Do not reintroduce a previously cleared diagnostic.
- A normal runtime `if` statement does not prevent C++ template instantiation. Unsupported
  API/dtype combinations must be isolated at compile time or placed in type-specific
  implementations using syntax supported by the current toolchain.
- Installed declarations and current compiler diagnostics are authoritative. Do not invent
  overloads or hide type errors with unsupported casts.
- In `analysis`, name the targeted error, the repair, and the cleared errors you preserved.
""".strip()


def render_repair_state(repair_state: dict[str, Any] | None) -> str:
    """Render only the run-local state relevant to currently open errors."""

    if not repair_state:
        return "(no active trajectory repair state)"
    lines = [
        f"Accepted attempt: {repair_state.get('accepted_attempt_id') or 'none'}",
        f"Latest rejected attempt: {repair_state.get('latest_rejected_attempt_id') or 'none'}",
        "",
        "Open errors:",
    ]
    open_errors = repair_state.get("open_errors", [])
    if open_errors:
        for item in open_errors:
            lines.append(
                "- "
                + str(item.get("error_id", "unknown"))
                + f"; symbol={item.get('symbol') or 'unknown'}; category={item.get('category') or 'unknown'}"
                + f"; evidence={item.get('normalized_message') or item.get('excerpt') or ''}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "Cleared errors that must not regress:"])
    cleared = repair_state.get("cleared_error_ids", [])
    lines.extend([f"- {item}" for item in cleared] or ["- none"])
    lines.extend(["", "Previous failed approaches relevant to current open errors:"])
    relevant = repair_state.get("failed_approaches_by_error", {})
    if not relevant:
        lines.append("- none")
    for error_id, approaches in relevant.items():
        lines.append(f"- {error_id}")
        for approach in approaches:
            attempts = ",".join(str(item) for item in approach.get("attempt_ids", []))
            lines.append(
                f"  - {approach.get('approach_id')} (family={approach.get('approach_family_id', 'unknown')}): {approach.get('summary', '')} "
                f"(outcome={approach.get('outcome')}; occurrences={approach.get('occurrences', 1)}; "
                f"family_occurrences={approach.get('family_occurrences', approach.get('occurrences', 1))}; "
                f"attempts={attempts or 'unknown'})"
            )
    lines.extend(["", "Escalation state:", json.dumps(repair_state.get("escalation", {}), ensure_ascii=False)])
    lines.extend(["", "Accepted frontier progress:", json.dumps(repair_state.get("accepted_progress", {}), ensure_ascii=False)])
    return "\n".join(lines)


def render_stage_context(selection: Any) -> str:
    """Render a stage-selected context without slicing through atomic entries."""

    payload = selection.to_dict() if hasattr(selection, "to_dict") else dict(selection)
    audience = str(payload.get("audience", "generator"))
    budget = payload.setdefault("budget", {})
    full_selected = budget.get("input_mode") == "full_selected"
    max_chars = int(budget.get("max_chars") or 12000)
    truncated: list[str] = []
    chunks = [
        "# Selected AscendC reference context",
        f"Audience: {audience}",
        "Stages: " + ", ".join(payload.get("stages", [])),
        "Primary skill: " + str(payload.get("task_facts", {}).get("primary_skill", "unknown")),
        "Route reason: " + str(payload.get("task_facts", {}).get("route_reason", "unknown")),
        "Secondary skill: " + str(payload.get("task_facts", {}).get("secondary_skill") or "none"),
        (
            "This is reference knowledge, not a workflow to execute. Resolve conflicts in this order: "
            + " > ".join(payload.get("authority_order", []))
            + "."
        ),
    ]

    quotas = (
        {
            "hard": 2500,
            "failure": 3000,
            "api": 1500,
            "skills": 4000,
            "other": 1000,
        }
        if audience == "planner"
        else {
            "hard": 3000,
            "failure": 4000,
            "api": 6000,
            "skills": 5000,
            "other": 2000,
        }
    )
    rendered_sections: list[str] = []

    def append_section(title: str, entries: list[str], quota: int) -> None:
        if not entries:
            return
        section = [f"\n## {title}"]
        used = len(section[0])
        accepted = 0
        for index, entry in enumerate(entries):
            item = entry.strip()
            addition = len(item) + 2
            projected_global = len("\n".join([*chunks, *section, item]))
            if (
                used + addition > quota
                or projected_global > max_chars
            ):
                truncated.append(f"{title}[{index}]")
                continue
            section.append(item)
            used += addition
            accepted += 1
            rendered_sections.append(f"{title}[{index}]")
        if accepted:
            chunks.extend(section)

    append_section(
        "Task and platform facts",
        [json.dumps(payload.get("task_facts", {}), ensure_ascii=False, indent=2)],
        quotas["other"],
    )
    append_section(
        "Hard project constraints",
        [
            json.dumps(item, ensure_ascii=False, indent=2)
            for item in payload.get("hard_constraints", [])
        ],
        quotas["hard"],
    )
    append_section(
        "Explicit exclusions",
        [f"- {item}" for item in payload.get("exclusions", [])],
        quotas["other"],
    )

    failure_entries = [
        json.dumps(item, ensure_ascii=False, indent=2)
        for item in payload.get("failure_guidance", [])
    ]
    append_section("Failure-specific guidance", failure_entries, quotas["failure"])

    api_entries = []
    runtime_facts = str(payload.get("runtime_facts", "")).strip()
    if runtime_facts:
        api_entries.append("Installed public-header facts:\n" + runtime_facts)
    api_entries.extend(
        json.dumps(item, ensure_ascii=False, indent=2)
        for item in payload.get("api_facts", [])
    )
    append_section("Exact API and runtime facts", api_entries, quotas["api"])

    skill_entries: list[str] = []
    for knowledge_module in payload.get("skill_knowledge_modules", []):
        skill_entries.append(
            "\n".join(
                [
                    f"### {knowledge_module.get('skill_id')} [{knowledge_module.get('stage')}; role={knowledge_module.get('role', 'support')}]",
                    f"Purpose: {knowledge_module.get('purpose', '')}",
                    f"Triggered because: {knowledge_module.get('trigger_reason', '')}",
                    f"Expected artifact: {knowledge_module.get('expected_artifact', '')}",
                ]
            )
        )
        for excerpt in knowledge_module.get("excerpts", []):
            headings = ", ".join(excerpt.get("headings", [])) or "complete knowledge module"
            skill_entries.append(
                f"#### {knowledge_module.get('skill_id')} source: {excerpt.get('source')} ({headings})\n"
                f"Origin: {excerpt.get('origin', 'cannbot')}; confidence=Level {excerpt.get('confidence_level', 1)}; provenance={excerpt.get('provenance', 'documented_skill')}\n"
                + str(excerpt.get("text", ""))
            )
    # Selection is the prompt-width boundary for embedded knowledge modules. Once a
    # knowledge module section is selected, render it atomically and completely.
    if skill_entries:
        chunks.append("\n## Relevant embedded knowledge modules")
        chunks.extend(item.strip() for item in skill_entries)
        rendered_sections.extend(
            f"Relevant embedded knowledge modules[{index}]"
            for index in range(len(skill_entries))
        )

    other_entries = [
        f"- {item}" for item in payload.get("design_patterns", [])
    ]
    append_section("Selected design patterns", other_entries, quotas["other"])
    append_section(
        "Compact provenance",
        [
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in payload.get("provenance", [])
        ],
        quotas["other"],
    )
    text = "\n".join(chunks)
    budget["used_chars"] = len(text)
    budget["estimated_tokens"] = (len(text) + 3) // 4
    budget["truncated_sections"] = truncated
    metadata = payload.setdefault("selection_metadata", {})
    metadata["rendered_sections"] = rendered_sections
    metadata["truncated_sections"] = truncated
    metadata["rendered_chars"] = len(text)
    metadata["rendered_estimated_tokens"] = (len(text) + 3) // 4
    metadata["input_mode"] = "full_selected" if full_selected else "bounded"
    metadata["input_truncated"] = bool(truncated)
    metadata["rendered_skill_ids"] = [
        str(item.get("skill_id"))
        for item in payload.get("skill_knowledge_modules", [])
        if item.get("skill_id")
    ]
    metadata["rendered_knowledge_module_sections"] = [
        f"{excerpt.get('knowledge_module_id')}#{excerpt.get('section_id')}"
        for item in payload.get("skill_knowledge_modules", [])
        for excerpt in item.get("excerpts", [])
        if excerpt.get("knowledge_module_id") and excerpt.get("section_id")
    ]
    if full_selected:
        metadata["rendered_structured_ids"] = [
            str(item.get("fact_id") or item.get("card_id"))
            for item in payload.get("api_facts", [])
            if item.get("fact_id") or item.get("card_id")
        ]
        metadata["rendered_runtime_fact_ids"] = list(
            metadata.get("runtime_fact_ids", [])
        )
    if hasattr(selection, "budget"):
        selection.budget = budget
    if hasattr(selection, "selection_metadata"):
        selection.selection_metadata = metadata
    return text


def _bundle_text(bundle: FileBundle | None) -> str:
    if bundle is None or not bundle.files:
        return "(no implementation exists yet)"
    chunks = []
    for path, content in sorted(bundle.files.items()):
        chunks.append(f"### {path}\n```\n{content}\n```")
    return "\n\n".join(chunks)


def build_prompt(
    *,
    reference_code: str,
    cases_text: str,
    current: FileBundle | None,
    previous_result: EvalResult | None,
    round_num: int,
    knowledge_context: str,
    phase: str = "BOOTSTRAP",
    plan_item: dict | None = None,
    failure_fingerprints: list[str] | None = None,
    full_input: bool = False,
    repair_state: dict[str, Any] | None = None,
    compile_repair_active: bool = False,
    knowledge_selection: Any = None,
    protected_regions: dict[str, Any] | None = None,
) -> str:
    if previous_result is None:
        feedback = "No previous evaluation. Generate the initial implementation."
    else:
        feedback = json.dumps(
            compact_evaluation(previous_result),
            ensure_ascii=False,
            indent=2,
        )
    action = (
        "Generate the initial AscendC implementation."
        if current is None or not current.files
        else "Repair or optimize the current implementation using the evaluation evidence. Preserve working files unless they need changes."
    )
    plan_text = (
        json.dumps(plan_item, ensure_ascii=False, indent=2)
        if plan_item
        else "(no active plan item; only valid for a historical EVAL checkpoint)"
    )
    selection_payload = (
        knowledge_selection.to_dict()
        if hasattr(knowledge_selection, "to_dict")
        else dict(knowledge_selection or {})
    )
    task_facts = selection_payload.get("task_facts", {})
    ownership = str(task_facts.get("failure_ownership") or "Kernel")
    active_profile = str(task_facts.get("active_profile") or "full evaluation / unspecified")
    profile_details = json.dumps(
        {
            "case_indices": task_facts.get("profile_case_indices", []),
            "features": task_facts.get("profile_features", []),
            "diagnostic_source_files": task_facts.get("diagnostic_source_files", []),
            "input_route": task_facts.get("input_route", {}),
        },
        ensure_ascii=False,
        indent=2,
    )
    protected_text = json.dumps(
        protected_regions or {"protected_region_ids": [], "protected_snippets": {}},
        ensure_ascii=False,
        indent=2,
    )
    forbidden = list(dict.fromkeys(selection_payload.get("exclusions", [])))
    forbidden_text = "\n".join(f"- {item}" for item in forbidden) or "- Follow the mandatory negative rules below."
    compilation_section = (
        f"## Compilation repair contract\n{COMPILATION_REPAIR_CONTRACT}\n\n"
        if compile_repair_active
        else ""
    )
    repair_section = f"## Open errors, cleared errors, and relevant failed approaches\n{render_repair_state(repair_state)}\n\n"
    return f"""# AscendC {phase.lower()} edit attempt {round_num}

## Current objective
{action}

Active plan item:
```json
{plan_text}
```

## Failure ownership
{ownership}

## Active evaluation profile
{active_profile}

## Failing or newly introduced case features
```json
{profile_details}
```

## Protected regions
```json
{protected_text}
```

{repair_section}## Must-satisfy semantic and ABI contracts
{RULES}

{compilation_section}## Exact installed/Verified facts and relevant knowledge modules
{knowledge_context}

## Explicit forbidden patterns
{forbidden_text}

## Reference PyTorch model and benchmark semantics (read-only)
```python
{reference_code}
```

## Test cases
```jsonl
{cases_text}
```

## Current implementation
{_bundle_text(current)}

## Previous evaluation
```json
{feedback}
```

## Output contract
{OUTPUT_CONTRACT}
"""


def build_plan_prompt(
    *,
    reference_code: str,
    cases_text: str,
    current: FileBundle | None,
    result: EvalResult | None,
    mode: str,
    history: list[dict],
    diagnosis_required: bool = False,
    knowledge_context: str = "",
    initial: bool = False,
    full_input: bool = False,
    repair_state: dict[str, Any] | None = None,
) -> str:
    feedback = (
        json.dumps(
            compact_evaluation(result),
            ensure_ascii=False,
            indent=2,
        )
        if result is not None
        else "No implementation has been evaluated yet."
    )
    ledger_lines: list[str] = []
    for record in history[-8:]:
        repair = record.get("repair_attempt", {}) if isinstance(record, dict) else {}
        item = record.get("plan_item", {}) if isinstance(record, dict) else {}
        ledger_lines.append(
            " | ".join(
                (
                    f"attempt={record.get('attempt_id', record.get('round'))}",
                    f"hypothesis={str(item.get('hypothesis', ''))[:240]}",
                    f"outcome={repair.get('outcome') or record.get('decision')}",
                    f"cleared={repair.get('cleared_error_ids', [])}",
                    f"new={repair.get('new_error_ids', [])}",
                    f"progress={repair.get('progress', {})}",
                )
            )
        )
    ledger = "\n".join(ledger_lines) or "(no completed attempts)"
    if initial:
        purpose = (
            "Create one complete implementation blueprint directly in AscendC terms before any source is generated."
        )
        execution_rules = (
            "The single plan item must describe a complete candidate, not a partial file or staged implementation. "
            "Reason from the reference semantics, cases, CANN constraints, and AscendC execution model only."
        )
        output_contract = INITIAL_PLAN_OUTPUT_CONTRACT
    else:
        purpose = (
            "Diagnose why the previous plan repeatedly failed and replace it with materially different approaches."
            if diagnosis_required
            else "Create the next evidence-driven implementation plan."
        )
        execution_rules = (
            "Return one vertical next step for the current accepted implementation. The item may\n"
            "touch multiple files when required by one primary hypothesis. Do not repeat a failed\n"
            "approach under a new name."
        )
        output_contract = PLAN_OUTPUT_CONTRACT
    repair_section = (
        f"## Current trajectory repair state\n{render_repair_state(repair_state)}\n\n"
        if repair_state and mode == "bootstrap" and not initial
        else ""
    )
    return f"""# AscendC {mode} planning

{purpose}
{execution_rules}

## Mandatory rules
{RULES}

The project's host launch ABI is fixed: pybind declares/calls extern "C" *_do wrappers,
and each wrapper is defined beside its __aicore__ kernel using kernel<<<blockDim,
nullptr, stream>>>. acl/acl_rt_launch.h and ACLRT_LAUNCH_KERNEL are unsupported.

## Reference PyTorch model (read-only)
```python
{reference_code}
```

## Test cases
```jsonl
{cases_text}
```

## Current implementation
{_bundle_text(current)}

## Latest evaluation evidence
```json
{feedback}
```

## Attempt ledger
{ledger}

## Stage-selected AscendC references
{knowledge_context or "(no additional reference material selected)"}

{repair_section}## Output contract
{output_contract}
"""


def parse_plan(
    text: str,
    *,
    min_items: int = 3,
    max_items: int = 5,
    require_evidence: bool = False,
) -> dict:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("planning response does not contain a JSON object")
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("planning response must contain an items list")
    if not min_items <= len(payload["items"]) <= max_items:
        expected = str(min_items) if min_items == max_items else f"{min_items} to {max_items}"
        raise ValueError(f"planning response must contain {expected} items")
    normalized = []
    required = ("id", "kind", "hypothesis", "change")
    for index, item in enumerate(payload["items"], 1):
        if not isinstance(item, dict) or any(not str(item.get(key, "")).strip() for key in required):
            raise ValueError(f"plan item {index} is incomplete")
        expected_signal = str(item.get("expected_signal", "")).strip()
        change = str(item["change"]).strip()
        if not expected_signal:
            embedded = re.search(
                r"(?ims)^\s*\d+[.)]\s*expected signal\s*$\n(?P<value>.+)\Z",
                change,
            )
            if embedded:
                expected_signal = embedded.group("value").strip()
                change = change[: embedded.start()].rstrip()
        if not expected_signal:
            raise ValueError(f"plan item {index} is incomplete")
        target_files = item.get("target_files", [])
        evidence_refs = item.get("evidence_refs", [])
        falsifies = item.get("falsifies", [])
        if not isinstance(target_files, list) or not all(isinstance(path, str) for path in target_files):
            raise ValueError(f"plan item {index} target_files must be a string list")
        if not isinstance(evidence_refs, list) or not all(isinstance(ref, dict) for ref in evidence_refs):
            raise ValueError(f"plan item {index} evidence_refs must be an object list")
        if not isinstance(falsifies, list) or not all(isinstance(value, str) for value in falsifies):
            raise ValueError(f"plan item {index} falsifies must be a string list")
        if require_evidence and (
            not evidence_refs
            or any(not str(ref.get("line_excerpt", "")).strip() for ref in evidence_refs)
        ):
            raise ValueError(f"diagnosis plan item {index} must cite exact evidence excerpts")
        if require_evidence and not falsifies:
            raise ValueError(f"diagnosis plan item {index} must identify a falsified prior approach")
        normalized.append(
            {
                **{key: str(item[key]).strip() for key in required if key != "change"},
                "change": change,
                "expected_signal": expected_signal,
                "target_files": [path.strip() for path in target_files if path.strip()],
                "edit_scope": str(item.get("edit_scope", "file")).strip() or "file",
                "allow_interface_change": bool(
                    item.get("allow_interface_change", item.get("allow_abi_change", False))
                ),
                "evidence_refs": evidence_refs,
                "falsifies": [value.strip() for value in falsifies if value.strip()],
                "order": int(item.get("order", index)),
            }
        )
    return {"diagnosis": str(payload.get("diagnosis", "")).strip(), "items": normalized}
