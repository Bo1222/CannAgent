from __future__ import annotations

import json
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
- Before returning, check for duplicate file-scope declarations, verify every API owner
  (for example TQue rather than TPipe for EnQue), and check every local helper call's arity.
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
      "expected_signal": "evaluation result that would validate the hypothesis"
    }
  ]
}
Return 3 to 5 independent items. Do not return source files in this response.
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
      "expected_signal": "static validation, compilation, all correctness cases, and a valid benchmark score succeed"
    }
  ]
}
Return exactly one complete baseline item. Do not return source files in this response.
Do not use or propose TileLang, another DSL, an intermediate implementation, or source-to-source conversion.
""".strip()


def render_stage_context(selection: Any) -> str:
    """Render a stage-selected context without slicing through atomic entries."""

    payload = selection.to_dict() if hasattr(selection, "to_dict") else dict(selection)
    audience = str(payload.get("audience", "generator"))
    budget = payload.setdefault("budget", {})
    max_chars = int(budget.get("max_chars", 12000))
    truncated: list[str] = []
    chunks = [
        "# Selected AscendC reference context",
        f"Audience: {audience}",
        "Stages: " + ", ".join(payload.get("stages", [])),
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
            if used + addition > quota or projected_global > max_chars:
                truncated.append(f"{title}[{index}]")
                continue
            section.append(item)
            used += addition
            accepted += 1
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
    for capsule in payload.get("skill_capsules", []):
        skill_entries.append(
            "\n".join(
                [
                    f"### {capsule.get('skill_id')} [{capsule.get('stage')}]",
                    f"Purpose: {capsule.get('purpose', '')}",
                    f"Triggered because: {capsule.get('trigger_reason', '')}",
                    f"Expected artifact: {capsule.get('expected_artifact', '')}",
                ]
            )
        )
        for excerpt in capsule.get("excerpts", []):
            headings = ", ".join(excerpt.get("headings", [])) or "bounded file excerpt"
            skill_entries.append(
                f"#### {capsule.get('skill_id')} source: {excerpt.get('source')} ({headings})\n"
                + str(excerpt.get("text", ""))
            )
    append_section("Selected CANNBot skill capsules", skill_entries, quotas["skills"])

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
    budget["truncated_sections"] = truncated
    if hasattr(selection, "budget"):
        selection.budget = budget
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
) -> str:
    if previous_result is None:
        feedback = "No previous evaluation. Generate the initial implementation."
    else:
        feedback = json.dumps(compact_evaluation(previous_result), ensure_ascii=False, indent=2)
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
    return f"""# AscendC {phase.lower()} edit attempt {round_num}

{action}

## Mandatory rules
{RULES}

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

## Previous evaluation
```json
{feedback}
```

## Active plan item
```json
{plan_text}
```

Previously observed normalized failure fingerprints: {json.dumps(failure_fingerprints or [])}

## Stage-selected AscendC references
{knowledge_context}

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
) -> str:
    feedback = (
        json.dumps(compact_evaluation(result), ensure_ascii=False, indent=2)
        if result is not None
        else "No implementation has been evaluated yet."
    )
    compact_history = []
    for record in history[-8:]:
        evaluation = record.get("evaluation", {}) if isinstance(record, dict) else {}
        compact_history.append(
            {
                "attempt_id": record.get("attempt_id", record.get("round")),
                "budget_phase": record.get("budget_phase"),
                "decision": record.get("decision"),
                "score": evaluation.get("score"),
                "failure_stage": evaluation.get("failure_stage"),
                "failure_code": evaluation.get("failure_code"),
                "diagnostics": evaluation.get("error_excerpt", "")[:4000],
                "failure_fingerprint": record.get("failure_fingerprint"),
                "plan_item": record.get("plan_item"),
            }
        )
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
            "The plan will be executed one item per candidate evaluation. Do not propose multiple\n"
            "edits inside one item and do not repeat a failed approach under a new name."
        )
        output_contract = PLAN_OUTPUT_CONTRACT
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

## Stage-selected AscendC references
{knowledge_context or "(no additional reference material selected)"}

## Recent settled attempts
```json
{json.dumps(compact_history, ensure_ascii=False, indent=2)}
```

## Output contract
{output_contract}
"""


def parse_plan(text: str, *, min_items: int = 3, max_items: int = 5) -> dict:
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
    required = ("id", "kind", "hypothesis", "change", "expected_signal")
    for index, item in enumerate(payload["items"], 1):
        if not isinstance(item, dict) or any(not str(item.get(key, "")).strip() for key in required):
            raise ValueError(f"plan item {index} is incomplete")
        normalized.append({key: str(item[key]).strip() for key in required})
    return {"diagnosis": str(payload.get("diagnosis", "")).strip(), "items": normalized}
