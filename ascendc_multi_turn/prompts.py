from __future__ import annotations

import json
from pathlib import Path

from .models import EvalResult, FileBundle


REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_FILES = (
    REPO_ROOT / "skills/ascendc/ascendc-translator/references/dsl2Ascendc.md",
    REPO_ROOT / "skills/ascendc/ascendc-translator/references/TileLang-AscendC-API-Mapping.md",
    REPO_ROOT / "skills/ascendc/ascendc-translator/references/AscendCVerification.md",
)

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
- Keep PYBIND11_MODULE's literal module name consistent with the module imported by model_new_ascendc.py.
- Cover every dtype, shape and attribute represented by the supplied cases.
- Prefer vectorized AscendC operations, aligned transfers and bounded UB usage.
""".strip()


def _references() -> str:
    chunks = []
    for path in REFERENCE_FILES:
        if path.is_file():
            chunks.append(f"## {path.name}\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(chunks)


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
) -> str:
    if previous_result is None:
        feedback = "No previous evaluation. Generate the initial implementation."
    else:
        feedback = json.dumps(previous_result.to_dict(), ensure_ascii=False, indent=2)
    action = (
        "Generate the initial AscendC implementation."
        if current is None or not current.files
        else "Repair or optimize the current implementation using the evaluation evidence. Preserve working files unless they need changes."
    )
    return f"""# AscendC generation round {round_num}

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

## AscendC references
{_references()}

## Output contract
{OUTPUT_CONTRACT}
"""
