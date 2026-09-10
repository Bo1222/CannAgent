---
name: ascendc-translator
description: Directly implement and optimize an AscendC kernel for a PyTorch reference model.
argument-hint: Input an output_dir containing model.py and cases; generate kernel/ and model_new_ascendc.py.
---

# Direct AscendC Kernel Skill

Implement the reference operator directly in AscendC. Do not require or generate a TileLang
or other DSL intermediate. Keep core tensor computation in one custom AscendC operator;
Python and Host code may validate metadata, allocate output/workspace, prepare tiling, and launch.

Before generation, read `@references/AscendC_knowledge/project_guides/host_binding.md`.
Select only relevant project patterns and exact API documents from
`@references/AscendC_knowledge/`. Installed CANN declarations and compiler/runtime evidence
override fallback documentation.

Generate `kernel/` sources plus `model_new_ascendc.py`, then run
`@references/evaluate_ascendc.sh {output_dir}`. Preserve working code between repairs, make
the smallest evidence-driven change, and never fall back to PyTorch/ATen computation.
