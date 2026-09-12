# Stage-aware context schema

## Goal

Deliver the smallest sufficient, highest-authority context to each LLM call.
The schema is an online projection over existing CannAgent knowledge plus
selected CANNBot domain modules; it is not a new knowledge store.

## 1. Selection input

```yaml
context_request:
  audience: planner | generator
  workflow_phase: bootstrap | optimization
  derived_stages:
    - operator_analysis
    - kernel_design
    - code_generation
    - compile_debug
    - runtime_debug
    - precision_debug
    - optimization
  operator:
    name: string
    families: [elementwise, reduction, broadcast, conversion, matmul, sort, flashattention, simt, unknown]
    semantics_summary: string
  cases:
    shapes: []
    dtypes: []
    attributes: {}
    alignment_hints: []
  platform:
    soc: string
    runtime_cann: string
    knowledge_cann: string
  source:
    paths: []
    exact_api_symbols: []
    relevant_calls: []
  plan:
    active_item: object | null
  failure:
    stage: string | null
    code: string | null
    subsystem: string | null
    concise_diagnostics: string
    related_symbols: []
    fingerprints: []
```

The implementation may initially infer family, shapes, and dtypes from text.
Unknown fields stay unknown; they must not be fabricated.

## 2. Selected context envelope

```yaml
selected_context:
  schema_version: 1
  audience: planner | generator
  stages: []
  authority_order:
    - current_evaluation
    - installed_headers
    - official_structured_facts
    - project_contracts
    - cannbot_practices
    - confirmed_experience
    - examples
  task_facts:
    operator: string
    families: []
    soc: string
    runtime_cann: string
    failure_stage: string | null
  hard_constraints: []
  api_facts: []
  design_patterns: []
  failure_guidance: []
  skill_capsules: []
  exclusions: []
  provenance: []
  budget:
    max_chars: integer
    used_chars: integer
    truncated_sections: []
  selection_trace: []
```

Each `skill_capsule` has this form:

```yaml
skill_capsule:
  skill_id: string
  stage: string
  purpose: string
  trigger_reason: string
  expected_artifact: string
  provided_context: []
  constraints: []
  exclusions: []
  excerpts:
    - source: relative/path
      headings: []
      text: string
```

## 3. Stage projections

| Stage | Must receive | May receive | Must not receive |
|---|---|---|---|
| `operator_analysis` | reference semantics summary, case shapes/dtypes/attributes, SoC identity, supported architecture facts | one operator-family classifier | API dumps, debug playbooks, performance variants, full templates |
| `kernel_design` | operator family, core/tile/tail strategy constraints, UB/buffer budget, dtype/alignment branches, project ABI | one family-specific tiling guide and exact API restrictions already implied by the design | unrelated families, runtime/plog guides, complete project scaffolds |
| `code_generation` | active plan, exact current API facts, installed-header facts, project contracts, one compatible structural template excerpt | family-specific CopyIn/Compute/CopyOut or launch example | other target platforms, CMake/run/test scaffolds, broad performance corpus, unrelated APIs |
| `compile_debug` | concise compiler diagnostics, failing source calls, exact header/API facts, ABI/source contracts | direct-invoke structural excerpt when launch/module layout is implicated | numerical precision guides, unrelated APIs/families, old full logs |
| `runtime_debug` | error code/subsystem, failing case, launch/tiling facts, relevant runtime decision branch | one error-code or kernel-binary section | performance guides, full API catalog, precision material unless the runtime fault is cleared |
| `precision_debug` | error symptoms/distribution, failing/passing dtype+shape cases, golden/output hints, dataflow and synchronization facts | one matching trap/instrumentation section and exact implicated APIs | environment/kernel-lookup guides, unrelated numerical traps, performance tuning |
| `optimization` | correct baseline, per-case latency, operator family, architecture, current tiling/buffer design, one testable hypothesis | one family-specific and at most one common optimization | debug playbooks, unmeasured optimizations, other families, changes that relax correctness |

## 4. Audience projections

### Planner / diagnoser

Purpose: decide what to do. Prefer constraints, formulas, classification, and
diagnostic decision branches.

Include:

- task/platform facts;
- project hard constraints;
- family-specific design guidance;
- failure cards and matching debug skill capsule when a failure exists;
- brief provenance identifiers.

Exclude:

- full API card JSON;
- long code examples;
- full current logs (already represented by compact evaluation evidence);
- redundant template/source files.

Recommended default knowledge budget: 12,000 characters.

### Generator / editor

Purpose: implement one active plan item. Prefer exact call constraints and a
small compatible code pattern.

Include:

- active plan and task/platform facts;
- exact API facts for source or plan symbols;
- all applicable project contracts;
- matching failure guidance;
- selected template/reference excerpts;
- explicit exclusions and authority order.

Exclude:

- retrieval trace and verbose provenance bodies;
- unrelated pattern cards;
- entire skill files or reference directories;
- workflow instructions from CANNBot.

Recommended default knowledge budget: 20,000 characters.

## 5. Stage derivation rules

Rules are ordered and deterministic:

1. No current implementation and planner audience:
   `operator_analysis + kernel_design`.
2. No current implementation and generator audience:
   `kernel_design + code_generation`.
3. Source/static/API-validation/build failure:
   `compile_debug` plus `code_generation` for the generator.
4. Correctness-stage failure with runtime code, device exception, or
   ACL/AICORE/MTE/RUNTIME subsystem:
   `runtime_debug`.
5. Correctness-stage mismatch, NaN/Inf, zero/random output, or tolerance signal
   without a runtime fault:
   `precision_debug`.
6. Valid correct baseline in optimization budget:
   `optimization`; generator also receives `code_generation` contracts but not
   the initial scaffold.
7. Ambiguous correctness failure may select runtime and precision capsules, but
   each gets half its normal excerpt budget and the planner must first classify
   the failure.

## 6. Budget and truncation policy

Use category quotas rather than truncating one serialized JSON blob:

| Priority | Category | Planner | Generator |
|---|---|---:|---:|
| 1 | authority statement + hard project constraints | 2,500 | 3,000 |
| 2 | current failure guidance | 3,000 | 4,000 |
| 3 | exact API/header facts | 1,500 | 6,000 |
| 4 | family design / CANNBot capsule | 4,000 | 5,000 |
| 5 | short examples and provenance IDs | 1,000 | 2,000 |

Within a category, preserve complete facts/sections. If the next item does not
fit, omit it and record the omission; never slice through an API signature,
constraint, or code block.

## 7. Selection and exclusion rules

- Exact API identity is mandatory for API semantics. Similar names do not
  inherit each other's facts.
- Operator-family routing precedes pattern/example routing.
- CANNBot references are allowlisted in `skill_mapping.yaml`; links discovered
  inside an excerpt are not recursively loaded.
- Only selected Markdown headings are loaded. Code templates are never loaded
  wholesale by default.
- A skill trigger must explain itself in `trigger_reason`.
- All rejected skill mappings and excerpts should be traceable with a reason.
- The same excerpt should not be repeated across planner and generator unless
  it is a hard constraint needed by both.
- If the CANNBot repository is unavailable, the adapter degrades to existing
  structured knowledge and records `source_unavailable`; it must not fail the
  generation workflow.

## 8. Prompt rendering contract

Render in this order:

```text
# Selected AscendC context
Audience: ...
Stages: ...

## Authority and conflict policy
...

## Task and platform facts
...

## Hard constraints
...

## Exact API facts
...

## Failure-specific guidance
...

## Selected CANNBot skill capsules
...

## Explicit exclusions
...
```

The prompt builder must identify this as reference context, not as a workflow to
execute. Existing mandatory rules and output JSON contracts remain unchanged.

## 9. Audit artifacts

For each call, persist:

- `planner_context.json` / `generator_context.json`: selected stage, skill,
  excerpts, exclusions, and character accounting;
- `planner_references.md` / `references.md`: exact rendered context;
- existing `knowledge_bundle.json` and `retrieval_trace.json`: underlying
  structured retrieval;
- token usage already recorded in `calls.jsonl` and summary.

These artifacts make baseline/new token and relevance comparisons possible
without inspecting provider traffic.
