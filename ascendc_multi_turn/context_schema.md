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
    active_profile: smoke | shape | dtype | full | benchmark | null
    profile_case_indices: []
    profile_features: [broadcast | tail | dtype]
  platform:
    soc: string
    runtime_cann: string
    knowledge_cann: string
    environment_fingerprint: string | null
  source:
    paths: []
    source_symbols: []
    relevant_calls: []
  plan:
    active_item: object | null
    planned_symbols: []
  failure:
    stage: string | null
    code: string | null
    subsystem: string | null
    concise_diagnostics: string
    failure_symbols: []
    fingerprints: []
  routing:
    primary_skill: string
    route_reason: string
    secondary_skill: string | null
    secondary_reason: string | null
    debug_category: string
    routing_confidence: direct | corroborated | fallback
    failure_ownership: Host | Host/Kernel boundary | Kernel
    diagnostic_source_files: []
    input_route: object
```

The implementation may initially infer family, shapes, and dtypes from text.
Unknown fields stay unknown; they must not be fabricated.

## 2. Selected context envelope

```yaml
selected_context:
  schema_version: 2
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
    failure_symbols: []
    source_symbols: []
    planned_symbols: []
    symbol_evidence:
      - symbol: string
        kinds: [failure | source | planned]
        sources: [evaluation | candidate | active_plan]
        domains: [ascendc_api | host_abi | launch_abi | operator_semantic | project_local | compiler_noise]
    primary_skill: string
    route_reason: string
    secondary_skill: string | null
    secondary_reason: string | null
    environment_fingerprint: object
    failure_ownership: string
    diagnostic_source_files: []
    active_profile: string | null
    profile_case_indices: []
    profile_features: []
    input_route: object
  hard_constraints: []
  api_facts: []
  design_patterns: []
  failure_guidance: []
  skill_knowledge_modules: []
  exclusions: []
  provenance: []
  budget:
    max_chars: integer
    used_chars: integer
    truncated_sections: []
  selection_trace: []
  selection_metadata:
    selected_skill_ids: []
    selected_structured_ids: []
    runtime_fact_ids: []
    selected_items:
      - id: string
        source: structured | runtime | cannbot | adapter
        evidence_symbol: string | null
        symbol_types: [failure | source | planned | fallback]
        confidence_level: 0 | 1 | 2 | 3
        provenance: string | []
    rendered_chars: integer
    rendered_estimated_tokens: integer
    rendered_sections: []
    truncated_sections: []
```

Each `skill_knowledge_module` has this form:

```yaml
skill_knowledge_module:
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
      knowledge_module_id: string
      section_id: string
      headings: []
      text: string
      origin: cannbot | adapter
      confidence_level: 0 | 1 | 2 | 3
      provenance: string
```

## 3. Stage projections

| Stage | Must receive | May receive | Must not receive |
|---|---|---|---|
| `operator_analysis` | reference semantics summary, case shapes/dtypes/attributes, SoC identity, supported architecture facts | one operator-family classifier | API dumps, debug playbooks, performance variants, full templates |
| `kernel_design` | operator family, core/tile/tail strategy constraints, UB/buffer budget, dtype/alignment branches, project ABI | one family-specific tiling guide and exact API restrictions already implied by the design | unrelated families, runtime/plog guides, complete project scaffolds |
| `code_generation` | active plan, exact current API facts, installed-header facts, project contracts, one compatible structural template excerpt | family-specific CopyIn/Compute/CopyOut or launch example | other target platforms, CMake/run/test scaffolds, broad performance corpus, unrelated APIs |
| `host_integration_debug` | Host diagnostic, current pybind source, local positive/negative ABI facts, installed/probed tensor+stream declarations | one direct-invoke structural excerpt | kernel math, tiling changes, CUDA patterns, unverified exact Host signatures |
| `compile_debug` | concise compiler diagnostics, failing source calls, exact header/API facts, ABI/source contracts | direct-invoke structural excerpt when launch/module layout is implicated | numerical precision guides, unrelated APIs/families, old full logs |
| `runtime_debug` | error code/subsystem, failing case, launch/tiling facts, relevant runtime decision branch | one error-code or kernel-binary section | performance guides, full API catalog, precision material unless the runtime fault is cleared |
| `precision_debug` | explicit dtype/cast/accumulation/rounding/epsilon/tolerance evidence, failing/passing cases, dataflow facts | one matching trap/instrumentation section and exact implicated APIs | generic large mismatch, indexing/data-movement errors, environment and performance guidance |
| `optimization` | correct baseline, per-case latency, operator family, architecture, current tiling/buffer design, one testable hypothesis | one family-specific and at most one common optimization | debug playbooks, unmeasured optimizations, other families, changes that relax correctness |

`code_generation` is selected for the initial candidate. Later pure Kernel correctness,
runtime, compile, and optimization repairs do not receive it automatically; their own
stage knowledge modules carry the applicable contracts. Selected embedded sections are deduplicated
by `knowledge_module_id + section_id` and are never partially truncated.

## 4. Audience projections

### Planner / diagnoser

Purpose: decide what to do. Prefer constraints, formulas, classification, and
diagnostic decision branches.

Include:

- task/platform facts;
- project hard constraints;
- family-specific design guidance;
- failure cards and matching debug skill knowledge module when a failure exists;
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
3. Source/static/API-validation/build failure involving CUDA contamination,
   NPU tensor checks, stream, Host wrapper/link/import evidence:
   `host_integration_debug`.
4. GM_ADDR/`__gm__`, descriptor transport, signature or cast evidence also routes
   to `host_integration_debug` with Host/Kernel boundary ownership. Other exact
   AscendC source/API/build failures route to `compile_debug`.
5. Correctness-stage failure with runtime code, device exception, or
   ACL/AICORE/MTE/RUNTIME subsystem:
   `runtime_debug`.
6. Correctness failure after successful compile/load/execute defaults to
   `kernel_design`. A broad numerical mismatch alone never selects precision.
7. Select `precision_debug` as primary only for direct precision evidence such
   as FP32 pass/FP16 fail, cast/accumulation dtype, rounding, epsilon,
   overflow/underflow, or a tolerance-boundary failure. It may be one optional
   secondary to `kernel_design` when the evidence supports both hypotheses.
8. Valid correct baseline in optimization budget:
   `optimization`.

There is exactly one primary route. There is at most one secondary route, and
it is empty unless current deterministic evidence states a second hypothesis.
No LLM classifier, magnitude scorer, distribution classifier, or numeric
confidence model participates in routing.

## 6. Budget and truncation policy

Use category quotas rather than truncating one serialized JSON blob:

| Priority | Category | Planner | Generator |
|---|---|---:|---:|
| 1 | authority statement + hard project constraints | 2,500 | 3,000 |
| 2 | current failure guidance | 3,000 | 4,000 |
| 3 | exact API/header facts | 1,500 | 6,000 |
| 4 | family design / CANNBot knowledge module | 4,000 | 5,000 |
| 5 | short examples and provenance IDs | 1,000 | 2,000 |

Within a category, preserve complete facts/sections. If the next item does not
fit, omit it and record the omission; never slice through an API signature,
constraint, or code block.

## 7. Selection and exclusion rules

- Exact API identity is mandatory for API semantics. Similar names do not
  inherit each other's facts.
- Symbol evidence is ordered `failure > source > planned`; all three sets are
  retained, and an empty set never hard-filters the remaining useful cards.
- Debug selection uses the union of failure, current-source, and relevant plan
  symbols. A bounded fallback remains available when no exact card exists.
- Operator-family routing precedes pattern/example routing.
- CANNBot references are allowlisted in `skill_mapping.yaml`; links discovered
  inside an excerpt are not recursively loaded.
- Only selected Markdown headings are loaded. Code templates are never loaded
  wholesale by default.
- A skill trigger must explain itself in `trigger_reason`.
- All rejected skill mappings and excerpts should be traceable with a reason.
- The same excerpt should not be repeated across planner and generator unless
  it is a hard constraint needed by both.
- Missing manifests/documents, hash drift, unsafe paths, unknown knowledge module IDs, or
  disallowed stages fail during Adapter initialization. Hybrid retains selected
  structured facts; Skills-only must not fall back to structured prompt knowledge.

## 8. Runtime knowledge provenance and invalidation

Runtime facts use four explicit levels:

| Level | Name | Meaning |
|---:|---|---|
| 3 | Verified | A minimal probe compiled with the current project toolchain and matching environment fingerprint. |
| 2 | Installed | Parsed from current installed CANN/torch_npu public headers, without an independent matching probe. |
| 1 | Documented | Official documentation, structured cards, or Skill references not proven against this exact installation. |
| 0 | Inferred | Unverified inference or header absence without direct compiler corroboration. |

The generator preference is `Verified > Installed > Documented >> Inferred`.
Level 0 is not sufficient grounding for a critical Host ABI or a previously
failing AscendC API. A verified fact is bound to CANN/toolkit root,
PyTorch/torch_npu versions, Host C++ and CANN `bisheng` compiler identities,
include roots, SoC, project build-contract hash, and probe-contract hash.
Fingerprint mismatch invalidates Level 3 and
forces re-probe or downgrade. Online/newest documentation never becomes Level
3 without the local probe.

## 9. Prompt rendering contract

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

## Selected CANNBot skill knowledge modules
...

## Explicit exclusions
...
```

The prompt builder must identify this as reference context, not as a workflow to
execute. Existing mandatory rules and output JSON contracts remain unchanged.

## 10. Audit artifacts

For each call, persist:

- `planner_context.json` / `generator_context.json`: selected stage, skill,
  excerpts, exclusions, and character accounting;
- `planner_references.md` / `references.md`: exact rendered context;
- existing `knowledge_bundle.json` and `retrieval_trace.json`: underlying
  structured retrieval;
- token usage already recorded in `calls.jsonl` and summary.

`calls.jsonl.prompt_metadata.knowledge_selection` records selected and rejected
IDs, symbol kinds, route reason, provenance/confidence, and exact rendered
sizes. Each trajectory round records observation-only stages A–H and evaluator
timings. `summary.json.stage_normalized_metrics` reports transition rates,
censored tokens/time-to-first-stage, calls/tokens per call, prompt component
sizes, and LLM/evaluator stage latency. An unreached milestone is `null`
(`N/A/censored`), never zero.

These artifacts make baseline/new token and relevance comparisons possible
without inspecting provider traffic.

## 11. 渐进评测、单调进展与接口契约

`EvalResult` 额外记录 `active_profile`、`passed_profiles`、`case_results` 和
`passed_case_indices`。评测顺序固定为 `smoke → shape → dtype → full → benchmark`；只有
`full` 通过才允许 `correctness=true`，只有随后获得有效正分数才建立 baseline。

每轮同时持久化以下规范门：`source_valid`、`compiled`、`loaded`、`kernel_started`、
`comparison_completed`、`full_correct`、`benchmarked`。Repair state 仅在未重现已清除错误且
门、profile、已通过 case 集或直接诊断严格前进时接受整个 bundle，不进行逐文件拼接。

首个被接受的已编译 bundle 生成 `interface_contract.json`。后续计划项默认
`allow_interface_change=false`；候选改变 pybind 模块、Host wrapper 声明/定义或 Kernel entry 时，
必须显式授权并重新通过编译。DIAGNOSE 计划还必须提供 `evidence_refs.line_excerpt` 和
`falsifies`，避免在没有直接证据时重复同族假设。
