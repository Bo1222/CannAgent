# Project Change Log

All repository modifications must be recorded here. Dates use `YYYY-MM-DD`.
Secrets must never be included.

## 2026-09-09

### Changes

- Added the M0 offline `knowledge_v2` compiler MVP with deterministic Markdown normalization, structured documents, provenance-bearing atomic parameter facts, a query API, real DataCopyPad coverage, and a phase TODO/report without changing runtime Agent behavior.
- Added the semantic knowledge target architecture and M0-M5 migration specifications under `repo_ascendc/` as the staged implementation contract for the offline compiler, snapshots, router, failure pipeline, validator, and experience loop.
- Began functionally decoupling `ascendc_multi_turn` knowledge routing from TileLang: direct routing now uses an `ascendc` domain, excludes all `dsl2Ascendc_*` supplements, accepts legacy router output for compatibility, and removes stale DSL supplements when loading resumable knowledge state.
- Added a dedicated one-item initial planning contract that requires a complete direct AscendC baseline blueprint and explicitly rejects TileLang, DSL intermediates, and source-to-source conversion.
- Changed new-task orchestration to select pure AscendC knowledge, create and persist the initial plan, then generate the first candidate; PLAN and generator reuse the same selected knowledge, while pending EVAL resumes still bypass model calls.
- Updated mock planning responses and compatibility coverage for the strict one-item initial plan, neutral AscendC domain, rejected DSL supplements, and legacy-state sanitation.
- Added end-to-end assertions for the new `knowledge_router -> planner -> generator` startup order, persisted bootstrap plan item, clean direct-reference context, and unchanged EVAL-checkpoint resume behavior.
- Added regressions with an isolated planner-failure fixture proving a failed initial planner pauses at the same non-counting checkpoint and resumes successfully, and proving already-persisted DSL supplements are removed on load.
- Updated the README and direct/non-Claude architecture guides to document the pure AscendC knowledge boundary, pre-generation baseline plan, later evidence-driven plans, artifact compatibility, and absence of TileLang conversion in `ascendc_multi_turn`.
- Added an optional `--max-total-rounds` limit spanning bootstrap and optimization evaluations, with explicit `stop_reason` reporting, so fixed-candidate diagnostic experiments terminate at an exact total without changing existing dual-budget defaults.

### Analysis

- `ascendc_multi_turn` has no TileLang import, compiler invocation, generated TileLang file, or TileLang evaluation stage. The removed coupling was prompt-level: five allowlisted `dsl2Ascendc_*` supplements could inject translation assumptions into otherwise direct AscendC generation.
- The shared versioned AscendC API corpus, runtime-header extraction, static validator, build tool, correctness verifier, and performance evaluator are direct AscendC dependencies and remain in use even though some are physically stored below the legacy translator Skill directory.
- The retained interactive TileLang/translator Skills are separate compatibility entry points; deleting them is unnecessary for functional isolation of the direct multi-turn runner.
- A fixed diagnostic window cannot be expressed by `--max-rounds` alone because that budget starts only after a valid baseline; the independent total cap is required to compare unsuccessful and successful trajectories on the same number of evaluated candidates.

### Validation

- Validated M0 against the real CANN 8.5.0 DataCopyPad page, including two context-preserving `blockLen` rows, schema round trips, source hashes, section evidence, and legacy repository regressions.
- Passed all 76 repository unit tests, including initial-plan failure/resume, legacy DSL-state sanitation, direct knowledge selection, planning, source safety, evaluator, and checkpoint coverage.
- Passed Python compilation, Ruff checks, and `git diff --check`.
- Completed a no-NPU mock run with call order `knowledge_router -> planner -> generator -> planner -> generator`; round one used the single `bootstrap-1` plan item, new knowledge artifacts recorded `domain=ascendc` without a `skill` field, and the generation knowledge/prompt artifacts contained no TileLang or `dsl2Ascendc` references.
- Removed the remaining direct-generator wording that implied an unplanned initial candidate or a runtime Skill layer, and aligned the architecture summary with the dual evaluation budgets.
- Added regression coverage for total-cap exhaustion before a baseline, total accounting across baseline and optimization, and invalid total limits; documented the resulting resume and summary semantics.

### Analysis

- `ascendc_multi_turn` has no TileLang import, compilation, verification, or generated-file stage. The coupling was limited to optional knowledge supplements and `ascendc-translator` naming; the direct evaluator continues to use only the AscendC validator, builder, correctness verifier, and performance harness.
- The shared versioned AscendC API corpus and validator remain valid direct-generation dependencies, so functional decoupling does not require deleting or duplicating the legacy interactive Skill directory.
- A pre-generation plan adds one bounded planner call per new task. It does not consume a bootstrap evaluation and is reused by the first generator call through the active plan item.

### Validation

- Passed all 74 repository unit tests, Python compilation, Ruff checks, and `git diff --check`.
- Completed a two-evaluation mock smoke run with call order `knowledge_router -> planner -> generator -> planner -> generator`; round 1 used the single `bootstrap-1` plan item and the run completed with a retained best candidate.
- Verified the mock round-1 knowledge prompt, rendered references, and generator prompt contain no `TileLang` or `dsl2Ascendc` material; `selected_knowledge.json` records `domain=ascendc` without a `skill` field.

## 2026-09-08

### Changes

- Replaced the direct AscendC same-round compiler-repair loop with an explicit bootstrap, PLAN, EDIT, EVAL, SETTLE, DIAGNOSE/REPLAN workflow. A correct benchmarked baseline now has an independent default budget of eight attempts, while `--max-rounds` counts only post-baseline performance candidates.
- Added schema-v3 phase checkpoints, structured plans, sticky baseline artifacts, resumable EVAL checkpoints, `blocked` exhaustion state, and v2 trajectory migration. Deprecated `--repair-*` options remain accepted without triggering a dedicated repair call.
- Added a deterministic Host launch ABI guard before compilation. Generated pybind code must call `extern "C" *_do` wrappers defined in AscendC kernel sources with `kernel<<<blockDim, nullptr, stream>>>`; unsupported launch headers/macros, unresolved local/CANN includes, absolute includes, and wrapper mismatches are rejected early.
- Updated direct-run documentation and tests for the dual-budget semantics, planning artifacts, failure decisions, infrastructure handling, and resume behavior without changing the fixed AscendC Skill corpus.
- Separated PLAN/DIAGNOSE from code-generation inference settings: planning now defaults to 8192 output tokens with thinking disabled, with independent CLI/environment overrides.

### Analysis

- The failed GELU trajectory did not repeat one compiler diagnostic verbatim, but rounds 2–5 shared an invalid Host ABI assumption: the generated code tried to launch kernels from pybind through `ACLRT_LAUNCH_KERNEL` and an `acl/acl_rt_launch.h` header absent from the installed CANN tree.
- Repository AscendC examples consistently place `*_do` Host wrappers beside `__aicore__` kernels and link those wrappers into pybind. All eight archived tasks containing pybind sources satisfy the new contract.
- A workflow state machine cannot guarantee that a zero-seed model reaches a valid kernel within a finite budget. It can prevent build-repair attempts from consuming performance rounds, reject known-invalid project contracts deterministically, diagnose repeated failures, and report resumable `blocked` rather than false completion.
- A live GELU short run showed that a structured PLAN inherited the 65536-token/high-thinking generator profile and consumed 37928 tokens. Planning needs an independent compact inference profile even when generation benefits from a large reasoning budget.

### Validation

- Passed all 73 repository unit tests, Python compilation, repository whitespace checks, and Ruff validation; source-contract coverage includes all eight archived Host launch layouts.

## 2026-09-07

### Changes

- Consolidated the pending direct AscendC multi-turn implementation: provider-aware token and thinking controls, progress reporting, evaluation-budget accounting, persistent knowledge routing, installed-CANN header evidence, structured diagnostics, conservative source validation, and same-round compiler repair are now covered by code, tests, and user documentation.
- Kept generated editor database caches out of the functional commit; they remain local workspace changes and are not part of the AscendC runtime behavior.

### Analysis

- The direct AscendC runner currently implements a `knowledge route -> generate -> evaluate -> select` loop rather than Triton AutoResearch's explicit `BASELINE -> PLAN -> EDIT -> DIAGNOSE/REPLAN` phase machine.
- Direct-run recovery persists the pending evaluation round and stable attempt ID, but not the exact in-round execution phase. An interruption after compilation or before compiler-repair completion therefore resumes the evaluation attempt rather than replaying from the precise completed substage.
- Aligning the workflows does not require changing the fixed Skill corpus or relying on Claude Code. The main remaining work belongs in orchestration state, experiment planning, checkpoint/replay, evaluation policy, and KEEP/DISCARD accounting.

### Validation

- Passed all 65 tests discovered by `python -m unittest discover -s tests -p 'test_*.py'`.
- Passed `git diff --check` and scanned the pending textual diff for common API-key, access-token, password, and private-key patterns without finding a candidate secret.

## 2026-09-06

### Changes

- Aligned direct AscendC LLM controls with the Triton AutoResearch operating model: evaluation rounds are now consumed only after a candidate reaches the evaluator, while routing, transport, empty-response, local-knowledge, and response-format failures pause a stable checkpoint for `--resume`.
- Added per-call output budgets (4096 router, 65536 generator, 65536 compiler repair), bounded transient retries, explicit DeepSeek thinking controls, and `high`/`max` reasoning effort defaults for generation and repair. Retained the old blanket token setting as a warned compatibility override.
- Changed the DeepSeek default to `deepseek-v4-flash`, exposed effective requested model/thinking/token settings in terminal progress, and recorded requested versus served model, finish reason, usage, reasoning-token metadata, and request options without copying reasoning text into aggregate logs.
- Made empty-final-response progress include the served model, token usage, requested thinking mode, and reasoning effort before retrying, so reasoning-only exhaustion is directly diagnosable from the terminal.
- Added invocation, orchestration-attempt, and pending-run audit files with timestamps; separated physical attempt IDs from evaluation-round numbers; preserved the last actual evaluator feedback across orchestration failures; and made legacy `LLM_FAIL`, `LOCAL_FAIL`, and `FORMAT_FAIL` records non-counting even when they contain synthetic evaluation data.
- Reworked knowledge retrieval around a compact symbol-bearing API manifest, one initial full route, bounded incremental routing, exact identifier matching, diagnostic-order priority, per-symbol header budgets, and a pure-Python public-header fallback when `rg` is unavailable.
- Added conservative AscendC source checks for unambiguous queue-owner mistakes and duplicate file-level constants before compilation, plus one targeted same-round repair and a regression guard that restores the pre-repair candidate when repair introduces additional diagnostics.
- Strengthened generator and repair prompts with API-owner, helper-arity, duplicate-definition, runtime-header-authority, and minimal-delta checks; updated direct-run documentation and environment examples for the new model, budgets, progress, resume, and artifact semantics.
- Removed stale source-line references from the architecture guide and documented the distinct pending-attempt and completed-evaluation resume behavior.

### Analysis

- Triton AutoResearch does not define a repository-level LLM output-token cap; it bounds evaluated rounds, wall time, and transient CLI retries. Direct AscendC therefore uses large per-call generation budgets while keeping routing small and treating infrastructure failures separately from evaluation rounds.
- Historical direct AscendC records may contain an `evaluation_attempts` entry for failures that never invoked the evaluator. Decision type must take precedence over that legacy field when migrating round counts.
- Installed CANN public headers are the authoritative source for exact API ownership, overloads, and parameter spelling. Bundled documentation remains useful for semantics but can differ from the installed patch release.
- Full API-corpus routing is useful once to seed a task working set; repeated full-index routing wastes tokens and can displace compiler-relevant declarations with generic matches. Subsequent selection should remain inside the working set unless concrete new symbols justify a bounded escape.

### Validation

- Regenerated the 115-document AscendC API manifest and verified duplicate API names retain unique document IDs.
- Passed 65 unit tests covering provider request metadata, thinking-mode responses, token configuration, evaluation-round migration, paused resume, deterministic/incremental routing, missing-`rg` fallback, source validation, compiler-repair rollback, progress, diagnostics, and wrapper safety.
- Verified the source guard reports the existing GELU candidate's duplicate `TANH_C1`/`TANH_C2` declarations and invalid `TPipe::EnQue` calls before compilation.
- Passed Python compilation and repository whitespace checks; completed a mock direct-run smoke test without external API or NPU access.

## 2026-09-05

### Changes

- Added DeepSeek and OpenAI provider selection for direct AscendC multi-turn generation. Both use the OpenAI-compatible Chat Completions wire format and independent `.env` credentials.
- Added CANN runtime-version detection and a version catalog for the bundled AscendC API knowledge. Exact matches are preferred; missing 8.x versions fall back to the newest bundled 8.x documentation with a warning; cross-major fallback is rejected.
- Added a Python-controlled `ascendc-translator` Skill and knowledge-routing step before every generation call. The router selects allowlisted supplementary guides and CANN API pages under configurable document-count and character budgets.
- Added per-round knowledge prompts, responses, selections, rendered references, call types, version metadata, and token totals grouped by call type.
- Added configuration, knowledge-version, routing, fallback, and multi-turn artifact tests.
- Preserved the pre-existing positional argument order of `RunConfig` while adding provider selection, and added explicit validation for invalid `LLM_PROVIDER` values.
- Updated the README and non-Claude architecture report to reflect the new DeepSeek/OpenAI scope and version-aware knowledge-routing phase.
- Installed `python-dotenv` 1.2.3 into the current user Python environment from `requirements.txt`.
- Fixed AscendC wrapper validation so calls through a confirmed extension alias, such as `_ext.gelu()`, are not misclassified as forbidden PyTorch Tensor methods; unreached checks now render as `SKIP` instead of `FAIL`.
- Captured Chat Completions `finish_reason` values and added one compact same-round generator retry when a response reaches the output-token limit, with separate retry artifacts and token accounting.
- Added regression coverage for extension methods that share PyTorch operation names, retained rejection of real Tensor/Functional fallbacks, and covered successful and exhausted truncation retries.
- Replaced AscendC extension filename heuristics with exact `PYBIND11_MODULE` identity matching and added source-aware torch/Tensor analysis across aliases, imported functions, operators, `nn.Module` attributes, and reachable helpers.
- Added an optional validator `--pybind-file` interface and made the local evaluator pass the task's pybind source explicitly.
- Added default stderr progress for direct AscendC runs, including round/stage transitions, 15-second heartbeats, LLM token completion details, KEEP/DISCARD decisions, and a `--quiet` opt-out while preserving stdout as pure JSON.
- Added full per-stage evaluation logs and structured failure diagnostics (`failure_stage`, `failure_code`, `error_excerpt`, and `details_path`) to round trajectories and final summaries.
- Converted knowledge-router/generator call failures into auditable failed rounds and prevented failed, performance-incomplete, or unscored candidates from becoming the best result.
- Added backward-compatible failure-stage and excerpt inference for trajectories produced before structured diagnostics were introduced, without claiming a full-log path when no historical stage log exists.
- Normalized incomplete or unexpectedly raised evaluator results into explicit failed rounds so an unsuccessful run always has an actionable final failure summary.
- Deduplicated repeated compiler diagnostics in terminal and summary excerpts while retaining the unmodified full stage log.
- Reworked direct AscendC knowledge routing into a persisted task-level working set: the first round sees the complete manifest, stable later rounds reuse existing documents without an LLM call, and new compiler/source symbols trigger bounded deterministic or incremental routing.
- Added unique API document IDs and a reproducible manifest generator so duplicate short API names no longer select an arbitrary page.
- Replaced repeated full-guide injection with compact invariant rules, section-aware API excerpts, an active set of at most five API pages, and a default 24000-character knowledge budget.
- Added installed-CANN public-header extraction and explicit runtime/document conflict notes; runtime declarations and compiler diagnostics now take precedence over fallback-version documents.
- Persisted `runtime_header_facts.json` even when installed public headers are unavailable, so every knowledge decision remains auditable.
- Added one same-round compiler-repair LLM call after a real AscendC build failure, followed by re-evaluation, separate repair artifacts, `compile_repair` token accounting, and repair-before/after `evaluation_attempts` history.
- Printed the concrete compiler excerpt and full-log path before starting same-round repair, so a subsequently successful repair does not hide the original failure.
- Corrected bundled guidance for the CANN 8.5.2 `DataCopyPadExtParams::paddingValue` spelling and made `CopyTiling` explicitly type/component-specific rather than a generic tiling-copy API.
- Added backward-compatible migration from historical per-round knowledge selections to `knowledge_state.json` and documented the new routing, runtime-header, repair, and artifact flow.

### Analysis

- The detailed bundled AscendC API corpus is tied to CANN Community Edition 8.5.0: page titles and source URLs use the 8.5.0/`850` version. `ascendc_dynamic_quant_kb.md` also explicitly identifies Atlas A2 with CANN 8.5.0.
- The API index declares 114 pages, while the repository currently contains 115 page Markdown files and 115 indexed title entries, plus 89 images. This mismatch is retained for audit rather than silently deleting material.
- AscendC API pages generally contain product support, behavior, function prototypes, parameters, return values, constraints, and examples. The corpus covers data movement, arithmetic, logic, reduction, matrix operations, conversion, sorting, activation, mathematical operations, and utility APIs.
- The AscendC knowledge hierarchy consists of the translator Skill, the main translation guide, TileLang-to-AscendC mapping, Vector/Cube/CV/host/cross-core guides, quantization patterns, verification scripts, and the versioned API index/pages.
- Before this change, direct multi-turn generation injected only `dsl2Ascendc.md`, `TileLang-AscendC-API-Mapping.md`, and `AscendCVerification.md` in full. It did not load individual CANN API pages. The Claude Skill workflow already instructed the agent to follow Mapping → INDEX → selected API pages.
- Prompt layers in the repository are: top-level Agent prompts in `agents/`, on-demand Skill prompts in `skills/**/SKILL.md`, the programmatic direct-LLM prompt in `ascendc_multi_turn/prompts.py`, and AutoResearch's `CLAUDE.md` + slash command + phase guidance + diagnosis subagent composition.
- Triton knowledge is organized into 6 Skills, 19 generator/hardware references, 15 latency-optimization references, and 21 design cases. This structure motivated the explicit Skill and selective-reference phase for direct AscendC generation.
- The AscendC validator previously recognized `_ext` as a compiled extension and then independently rejected `_ext.gelu()` by attribute name alone. Any extension export named like a forbidden Tensor method could therefore be blocked before compilation.
- Method names alone cannot establish whether an operation belongs to PyTorch. Reliable wrapper validation requires import provenance, Tensor value propagation, and an exact link between the Python extension import and the pybind module declaration.
- Evaluation subprocess output is intentionally captured for complete per-stage log persistence; the console exposes bounded diagnostic excerpts plus periodic heartbeats rather than streaming the entire compiler output.
- The bundled API archive contains 115 indexed pages but only 81 distinct short API names; names such as `Exp` occur multiple times, so short names are not stable routing keys. Page IDs are unique across all 115 documents.
- Previous generation rounds rendered roughly 60000 characters of references and repeatedly paid for the same fixed guides. The new mock smoke run rendered 4409-4842 characters per round and made one knowledge-router call across three rounds.
- CANN 8.5.2's installed public header declares `DataCopyPadExtParams<T>::paddingValue`; the bundled 8.5.0 DataCopyPad page describes `padValue`. Runtime header extraction is therefore required even for same-major documentation fallback.
- The public `CopyTiling` declaration found in CANN 8.5.2 is under the Matmul advanced API and does not establish support for arbitrary user tiling structs.

### Validation

- Installed `python-dotenv` at `/home/developer/.local/lib/python3.11/site-packages` and verified it imports with the active interpreter.
- Python compilation and Bash syntax checks passed.
- All 15 configuration, provider, CANN-version, knowledge-routing, safety, resume, and multi-turn tests passed.
- A two-round mock run completed successfully with exact CANN 8.5.0 knowledge selection, separate `knowledge_router`/`generator` token accounting, and all per-round knowledge artifacts present.
- OpenAI startup validation was exercised with an empty key and failed early with the expected `Missing environment variable: OPENAI_API_KEY` message.
- All 22 unit tests passed, including new validator and truncation-retry regressions.
- Python compilation checks passed for the modified runtime, validator, and test modules.
- The saved `outputs/1_GELU/model_new_ascendc.py` wrapper passed direct static validation after the extension-aware fix.
- All 34 unit tests passed after the source-aware validator refactor; current GELU and six non-degraded archived wrappers passed direct validation, while archived wrappers containing actual torch computation remained rejected.
- Progress, quiet-mode stdout/stderr separation, all evaluation-stage classifications, structured build diagnostics, full/timeout-log retention, heartbeat shutdown, LLM failure recording, last-round failure visibility, and invalid-best selection regressions passed in the direct multi-turn unit suite.
- All 54 unit tests passed, including unique document routing, bounded rendering, legacy-state migration, runtime-header conflicts, router reuse, and successful same-round compiler repair.
- Regenerated the 115-document API manifest, passed `git diff --check`, and passed Python compilation for the direct runner, manifest generator, and validator.
- A three-round mock smoke run used call types `knowledge_router, generator, generator, generator`, reused knowledge without later router calls, and kept every rendered reference context below 5000 characters.

## 2026-09-04

### Changes

- Added root `.env`/`.env.example` configuration, `python-dotenv` dependency declaration, centralized Python environment access, and a shared shell `.env` loader.
- Added configurable Claude model defaults for Triton batch generation, AscendC Claude batch generation, and AutoResearch while retaining command-line overrides.
- Added startup validation, `.env` ignore rules, configuration tests, and direct AscendC usage documentation.

### Analysis

- Documented the non-Claude AscendC execution loop, state/memory behavior, fixed-reference injection, generated artifacts, resume behavior, and evaluation boundaries in `docs/non-claude-execution-analysis.md`.
