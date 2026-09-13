# Project Change Log

All repository modifications must be recorded here. Dates use `YYYY-MM-DD`.
Secrets must never be included.

## 2026-09-13

### 本轮计划：编译修复与分层轨迹记忆

- 保持现有 Agent、Evaluator、B/D 知识源和 `generator_thinking=disabled` / `generator_max_tokens=32768` 不变，将真实 compiler diagnostic、同阶段部分修复和失败方法接入现有调用逻辑。
- 将尝试历史分为三层：磁盘保存完整 Attempt；`repair_state.json` 保存 accepted base、open/cleared errors 和去重后的失败方法；Prompt 只投递当前 open error 相关的至多三个不同方法。
- 严格区分 Stable Knowledge 与 Episodic Memory：Structured/runtime/Skill facts 继续作为稳定知识；单轮采用 `if(sizeof(T))` 等方法失败只属于当前 trajectory，不能写回 Skill、API fact 或 runtime fact。
- 仅在 bootstrap compile-repair Generator 调用中加入模板实例化、compiler authority 和 error-regression contract；initial、runtime/correctness 和 optimization Generator 不重复携带该段编译 Prompt。
- 保留 validated frontier 的原有含义，新增 repair workspace；同一 compile frontier 中确实清除旧错误且未引入历史已清除错误的候选以 `PARTIAL_KEEP` 累积，未改善或回归的候选才 rollback。
- 后续 Planner 每次只产生一个基于当前 accepted implementation 的垂直步骤，不再预生成 3–5 个横向候选。

### 编译修复与轨迹实现

- 集中定义并落盘实际 system prompt；`calls.jsonl` 记录 system prompt ID、hash 和 artifact path，因此 system/user request 均可追溯。system prompt 内容保持“AscendC expert + JSON schema”，编译修复规则不被无差别放入所有调用。
- 新增 compiler diagnostic normalization：从真实 compiler/evaluator output 生成逐条、忽略路径行号变化的 error ID，并记录 category、symbol、source file 和 evidence origin。该逻辑只在编译后比较结果，不参与 source validation 或源码 regex 拒绝。
- 新增 `repair_state.py` 和 `.llm_state/attempt_ledger.jsonl` / `attempts/attempt_N.json` / `repair_state.json` artifacts；完整 diff、知识选择和结果保存在磁盘，Prompt projection 只包含当前 open/cleared errors 与相关去重方法。
- Runner 现在区分 current accepted bundle、与其匹配的 base evaluation 和 latest rejected attempt；生成前保存 `round_N/base.json`，resume EVAL checkpoint 时不再把待评估 candidate 误当成它自己的 base。
- 同一 compile stage 的 `{AddTiling, Muls} → {Muls}` 被识别为 `PARTIAL_KEEP`，下一轮从已修复 AddTiling 的 source 继续；已清除错误重新出现或退回 source validation 时识别为 regression 并恢复 accepted repair base。
- Debug route 只使用直接 failure evidence。current source 中合法的 `c10_npu`、`aclrtStream` 或 `getCurrentNPUStream` 只用于 source-symbol retrieval，不再把 Muls 等 kernel compiler failure 污染成 `host_integration_debug`。
- 从生成 Prompt 和 knowledge-state 更新中移除了旧的 failure-fingerprint episode 注入；稳定 failure guidance、Skill capsules 与 run-local failed approaches 使用独立数据路径和独立 Prompt 章节。
- Compile-repair Prompt 增加 accepted base、open/cleared errors、普通运行时 `if` 无法隔离 C++ 模板实例化、不得虚构 overload/cast 等约束；optimization Prompt 明确不注入该 contract 或历史 compile attempts。
- Planner/Diagnose 的非初始输出改为精确一个下一步 item；单一假设可以修改必要的多个文件，每轮评估后基于最新 accepted candidate 重新规划。

### 本轮验证

- 新增真实 Add compiler failure regression，确认 AddTiling/Muls error ID 不随源码行号变化，重复 `if(sizeof(T))` 方法在 Prompt 中聚合为一条并保留 occurrence/attempt IDs。
- 新增 Runner integration trajectory，确认 `FAIL → PARTIAL_KEEP → BASELINE_KEEP` 三轮中 Round 3 的 Prompt 和代码基线来自 Round 2，attempt 2 的 `base_attempt_id=1`，并生成完整 attempt ledger 与 system-prompt artifact。
- 新增路由回归，确认 current source 同时包含合法 NPU stream API 和 Muls 调用时，直接 Muls compiler diagnostic 仍选择 `api_compile_debug`；真实 CUDA contamination 仍选择 Host capsule。
- 新增 phase-aware Prompt 回归，确认 compilation contract 和 Episodic summary 只出现在 compile repair，optimization 不包含这两部分。
- 通过 `python -m unittest discover -s tests -p 'test_*.py'`：122 项。
- 通过 `python -m unittest discover -s ascendc_multi_turn/structured_knowledge/tests -p 'test_*.py'`：25 项。
- 通过修改模块的 Python compilation 检查和 `git diff --check`。本轮没有执行真实 Add、完整 B/D 或额外 NPU 实验。

### Follow-up implementation

- Added the experimental `knowledge_input_mode=full_selected` switch. It preserves stage/symbol routing while disabling Agent-side reference, section, runtime-declaration, structured-field, and total-character truncation; the default remains `bounded`.
- Added full-input prompt observability with UTF-8 byte counts, prompt SHA-256, selected/rendered knowledge IDs, source-specific character counts, and an explicit `input_truncated` flag.
- Added deterministic recovery when a Planner places an explicit numbered `Expected signal` section inside `change`, plus resume reuse of the already persisted response so recovery does not trigger another Planner sample.
- Removed heuristic duplicate variable/constant detection from generated-source validation after a realistic file-scope function signature with a `const` parameter triggered catastrophic regex backtracking; duplicate C++ declarations are now left to the CANN compiler, while deterministic CUDA, Host ABI, include, queue-owner, and wrapper-contract checks remain.
- Began the grounded Hybrid/Skills-only follow-up by replacing the generated-source duplicate-symbol heuristic with a comment/literal-aware scope walk and adding deterministic CUDA Host contamination checks.
- Added regression coverage for distinct-kernel locals, namespace/record scopes, literal/comment false positives, and CUDA-only bindings.
- Added provenance-preserving failure/source/planned symbol extraction and an observation-only A-H evaluation ledger with explicit unknown/not-reached states.
- Added environment-fingerprinted runtime facts and auditable compile-probe primitives; installed header matches now remain Level 2 unless the exact fact has a matching successful probe artifact.
- Extended the adapter schema compatibly with local, path-confined adaptation capsules, primary/secondary/support roles, symbol provenance, and fact confidence metadata.
- Made debug routing deterministic: generic correctness failures now route to kernel design, runtime/Host failures use direct evidence, and precision is selected only for explicit hypotheses; API selection now prioritizes failure, source, then planned symbols without hard-empty filtering.
- Added bounded CannAgent-local Host ABI and GELU/LayerNorm/Permute semantic capsules plus a provenance/probe manifest; exact environment-dependent API forms remain conditional on runtime facts.
- Updated stage mapping with local Host/invariant support capsules and removed generic numerical-mismatch triggers from precision debugging.
- Wired active-plan, current-source, and failure evidence independently into routing and made both knowledge modes consume the same environment-fingerprinted runtime fact source.
- Made rendered prompts expose the deterministic route and capsule confidence/provenance, while recording section-level rendering, truncation, character, and token-estimate telemetry.
- Added non-invasive progress timing events, per-call prompt metadata support, and censored stage-normalized token/time milestones without changing evaluator behavior.
- Persisted prompt composition, per-evaluation A-H observations, stage timings, and tokens/time-to-first-stage metrics in the existing runner artifacts.
- Added runtime grounding/probe tests and full failure-evidence-to-rendered-context regressions for ReduceSum overloads, CUDA Host contamination, and broad Permute correctness failures.
- Attached the exact selected/rendered knowledge metadata to Planner and Generator call records and round trajectories, including retries, so delivery failures can be separated from generation failures.
- Corrected installed-header candidate ranking so core torch_npu declarations and basic AscendC tensor definitions outrank third-party call sites or incidental advanced-API uses.
- Added explicit environment-bound negative facts when both the installed public-header scan and the current compiler diagnostic reject a symbol; header absence alone remains non-authoritative Level 0.
- Runtime fact rendering now retains a bounded set of distinct installed overload declarations instead of silently presenting only the first match.
- Added reusable probe-manifest writing and an optional `CANNAGENT_PROBE_MANIFEST` path so fingerprint-bound grounding can be prepared once and shared by paired runs without adding per-generation adapter work.
- Updated adapter regressions for the deliberate correctness-routing change and retained fail-open reporting when the external CANNBot source is unavailable.
- Included full compiler and verification text in failure-symbol extraction, ordered exact structured matches by failure/source/planned evidence priority, and attached per-item evidence/provenance metadata to the selected context.
- Narrowed Host routing so a generic linker symbol does not imply an ABI problem while locally observed NPU/CUDA tensor and stream identifiers remain deterministic Host triggers.
- Corrected the probe-manifest regression fixture so its environment payload is asserted before temporary artifacts are released.
- Expanded stage-normalized reporting with prompt-component totals, evaluator-stage latency, censored milestone aliases, and A→H transition rates; binding failures and optimized correctness regressions now have distinct observational states.
- Revised the Skill Adapter stage table to match the implemented deterministic correctness/Host routing and failure/source/planned evidence priority.
- Documented the four runtime-knowledge confidence levels, environment fingerprint invalidation, new observability artifacts, and the separation of explicit compile probes from online generation.
- Added the current two-phase B/D screening and conditional replacement-confirmation gate without scheduling or executing new operator runs.
- Updated the context schema to version 2 with deterministic route fields, three-source symbol evidence, Level 0–3 runtime provenance, per-item selection metadata, Host routing, and censored stage metrics.
- Aligned the README and fallback wording with strict Skills-only isolation, new trajectory artifacts, and the conditional two-phase B/D protocol.
- Added a standalone environment-bound Host tensor/stream syntax probe command that discovers headers without importing torch, emits its compiler command, and writes a reusable manifest.
- Documented one-time Host probe preparation and shared manifest use for fair paired runs.
- Added a thin `probe_knowledge` module entry point so the standalone probe runs without the package import-order warning.
- Improved installed-header declaration ranking to ignore comment-only symbol hits and retain bounded count-based overloads needed by failed arithmetic calls.
- Excluded comment matches before header excerpt selection and stopped declaration excerpts from absorbing a preceding unrelated `__aicore__` declaration.
- Bounded installed inline-function excerpts at their closing brace instead of consuming the rest of a header while searching for a semicolon.
- Added regressions proving compiler-output-only API symbols are failure evidence and that header absence becomes a negative fact only when the active compiler directly rejects the symbol.
- Clarified that non-Host operator API probes are targets rather than already-required or completed verification artifacts.
- Added a reusable environment-bound AscendC CMake probe for the installed Add/Mul/Muls/Cast/Sqrt/ReduceSum/Tanh/GlobalTensor contracts so successful results can enter the same manifest as Host facts.
- Added a unit regression ensuring kernel API facts are emitted only after both configure and build report success.
- Recorded the installed/probed required include alongside each runtime declaration, using `kernel_operator.h` for basic APIs and the exact relative header for advanced and torch_npu facts.

### Follow-up validation

- Ran one authorized Add/Hybrid/full-selected experiment with non-thinking generation, a 32768 output-token cap, and at most three candidate evaluations. All five LLM requests accepted their full prompts; the largest request used 38,922 prompt tokens, every call reported `finish_reason=stop`, and every context recorded `truncated_sections=[]`.
- The three-round Add experiment was a focused input-capability test rather than a B/D comparison. `knowledge_input_mode=full_selected` preserved Hybrid stage/symbol routing but disabled Agent-side reference, per-section, runtime-declaration, structured-field, evaluation-evidence, and total-character truncation for the already selected knowledge. It did not inject the unselected knowledge corpus and could not remove the provider's hard context-window limit. The fixed controls were DeepSeek `deepseek-v4-flash`, temperature `0.2`, Planner thinking disabled with 8192 output tokens, Generator thinking disabled with 32768 output tokens, CANN 8.5.2, Ascend910B3, device 0, and the real local evaluator over the Add fixture's 50 cases.
- The success gate for that experiment was an end-to-end source-valid, compile, load/binding, NPU-execution, and 50/50 correctness pass. Token and latency milestones that were never reached are treated as censored rather than zero. Its purpose was to determine whether earlier failures were caused by Agent-side input truncation; it was not intended to establish Hybrid/Skills-only replacement or performance superiority.
- The Add experiment did not produce a correct candidate: round 1 stopped at source validation, while rounds 2 and 3 reached the real CANN build and failed compilation. The final error was unsupported `Muls<bfloat16_t>` even though the round-3 prompt contained the complete structured `Muls.T` dtype restriction and Level 3 runtime declaration. This is recorded as knowledge-delivered/generation-reasoning failure, not evidence of input truncation.
- The same run exposed an independent source-validation false positive: round 1 contained an actual `kernel<<<...>>>` launch after control flow in the `*_do` wrapper, but the regex-bounded wrapper body ended early. It was recorded without extending or rerunning the experiment.
- Across the five accepted calls, cumulative usage was 110,707 prompt tokens and 15,273 completion tokens. The first source-valid candidate appeared in evaluation round 2 after 83,147 cumulative tokens and 69.14 seconds; compile, load, NPU execution, correctness, and benchmark milestones remained censored. The result shows that the service accepted the selected full input in this Add trajectory, but removing Agent-side input budgets alone was insufficient to produce a correct operator in three candidate evaluations.

#### Fresh Add full-selected ten-round capability run

- Re-ran Add from a clean output directory under the current validator instead of resuming the earlier three-round trajectory. The run used real local compilation and NPU evaluation, Hybrid knowledge, `knowledge_input_mode=full_selected`, DeepSeek `deepseek-v4-flash`, temperature 0.2, Planner thinking disabled with 8192 output tokens, Generator thinking disabled with 32768 output tokens, CANN 8.5.2, and Ascend910B3. Bootstrap, optimization, and total candidate caps were all 10; the total cap, rather than a ten-item Planner response, bounded the experiment.
- Execution remained evidence-driven. The initial Planner emitted exactly one complete implementation. Later repair/diagnosis plans contained five hypotheses, but only three consecutive failing items were consumed before the remaining items were discarded and a fresh DIAGNOSE/REPLAN was requested. A correct candidate would have been saved as the baseline, cleared the bootstrap plan, switched routing to optimization, and used the remaining total budget to optimize from the correct best frontier. No correct baseline was reached, so all ten evaluations remained bootstrap attempts and optimization was never entered.

The following command was executed once from a clean directory. Standard output and error were appended to `outputs/nonthinking_add_full_selected_10round_20260913.run.log` and observed with `tail -n 30 -F` until exit.

```bash
export CANNAGENT_PROBE_MANIFEST=/mnt/workspace/CannAgent/outputs/nonthinking_capability_20260913/artifacts/probe_manifest.json

python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/3_Add.py \
  --output-dir outputs/nonthinking_add_full_selected_10round_20260913 \
  --provider deepseek \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --temperature 0.2 \
  --max-bootstrap-rounds 10 \
  --max-rounds 10 \
  --max-total-rounds 10 \
  --timeout 600 \
  --device 0 \
  --soc-version Ascend910B3 \
  --cann-version 8.5.2 \
  --knowledge-mode structured \
  --knowledge-source hybrid \
  --knowledge-input-mode full-selected \
  --skill-adapter \
  --router-max-tokens 4096 \
  --router-thinking disabled \
  --generator-thinking disabled \
  --generator-max-tokens 32768 \
  --generator-reasoning-effort high \
  --planner-thinking disabled \
  --planner-max-tokens 8192 \
  --planner-reasoning-effort low \
  --repair-thinking enabled \
  --repair-max-tokens 65536 \
  --repair-reasoning-effort max \
  --llm-transient-retries 3
```

Results by evaluation round:

| Round | Plan source | Source/API/static | Compile result | Principal diagnostic |
|---:|---|---|---|---|
| 1 | Initial one-item plan | pass/pass/pass | fail | `AddTiling` GM-copy construction and `Muls` overload |
| 2 | Planner v2 item 1 | pass/pass/pass | fail | `Muls` overload; the `AddTiling` field-read edit removed its constructor error |
| 3 | Planner v2 item 2 | pass/pass/pass | fail | `AddTiling` GM-copy construction and `Muls` overload returned after rollback |
| 4 | DIAGNOSE v3 item 1 | pass/pass/pass | fail | `Muls` overload |
| 5 | DIAGNOSE v3 item 2 | pass/pass/pass | fail | `AddTiling` GM-copy construction and `Muls` overload |
| 6 | DIAGNOSE v3 item 3 | pass/pass/pass | fail | `Muls` overload |
| 7 | DIAGNOSE v4 item 1 | pass/pass/pass | fail | `AddTiling` GM-copy construction and `Muls` overload |
| 8 | DIAGNOSE v4 item 2 | pass/pass/pass | fail | `Muls` overload |
| 9 | DIAGNOSE v4 item 3 | pass/pass/pass | fail | `AddTiling` GM-copy construction and `Muls` overload |
| 10 | DIAGNOSE v5 item 1 | pass/pass/pass | fail | `Muls<bfloat16_t>` static assertion; supported types are half/float/int16/int32 |

- Final status was `blocked` at `max_total_rounds`: source validation 10/10, API-constraint validation 10/10, static validation 10/10, compile 0/10, load 0/10, NPU execution 0/10, correctness 0/10, baseline 0, and optimization 0. All stages after source-valid are therefore censored rather than zero-cost successes.
- The run made 15 LLM calls: two Planner calls, three DIAGNOSE calls, and ten Generator calls. Every call used disabled thinking, reported `finish_reason=stop`, recorded `knowledge_input.mode=full_selected`, `input_truncated=false`, and `truncated_sections=[]`; selected and rendered Skill, structured-card, and runtime-fact ID counts matched on every call. There was no response-length or provider context-limit failure.
- Actual token use was 1,053,495 prompt + 66,205 completion = 1,119,700 total. The largest input was the round-10 DIAGNOSE call at 153,693 prompt tokens; the largest Generator input was round 8 at 64,623 prompt tokens. Average use was 74,646.7 tokens per LLM call. The rendered prompts contained 3,203,726 characters in aggregate, including 1,587,346 knowledge-reference characters, 192,633 source-code characters, and 433,418 compiler/evaluator-evidence characters.
- Measured LLM latency was 229.52 seconds and evaluation latency was 271.10 seconds, of which 266.21 seconds was compilation. Wall-clock runtime from the first log event to final summary was approximately 610 seconds. The first source-valid candidate arrived in round 1 after 55,371 tokens and 60.23 seconds; compile/load/execute/correct/benchmark/optimization milestones were all censored.
- The selected knowledge was not missing the failing contract. Every Generator reference included `Muls.T` and a Level 3 runtime `Muls` fact; rounds 2-10 also rendered the explicit restriction that BF16 is unsupported. Round 10 nevertheless instantiated `AddContiguous<bfloat16_t>` and used an ordinary runtime `if (sizeof(T) == 4)` around `Muls<T>`. Because both template branches are compiled, this still instantiated the forbidden BF16 overload. This is direct evidence of generation/implementation reasoning failure after successful knowledge availability, retrieval, and delivery.
- A separate workflow limitation prevented independent compile repairs from accumulating. The frontier manifest remained at round 1's `source` bundle. Round 2 removed the invalid `AddTiling` GM-copy but still failed at `Muls`, so the runner rolled back; round 3 then applied its different repair to the old source and reintroduced the tiling error. The alternating `AddTiling`/`Muls` diagnostics across later plans show the same pattern. Under the current frontier policy, one candidate had to repair every compile blocker at once in order to advance beyond the source frontier.
- Debug routing also contributed noise: round 1 used `kernel_design`, but every compiler-repair Generator in rounds 2-10 selected `host_integration_debug` with `host_or_cuda_compile_evidence`, even though the active blockers were device-kernel `Muls` and tiling C++ construction. The correct API facts remained present, but the Primary Skill did not represent the dominant failure layer.
- The requested vertical optimization behavior was not reached, not because the Agent executed ten preplanned alternatives, but because no candidate compiled and passed correctness. Planner cadence behaved as configured: one initial item, then replanning at rounds 4, 7, and 10 after each three-failure window. The experiment therefore shows that removing Agent-side input limits alone is insufficient to measure the model's end-to-end correct-kernel capability cleanly while failed-candidate rollback and compile-error routing remain confounders.
- Artifacts are retained under `outputs/nonthinking_add_full_selected_10round_20260913/`; the raw tailed terminal record is `outputs/nonthinking_add_full_selected_10round_20260913.run.log`. No additional operator, B/D, or retry run was started.
- Passed the environment-bound Host syntax probe and AscendC kernel API CMake probe on CANN 8.5.2 / Ascend910B3 with final fingerprint `06b8e945d4cd781bd1b4`.
- Confirmed the shared manifest promotes 11 exact Host/kernel facts to Level 3: `is_npu`, `getCurrentNPUStream`, `aclrtStream`, `Add`, `Mul`, `Muls`, `Cast`, `Sqrt`, `ReduceSum`, `Tanh`, and `GlobalTensor`.
- Extended environment identity with the CANN `bisheng` path/version and torch header roots, preventing Verified kernel/Host facts from surviving a relevant compiler or include-tree change.
- Bound probe manifests to the probe implementation hash itself and removed duplicate Host include flags, so a changed probe contract also invalidates prior Level 3 results.
- Cleaned new lint findings in evidence composition and runtime declaration ranking without changing routing behavior.
- Applied import-only Ruff formatting to the newly changed probe, validation, and regression modules.
- Decoupled runtime fact discovery/provenance from prompt rendering budget so later installed symbols are no longer misclassified as missing when the text quota is exhausted.
- Final validation passed 108 fast/unit/integration regressions, including nine targeted source-validation cases; the repository-wide archived-source traversal was intentionally not repeated after its earlier no-output stall.
- Added an explicit integration assertion that structured API cards are reordered by failure, current-source, then planned evidence rather than their storage order.
- Python compilation, Ruff checks for every new/changed implementation and regression module, and `git diff --check` passed; no full Hybrid/Skills-only operator experiment was run.

### Changes

- Completed the controlled B/D knowledge-source ablation on the real local
  Ascend910B3 environment for GELU, LayerNorm, and Permute. Mode B used
  `knowledge_source=hybrid` (Structured Knowledge A + CANNBot Skill Knowledge
  B); Mode D used `knowledge_source=skills` (CANNBot Skill Knowledge B only).
- Used the same Agent implementation and evaluation workflow in every arm.
  No runtime source, workflow, generator, evaluator, structured knowledge
  build, or CANNBot Skill corpus was changed during the experiment.
- Standardized every arm on DeepSeek `deepseek-v4-flash`, temperature `0.2`,
  five bootstrap evaluations, two post-baseline optimization evaluations,
  Planner thinking disabled with 8192 output tokens, and Generator thinking
  disabled with 32768 output tokens. The evaluator was local rather than mock,
  with device `0`, `Ascend910B3`, CANN auto-detection, and a 600-second stage
  timeout.
- Stored the six run directories, live terminal logs, and elapsed-time records
  under `outputs/ablation_bd_final_20260912/`.

### Commands

The following command shape was executed once for each operator/mode pair. The
operator paths were `benchmarks/NPUKernelBench/level1/1_GELU.py`,
`benchmarks/NPUKernelBench/level1/10_LayerNorm.py`, and
`benchmarks/NPUKernelBench/level1/12_Permute.py`; `MODE` was `hybrid` for B and
`skills` for D, and every pair used a clean mode-specific output directory.

```bash
python -m ascendc_multi_turn \
  --op-file OP_FILE \
  --output-dir outputs/ablation_bd_final_20260912/OPERATOR/MODE \
  --provider deepseek \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --temperature 0.2 \
  --max-bootstrap-rounds 5 \
  --max-rounds 2 \
  --timeout 600 \
  --soc-version Ascend910B3 \
  --device 0 \
  --cann-version auto \
  --knowledge-mode structured \
  --knowledge-source MODE \
  --cannbot-skills-root /mnt/workspace/cannbot-skills/ops \
  --generator-thinking disabled \
  --generator-max-tokens 32768 \
  --generator-reasoning-effort high \
  --planner-thinking disabled \
  --planner-max-tokens 8192 \
  --planner-reasoning-effort low
```

Each command redirected stdout/stderr to its `logs/*.live.log` file and was
observed through `tail -n +1 -F` until the Agent process exited. The arms ran
sequentially so they did not contend for the NPU.

### Results

An operator counts as compiled/correct when at least one of its five candidates
passes that gate. `First compile` is the first evaluated candidate reaching a
successful build; `-` means the arm never reached it.

| Operator | Mode | Compiled candidates | Correct candidates | First compile | Bootstrap / optimization | Tokens (prompt + completion) | E2E seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| GELU | B Hybrid | 0/5 | 0/5 | - | 5 / 0 | 94,357 (69,113 + 25,244) | 100 |
| GELU | D Skills-only | 0/5 | 0/5 | - | 5 / 0 | 114,186 (85,030 + 29,156) | 216 |
| LayerNorm | B Hybrid | 0/5 | 0/5 | - | 5 / 0 | 133,123 (106,671 + 26,452) | 130 |
| LayerNorm | D Skills-only | 0/5 | 0/5 | - | 5 / 0 | 162,541 (127,492 + 35,049) | 203 |
| Permute | B Hybrid | 0/5 | 0/5 | - | 5 / 0 | 165,079 (144,663 + 20,416) | 273 |
| Permute | D Skills-only | 3/5 | 0/5 | 2 | 5 / 0 | 169,397 (143,645 + 25,752) | 554 |

Aggregate operator-level results:

| Metric | B Hybrid | D Skills-only |
|---|---:|---:|
| Compile success rate | 0/3 (0%) | 1/3 (33.3%) |
| Correctness rate | 0/3 (0%) | 0/3 (0%) |
| Candidate compile rate | 0/15 (0%) | 3/15 (20.0%) |
| Candidate correctness rate | 0/15 (0%) | 0/15 (0%) |
| Candidate evaluations | 15 | 15 |
| Optimization evaluations | 0 | 0 |
| Prompt tokens | 320,447 | 356,167 |
| Completion tokens | 72,112 | 89,957 |
| Total tokens | 392,559 | 446,124 |
| Summed end-to-end time | 503 s | 973 s |

The average rendered knowledge-reference sizes below are characters, measured
from the per-round `references.md` and `planner_references.md` artifacts.

| Operator | Mode | Generator references | Planner references |
|---|---|---:|---:|
| GELU | B Hybrid | 9,635 | 6,154 |
| GELU | D Skills-only | 7,855 | 4,227 |
| LayerNorm | B Hybrid | 7,854 | 5,930 |
| LayerNorm | D Skills-only | 6,924 | 4,273 |
| Permute | B Hybrid | 7,764 | 5,595 |
| Permute | D Skills-only | 6,830 | 3,451 |

### Analysis

- Skills-only improved evaluation depth in this limited sample, but did not
  establish that Skills can replace structured knowledge. GELU Skills-only
  reached real builds in four rounds while Hybrid remained at source
  validation; LayerNorm Skills-only reached real builds in three rounds while
  Hybrid reached one; Permute Skills-only compiled in rounds 2, 3, and 5 while
  Hybrid compiled none. Nevertheless, neither mode produced one correct
  operator, so the replacement hypothesis is not verified.
- Permute Skills-only round 2 compiled but its binding rejected the NPU input
  with `Input must be a CUDA tensor`. Round 3 compiled and executed all 149 NPU
  cases, but failed with large numerical errors across later FP16/FP32 cases,
  demonstrating an incorrect index/data-movement implementation. Round 5
  compiled but again rejected the input in its binding. Hybrid's later Permute
  candidates repeatedly used unavailable or incorrectly declared
  `aclrtStream`/`aclrtGetCurrentStream` interfaces.
- GELU's Skills-only build failures were dominated by invalid AscendC
  arithmetic overloads such as `Muls`, `Add`, `Mul`, `Tanh`, and cast usage.
  LayerNorm's build failures included unsupported `sqrtf`, `ReduceSum`, and
  invalid `GlobalTensor::Get` usage. The selected practices/templates therefore
  did not supply enough exact installed-version API and Host ABI detail for a
  correct implementation.
- Skills-only reduced the rendered knowledge-reference characters for every
  tested operator, but used 53,565 more total tokens than Hybrid (+13.6%). The
  token increase cannot be attributed to knowledge size alone: Skills-only
  trajectories reached later build/correctness stages and accumulated larger
  source, plan, and evaluator evidence. Its 470-second aggregate latency
  increase is likewise confounded by additional real compilation and NPU
  correctness work, so it is not an Adapter routing-overhead measurement.
- No arm entered optimization because the workflow correctly requires a
  compiled, correct, benchmarked baseline first. The requested two optimization
  rounds were configured identically; they were not manually skipped.
- This is one paired run for each of three operators, as requested to avoid
  redundant mass testing. It is sufficient to expose failure transitions, but
  not to establish statistical superiority for a nondeterministic LLM.

### Validation

- Compared each B/D `trajectory.json` configuration after excluding only
  `knowledge_source` and the required mode-specific `output_dir`; no other
  configuration differences were found.
- Audited every Skills-only Planner and Generator knowledge bundle. Structured
  `api_semantics`, `relevant_facts`, `examples`, `failure_cards`,
  `project_contracts`, and `provenance` were empty; structured working document
  IDs/supplements were empty and full/incremental structured route counts were
  zero. All summaries recorded `structured_prompt_enabled=false` and
  `skills_enabled=true`, so Mode D did not fall back to Knowledge A.
- Audited all 48 LLM calls: every call ended with `finish_reason=stop`. All 30
  Generator calls requested thinking disabled and 32768 maximum output tokens,
  and none recorded reasoning content. The prior reasoning-only
  `finish_reason=length` failure did not recur.
- Confirmed all six trajectories used `evaluator=local` and `mock=false`.
  Permute Skills-only produced successful real CANN builds and entered real NPU
  correctness execution, proving the run was not a mock result.
- Final conclusion: keep structured knowledge available behind the runtime
  switch. Skills-only shows a promising compile-frontier improvement, but zero
  correctness means there is currently no evidence to delete or replace
  Structured Knowledge A. The next change should target exact Host ABI and
  installed-CANN API constraints inside the selected Skill capsules, followed
  by another controlled evaluation rather than removing either source.

## 2026-09-11

### Changes
- 做关于cannskills中生成ascendc算子的skills 的适配器
主要目的就是改善cannagent的知识库，当前知识库的内容太差，llm利用不到有效的知识内容
## 2026-09-10

### Changes

- Replaced eight unused TileLang-to-AscendC and outdated verification documents with direct, authority-labelled Host, Vector, Cube, C/V, and cross-core project guides under `AscendC_knowledge`.
- Added deterministic project contract, failure card, and pattern card compilation with evidence provenance; document routing now always includes the Host contract and can select the direct kernel patterns.
- Reframed the retained `ascendc-translator` Skill as a direct AscendC entry point without a TileLang prerequisite.
- Updated repository workflow documentation and the retained TileLang Skill boundary so no active instruction references the deleted translation documents or treats TileLang output as an AscendC generation prerequisite.
- Replaced ambiguous knowledge names across the CLI, Python package, build layout, tests, and architecture documents: `document` now means raw Markdown retrieval, `structured` means compiled fact retrieval, and immutable outputs are `knowledge builds` rather than snapshots.
- Made `structured` the default knowledge mode, added repository-local source/store defaults, and publish `current.json` after a validated build so ordinary Kernel runs do not require a build ID.
- Renamed `knowledge_v2` to `structured_knowledge`, `KnowledgeRouterV2` to `StructuredKnowledgeRouter`, and the compile-time checker to `ApiConstraintValidator`; added a detailed install/update and runtime-data-flow guide.
- Added an AscendC multi-turn README covering execution, phase behavior, every knowledge-build artifact, per-round routing, trajectory analysis, and knowledge-update validation.
- Expanded the multi-turn README with the complete output-directory contract: final source files, resumable global state, plans, candidate bundles, evaluation frontiers, incident records, per-round knowledge artifacts, conditional validation logs, and generated build outputs.

### Analysis

- Physical documents outside the direct runner allowlist do not affect its prompt. The actionable gap was that the API corpus contained Kernel API pages but no systematic Host binding/launch contract.
- Migrated project-authored guidance remains `PROJECT_CONTRACT`; it is not promoted to official CANN truth. Official API Markdown and its image evidence remain intact.
- Tests retain removed DSL filenames only as negative migration and allowlist assertions; those strings are not runtime knowledge sources.
- The full 8.5.0 source rebuild produced 121 normalized documents, 930 atomic facts, 341 API cards, 3 project contracts, 2 failure cards, and 5 pattern cards without indexing project-guide headings as API symbols.
- Structured knowledge compilation remains an installation/update operation, not a per-Kernel step: it normalizes sources, extracts and validates facts/cards/indexes, publishes an immutable content-addressed build, and advances `current.json` only after validation succeeds.
- The generated `knowledge_store/` is local runtime data and is ignored by Git; `--knowledge-build-id` exists only for reproducing an older run.
- `ARARARAR` is a valid axis-pattern enum member found in compressed official Reduce tables, but its presence in `symbols.json` exposed an API-identity extraction bug rather than an invalid official-document value.

### Validation

- Built and validated a complete CANN 8.5.0 knowledge build: zero fact issues, zero conflicts, and no project-guide symbol-index pollution.
- Passed all 24 structured knowledge tests and all 79 repository tests; Python bytecode compilation also passed.
- Verified deleted-document references remain only in four negative regression assertions. Ruff still reports pre-existing repository-wide lint findings; the files changed for structured compilation and routing pass when the existing `TRY004` policy violation in document-response parsing is excluded.
- Passed all 25 structured-knowledge tests and all 80 repository tests after the rename.
- Built the complete CANN 8.5.0 source into knowledge build `df3113d5fa2f6c82f46f1dfc`: 121 documents, 930 facts, 341 API cards, zero validation issues, and zero conflicts; verified `current.json` selects it.
- Ran the GELU mock workflow without specifying `--knowledge-mode`; it completed through the default structured path with only planner/generator calls and no document-router LLM call.
- Verified the new README against the active CLI, knowledge-build tree, runtime router, and the source evidence responsible for the `ARARARAR` index entry; no structured output or compiler behavior was changed.
- Verified the documented output names and conditional creation rules against the runner, trajectory logger, bundle manager, evaluator, frontier manager, and incident persistence implementation.

## 2026-09-09

### Changes

- Deleted the legacy standalone `ascendc_multi_turn/knowledge.py` layout, replaced it with the `ascendc_multi_turn/knowledge/` package, and exposed the documented `python -m ascendc_multi_turn.knowledge.build` snapshot CLI; repository history is the record of the removed layout.
- Added the M5 experiment knowledge loop with incident records, bounded repair diffs, resolved-fact links, correctness-and-signal-gated confirmed experience candidates, persistent evaluation frontiers, and automatic rollback of non-advancing candidates.
- Added the M4 CallSemanticsResolver and semantic source validator, including exact API/structure/overload binding, provenance-bearing resolved calls, generic unit/alignment/project-contract checks, and a pre-compile semantic evaluation stage for semantic mode.
- Added the M3 StructuredFailure schema and deterministic MTE/AI Core/ACL/compiler/runtime diagnostic parsing, persisted structured failures in evaluations and trajectories, connected them to semantic retrieval, and removed local details paths from LLM-facing compact evidence.
- Added the M2 context-aware semantic router, snapshot view, phase-specific knowledge bundle and trace schemas, exact API isolation, supplemental metadata/lexical/vector ranking, semantic runtime artifacts, and opt-in `--knowledge-mode semantic` while preserving legacy routing.
- Added the M1 immutable semantic snapshot builder, API/failure/pattern card schemas, fact validation, context-sensitive conflict reporting, exact symbol indexes, squashed-declaration symbol recovery, load API, offline CLI, tests, and a completed phase report.
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

- A failed candidate is useful only when it proves a deeper evaluation capability. Frontier rank therefore controls the repair baseline independently from correctness/performance best selection; repeated failure at the same rank restores the earlier stable candidate.
- Runtime experiments produce evidence under `CONFIRMED_EXPERIENCE`, never new official API truth. Promotion requires both a passing correctness result and disappearance of the addressed structured failure signature.
- A field name alone cannot define its unit: semantic validation now selects facts through API identity and parameter structure before comparing a source expression, and unresolved calls do not borrow constraints from similarly named APIs.
- Local `details_path` values remain useful for audit but cannot be a model's primary diagnostic input; semantic routing now consumes persisted failure fields and related API symbols instead.
- API identity must be resolved before similarity retrieval: prefix-related names such as DataCopy, DataCopyPad, and DataCopyExt are explicitly rejected when only a different exact source symbol is present.
- Snapshot conflict identity includes the complete applicability context; two `blockLen` facts in different parameter structures are context differences rather than mutually exclusive global facts.
- `ascendc_multi_turn` has no TileLang import, compiler invocation, generated TileLang file, or TileLang evaluation stage. The removed coupling was prompt-level: five allowlisted `dsl2Ascendc_*` supplements could inject translation assumptions into otherwise direct AscendC generation.
- The shared versioned AscendC API corpus, runtime-header extraction, static validator, build tool, correctness verifier, and performance evaluator are direct AscendC dependencies and remain in use even though some are physically stored below the legacy translator Skill directory.
- The retained interactive TileLang/translator Skills are separate compatibility entry points; deleting them is unnecessary for functional isolation of the direct multi-turn runner.
- A fixed diagnostic window cannot be expressed by `--max-rounds` alone because that budget starts only after a valid baseline; the independent total cap is required to compare unsuccessful and successful trajectories on the same number of evaluated candidates.

### Validation

- Validated the stable knowledge-build module entry point with `--help` and reran all runtime imports after replacing the standalone knowledge module.
- Validated M5 frontier advancement, implied frontier materialization, no-frontier source rejection, same-stage rollback, score-gated performance replacement, incident persistence, and confirmed-experience authority isolation in unit and end-to-end runner tests.
- Validated DataCopyPad overload and source-fact resolution without DataCopy contamination, pre-compile `blockLen` unit rejection, and generic required/forbidden project-contract enforcement.
- Validated MTE illegal-configuration extraction with runtime code, core/block IDs and DataCopyPad symbols, compile/ACL classification, structured trajectory data, and omission of private log paths from compact model evidence.
- Validated semantic snapshot disambiguation, exact similar-name rejection, context-filtered facts, retrieval traces, and a mock multi-turn run whose call log contains no knowledge-router LLM request.
- Validated deterministic snapshot IDs and rebuild reuse, snapshot load integrity, required DataCopy/DataCopyPad/TPipe/TQue symbol cards, provenance rejection, and context-sensitive conflict handling.
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
