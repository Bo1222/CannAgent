# CannAgent 检索链路（RAG）分析报告

- 分析对象：`CannAgent/ascendc_multi_turn/` 全链路，重点 `devkit_retrieval.py`
- 语料：`asc-devkit-9.1.0/`（固定 commit `c785b5f7…`）+ `knowledge_modules/cannbot_a08c4970_knowledge_base/`（固定 commit `a08c4970…`）
- 分析日期：2026-09-19
- 数据来源：源码静态阅读 + 本机实测 + `outputs/e2e_add_diagnosefix_20260918` 真实产物回放

---

## 0. 结论摘要

1. **系统里没有 BM25，也没有任何向量 / embedding / rerank 检索。** 全仓库 grep 零命中。检索是纯确定性的启发式打分与条件门控。
2. **`devkit_retrieval.py` 不在路由决策路径上**，它是路由**下游**的一个并列证据源（第 5 层），输入由前面的路由层产出，输出与 CANNBot 知识库（第 4 层）融合。
3. **`devkit_retrieval.py` 是整条链路最大的性能瓶颈**：单次 `retrieve()` 实测 **4.3–4.5 秒**（冷缓存首次 6.8 秒），其中 **75% 是打分循环**（一个可一行修复的 `lower()` 重复计算）。
4. **检索结果 62% 被白算**：每轮选出 24 个 chunk（每 root 6 个），但按生产配置 `max_knowledge_chars=24000` 只有 **9 个**能进 prompt，`declaration` 和 `implementation` 两个 root **存活数为 0**。
5. **结构化 API 通道完全失效**：真实产物里 `api_facts` 有 24 条，但 `精确 API 与 runtime 事实` section 的 **25 个条目全部被 budget 截断**，一条都没进 prompt。
6. **L1–L8 每轮跑两次，不是一次**：planner 与 generator 各建一遍，且 planner 的结果**从不复用**给 generator（`runner.py:1413`）。实测 9 轮共 **17 次** L1–L8。
7. **L1–L8 完全不需要 LLM**，是纯 Python 确定性逻辑。LLM 是 L8 输出的**消费者**。实测 17 次模型调用中 `planner` 3 次 / `diagnose` 6 次 / `generator` 8 次。
8. **存在一套从不被调用的 LLM 知识路由器残骸**：`knowledge_router` 的 system prompt、call config、乃至 `--router-thinking` / `--router-max-tokens` CLI 开关都还在，但全仓库无任何调用点，实测调用 0 次。
9. **【最严重】检索输出对失败演化完全不敏感**：`build.log` 每轮都在变（CMake 错误在演化），但 `references.md` **从 round_03 到 round_08 逐字节完全相同**，连续 6 轮注入同一份 9 个 excerpt。这是质量问题而非性能问题，且无法靠调参修复（§5.9）。
10. **L5 的重复计算率 82%**：17 次调用只产出 **3 个不同**的 `EvidenceBundle`，其余 14 次是完全相同的重算（§6.4）。

> 第 6/7/8 条的完整证据（调用点、条件、I/O 契约、逐轮验证）见 §3 的「执行覆盖」「LLM 依赖」「L1–L8 输入 / 输出契约总表」「遗留物」四节。

---

## 1. 实测数据量

### 1.1 四个检索 root 的规模

`DevkitRetriever.ROOTS`（`devkit_retrieval.py:46-51`）定义 4 个 root，`retrieve()` 每次调用**全量遍历并全文读取**：

| root | 优先级 | 目录 | 全部文件 | 文本文件 | 文本字节 | 图片等被跳过 |
|---|---|---|---|---|---|---|
| `api_doc` | 0 | `docs/api` | 3 942 | **2 720** | 11.38 MB | 1 222 |
| `example` | 1 | `examples` | 1 809 | **1 639** | 7.38 MB | 170 |
| `declaration` | 2 | `include` | 616 | **616** | 3.47 MB | 0 |
| `implementation` | 3 | `impl` | 4 444 | **4 421** | 36.75 MB | 23 |
| **合计** | | | **10 811** | **9 396** | **58.98 MB** | 1 415 |

> 文本文件 = 后缀落在 `_TEXT_SUFFIXES`（`devkit_retrieval.py:11`）内的文件。图片后缀在 `_IMAGE_SUFFIXES`（`:12`）中被过滤，符合 README「图片不会进入上下文」的承诺。

### 1.2 `docs/api` 内部构成

```
docs/api               3 942 文件 / 49.38 MB
├── SIMD-API           1 815 文件   ← 主体
├── figures            1 159 文件   ← 全部是 .png
├── SIMT-API             743 文件
├── Utils-API            209 文件
├── 附录                   9 文件
└── AI-CPU-API             5 文件

按后缀：  .md 2 720  |  .png 1 222
```

### 1.3 文档体积分布（`docs/api` 的 2 720 个 md）

| 指标 | 值 |
|---|---|
| 长度中位数 | 1 752 字符 |
| 长度均值 | 3 176 字符 |
| **最大** | **226 108 字符**（`docs/api/README.md`，是中位数的 **129 倍**） |
| 次大 | 156 797 字符（`Cast-45.md`） |

**这个 129 倍的离散度直接击穿了打分公式**（见 §5.3）。

---

## 2. `devkit_retrieval.py` 在路由中的位置

### 2.1 完整分层图

调用链全部收敛在 `runner.py:865-971` 的 `_build_stage_knowledge()` 内，按执行顺序：

```
 входные данные: 项目参考 / 测试用例 / 当前源码 / 上一轮失败日志 / 当前 plan
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L1  路由决策层      ContextSelector.derive_route()                  │
│     输入: failure_stage + 失败文本                                   │
│     输出: primary_skill / debug_category / failure_ownership        │
│     机制: 关键字子串匹配规则树                      context_selector.py:140 │
└─────────────────────────────────────────────────────────────────────┘
        │  primary_skill
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L2  Stage 展开层    ContextSelector.derive_stages()                 │
│     输出: ["kernel_design","code_generation"] 等 stage 列表          │
│     机制: 白名单映射表                            context_selector.py:250 │
└─────────────────────────────────────────────────────────────────────┘
        │  stages
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L3  请求装配层      ContextSelector.request()                       │
│     产出 SkillAdapterContext：算子族 / shape 形态 / 风险旗标 /        │
│     符号证据 / 符号域 / profile 特征              context_selector.py:290 │
│     机制: 词表子串匹配 + 正则 + 启发式             skill_adapter.py:311  │
└─────────────────────────────────────────────────────────────────────┘
        │  SkillAdapterContext
        ├──────────────────────────────┬──────────────────────────────┐
        ▼                              ▼                              ▼
┌───────────────────────┐  ┌────────────────────────┐  ┌────────────────────────┐
│ L4 CANNBot 知识选择    │  │ L5 DevKit 证据检索     │  │ L6 运行时头文件事实     │
│ SkillAdapter.select() │  │ DevkitRetriever        │  │ collect_runtime_facts()│
│                       │  │   .retrieve()          │  │                        │
│ 60 个固定 md 模块      │  │ 9 396 个文件 / 59 MB   │  │ rg 实机 CANN/torch_npu │
│ 条件门控（布尔合取）   │  │ 词项加权打分           │  │ 头文件                 │
│                       │  │                        │  │ 手工优先级打分         │
│ skill_adapter.py:387  │  │ devkit_retrieval.py:73 │  │ runtime_knowledge.py   │
│                       │  │                        │  │                    :219│
└───────────────────────┘  └────────────────────────┘  └────────────────────────┘
        │                              │                              │
        └──────────────┬───────────────┴──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L7  融合层          ContextSelector.select()                        │
│     合并三路来源 / 去重 / 标 confidence_level / 生成 UNVERIFIED_API 告警│
│                                                  context_selector.py:522 │
└─────────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│ L8  渲染层          render_stage_context()  +  render_evidence()    │
│     两段拼接 → references.md / planner_references.md                │
│     prompts.py:238                runner.py:954                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 关键定位

**`devkit_retrieval.py` = L5，与 L4 并列，不在路由决策路径上。**

具体证据链：

- 调用点在 `runner.py:914`：`DevkitRetriever(self.devkit_root).retrieve(operator=…, symbols=…, query_text=…)`
- 它**消费**的 `symbols` 由 L3 产出（`runner.py:899-906` 从 `stage_request` 的三个符号列表 + `extract_api_symbols(evidence)` 合并）
- 它**不回头影响** L1/L2/L3 的任何决策。`derive_route` / `derive_stages` 在它之前就已执行完毕，不读它的输出
- 它的输出走**两条互不相干的路**：
  - → `ContextSelector.select()`（L7）进入 `api_facts`，条件是 `include_api = audience=="generator" or debug_stage`（`context_selector.py:532`）
  - → `render_evidence()`（L8）**无条件**追加到 `knowledge_text`（`runner.py:954`）

**这个"两条路"是后续多个缺陷的根源**，见 §5.5。

---

## 3. 各层路由逻辑拆解

### L1 路由决策层 —— `ContextSelector.derive_route()`（`context_selector.py:140`）

纯规则树，无任何打分。判定顺序：

```
previous is None or 无 error
  ├─ workflow_phase == "optimization" → optimization / correct_baseline_available
  └─ 否则                              → kernel_design / initial_or_unresolved_kernel_design

failure_stage ∈ _COMPILE_STAGES（8 个编译类 stage）
  ├─ 命中 _LAUNCH_ABI_MARKERS 或 diagnostics 有 launch_abi → host_integration_debug
  ├─ 命中 _HOST_MARKERS 或 diagnostics 有 host_abi/host_cpp   → host_integration_debug
  └─ 否则                                                     → api_compile_debug

failure_stage ∈ {correctness, runtime, acl_runtime}
  ├─ runtime_code / device_exception / _RUNTIME_MARKERS / 设备错误码正则 → runtime_debug
  ├─ _PRECISION_PRIMARY_MARKERS                                      → precision_debug
  └─ 否则 → kernel_design（命中 _PRECISION_SECONDARY_MARKERS 时追加 secondary=precision_debug）

failure_stage == "performance" → optimization
其余                          → kernel_design（reason=unmapped_failure_stage）
```

标记表定义在 `context_selector.py:17-94`。设备错误码用**族匹配** `\b(?:50|56)\d{4}\b`（`:42`），注释明确说明逐码枚举会漏掉 507034。

### L2 Stage 展开层 —— `derive_stages()`（`:250`）

映射表，逻辑简单：

| 条件 | 输出 stages |
|---|---|
| 有上一轮 error，primary ∈ {host_integration_debug, api_compile_debug} | `["host_integration_debug" 或 "compile_debug"]` + secondary |
| 有 error，primary == optimization | `["optimization"]`（仅当 phase 也是 optimization） |
| 有 error，其他 primary | `[primary_skill]` + secondary |
| phase == optimization | `["optimization"]` |
| 无 previous，`current` 不存在 | planner → `["operator_analysis","kernel_design"]`；generator → `["kernel_design","code_generation"]` |
| 其余 | `["kernel_design"]` |

### L3 请求装配层 —— `ContextSelector.request()`（`:290`）

四项派生，全部是启发式：

| 派生项 | 方法 | 机制 |
|---|---|---|
| `operator_families` | `skill_adapter.py:311` | 算子名归一化（`_`→空格）后与 `operator_families` 词表（`skill_mapping.yaml:5-14`）做**子串包含**；不中则对 evidence 做**命中词计数**，分高者胜 |
| `shape_regime` | `context_selector.py:432` | 正则 `\[\s*\d+(?:\s*,\s*\d+)*\s*\]` 数维度；`dynamic shape/symbolic` 判动态；族→关系映射 |
| `risk_flags` | 同上 | 9 项布尔旗标（reduction/broadcast/dynamic_shape/tail/multi_output/atomic/matmul_cube/host_kernel_coupling） |
| `symbols` | `diagnostics.extract_symbol_evidence()` | 从 failure/source/planned 三路文本提符号，`classify_symbol_domain()` 分域 |

### L4 CANNBot 知识选择层 —— `SkillAdapter.select()`（`skill_adapter.py:387`）

**这是真正意义上的"知识库路由"**，但机制是条件门控而非检索：

1. 按 `context.stages` 取出 `skill_mapping.yaml` 里对应的 skill 组（外加 `standing_contracts`）
2. 对每个 skill 跑 `_condition_matches()`（`:332`）——**合取判定，全部条件必须满足**：
   `always` / `requires_correct_baseline` / `failure_stages` / `any_terms` / `when_any` / `operator_families` / `when_domains` / `profiles` / `profile_features` / `risk_flags` / `shape_relations`
3. 对每条 reference 再跑一次同样的判定 + `operators` 白名单（`_reference_matches()`，`:380`）
4. 命中后用 `_extract_markdown_sections()`（`:140`）按 `headings` 切 Markdown 段落，**整段注入、绝不截断**
5. 两级去重：`(module_id, section_id)` + 内容 sha256
6. 预算贪心装箱：`planner=12000 / generator=20000` 字符，超预算**整组回退**并释放去重槽位

知识库固定 60 个 md 模块，`skill_adapter.py:202` 校验 vendor manifest 的 sha256，`:247` 再校验每个模块的 sha256，任一对不上直接抛异常。

### L5 DevKit 证据检索层 —— `DevkitRetriever.retrieve()`（`devkit_retrieval.py:73`）

见 §4 深度剖析。

### L6 运行时头文件事实层 —— `collect_runtime_facts()`（`runtime_knowledge.py:219`）

不是知识库检索，是**在真实安装的 CANN / torch_npu 头文件里 grep 符号**：

- `_matching_locations()`（`:86`）用 `rg --fixed-strings --word-regexp` 找最多 512 处命中
- `priority()`（`:144`）**手工加权**排序：注释行 +80（惩罚）、struct/class −40、函数声明 −30、`/third_party/` +30、`op_frame`/`_impl.h` +20、`/basic_api/` −10
- `_excerpt()`（`:179`）用**括号/花括号/尖括号配平**确定摘录边界（比固定行窗精确）
- 输出四级置信度：`Level 3 Verified`（compile probe 通过）> `Level 2 Installed`（头文件命中）> `Level 1 Documented` > `Level 0 Inferred`
- 带环境 fingerprint，缓存需 `fingerprint_id` 匹配才复用

### L7 融合层 —— `ContextSelector.select()`（`context_selector.py:522`）

三路来源合并，产出 `SelectedStageContext`。关键在于 `include_api` 门控：

```python
include_api = request.audience == "generator" or debug_stage   # :532
```

非 debug stage 的 planner **不会**拿到 `api_facts`。

### L8 渲染层 —— 两次独立的预算裁剪

```python
rendered = render_stage_context(selected_context)                    # runner.py:945
selected_context.budget["max_chars"] = min(budget, max_knowledge_chars)
knowledge_text = rendered + "\n\n" + render_evidence(                  # runner.py:954
    evidence_bundle, max_chars=max_knowledge_chars)
```

`render_stage_context()`（`prompts.py:238`）内部按 section 配额裁剪：

| section | planner 配额 | generator 配额 |
|---|---|---|
| 任务与平台事实 (`other`) | 1 000 | 2 000 |
| 项目硬约束 (`hard`) | 2 500 | 3 000 |
| 明确禁止项 (`other`) | 1 000 | 2 000 |
| 失败专用指引 (`failure`) | 3 000 | 4 000 |
| **精确 API 与 runtime 事实 (`api`)** | **1 500** | **6 000** |
| 相关内置知识模块 (`skills`) | 4 000 | 5 000 |

### 执行覆盖：每轮都会经过 L1–L8 吗

**不是一次，也不是固定次数。** L1–L8 的唯一入口包装是 `Runner._knowledge_context()`（`runner.py:858`），它一共有 **3 个调用点**：

| 调用点 | audience | 触发条件 |
|---|---|---|
| `runner.py:1309` | `planner` | 在 `if need_plan:` 内（`:1304`）**且** `if pending_phase != "EVAL"`（`:1307`） |
| `runner.py:1058` | `generator` | `_prepare_candidate()` 中 `prepared_knowledge is None` 时（`:1057`） |
| `runner.py:1034` | `generator` | `pending_phase == "EVAL"` 且存在 `candidate.json` 检查点时，从 checkpoint 恢复候选后重建 selection 供日志使用 |

**关键实现细节 —— planner 那次的结果从不复用给 generator**（`runner.py:1413`）：

```python
generator_knowledge = (
    None if self.context_selector is not None else prepared_knowledge
)
```

而 `self.context_selector = ContextSelector(self.skill_adapter)` 在 `runner.py:99` **恒被赋值**，因此 `generator_knowledge` 恒为 `None`，`_prepare_candidate` 必然以 `audience="generator"` **从零重跑一遍 L1–L8**。

**实测覆盖（`outputs/e2e_add_diagnosefix_20260918`，9 轮）**：

| 轮次 | planner 侧 L1–L8 | generator 侧 L1–L8 |
|---|---|---|
| `round_01` … `round_08` | ✓ | ✓ |
| `round_09` | ✓ | ✗（规划完成后停止，未生成） |
| **合计** | **9 次** | **8 次** |

> 与产物完全吻合：`devkit_evidence_planner.json` 9 个 + `devkit_evidence_generator.json` 8 个 = **17 个**。

**L6 是唯一条件执行的层**（`runner.py:927`）：

```python
if audience == "generator" or any(stage.endswith("_debug") for stage in stage_request.stages):
    runtime_facts = collect_runtime_facts(...)
```

实测逐轮验证（`runtime_header_facts_*.json` 的存在性）：

| round | planner 侧 L6 | generator 侧 L6 | planner stages | generator stages |
|---|---|---|---|---|
| `round_01` | **无** | 有 | `operator_analysis, kernel_design` | `kernel_design, code_generation` |
| `round_02`…`round_09` | 有 | 有 | `compile_debug` | `compile_debug` |

`round_01` 的 planner stage 不是 debug stage，`include_api` 为假 → L6 被跳过。**与代码条件逐轮精确吻合。**

### LLM 依赖：L1–L8 零模型调用

**L1 到 L8 全部是纯 Python 确定性逻辑，一次 LLM 调用都没有。**

- 全仓库 `call_type=` 的实际调用点只有 4 处：`planner`（`runner.py:803`，`call_type` 由 `:765` 决定）、`diagnose`（同上）、`generator`（`:1105`）、`generator_retry`（`:1120`）
- LLM 是 **L8 输出的消费者，不是 L1–L8 的产出者**：`_create_plan(knowledge_context=planning_knowledge)` 与 `_prepare_candidate()` 把 L8 渲染出的字符串作为 prompt 的一部分发出去

**实测（`calls.jsonl`，17 次调用 / 总 190.1 s）**：

| call_type | 次数 | 总耗时 | 单次中位 |
|---|---|---|---|
| `planner` | 3 | 21.0 s | 5.8 s |
| `diagnose` | 6 | 49.6 s | 8.0 s |
| `generator` | 8 | 119.5 s | 16.2 s |
| **`knowledge_router`** | **0** | — | — |

对照：L1–L8 共跑 17 次，其中 L5 单次约 4.3–4.5 s → **纯 CPU 检索约 76 s**，占 LLM 总耗时（190 s）的 **40%**。

### L1–L8 输入 / 输出契约总表

| 层 | 实现位置 | 输入 | 输出 | LLM |
|---|---|---|---|---|
| **L1** 路由决策 | `context_selector.py:140` `derive_route()` | `workflow_phase: str`、`current_exists: bool`、`previous: EvalResult \| None`、`evidence: str`（失败日志文本） | `RoutingDecision`：`primary_skill`、`route_reason`、`secondary_skill`、`secondary_reason`、`debug_category`、`routing_confidence`、`route_evidence_origin`、`matched_route_trigger`、`failure_ownership` | ✗ |
| **L2** Stage 展开 | `context_selector.py:250` `derive_stages()` | 同 L1 四项 | `list[str]`：stage 名列表 | ✗ |
| **L3** 请求装配 | `context_selector.py:290` `request()` | `audience`、`workflow_phase`、`operator`、`soc`、`runtime_version`、`knowledge_version`、`current_exists`、`previous`、`evidence`、`source_evidence`、`planned_evidence`、`active_profile`、`profile_case_indices`、`profile_features`、`environment_fingerprint` | `SkillAdapterContext`（**20 个字段**：算子族、`shape_regime`、`risk_flags`、符号三路 `failure/source/planned_symbols`、`symbol_domains`、路由字段…） | ✗ |
| **L4** CANNBot 知识选择 | `skill_adapter.py:387` `select()` | `SkillAdapterContext` | `SkillAdapterSelection`：`knowledge_modules[]`、`trace[]`、`source_root`、`source_available`、`knowledge_base_id` | ✗ |
| **L5** DevKit 证据检索 | `devkit_retrieval.py:73` `retrieve()` | `operator: str`、`symbols: list[str]`、`query_text: str` | `EvidenceBundle`：`operator`、`symbols`、`evidence[]`（**固定 24 个** `EvidenceChunk`）、`retrieval_trace[]`、`version`、`commit` | ✗ |
| **L6** 运行时头文件 | `runtime_knowledge.py:219` `collect_runtime_facts()` | `symbols`、`runtime_version`、`cache_path`、`max_chars`、`soc_version`、`project_root`、`probe_manifest`、`failure_evidence` | `RuntimeFacts`：`symbols`、`text`、`conflicts`、`include_root`、`facts[]`（含 `confidence_level` 0–3）、`environment_fingerprint` | ✗ |
| **L7** 融合 | `context_selector.py:522` `select()` | `bundle: EvidenceBundle`、`request: SkillAdapterContext`、`runtime_facts` | 元组 `(SelectedStageContext, SkillAdapterSelection)`。`SelectedStageContext` 含 `api_facts`、`skill_knowledge_modules`、`exclusions`、`provenance`、`selection_trace`、`selection_metadata` | ✗ |
| **L8** 渲染 | `prompts.py:238` `render_stage_context()` + `devkit_retrieval.py:123` `render_evidence()` | `SelectedStageContext` / `EvidenceBundle` | `str` → 落盘为 `references.md`（generator）/ `planner_references.md`（planner） | ✗ |

**实测耗时（各层单独计时）**：

| 层 | 耗时 | 说明 |
|---|---|---|
| L1 + L2 + L3 | < 1 ms | 纯规则/字典查表 |
| **L4** `SkillAdapter` 构造（60 模块 + 全量 sha256 校验） | **29 ms** | 一次性，但每轮重建 |
| **L4** `select()` | **0.3 ms** | 条件判定 + 段落切分 |
| **L5** `retrieve()` | **4 300 – 6 800 ms** | **绝对瓶颈**（见 §5.1） |
| L6 | 50 – 500 ms（含 rg 扫描 + 缓存读写） | 条件执行 |
| L7 | < 1 ms | 内存合并去重 |
| **L8** `render_evidence()` | **< 0.1 ms** | 纯字符串拼接 |

**L4 的真实产出规模（实测产物）**：

| round | audience | 选中知识模块 | 注入字符 | trace（selected/rejected） |
|---|---|---|---|---|
| `round_01` | planner | 2 | 10 019 | 5 / 25 |
| `round_01` | generator | 3 | 14 732 | 10 / 20 |
| `round_02` – `round_05` | 两者 | **1** | **6 417** | 4 / 5 |

值得注意：在 `compile_debug` 阶段，**L4（真正的知识库路由）只产出 6 417 字符，而 L5 的原始转储有 23 488 字符** —— 确定性知识库被开发套件的无结构转储以 **3.7 : 1** 的比例压过。这与 §5.4 / §5.5 的预算缺陷是同一个病根。

### 遗留物：被替换掉的 LLM 知识路由器（死代码）

仓库里残留着一套**完整但从不被调用**的 LLM 知识路由配置：

| 位置 | 内容 |
|---|---|
| `llm.py:35-38` | call_type `knowledge_router` 的 system prompt：id `ascendc-knowledge-router-v2`，正文「你是 AscendC 中文知识库路由器。只从给定索引中选择下一轮直接需要的知识，保持 doc_id 原样，不得生成代码、改写知识原文或返回文件路径。」 |
| `models.py:220-226` | 专用 `LLMCallConfig`：`router_max_tokens`（默认 4096）、`router_thinking` |
| `__main__.py:54, 59` | **CLI 开关 `--router-max-tokens` / `--router-thinking` 仍在暴露** |
| `models.py:166, 192, 209` | 参数校验与环境变量解析仍在执行 |
| `runner.py:1331` | 仅剩一个异常标签字符串 `stage="llm_knowledge_router"` |

**但全仓库不存在任何 `_call_llm(call_type="knowledge_router")` 调用点，实测 `calls.jsonl` 中 0 次。**

结论：这是**被 `SkillAdapter` 确定性条件门控替换掉的 LLM 路由器的残骸**。用户传 `--router-thinking enabled` 或设置 `ASCENDC_ROUTER_MAX_TOKENS` **不会产生任何效果** —— 参数被解析、校验、存入配置对象，然后被丢弃。若要彻底清洁，可删除 `llm.py:35-38`、`models.py:220-226` 与两个 CLI 开关；若计划恢复 LLM 路由能力，则 `runner.py:1331` 的标签说明曾经的设计意图仍在。

### 一轮的完整时序（实测合成）

以 `round_05`（stage = `compile_debug`）为例：

```
round N
 ├─ L1 → L2 → L3   (planner 参数)          < 1 ms
 ├─ L4  SkillAdapter.select()               0.3 ms
 ├─ L5  DevkitRetriever.retrieve()       ≈ 4 500 ms   ← 瓶颈
 ├─ L6  collect_runtime_facts()          ≈ 50–500 ms
 ├─ L7  ContextSelector.select()            < 1 ms
 ├─ L8  render_stage_context + render_evidence  < 5 ms
 ├─ ▸ LLM call_type = planner | diagnose   ≈ 5.8–8.0 s
 │
 ├─ L1 → L2 → L3   (generator 参数)        < 1 ms    ← 全部重算，不复用 planner 结果
 ├─ L4                                      0.3 ms
 ├─ L5  DevkitRetriever.retrieve()       ≈ 4 500 ms   ← 第二次全量扫描
 ├─ L6                                   ≈ 50–500 ms
 ├─ L7                                       < 1 ms
 ├─ L8                                       < 5 ms
 └─ ▸ LLM call_type = generator            ≈ 16.2 s
```

**单轮纯 CPU 检索开销约 9–10 s，其中 90% 是 L5 的两次数 9 396 文件的全量扫描。**

---

## 4. `devkit_retrieval.py` 深度剖析

文件共 138 行。四个阶段：

### 4.1 阶段一：token 提取 —— `_tokens()`（`:58`）

```python
values = [*symbols, *re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", text)]
ignored = {"model","tensor","return","shape","dtype","input","output"}
return list(dict.fromkeys(v for v in values if v.lower() not in ignored))[:32]
```

- 符号名 + 查询文本中所有 **≥3 字符**的标识符
- 保序去重，**硬截断 32 个**
- stopword 表只有 7 个词
- **无 IDF、无加权、无词性/词形处理**

实测一次典型调用提到 11 个 token：`['DataCopyPadExtParams','TQue','LocalTensor','EnQue','DeQue','elementwise','add','fp16','tail','unaligned','DataCopyPad']`。

### 4.2 阶段二：逐 root 遍历打分（`:76-96`）

```python
for priority, (source_kind, relative_root) in enumerate(self.ROOTS):
    for path in root.rglob("*"):                       # 全量遍历
        if 后缀不在 _TEXT_SUFFIXES: continue
        name = path.name.lower()
        name_score    = sum(40 for token in tokens if token.lower() in name)      # :86
        text          = path.read_text(...)                # 全文读入
        content_score = sum(min(text.lower().count(token.lower()), 5) * 3
                            for token in tokens)                                   # :91
        score = name_score + content_score - priority                              # :92
        if score > 0: candidates.append((score, path, text))
    candidates.sort(key=lambda item: (-item[0], item[1].as_posix()))
    selected = candidates[: self.max_files_per_root]        # 默认 6，每 root 独立
```

打分量级：

| 项 | 最大值 | 说明 |
|---|---|---|
| `name_score` | 32 × 40 = **1 280** | 每个命中 token 固定 +40，不区分 token 重要性 |
| `content_score` | 32 × 15 = **480** | 每 token 上限 `min(count,5)×3 = 15` |
| `-priority` | −3 … 0 | **相对前两项可忽略，实质是平局时的 tie-breaker** |

### 4.3 阶段三：摘录窗口 —— `_excerpt()`（`:64`）

```python
positions = [lowered.find(token.lower()) for token in tokens]   # -1 表未命中
positions = [p for p in positions if p >= 0]
start = max(0, (min(positions) if positions else 0) - limit // 4)
end   = min(len(text), start + limit)                            # limit=2400
```

- 窗口锚定在**最早**的任意 token 命中位置，向前留 600 字符
- **每个文件只取一个窗口**，`limit=2400` 硬截断
- 若命中 token 分散在 226 KB 的文件各处，只展示其中一处

### 4.4 阶段四：渲染 —— `render_evidence()`（`:123`）

```python
sections = ["# Asc DevKit 官方证据\n\nversion=…; commit=…"]
used = len(sections[0])
for item in bundle.evidence:                # 按 root 顺序线性消费
    section = f"\n\n## {item.source_kind}: {item.source_path}\n" ... + item.excerpt
    if used + len(section) > max_chars: break          # ← 全局硬截断，直接 break
    sections.append(section); used += len(section)
```

**按 root 顺序消费 + 全局 cap + `break`** —— 三个因素叠加导致后面的 root 系统性饿死（见 §5.4）。

---

## 5. 实测缺陷清单

### 5.1 【P0】`text.lower()` 在 token 循环内重复计算 —— 3.8–3.9× 性能损失

`devkit_retrieval.py:91`：

```python
content_score = sum(min(text.lower().count(token.lower()), 5) * 3 for token in tokens)
```

`text.lower()` 对**每个 token 各执行一次**。9 396 个文件 × 最多 32 个 token = 最多 **30 万次**对平均 3 KB 字符串的完整 lower 操作。

**实测**（两次独立运行，附录 §8 可复现）：

```
载入 9 396 个文本文件
  原打分循环 = 3 164 – 3 264 ms    优化后 = 834 – 837 ms    加速 3.8 – 3.9x
```

整次 `retrieve()` 的时间分解：

| 阶段 | 耗时 | 占比 |
|---|---|---|
| 纯 I/O（`rglob` + 9 396 次 `read_text`） | 1 133 – 1 145 ms | 25% |
| **打分 + 排序** | **3 164 – 3 391 ms** | **75%** |
| 合计 | ~4 300 – 4 500 ms | |

**修复**（等价变换，不改变任何分数）：

```python
lowered = text.lower()                      # 每文件一次
content_score = sum(min(lowered.count(tok), 5) * 3 for tok in lowered_tokens)
```

其中 `lowered_tokens = [t.lower() for t in tokens]` 在 root 循环外预计算。

### 5.2 【P0】`name_score` 用子串匹配 —— 短 token 误匹配率高达 96%

`devkit_retrieval.py:86`：`token.lower() in name` 是**子串**包含，不是词边界匹配。

**实测误匹配率（`docs/api` 的 2 720 个 md 文件名）**：

| token | 子串命中 | 词边界真命中 | **误匹配率** | 误匹配样例 |
|---|---|---|---|---|
| `sin` | 49 | 2 | **95.9%** | `GetWindowsInAddr.md`、`SetSingleShape.md` |
| `cos` | 35 | 2 | **94.3%** | `ComputeCost.md` |
| `sort` | 19 | 2 | **89.5%** | `asc_mrgsort4.md`、`asc_bitsort.md` |
| `mul` | 58 | 11 | **81.0%** | `asc_cumulative_histogram.md`、`asc_mulls.md` |
| `exp` | 30 | 7 | **76.7%** | `aclrtcAddNameExpr.md`、`GetExpTmpBufferFactorSize.md` |
| `add` | 81 | 20 | **75.3%** | `SetFixPipeAddr.md`、`AddInputTd.md` |
| `pad` | 23 | 8 | **65.2%** | `asc_set_l13d_padding.md` |
| `sum` | 20 | 8 | 60.0% | `ReduceSum-90.md` |
| `sub` | 32 | 12 | 62.5% | `asc_sync_subblock_wait.md` |

**生产实例**：一次以 `add` 算子为主题的调用中，`api_doc` 分数第一名是
`docs/api/SIMD-API/基础API/cube_compute_ISASI/矩阵搬出辅助配置接口/SetFixPipeAddr.md`（score 118）
—— 命中原因只是 `Addr` 里含有 `Add`，与加法算子毫无关系，却占据了 6 个席位之一。

**修复**：预编译词边界正则（`_matching_locations()` 已在用同样手法，见 `runtime_knowledge.py:90`）：

```python
pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])", re.I)
name_score = sum(40 for pat in patterns if pat.search(name))
```

### 5.3 【P1】无 IDF + 无长度归一化 —— 长文件与高频词双重偏置

**（a）无 IDF。** `add`（语料中出现上万次）与 `DataCopyPadExtParams`（极稀有、极有区分度）在公式里权重**完全相同**，各占 40 / 15 分。这正是 BM25 的 IDF 项要解决的问题。

**（b）无长度归一化。** `content_score = Σ min(count,5)×3` 对每个 token 的满分固定 15 分，与文件长度无关；但**长文件命中任意 token 的概率趋近 1**，且多 token 累积机会更多。

`docs/api` 的 md 长度分布：

```
中位数 1 752 字符  |  均值 3 176  |  最大 226 108（README.md，中位数的 129 倍）
```

长文件典型受害者/受益者：

| 文件 | 长度 | 后果 |
|---|---|---|
| `docs/api/README.md` | 226 KB | 索引/目录文件，实测 score 96，挤进 `api_doc` 前 6 |
| `docs/api/SIMD-API/…/Cast-45.md` | 157 KB | 类型转换全表 |
| `docs/api/…/API列表.md` | 69 KB | 纯列表，无实质内容 |

且这些超长文件被 `_excerpt` 截到 **2 400 字符**（占 README 的 **1.06%**），窗口锚定位置还不受控 —— 花掉一个宝贵席位，实际注入的却是任意 1% 片段。

### 5.4 【P0】62% 检索结果被丢弃，`declaration` / `implementation` 两个 root 存活数为 0

三重因素叠加：

1. `retrieve()` 每 root 独立取 top-6 → 固定产出 **24 个 chunk**（实测每 root 恰好 6 个，总计 57 587 字符）
2. 生产配置 `max_knowledge_chars = 24000`（`runner.py:1224` 硬编码）
3. `render_evidence()` **按 root 顺序线性消费 + 全局 cap + `break`**（`:134`）

后果：`api_doc` 的 6 个先吃掉约 15 600 字符，`example` 再吃 3 个撑满 24 000，到 `declaration` / `implementation` 时**预算已尽，直接 break**。

**实测（合成查询，`max_chars=24000`）**：

| | api_doc | example | declaration | implementation | 合计 |
|---|---|---|---|---|---|
| 检索选中 | 6 | 6 | 6 | 6 | **24** |
| **实际渲染** | 6 | 3 | **0** | **0** | **9** |
| 丢弃 | 0 | 3 | 6 | 6 | **15（62%）** |

**真实产物交叉验证**（`outputs/e2e_add_diagnosefix_20260918`，round_02 / 05 / 07 三轮完全一致）：

```
round_05 planner_references.md  总 31 970  devkit段 23 488 ( 73%)  chunks= 9/24  roots={api_doc:6, example:3}
round_05 references.md          总 36 974  devkit段 23 488 ( 64%)  chunks= 9/24  roots={api_doc:6, example:3}
round_02 / round_07             同上，逐字节一致
```

`postmortem`：**`include/` 目录里放着真实安装的 API 声明头**（`kernel_tpipe.h`、`kernel_operator_data_copy_intf.h`…），对"这个符号的准确签名是什么"这类问题本应是最高价值证据，却被系统性零注入。而 `devkit 段` 占了 prompt 的 **64%–73%** —— **prompt 的绝大部分预算花在了一个只装得下 37.5% 结果、且缺失关键 root 的通道上。**

### 5.5 【P0】结构化 API 通道 100% 失效（与 5.4 相互独立）

`render_stage_context()` 的 `api` section 把 `runtime_facts` 与 `api_facts` **塞进同一个配额桶**（`prompts.py:328-335`）：

```python
api_entries = []
if runtime_facts:
    api_entries.append("已安装 public header 事实：\n" + runtime_facts)   # 最长 4 000 字符
api_entries.extend(json.dumps(item, …) for item in payload["api_facts"])  # 24 × ~2 600 字符
append_section("精确 API 与 runtime 事实", api_entries, quotas["api"])     # 配额仅 1 500 / 6 000
```

**供需严重失衡**：单是 `runtime_facts` 就实测 **3 931 字符**，已经超过 planner 的 `api` 配额（1 500），并吃掉 generator 配额（6 000）的 **65%**。剩余空间装不下任何一条 ~2 600 字符的 devkit 条目。

**真实产物证据**（round_05）：

```
generator_context.json  api_facts 条数 = 24
  truncated_sections = ['任务与平台事实[0]',
                        '精确 API 与 runtime 事实[1]' … '精确 API 与 runtime 事实[24]',   ← 24 条全丢
                        '紧凑 provenance[8]' … '紧凑 provenance[11]']
planner_context.json    api_facts 条数 = 24
  truncated_sections = [..., '精确 API 与 runtime 事实[0]' … '精确 API 与 runtime 事实[24]', ...]
                                                                    ↑ 含 runtime_facts[0] 在内，25 条全丢
```

即：**`api_facts` 检索出 24 条、`selection_metadata.selected_evidence_ids` 记录了 24 条，但结构化通道一条都没进 prompt。** 同时 `紧凑 provenance` 段还泄漏了被丢弃条目的 `evidence_id` / `source_path` —— 模型看到"证据编号"却看不到证据内容。

### 5.6 【P1】planner 门控与渲染路径自相矛盾

`context_selector.py:532` 明确门控：

```python
include_api = request.audience == "generator" or debug_stage
```

但 `runner.py:954` 的 `render_evidence()` **无条件追加**，不看 `include_api`：

```python
knowledge_text = rendered + "\n\n" + render_evidence(evidence_bundle, max_chars=max_knowledge_chars)
```

结果：非 debug stage 的 planner，其 `api_facts` 被 L7 刻意清空，却在 L8 拿到最多 24 000 字符的原始 devkit 转储。同时该转储无 `confidence_level` 标注，绕过了 L7 的置信度分级与 `UNVERIFIED_API` 告警机制。

### 5.7 【P1】无任何缓存，每轮重算

- `DevkitRetriever` 无 memoization，每次 `DevkitRetriever(root).retrieve(...)` 全新实例（`runner.py:914`）
- 每轮 **2 次**调用（planner + generator）
- 实测单次 4.3–4.5 秒（热缓存），冷缓存首次 6.8 秒

**真实 run 的开销**：`outputs/e2e_add_diagnosefix_20260918` 有 **9 个 round**、**17 个 `devkit_evidence_*.json`**：

```
9 轮 × 2 audience × ~4.5 s ≈ 81 秒     （且随 round 数线性增长）
```

会话内 `include`/`impl` 的 5 036 个文件内容在 9 轮里被**重复读取约 90 600 次**。

### 5.8 【P2】`max_api_docs` 是死参数

`runner.py:1224` 赋值 `max_api_docs, max_knowledge_chars = 8, 24000`，随后逐层透传：

```
runner.py:1224 → :1315/:1422 → _knowledge_context(:1017) → _build_stage_knowledge(:870)
```

但 **`devkit_retrieval.py` 从未读取它** —— 它用的是自己的 `max_files_per_root=6` 默认值（`:53`），且没有任何调用点覆盖该默认值。**上层的 8 与实际生效的 6 不一致，调节该参数完全无效果。**

### 5.9 【P0】检索输出对失败演化完全不敏感 —— 6 轮逐字节冻结

这是本次补充分析中发现的最严重的一条，**不是性能问题而是质量问题**。

**实测**：对同一 run 的逐轮产物做 sha256（`outputs/e2e_add_diagnosefix_20260918`）：

| round | `build.log`（失败证据） | `references.md`（L1–L8 最终输出） | `prompt.txt` |
|---|---|---|---|
| `round_02` | `aa155f656d1d` | `14e5be9ec29d` | `2d86b3605cfd` |
| `round_03` | `ad95f0c295ae` | `5bb4b8b6280a` | `0d3b8f3c489e` |
| `round_04` | —（缺） | `5bb4b8b6280a` | `4dafbcc81864` |
| `round_05` | `fee6903bce94` | `5bb4b8b6280a` | `416e71ef69ff` |
| `round_06` | `3e9d6f8fd565` | `5bb4b8b6280a` | `f776f072e7bc` |
| `round_07` | `fac5062aeaa0` | `5bb4b8b6280a` | `785915b3044e` |
| `round_08` | `da4609d5204d` | `5bb4b8b6280a` | `44882f06e221` |

**`build.log` 每轮都不同**（失败的 CMake 错误在演化）：

```
round_02: CMake Error: Could not find cmake module file: CMakeDetermineASCCompiler.cmake …
round_05: CMake Error at CMakeLists.txt:26 (find_package): By not providing "FindTorch.cmake" …
round_08: … Permission mismatch: The owner of …/torch_npu/lib/libop_plugi…
```

**但 `references.md` 从 `round_03` 到 `round_08` 逐字节完全相同**（`5bb4b8b6280a`，连续 6 轮）。`prompt.txt` 因携带源码与失败日志而在变，**唯独知识/证据部分被冻结**。

**后果**：generator 在调试一个持续演化的编译失败时，连续 6 轮拿到的是**完全相同的 9 个 excerpt**（§5.4）。`round_04` 甚至连 `build.log` 都没落盘，`references.md` 依然与前一轮一致 —— 说明整个 L1–L8 的产出与失败内容已经解耦。

> **不是重放**：`round_04` 是真实执行的一轮 —— 它在 `calls.jsonl` 中有 `diagnose`（7.6 s）与 `generator`（16.3 s）两次调用，`prompt.txt` 哈希（`4dafbcc81864`）也与其他轮不同（携带了变化的源码与修复状态）。**唯独知识段一字未变**，这正是本条缺陷的核心。

**根因**（三者叠加）：

1. **`symbols` 主导了 token 列表**：实测稳定为 12 个符号，`_tokens()` 的 `[:32]` 截断后，失败文本贡献的 token 排在符号之后
2. **打分极其粗糙**：子串匹配（§5.2）+ `min(count,5)` 硬饱和（§5.1），不同失败文本产生的分数差异不足以改变 top-24 排序，连 `retrieval_trace` 的 `candidates` 计数都完全一致
3. **没有反馈通道**：L5 只看 `symbols` 与 `query_text`，前者来自 L3 的符号提取、后者来自固定的证据拼接；失败文本的变化无法传导到检索

**这一条无法通过调参修复** —— 需要 §6.2 方案 2（词边界）+ 方案 5（IDF）组合，让失败文本中新出现的稀有标识符（如 `FindTorch.cmake`、`CMakeDetermineASCCompiler`）获得足够权重进入 top-N。**建议把「连续轮次 `references.md` 的哈希是否变化」作为该修复的验收指标。**

---

## 6. 优化方案

### 6.1 收益排序总览

| # | 优先级 | 优化项 | 预期收益 | 风险 | 改动量 |
|---|---|---|---|---|---|
| 1 | **P0** | `lower()` 外提 | 打分循环 **3.8–3.9×**；单次 retrieve ~4.5 s → ~1.6 s | 无（等价变换） | 3 行 |
| 2 | **P0** | 词边界匹配 + IDF 组合（治 §5.9） | 消除 sin/cos/sort 等 90%+ 误匹配；让失败文本中新出现的稀有标识符能进入 top-N | 低 | 5 行 + §6.2 方案 5 |
| 2b | **P0** | 检索对失败演化脱敏（§5.9） | 连续 6 轮注入同一份 excerpt → 恢复随失败变化 | 中 | 依赖方案 2 + 5 |
| 3 | **P0** | 全局排序后截断 + root 配额 | 丢弃率 62% → 0，`declaration` 从 0 → 有席位 | 中（需重定配额） | ~30 行 |
| 4 | **P0** | `api` 配额重建 | 结构化通道从 0/24 → 有效注入 | 中 | ~20 行 |
| 5 | **P1** | IDF 加权 | 高频词降权，稀有符号升权 | 低 | ~25 行 + 一次 df 统计 |
| 6 | **P1** | 长度归一化 | 消除 129× 长度离散度偏置 | 低 | 5 行 |
| 7 | **P1** | 进程内缓存 | 9 轮 × 2 次重扫 → 首轮一次 | 低 | ~40 行 |
| 8 | **P1** | planner 结果复用给 generator（见 §6.4） | 每轮 L1–L8 从 2 次 → 1 次，**省约 4.5 s/轮** | 中（两 audience 的 stage 不同，见 §6.4） | ~15 行 |
| 9 | **P2** | `max_api_docs` 接线或删除 | 消除死参数 | 无 | 2 行 |
| 10 | **P2** | 清理 `knowledge_router` 残骸（见 §6.5） | 消除误导性 CLI 开关与死配置 | 无 | 删 ~10 行 |

### 6.2 详细方案

#### 方案 1（P0）：`lower()` 外提

`devkit_retrieval.py:73-96`，在 root 循环前预计算，文件循环内只 lower 一次：

```python
def retrieve(self, *, operator, symbols, query_text):
    tokens = self._tokens(symbols, query_text)
    lowered_tokens = [t.lower() for t in tokens]            # 新增：预计算一次
    ...
        lowered_name = path.name.lower()
        name_score = sum(40 for t in lowered_tokens if t in lowered_name)
        ...
        lowered_text = text.lower()                          # 新增：每文件一次
        content_score = sum(min(lowered_text.count(t), 5) * 3 for t in lowered_tokens)
```

- **收益**：3 164–3 264 ms → 834–837 ms（实测 3.8–3.9×）
- **验证**：分数逐字节不变，`tests/test_ascendc_91_architecture.py:118` 天然通过

#### 方案 2（P0）：词边界匹配

`devkit_retrieval.py:86`：

```python
import re
patterns = [(t, re.compile(rf"(?<![A-Za-z0-9_]){re.escape(t)}(?![A-Za-z0-9_])", re.I))
            for t in tokens]
name_score = sum(40 for _, pat in patterns if pat.search(path.name))
```

**注意**：这会改变分数分布，需要重新校准 §6.3 的阈值，并重跑一次 e2e 基线对比。

#### 方案 3（P0）：改为全局排序 + 显式 root 配额

**当前问题**：每 root 独立 top-6 → 24 条 → 按 root 顺序消费 → 后面饿死。

**目标形态**：一次性取全局 top-N，N 由预算反推，且保证每个 root 有下限。

```python
# 1. 收集全部候选（不再按 root 截断）
all_candidates.sort(key=lambda it: (-it[0], it[1].as_posix()))

# 2. 按预算反推 N：每条 excerpt 上限 max_excerpt_chars，扣除 header 开销
budget_chars = ...                                    # 由调用方传入
per_chunk = self.max_excerpt_chars + 160              # header ~160
max_n = max(1, budget_chars // per_chunk)

# 3. 显式配额：保证高价值 root 下限，其余按分数竞争
ROOT_FLOOR = {"api_doc": 3, "example": 1, "declaration": 3, "implementation": 1}
```

- **收益**：`declaration`（真实 API 声明头）从 0 → 至少 3 个席位；总丢弃率 62% → 0
- **需同步修改**：`render_evidence()` 的 `break` 改为跳过超预算条目而非终止（`:134`），否则配额调整无效
- **验证**：用 `outputs/e2e_add_diagnosefix_20260918` 做 A/B，检查 `references.md` 里 `## declaration:` 计数 > 0

#### 方案 4（P0）：`api` 配额重建

`prompts.py:328-335` 的问题是把 `runtime_facts`（最长 4 000 字符）和 24 条 `api_facts` 挤进同一个 1 500/6 000 的桶。

**推荐拆分为两个 section，各自独立配额**：

```python
append_section("已安装 public header 事实", [runtime_facts], quotas["runtime"])
append_section("精确 API 证据", api_entries, quotas["api"])
```

并在 `quotas` 里新增：

| audience | `runtime` | `api` | 说明 |
|---|---|---|---|
| planner | 2 000 | 4 000 | planner 只需符号签名，不需要完整 excerpt |
| generator | 4 000 | 12 000 | generator 需要完整签名与用法 |

同时**降低单条 `api_facts` 的体积**：`_compact_evidence()`（`context_selector.py:497`）可把 `excerpt` 从 2 400 字符裁到 ~800（结构化通道只承载"签名级"信息，完整用法交给 §5.4 的 dump 通道），这样 12 000 配额可装 ~10 条而非 2 条。

#### 方案 5（P1）：IDF 加权

这是 BM25 中**唯一真正必要**的部分。在 `SkillAdapter` 或 `DevkitRetriever` 初始化时对语料建一次 df 表：

```python
# 一次性：统计每个 token 出现在多少个文件中
df[token] = 命中文件数
N = 9 396

import math
def idf(t): return math.log(1 + N / max(1, df.get(t, 0)))
```

打分改为：

```python
name_score    = sum(40 * idf(t) for t in tokens if pat(t).search(name))
content_score = sum(min(lowered.count(t), 5) * 3 * idf(t) for t in tokens)
```

**预期效果**：`add`（df 极高）权重被压到接近 0，`DataCopyPadExtParams`（df 极低）权重被放大数倍 —— 正好修正 §5.3(a) 的偏置。

- **代价**：启动时一次全语料 df 统计（实测遍历 9 396 文件约 1.1 s I/O），或固化为离线文件
- **注意**：`idf` 会放大 `name_score` 与 `content_score` 的量纲差，需重新校准 40 / 3 两个系数

#### 方案 6（P1）：长度归一化

采用 BM25 的长度归一化形式，加在 `content_score` 上：

```python
avg_len = 3 176                                   # 实测均值
b = 0.75
norm = (1 - b + b * len(text) / avg_len)
content_score = int(sum(min(lowered.count(t), 5) * 3 * idf(t) for t in tokens) / norm)
```

**预期效果**：226 KB 的 `README.md` 归一化后权重降到约 1/54，让位于中位长度的精准 API 文档。

#### 方案 7（P1）：进程内缓存

在 `runner.py:914` 处复用实例而非每轮新建：

```python
class DevkitRetriever:
    _corpus_cache: dict[str, list[tuple[Path, str]]] = {}    # root → [(path, text)]
    _df_cache: dict[str, dict[str, int]] = {}

    @classmethod
    def cached(cls, devkit_root, **kw): ...
```

- **收益**：9 轮 × 2 次 × ~4.5 s ≈ **81 s → 首轮 ~4.5 s + 后续 <0.5 s**
- **注意**：DevKit 是固定 commit 只读树，缓存安全；但需在 `inspect_devkit()` 健康检查失败时失效

#### 方案 8（P2）：`max_api_docs` 接线或删除

二选一：
- **接线**：`runner.py:914` 改为 `DevkitRetriever(self.devkit_root, max_files_per_root=max_api_docs).retrieve(...)`
- **删除**：从 `:870 / :1017 / :1044 / :1068 / :1224 / :1315 / :1422` 七处移除

推荐**删除**，因为方案 3 会用它替代 root 配额机制。

### 6.3 变更风险与回归验证

**测试覆盖情况**（`tests/test_ascendc_91_architecture.py:118-125`）：

```python
def test_retrieval_priority_metadata_and_image_exclusion(self):
    bundle = DevkitRetriever(root).retrieve(operator="add", symbols=["Add"], query_text="Add")
    self.assertEqual(bundle.evidence[0].source_kind, "api_doc")          # ← 钉住 root 优先级
    self.assertTrue(all(not item.source_path.endswith(".png") ...))      # ← 钉住图片过滤
    self.assertTrue(all(item.commit == ASC_DEVKIT_COMMIT ...))           # ← 钉住版本戳
```

**该测试只钉住三件事：root 优先级顺序、图片过滤、version/commit 戳记。** 打分公式、摘录窗口、token 提取**均未被测试锁定** —— 因此方案 1/2/5/6 属于测试安全区间，只要保持 `ROOTS` 顺序、图片过滤、戳记不变。

**必须保留的不变量**：

1. `EvidenceBundle.retrieval_trace` 的结构（`root` / `candidates` / `selected`）—— 被 `runner.py:923` 写成 `retrieval_trace_{audience}.json`，属于可观测性契约
2. `EvidenceChunk` 的 8 个字段 —— 被 `_compact_evidence()`（`context_selector.py:497`）和 `selected_items` 消费
3. `version=ASC_DEVKIT_VERSION` / `commit=ASC_DEVKIT_COMMIT` 戳记
4. 图片不进上下文（README 明文承诺）

**建议回归方式**：

```bash
# 1. 单元测试
cd /mnt/workspace/CannAgent && python -m pytest tests/test_ascendc_91_architecture.py -v

# 2. 检索层 A/B（不调 LLM）：固定 round_05 的 symbols 与 query_text，
#    对比改造前后 evidence 的 (source_path, score) 序列与 roots 分布
# 3. e2e A/B：以 outputs/e2e_add_diagnosefix_20260918 为基线，
#    对比 references.md 的 root 分布、轮数、最终 benchmark 分数
```

### 6.4 详案：planner 结果复用给 generator

**现状**（`runner.py:1413`）：

```python
generator_knowledge = (
    None if self.context_selector is not None else prepared_knowledge
)
```

`self.context_selector` 恒非 `None`（`runner.py:99`），所以 planner 那次 `_knowledge_context(audience="planner")` 的结果**被无条件丢弃**，generator 必定重跑一遍完整 L1–L8。

**不能直接复用的原因**：两侧的 `audience` 与 `stages` 不同，会改变 L4/L6/L8 的行为：

| 差异点 | planner | generator |
|---|---|---|
| `audience` | `"planner"` | `"generator"` |
| `stages`（首轮） | `operator_analysis, kernel_design` | `kernel_design, code_generation` |
| L4 预算 | 12 000 字符 | 20 000 字符 |
| L6 | 仅 debug stage | **总是**执行 |
| L8 section 配额 | `api`=1 500 | `api`=6 000 |

**可行的复用策略（推荐）**：把昂贵的 **L5 结果**（`EvidenceBundle`，含 24 个 chunk）按 `(operator, tuple(symbols), query_text)` 缓存复用。L4/L6/L7/L8 依然各跑各的。

**实测重复度 —— 收益比预期大得多**。对 9 轮产物做整包 sha256，**17 次 L5 调用只产出 3 个不同的 bundle**：

| round | planner 侧哈希 | generator 侧哈希 | 两侧相同？ | planner symbols | generator symbols |
|---|---|---|---|---|---|
| `round_01` | `62e23f9f894e` | `dc9d3371408a` | **✗** | **0** | **5** |
| `round_02` … `round_08` | `2f27fe863b42` | `2f27fe863b42` | ✓ | 12 | 12 |
| `round_09` | `2f27fe863b42` | —（未生成） | | 12 | |

```
不同哈希值的出现次数：
  2f27fe863b42  出现 15 次     ← round_02..09 的两侧 + round_09 planner
  62e23f9f894e  出现  1 次     ← round_01 planner
  dc9d3371408a  出现  1 次     ← round_01 generator

总 L5 调用 = 17 次 ；独立 bundle = 3 个 ；可省 = 14 次
```

**`round_01` 两侧不同的原因**（我最初误判为「输入恒等」，实测推翻）：planner 的 L5 跑在 **plan 生成之前**，`planned_evidence` 还是空；generator 的 L5 跑在 **plan 生成之后**，`extract_api_symbols(planned_evidence)` 能从计划里提取出 5 个 API 符号。所以两侧 symbols 由 `0` vs `5` 变为 `12` vs `12`。**这说明缓存键必须包含 `symbols`**，不能只按 round 缓存。

- **收益**：17 次 → **3 次** L5 调用，省 **14 × ~4.5 s ≈ 63 秒**（占该 run LLM 总耗时 190 s 的 33%）
- **叠加方案 7（进程内语料缓存）后**：3 次调用的 I/O 也只剩首次真实读盘
- **注意**：`EvidenceBundle` 是可变 dataclass，复用前需确认下游（`ContextSelector.select()` 的 `_compact_evidence()`、`_DEVICE_ERROR_CODE` 相关逻辑）不会就地修改它 —— 经检查 L7 只读取，可以安全共享
- **额外收益**：缓存键命中率是一个**天然的检索健康度指标**。命中率长期 100%（如本例）说明检索对失败演化不敏感 —— 正是 §5.9 的问题

```python
# 在 _build_stage_knowledge 中，L5 调用点（runner.py:914）改为：
cache_key = (Path(self.config.op_file).stem, tuple(symbols), evidence)
evidence_bundle = self._devkit_cache.get(cache_key)
if evidence_bundle is None:
    evidence_bundle = DevkitRetriever(self.devkit_root).retrieve(...)
    self._devkit_cache[cache_key] = evidence_bundle
```

- **收益**：每轮 L5 从 2 次 → 1 次，**省约 4.5 s/轮**；9 轮省约 40 s
- **叠加方案 7（进程内语料缓存）后**：每轮 L5 从 2 次 → 0 次（首轮之后），9 轮省约 76 s
- **注意**：`EvidenceBundle` 是可变 dataclass，复用前需确认下游（`ContextSelector.select()` 的 `_compact_evidence()`、`_DEVICE_ERROR_CODE` 相关逻辑）不会就地修改它 —— 经检查 L7 只读取，可以安全共享

### 6.5 详案：清理 `knowledge_router` 残骸

这是**纯清理**，无功能影响（因为该路径本就不执行）。删除清单：

| 位置 | 内容 |
|---|---|
| `llm.py:35-38` | `_SYSTEM_PROMPTS["knowledge_router"]` 条目 |
| `models.py:220-226` | `call_config()` 中的 `knowledge_router` 分支 |
| `models.py:109, 113` | `router_max_tokens` / `router_thinking` 字段 |
| `models.py:166, 192-193, 209` | 上述字段的校验与默认值解析 |
| `__main__.py:54, 59, 140, 144` | `--router-max-tokens` / `--router-thinking` 开关及其透传 |
| `runner.py:1331` | `stage="llm_knowledge_router"` 标签 → 改为 `stage="llm_knowledge_context"`（该处是真实的异常路径，标签本身有保留价值，只是名字误导） |

**验证**：`tests/test_ascendc_91_architecture.py:129-134` 的 `test_legacy_knowledge_flags_are_absent` 断言了 `--knowledge-mode` / `--knowledge-source` / `--knowledge-store` / `--skill-adapter` 四个遗留开关**不在** help 中 —— 删除 `--router-*` 与既有测试意图一致，建议同步把这两个开关加入该断言列表：

```python
for option in ("--knowledge-mode", "--knowledge-source", "--knowledge-store",
               "--skill-adapter", "--router-max-tokens", "--router-thinking"):
    self.assertNotIn(option, help_text)
```

> **反向选择**：如果团队的路线图里还打算恢复 LLM 路由（`ascendc-knowledge-router-v2` 的 prompt 写得很明确），那么应保留配置、只在 `runner.py:1331` 加注释说明当前未启用，避免后续维护者误以为它在工作。

---

## 7. 关于 BM25 / 向量检索的判断

### 7.1 是否应该引入 BM25

**部分应该 —— 但只是 IDF 那一部分。**

当前公式与 BM25 的差距：

| BM25 组件 | 当前实现 | 是否需要 |
|---|---|---|
| **IDF** | ❌ 完全缺失 | ✅ **必须补**（§6.2 方案 5） |
| **tf 饱和** | ⚠️ `min(count,5)` 硬截断，是粗糙替身 | 🔸 已有，可保留 |
| **长度归一化** | ❌ 完全缺失 | ✅ **必须补**（§6.2 方案 6） |
| k1 / b 调参 | — | ❌ 当前语料与查询形态下调参收益远低于前三项 |
| 倒排索引 | ❌ 每文件线性扫描 | 🔸 可用 df 表顺带承担部分剪枝 |

**建议**：实现 **IDF + 长度归一化 + tf 饱和**的 BM25-lite（不需要建倒排索引、不需要 k1 调参），即可拿到 BM25 绝大部分收益。完全体 BM25 的边际收益在这个语料规模下不明显。

### 7.2 是否应该引入向量检索

**不应该。** 理由：

1. **规模不匹配**：9 396 个文件、59 MB 语料。向量检索的收益要在十万级文档以上才显现，此处属于杀鸡用牛刀
2. **破坏离线与确定性**：README 明确承诺"普通运行不访问网络，只读托管缓存"。embedding 模型引入额外的权重文件、GPU/NPU 依赖与推理延迟，与离线约束冲突
3. **破坏可审计性**：当前每个 chunk 携带 `version` + `commit` + `source_path` + `score`，`retrieval_trace` 完整可复现。向量相似度分数难以给出同等强度的可解释性
4. **现有问题用不到向量**：§5 的四个 P0 缺陷（`lower()` 重复、子串误匹配、预算饿死、结构化通道失效）全部是工程缺陷，**换成向量检索一个都解决不了**，反而会把"为什么这条没被选中"变成不可调试的黑盒
5. **混合检索的前提不成立**：通常 hybrid（BM25 + 向量）用于"词面不匹配但语义相近"的场景。本场景的查询是**编译器诊断 / API 符号名 / 失败日志**，与文档是**强词面重叠**关系 —— 正是 BM25 类方法的优势区间

---

## 8. 复现命令附录

```bash
cd /mnt/workspace/CannAgent/ascendc_multi_turn

# 数据量统计
for d in docs/api examples include impl; do
  echo "$d: $(find asc-devkit-9.1.0/$d -type f | wc -l) files"
done

# 单次 retrieve() 计时与 root 分布
# 注意：本模块的包内 logging.py 会遮蔽 stdlib logging，直接 import 包会失败，
# 需按文件路径加载 devkit_retrieval 并注入 .devkit 桩模块（见下方自包含片段）
python3 - <<'PYEOF'
import sys, types, importlib.util, time
from pathlib import Path
PKG = Path('/mnt/workspace/CannAgent/ascendc_multi_turn')
pkg = types.ModuleType("ascendc_multi_turn"); pkg.__path__ = [str(PKG)]
sys.modules["ascendc_multi_turn"] = pkg
stub = types.ModuleType("ascendc_multi_turn.devkit")
stub.ASC_DEVKIT_COMMIT, stub.ASC_DEVKIT_VERSION = "s", "9.1.0"
sys.modules["ascendc_multi_turn.devkit"] = stub
spec = importlib.util.spec_from_file_location(
    "ascendc_multi_turn.devkit_retrieval", PKG / "devkit_retrieval.py")
m = importlib.util.module_from_spec(spec)
sys.modules["ascendc_multi_turn.devkit_retrieval"] = m
spec.loader.exec_module(m)

r = m.DevkitRetriever(PKG / 'asc-devkit-9.1.0')
t0 = time.perf_counter()
b = r.retrieve(operator="add",
               symbols=["DataCopyPadExtParams","TQue","LocalTensor","EnQue","DeQue"],
               query_text="elementwise add fp16 tail unaligned DataCopyPad")
print(f"retrieve() = {(time.perf_counter()-t0)*1000:.0f} ms")
for t in b.retrieval_trace: print(" ", t)

# 生产配置 24000 下的 root 存活分布
from collections import Counter
out = m.render_evidence(b, max_chars=24000)
used, kept = len(out[:out.find('##')]), []
for e in b.evidence:
    sec = len(e.excerpt) + 160
    if used + sec > 24000: break
    used += sec; kept.append(e.source_kind)
print("检索选中:", dict(Counter(e.source_kind for e in b.evidence)))
print("实际渲染:", dict(Counter(kept)))
PYEOF

# 子串误匹配率（§5.2 数据来源）
python3 - <<'PYEOF'
import re
from pathlib import Path
root = Path('asc-devkit-9.1.0/docs/api')
md = list(root.rglob('*.md'))
for tok in ["add","sub","mul","div","exp","sin","cos","abs","mean","sum","sort","pad"]:
    files = [p.name for p in md if tok in p.name.lower()]
    real  = [f for f in files if re.search(rf"(?<![a-z0-9]){tok}(?![a-z0-9])", f.lower())]
    bad   = [f for f in files if f not in real]
    pct   = 100 * len(bad) / max(1, len(files))
    print(f"{tok:6s} 子串={len(files):3d} 真命中={len(real):3d} 误匹配={pct:5.1f}%  例: {bad[:2]}")
PYEOF

# 时间分解 + lower() 外提加速比（§5.1 数据来源）
python3 - <<'PYEOF'
import sys, types, importlib.util, time
from pathlib import Path
PKG = Path('/mnt/workspace/CannAgent/ascendc_multi_turn')
pkg = types.ModuleType("ascendc_multi_turn"); pkg.__path__ = [str(PKG)]
sys.modules["ascendc_multi_turn"] = pkg
stub = types.ModuleType("ascendc_multi_turn.devkit")
stub.ASC_DEVKIT_COMMIT, stub.ASC_DEVKIT_VERSION = "s", "9.1.0"
sys.modules["ascendc_multi_turn.devkit"] = stub
spec = importlib.util.spec_from_file_location(
    "ascendc_multi_turn.devkit_retrieval", PKG / "devkit_retrieval.py")
m = importlib.util.module_from_spec(spec)
sys.modules["ascendc_multi_turn.devkit_retrieval"] = m
spec.loader.exec_module(m)

root, TEXT = PKG / 'asc-devkit-9.1.0', m._TEXT_SUFFIXES
tokens = m.DevkitRetriever._tokens(
    ["DataCopyPadExtParams","TQue","LocalTensor","EnQue","DeQue"],
    "elementwise add fp16 tail unaligned DataCopyPad")

files, t_io = [], 0.0
for _, rel in m.DevkitRetriever.ROOTS:
    t0 = time.perf_counter()
    for p in (root / rel).rglob("*"):
        if p.is_file() and p.suffix.lower() in TEXT:
            try: files.append((p, p.read_text(encoding="utf-8", errors="replace")))
            except OSError: pass
    t_io += time.perf_counter() - t0
print(f"纯 I/O = {t_io*1000:.0f} ms ({len(files)} 文件)")

t0 = time.perf_counter()
for p, t in files:
    sum(min(t.lower().count(tk.lower()), 5) * 3 for tk in tokens)      # 原实现
t_old = time.perf_counter() - t0

low = [tk.lower() for tk in tokens]
t0 = time.perf_counter()
for p, t in files:
    lo = t.lower()                                                     # 新实现
    sum(min(lo.count(tk), 5) * 3 for tk in low)
t_new = time.perf_counter() - t0
print(f"打分循环 原 = {t_old*1000:.0f} ms | 优化后 = {t_new*1000:.0f} ms | 加速 {t_old/t_new:.1f}x")
PYEOF

# §6.4 / §5.9：L5 重复度 + 检索是否随失败演化
cd /mnt/workspace/CannAgent/outputs/e2e_add_diagnosefix_20260918/.llm_state
python3 - <<'PYEOF'
import json, hashlib, collections, pathlib
rows = []
for rnd in range(1, 10):
    p = f"round_{rnd:02d}"
    rec = {}
    for aud in ("planner", "generator"):
        f = pathlib.Path(f"{p}/devkit_evidence_{aud}.json")
        if not f.is_file(): rec[aud] = None; continue
        d = json.load(open(f, encoding="utf-8"))
        rec[aud] = hashlib.sha256(
            json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    rows.append((p, rec))
c = collections.Counter(v for _, r in rows for v in r.values() if v)
print(f"L5 总调用 = {sum(c.values())} ; 独立 bundle = {len(c)} ; 可省 = {sum(c.values())-len(c)}")
for h, n in c.most_common(): print(f"  {h} 出现 {n} 次")

print("\n失败证据 vs 检索输出 是否随轮次变化：")
for rnd in range(2, 9):
    r = pathlib.Path(f"round_{rnd:02d}")
    def h(n):
        f = r / n
        return hashlib.sha256(f.read_bytes()).hexdigest()[:12] if f.is_file() else '—'
    print(f"  round_{rnd:02d}  build.log={h('build.log'):14s} references.md={h('references.md')}")
PYEOF

# 真实产物的 root 存活分布（已验证 3 轮逐字节一致）
cd /mnt/workspace/CannAgent
python3 - <<'PYEOF'
import re
from collections import Counter
for rnd in ("round_02","round_05","round_07"):
    t=open(f"outputs/e2e_add_diagnosefix_20260918/.llm_state/{rnd}/references.md",encoding="utf-8").read()
    m=re.search(r"# Asc DevKit 官方证据", t)
    tail=t[m.start():]
    print(rnd, dict(Counter(re.findall(r"## (\w+):", tail))))
PYEOF
```

---

## 9. 附：本次分析未覆盖的部分

- `diagnostics.py`（798 行）：失败分类与符号域判定，作为 L3 的上游输入，未展开
- `evaluator.py`（646 行）：验证前沿与评分，与检索链路解耦
- `benchmark.py`（498 行）：性能测量，独立于检索
- `repair_policy.py` / `repair_state.py`：修复策略状态机，未展开
- `runtime_knowledge.py` 的 `priority()` 加权表（`:144-174`）内部系数（如 `+80` 惩罚注释行）的合理性未做实证校准
- `prompts.py`（868 行）中 L8 之外的 prompt 组装逻辑：`_create_plan` / `_prepare_candidate` 如何把 `knowledge_text` 与源码、用例、失败日志拼成最终 prompt，以及四个 call_type 的 system prompt 内容差异，未展开
- LLM 输出侧：`parse_plan` 的解析与校验、`generator_retry` 的重试判定条件，未展开
- 本次补充分析（L1–L8 执行覆盖 / LLM 依赖 / I/O 契约 / `knowledge_router` 残骸）基于 `outputs/e2e_add_diagnosefix_20260918` 单个 run 的 9 轮 17 次产物，未跨 run 交叉验证
