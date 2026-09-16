# CANNBot Skills Adapter for CannAgent

本文说明 `ascendc_multi_turn` 如何把内置 CANNBot AscendC 精选知识库作为领域知识模块接入
CannAgent。Adapter 只负责向 Planner 和 Generator 提供知识，不执行 CANNBot 工作流，也不替换
CannAgent Agent 框架或代码生成器。Evaluator 的渐进 profile 与 Runner 的单调验收属于独立的
本地可靠性机制，不由 CANNBot 文档驱动。

生产运行不再读取外部 CANNBot 仓库。包内 knowledge base 固定到 upstream commit
`a08c49706e35a400d7c77e0875bc7c72a3a79012`，保留 CANN Open Software License
Agreement 2.0、来源说明和逐文档 manifest。所有静态 CANNBot 内容默认仅为 Level 1
Documented；installed header、项目源码和本地 compile/correctness evidence 具有更高 authority。

## 1. Stage 路由表及与 Baseline 的详细对比

### 1.1 三种 prompt knowledge source 的根本区别

`--knowledge-source structured`（默认）保持原 baseline：CannAgent 使用现有 `StructuredKnowledgeRouter`。Runner 在规划前生成一份 structured `KnowledgeBundle`，将其渲染为知识上下文交给 Planner；同一份 prepared knowledge 随后可被 Generator 复用。

```text
Baseline

operator/cases/current source/previous result
                   |
                   v
       StructuredKnowledgeRouter
                   |
                   v
        rendered KnowledgeBundle
                   |
             +-----+-----+
             |           |
             v           v
          Planner     Generator
```

`--knowledge-source skills` 只向 prompt 注入 CANNBot Skill knowledge modules：Planner 和 Generator 分别推导 stage，official `KnowledgeBundle` 为空，但固定规则、真实 evaluator evidence 和 installed-header facts 保留。`--knowledge-source hybrid` 则先路由 structured official knowledge，再与对应 Skill knowledge modules 合并。两个 Skills 模式都不会默认共享 Planner 和 Generator 的最终知识文本。旧 `--skill-adapter` 等价于 `--knowledge-source hybrid`。

```text
Skills / Hybrid

                 workflow state + evidence
                            |
             +--------------+--------------+
             |                             |
             v                             v
     derive Planner stages         derive Generator stages
             |                             |
             v                             v
     optional structured facts    optional structured facts
          + Planner Skills              + Generator Skills
             |                             |
             v                             v
       planner_context               generator_context
             |                             |
             v                             v
          Planner                       Generator
```

Hybrid 在现有 structured bundle 之上增加按 stage、audience 和失败证据选择的 knowledge modules；Skills-only 则保留相同 selector、runtime facts 和本地适配 knowledge module，但给 prompt 的旧 structured `KnowledgeBundle` 明确为空，不发生 fallback。两者都不删除 Structured Knowledge A。

### 1.2 Workflow 状态到 stage 的路由

Stage 由 `ContextSelector.derive_stages()` 根据 audience、workflow phase、当前是否已有实现以及上一轮真实评测结果确定。

| 当前状态 | Structured baseline | Hybrid / Skills-only Planner stages | Hybrid / Skills-only Generator stages |
|---|---|---|---|
| 首次生成，没有当前实现 | 通用 `plan_generate` | `operator_analysis`, `kernel_design` | `kernel_design`, `code_generation` |
| 已有实现，没有明确失败 | 通用 `plan_generate` | `kernel_design` | `kernel_design` |
| pybind/include/ATen container/NPU tensor/stream Host 失败 | 通用 diagnose bundle | `host_integration_debug` | `host_integration_debug` |
| GM_ADDR/`__gm__`/descriptor/signature/cast boundary 失败 | 通用 diagnose bundle | `host_integration_debug` | `host_integration_debug` |
| 精确 AscendC overload/template/owner 编译失败 | 通用 diagnose bundle | `compile_debug` | `compile_debug` |
| 507001/507035、ACL、AICore、MTE、address/device exception | 通用 diagnose bundle | `runtime_debug` | `runtime_debug` |
| 已 compile/load/execute，但出现一般 correctness mismatch | 通用 diagnose bundle | `kernel_design` | `kernel_design` |
| 已有 dtype/cast/accumulation/rounding/epsilon/tolerance 直接证据 | 通用 diagnose bundle | `precision_debug`，或带明确提示的 secondary | 同 Planner |
| 已获得正确 baseline，进入优化 | 通用 optimization knowledge | `optimization` | `optimization` |

编译类 stage 包括：`bundle_validation`、`ascendc_source_validation`、`api_constraint_validation`、`static_validation`、`ascendc_build`、`compile` 和 `response_format`。

每轮恰有一个 Primary Skill。一般 numerical mismatch 不能自动等同于精度故障：Permute 等 indexing/data-movement 算子的广泛误差默认属于 `kernel_design`。只有明确出现 FP32 pass/FP16 fail、cast/accumulation dtype、rounding、epsilon、overflow/underflow 或 tolerance-boundary 证据时，才使用 `precision_debug`；Secondary 每轮最多一个，没有确定证据时为空。路由完全基于已有 failure stage、日志、当前源码、算子身份和 correctness 结果，不调用 LLM 分类器，也不产生未经校准的数值 confidence。

### 1.3 Stage 到 CANNBot Skill 的映射

| CannAgent stage | CANNBot Skill | 主要触发条件 | 提供给模型的目标知识 | 明确排除 |
|---|---|---|---|---|
| `operator_analysis` | `npu-arch` | 初始 Planner，stage 存在即触发 | SoC 身份、运行时查询要求、能力约束、兼容性分支 | Triton 指南、无关 SKU 表、硬编码 core/UB 参数、完整架构白皮书 |
| `kernel_design` | `ascendc-tiling-design` | stage 存在即触发，再按 operator family 选引用 | 多核切分、UB tiling、buffer 生命周期、tail、tiling fields 和分支覆盖 | 无关算子族、一次注入所有 tiling 方案、正确性前的性能调优 |
| `kernel_design` | `ascendc-api-best-practices` | stage 存在即触发，引用再按 API/关键词过滤 | 与当前设计相关的 API、对齐、buffer、精度和 pipeline 约束 | 完整 API 目录、源码中未涉及的 API、平台不支持的高级 API |
| `code_generation` | direct-launch + `cannagent-host-abi` | 仅首次完整候选 | CannAgent ABI 兼容的 wrapper、host、kernel、launch、descriptor 和数据流结构 | CMake、复制命令、运行脚本、测试工程、完整模板目录、PyTorch fallback |
| `host_integration_debug` | `cannagent-host-abi`（Primary route） | Host binding、NPU tensor/stream 或 CUDA contamination 的确定性 evidence | 一个最小 Host ABI 修复及 positive/negative facts | Kernel 数学、tiling、无关 API cards |
| `compile_debug` | `ascendc-api-best-practices` | 上一轮属于源码/API/编译失败 | 失败调用点对应的 signature、owner、overload、参数或约束修复依据 | 无关 API、整份 API 文档、精度和性能建议、与 installed headers 冲突的签名 |
| `host_integration_debug` | direct-launch ABI leaves | GM_ADDR、`__gm__`、descriptor、wrapper signature 或 cast evidence | 三方 ABI 与 Host/device memory boundary | 工程重写、冲突 stream 示例、未经验证强转 |
| `runtime_debug` | `ascendc-runtime-debug` | runtime/correctness failure 且包含 ACL、AICore、MTE、错误码、tiling、kernel lookup 等信号 | 错误分类、排序后的检查项、下一诊断动作及确认信号 | 完整错误码目录、无环境信号时的环境排查、性能建议 |
| `precision_debug` | `ascendc-precision-debug` | correctness failure 且存在明确 dtype/cast/accumulation/rounding/epsilon/tolerance 证据 | 一项可验证的精度假设、最小 instrumentation/fix 和预期信号 | 仅凭大幅 mismatch 触发、index/data movement 错误、runtime/performance 流程 |
| `optimization` | `ascendc-performance-best-practices` | 已存在正确 baseline | 一个有证据支持的优化假设、适用条件、最小改动和期望性能信号 | 无关算子族、没有证据的优化、放宽正确性、模板重写 |
| `optimization` | `ascendc-tiling-design` | 已正确且证据涉及 tile、UB、core、tail、balance 或 occupancy | 有测量依据的 tiling 变化及更新后的 UB/core 公式 | 初始 tiling 教程、无关算子族、同时引入多个 tiling 策略 |

Skill 触发分为两层：

1. Stage-level trigger 决定某个 Skill 是否参与当前轮次，例如 `compile_debug` 只接受编译类失败，`optimization` 必须已经有正确 baseline。
2. Reference-level trigger 再按 operator family、错误文本和 API 关键词决定读取 Skill 中的哪些文档片段。

因此“选中一个 Skill”不代表把该 Skill 的所有文档放入 prompt。Skill 可以被选中，但其中不匹配当前证据的 reference 仍会被拒绝并记录原因。

### 1.4 Planner 和 Generator 的知识差异

Planner 面向方案决策，默认只需要任务事实、项目硬约束、少量设计模式和与 stage 匹配的 Skill knowledge modules。非 debug Planner 不会接收完整 API facts。

Generator 面向代码实现，会接收 API facts、API cards、installed public-header facts、更多设计模式以及 `code_generation` Skill。进入 debug stage 后，API facts 按 `failure_symbols > source_symbols > planned_symbols` 排序。三类 evidence 分别来自当前 compiler/linker/runtime/correctness 失败、candidate source 和 active plan。某一集合为空不会把其余 API cards 清空；selector 仍保留小额 fallback，并在 metadata 中记录每项知识命中的 symbol 类型。

默认字符预算为：

| Audience | 总预算 | 重点配额 |
|---|---:|---|
| Planner | 12,000 字符 | hard constraints、failure guidance、精简 API、Skill knowledge modules |
| Generator | 20,000 字符 | hard constraints、failure guidance、API/header facts、Skill knowledge modules |

Structured/runtime 条目始终受 Runner 的确定性字符预算约束，`full_selected` 不再关闭预算。
Embedded Skill section 的宽度在 selection 阶段由 stage、failure ownership、operator family、profile
和 evidence 控制；超出预算时拒绝整个知识模块，一旦 selected 就完整、原子投递，不从正文中间截断，并记录实际渲染的
`knowledge_module_id#section_id`。

当前算子的最小语义契约通过 `standing_contract` 叠加到主路由。例如 Add 在进入 Host、编译或
runtime 调试后仍保留广播 offset/stride 约束；完整 Host/API/性能教程仍按当前证据选择，不会全部常驻。

## 2. 对 CANNBot Skill 内容所做的适配

### 2.1 当前实现不是动态执行完整 Skill

Adapter 不会在算子生成期间解释或执行 CANNBot 的完整 Skill workflow。精选 Skill 的
purpose、trigger condition、required context、expected artifact、knowledge type、leaf
references 和 exclusions 已人工提取并固化在 `skill_mapping.yaml` 中。

Runner 初始化时一次性读取、校验并缓存 mapping 中明确允许的 package-local reference，
后续轮次不再读取 Markdown 文件。以下内容不会被执行或复制：

- CANNBot workflow 和 agent orchestration；
- Skill 中的 shell/Python 脚本；
- CMake、编译和运行脚本；
- 测试工程和完整模板目录；
- 示例工程的模块命名和 registry 流程；
- 与目标 SoC、operator family 或当前失败无关的材料；
- PyTorch fallback 实现。

### 2.2 从 Skill 文档到 knowledge module

每个选中的 Skill 被转换为以下结构：

```text
SkillKnowledgeModule
  skill_id
  stage
  purpose
  trigger_reason
  expected_artifact
  knowledge_type[]
  provided_context[]
  exclusions[]
  excerpts[]
    knowledge_module_id
    section_id
    source                     # package-local provenance path
    headings[]
    complete selected text
```

这一转换把 CANNBot 中面向其自身工作流的说明，约束为 CannAgent 可以直接放入 Planner/Generator prompt 的领域知识：

- 架构 Skill 输出硬件约束，而不是启动硬件分析工作流；
- tiling Skill 输出设计规则和公式，而不是生成独立 tiling 工程；
- template Skill 输出最小代码结构提示，而不是复制模板项目；
- debug Skill 输出一个可验证的诊断动作，而不是运行调试工具链；
- performance Skill 输出单个证据驱动假设，而不是展开所有优化方案。

### 2.3 文档完整性与选择边界

Reference 必须同时满足：

- knowledge module ID 存在于 package-local manifest；
- manifest 路径位于对应 package knowledge base/local knowledge module root 内；
- 文件扩展名为 `.md`；
- 文件真实存在；
- 内容 SHA256 与 manifest 一致；
- mapping stage 位于该 knowledge module 的 allowed stages；
- operator family 与 mapping 匹配；
- 配置了 `when_any` 时，当前 evidence 必须命中至少一个关键词。

通过检查后，Adapter 只抽取 mapping 指定的完整 Markdown heading；没有 heading 时选择完整
knowledge module。选择按 `knowledge_module_id + section_id` 去重，不使用 per-reference `max_chars`。

manifest 缺失、越界、hash drift 或 mapping 非法会在 Runner 初始化时 fail fast；条件不匹配或
heading 不存在则在 selection trace 中记录确定性拒绝原因。Hybrid 保留已路由的 Structured A；
Skills-only 仍禁止 fallback 到 Structured A。

### 2.4 权威顺序

合并上下文时使用以下顺序：

1. 当前环境的 compiler、runtime 和 evaluator 证据；
2. 与当前 environment fingerprint 绑定并通过项目 toolchain probe 的 Level 3 Verified facts；
3. 当前安装 CANN/torch_npu headers 中的 Level 2 Installed facts；
4. 版本匹配的 official/Skill Level 1 Documented facts；
5. Level 0 Inferred facts；不得将其作为关键 ABI/API 调用的唯一依据。

CANNBot Skill 不能覆盖真实编译错误、installed header signature、CannAgent host ABI 或 evaluator 结果。

Runtime fingerprint 包含 CANN/toolkit root、PyTorch/torch_npu 版本、Host C++ 与 CANN `bisheng` compiler、include roots、SoC、项目 build contract 和 probe contract hash。旧 probe 的 fingerprint 不匹配时不会继续标记为 Verified，而是回落到 Installed/Documented。网络搜索或最新版文档不能直接升级为 Level 3。

## 3. 对应代码逻辑

### 3.1 启用和初始化

CLI 使用 `--knowledge-source structured|skills|hybrid` 选择 prompt knowledge source。默认是 `structured`；旧 `--skill-adapter` 保留为 `hybrid` 的兼容别名。`RunConfig.knowledge_source` 保存规范化后的有效模式，`uses_structured_prompt` 和 `uses_skills` 分别控制两条路由。

`knowledge_mode=document` 继续保留为旧原始 Markdown 路由，但不能与 `skills` 或 `hybrid` 组合。`skills` 模式仍使用结构化的 `SelectedStageContext`，只是不给 prompt 注入旧 `knowledge_store` 的 `KnowledgeBundle` 内容。

Runner 仅在 `knowledge_source` 为 `skills` 或 `hybrid` 时创建：

```text
SkillAdapter()  # validates and freezes embedded manifests/mapping/documents
ContextSelector(skill_adapter)
```

`--cannbot-skills-root`、`--skill-mapping` 或 `CANNBOT_SKILLS_ROOT` 一旦出现即返回迁移错误；
不存在外部 source 不可用后静默生成空 knowledge modules 的 fallback。

`structured` 模式下两者均为 `None`，原有知识和 prompt 路径保持不变。`skills` 模式为 official knowledge 构造带 source-policy trace 的空 bundle；`hybrid` 模式路由真实 official bundle，再与 Skill knowledge modules 合并。

### 3.2 每个 audience 的调用链

```text
Runner._knowledge_context()
  |
  +-- 拼接 reference model、cases、current source、previous result
  |
  +-- ContextSelector.request()
  |     +-- derive_stages()
  |     +-- detect_operator_families()
  |     +-- 提取 failure stage/code/evidence/symbols
  |
  +-- StructuredKnowledgeRouter.route()  # structured/hybrid；skills 跳过
  |     +-- official facts
  |     +-- API cards
  |     +-- failure/pattern cards
  |     +-- project contracts
  |
  +-- collect_runtime_facts()
  |     +-- Generator 或 debug stage 才执行
  |     +-- 从 installed public headers 提取相关事实
  |
  +-- ContextSelector.select()
        +-- SkillAdapter.select()
        |     +-- _condition_matches()
        |     +-- _reference_matches()
        |     +-- _extract_markdown_sections()
        |     +-- knowledge_module_id + section_id deduplication
        +-- 投影 official bundle
        +-- 合并 Skill knowledge modules
        +-- 建立 authority/exclusions/budget/trace
  |
  +-- render_stage_context()
  |
  +-- build_plan_prompt() 或 build_prompt()
```

启用 Adapter 后，Runner 会在生成前将 Planner 的 prepared knowledge 传给 Planner，但不会把它直接复用给 Generator。Generator 会用 `audience=generator` 再调用一次 `_knowledge_context()`，获得自己的 stage 和上下文。

### 3.3 主要实现文件

| 文件 | 作用 |
|---|---|
| `__main__.py` | 暴露 knowledge-source；旧外部路径参数只用于给出迁移错误 |
| `models.py` | 保存配置、限制 Adapter mode 并拒绝外部知识路径/环境变量 |
| `runner.py` | 初始化 Adapter，分别准备 Planner/Generator knowledge，并写审计文件 |
| `context_selector.py` | stage 推导、operator family 检测、official knowledge 投影和 audience 隔离 |
| `skill_adapter.py` | 双 manifest/hash 校验、immutable registry、内存条件选择、section 抽取和去重 |
| `skill_mapping.yaml` | schema v3 stage-to-knowledge module/section、trigger、input 和 exclusion 配置 |
| `prompts.py` | 按 authority 渲染 Structured facts，并完整投递 selected embedded sections |
| `knowledge_probe.py` | 生成环境 fingerprint，执行受控最小 compile probe，并写入可审计 manifest |
| `runtime_knowledge.py` | 从本机 installed headers 提取声明，按 matching probe 标记 Level 2/3 |
| `source_validation.py` | comment/literal-aware、brace/scope-aware 的源码检查及 Host negative checks |
| `diagnostics.py` | failure/source/planned symbol evidence 和 A-H 观察状态 |
| `logging.py` | stage-normalized token/latency/transition 指标 |

### 3.4 审计产物

每轮 Adapter 运行会保存：

| 文件 | 内容 |
|---|---|
| `knowledge_bundle_planner.json` | Planner official bundle；`skills` 模式为空并带 source-policy trace |
| `knowledge_bundle_generator.json` | Generator official bundle；`skills` 模式为空并带 source-policy trace |
| `retrieval_trace_planner.json` | Planner official knowledge 路由记录 |
| `retrieval_trace_generator.json` | Generator official knowledge 路由记录 |
| `planner_context.json` | Planner 最终结构化上下文 |
| `generator_context.json` | Generator 最终结构化上下文 |
| `skill_selection_planner.json` | Planner Skill/reference 的选择与拒绝原因 |
| `skill_selection_generator.json` | Generator Skill/reference 的选择与拒绝原因 |
| `planner_references.md` | 实际进入 Planner prompt 的渲染文本 |
| `references.md` | 实际进入 Generator prompt 的渲染文本 |
| `runtime_header_facts_*.json` | 从 installed headers 提取的事实和状态 |
| `calls.jsonl` 的 `prompt_metadata` | selected IDs、symbol 来源、provenance/confidence、渲染字符/token 估计 |
| `trajectory.json` 每轮 `stage_observation` | A source-valid 到 H optimized-correct 的 pass/fail/unknown/not-reached |
| `summary.json.stage_normalized_metrics` | transition rates、censored milestones、prompt 分量和 evaluator stage 时延 |

为兼容现有下游工具，Generator 仍会写通用的 `knowledge_bundle.json`、`retrieval_trace.json` 和 `selected_knowledge.json`。

## 4. Adapter 是否可以脱离 CLI 和在线生成过程

### 4.1 是否依赖 CLI

Adapter 不依赖 CLI。CLI 只是一个配置入口，Python 调用方可以直接使用：

```python
RunConfig(
    ...,
    knowledge_mode="structured",
    knowledge_source="hybrid",
)
```

因此 Adapter 可以由测试、服务入口或其他 orchestration 层启用，不要求通过 `python -m ascendc_multi_turn` 启动。

### 4.2 当前哪些工作发生在生成过程中

Runner 创建时读取并冻结 mapping、两个 manifests 和所有已校验 Markdown。每个
Planner/Generator knowledge preparation 只执行：

- stage 推导和 operator-family 关键词匹配；
- structured knowledge build 加载和路由（仅 `hybrid`；`skills` prompt 跳过）；
- embedded knowledge module/section 条件匹配与内存 heading 投影；
- installed-header facts 收集，并只消费预先生成、fingerprint 匹配的 probe manifest；
- context 合并和字符预算渲染；
- 审计 JSON/Markdown 写入。

这些操作全部在本机完成。Skill Adapter 本身不调用 LLM、不访问网络，也不运行 CANNBot 脚本。Compile probes 是显式的准备/验证步骤，不在每轮生成路径中执行；运行时只读取 `CANNAGENT_PROBE_MANIFEST` 指向的结果。因此 probe 编译耗时不会叠加到每个算子生成 round。此前持续数分钟的 `generator still running` 对应远程 Generator LLM 请求，不是 Adapter 解析 Skill。

当前没有针对 Adapter 各步骤的独立计时，因此不能在没有测量的情况下声称其开销只有固定的毫秒数。特别是 `hybrid` 的 structured knowledge 分 audience 路由、installed-header 扫描和审计文件写入也应在后续评测中单独计时；`skills` 模式可以用于隔离这部分 prompt-routing 开销。

### 4.3 已内置离线化的部分

静态 Skill 来源已经以只读 knowledge base + manifest 形式脱离算子生成命令：

```text
CANNBot upstream Markdown + curated allowlist
                    |
                    v
           explicit knowledge base refresh
                    |
                    v
        knowledge_base_manifest.json + skill_mapping.yaml
               - knowledge module/section IDs
               - family/term/domain routes
               - source hashes and provenance
               - stage/conflict allowlists
                    |
                    v
        Runner 初始化校验并加载一次，随后内存选择
```

适合离线处理的内容包括：

- CANNBot Markdown 解析；
- heading 查找和片段抽取；
- 单 reference 静态字符裁剪；
- purpose、artifact、knowledge type 和 exclusion 标准化；
- operator-family/reference 倒排索引；
- source hash、版本和 provenance 生成；
- mapping 与文件安全性预校验。

### 4.4 必须保留在运行时的部分

动态 stage routing 依赖当前轮真实状态，无法完全离线：

- 当前属于 bootstrap 还是 optimization；
- 是否已经获得正确 baseline；
- 上一轮 failure stage/code；
- runtime 和 precision 信号分类；
- 当前候选源码中出现的 API symbols；
- compiler/runtime/evaluator 的最新错误证据；
- Planner 与 Generator 的独立 audience 投影；
- installed headers 与 Skill 建议的冲突处理。

因此合理的目标不是完全移除运行时 Adapter，而是把运行时工作缩小为“加载预编译 catalog 后进行内存中的确定性过滤与合并”。

### 4.5 当前状态与未来建议

当前已实现：

- schema v3 `skill_mapping.yaml` knowledge module/section 映射；
- 本地确定性 stage/Skill/reference 路由；
- manifest/hash/stage 校验和跨轮 immutable document cache；
- selected section 原子完整投递与 knowledge module/section 去重；
- Planner/Generator 独立投影；
- 审计产物。

仍未实现的优化包括 `KnowledgeBuild` 的 run-level 共享缓存和 Adapter 分阶段耗时埋点；
这不影响 package knowledge base 的完整性或 fail-fast 行为。

上述未实现项属于后续性能优化方向，不应被当作现有能力。即使未来离线化，也必须通过 source hash 或 build ID 检测 CANNBot 文档和 mapping 是否发生变化，避免使用陈旧 catalog。

## 5. Adapter 在实际算子生成流程中的位置

```text
PyTorch reference + cases + SoC/CANN
                 |
                 v
       CannAgent workflow state
  BOOTSTRAP / DEBUG / OPTIMIZATION
                 |
                 v
     Planner stage/context routing
                 |
                 v
              Planner
                 |
                 v
     Generator stage/context routing
                 |
                 v
             Generator
                 |
                 v
       AscendC source validation
                 |
                 v
       API constraint validation
                 |
                 v
              Compile
                 |
                 v
            Correctness
                 |
                 v
            Performance
                 |
                 v
    真实证据驱动下一轮 stage
```

Adapter 只作用于图中的两个 context routing 节点。其后的 source validation、API validation、
compile、渐进 correctness、performance、frontier 和 settle 均由 CannAgent 本地逻辑执行；
知识模块不能绕过这些确定性检查。

以当前 GELU baseline 的错误为例：

```text
failure_stage = ascendc_source_validation
failure = missing_kernel_launch
                 |
                 v
Planner stages = [compile_debug]
Generator stages = [compile_debug]
                 |
                 v
ascendc-api-best-practices
current compile knowledge module selected by diagnostic ownership
                 |
                 v
只选择命中 launch/kernel/stream/blockDim 的结构片段
```

这使下一轮知识集中在 CannAgent ABI 和 direct kernel launch 结构，而不会同时注入 precision-debug 或 performance materials。该例说明的是路由行为；是否能实际解决错误仍必须以真实编译结果判断。

## 6. 历史三组命令参考（不属于当前 B/D）

本节保留早期 Structured/Skills-only/Hybrid 命令形状用于审计，不是下一轮执行计划。当前消融只比较 B Hybrid 与 D Skills-only，并采用下方第 7 节的两阶段门槛。

### 6.1 实验约束

三组必须使用相同的：

- 代码 commit；
- reference operator 和 cases；
- provider、model、temperature 和 token budgets；
- bootstrap、optimization 和 total round budgets；
- SoC、CANN、device 和 local evaluator；
- 干净且互不复用的输出目录。

当前被中断的 `outputs/ab_gelu_20260911/baseline` 只保留为诊断证据，不作为正式 A/B baseline。

### 6.2 Structured 命令

```bash
cd /mnt/workspace/CannAgent

stdbuf -oL -eL python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/1_GELU.py \
  --output-dir outputs/ab_gelu_clean/baseline \
  --provider deepseek \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --temperature 0.2 \
  --max-bootstrap-rounds 8 \
  --max-rounds 5 \
  --max-total-rounds 5 \
  --timeout 600 \
  --soc-version Ascend910B3 \
  --device 0 \
  --knowledge-mode structured \
  --knowledge-source structured \
  > outputs/ab_gelu_clean_baseline.live.log 2>&1
```

实时查看：

```bash
cd /mnt/workspace/CannAgent
tail -n +1 -F outputs/ab_gelu_clean_baseline.live.log
```

### 6.3 Skills-only 命令

```bash
cd /mnt/workspace/CannAgent

stdbuf -oL -eL python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/1_GELU.py \
  --output-dir outputs/ab_gelu_clean/skills \
  --provider deepseek \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --temperature 0.2 \
  --max-bootstrap-rounds 8 \
  --max-rounds 5 \
  --max-total-rounds 5 \
  --timeout 600 \
  --soc-version Ascend910B3 \
  --device 0 \
  --knowledge-source skills \
  > outputs/ab_gelu_clean_skills.live.log 2>&1
```

实时查看：

```bash
cd /mnt/workspace/CannAgent
tail -n +1 -F outputs/ab_gelu_clean_skills.live.log
```

### 6.4 Hybrid 命令

```bash
cd /mnt/workspace/CannAgent

stdbuf -oL -eL python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/1_GELU.py \
  --output-dir outputs/ab_gelu_clean/adapter \
  --provider deepseek \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --temperature 0.2 \
  --max-bootstrap-rounds 8 \
  --max-rounds 5 \
  --max-total-rounds 5 \
  --timeout 600 \
  --soc-version Ascend910B3 \
  --device 0 \
  --knowledge-mode structured \
  --knowledge-source hybrid \
  > outputs/ab_gelu_clean_adapter.live.log 2>&1
```

实时查看：

```bash
cd /mnt/workspace/CannAgent
tail -n +1 -F outputs/ab_gelu_clean_adapter.live.log
```

### 6.5 TODO

- [ ] 在新的干净目录完成 GELU baseline。
- [ ] 使用完全相同参数完成 GELU Skills-only arm。
- [ ] 使用完全相同参数完成 GELU Hybrid arm。
- [ ] 确认三组均使用 local evaluator，而不是 `--mock`。
- [ ] 确认真实执行到 AscendC compile 和 NPU correctness。
- [ ] 汇总三组 `calls.jsonl`、`trajectory.json`、`summary.json` 和 evaluator logs。
- [ ] 汇总 Adapter 的 `planner_context.json`、`generator_context.json` 和 `skill_selection_*.json`。
- [ ] 分离记录 Adapter routing、LLM、compile、correctness 和 performance 耗时。
- [ ] GELU smoke 有效后扩展到现有 operator case 集。
- [ ] 对非确定性模型每个算子至少进行三次 paired repeats。
- [ ] 不用配置预算之外的人工重试替换失败样本。

### 6.6 比较指标

| 指标 | 定义 |
|---|---|
| Compile success rate | 至少一个候选成功编译的算子数 / 总算子数 |
| Correctness rate | 至少一个候选通过全部正确性用例的算子数 / 总算子数 |
| Iterations to first compile | 到第一个编译成功候选为止的真实 candidate evaluations 数 |
| Iterations to first correctness | 到第一个正确候选为止的真实 candidate evaluations 数 |
| Token consumption | prompt、completion、total、cache hit/miss，并按 planner/diagnose/generator 分类 |
| End-to-end latency | 从 Agent 启动到完成或预算耗尽的总时间 |
| Kernel latency | 对相互比较且都正确的模式，使用最佳正确候选的 evaluator performance latency |
| Injected context size | Planner/Generator 每轮实际注入字符数和截断条目 |
| Skill relevance | 每轮 selected/rejected Skill/reference 数量及原因 |
| Repeated-context ratio | 相邻轮次重复注入内容占比 |
| Failure transition | 是否从 source failure 推进到 compile、correctness 或 optimization |
| Adapter overhead | stage derivation、structured routing、Skill selection、reference/catalog loading 和 context rendering 的耗时 |

性能比较只对相互比较且都已经正确的算子有效。Compile/correctness 是首要门槛；不能用更快但错误的 kernel latency 证明 Adapter 有效。

Adapter overhead 必须与以下耗时分开统计：

```text
total Agent latency
  = adapter/local knowledge time
  + LLM request time
  + source/API validation time
  + compile time
  + NPU correctness time
  + NPU performance time
```

这样可以判断卡顿究竟来自本地 Adapter、远程 LLM、编译器还是 NPU evaluator，避免仅根据终端长时间没有新输出做错误归因。

## 7. 当前 B/D 两阶段实验设计（本次实现不执行）

### 7.1 Phase 1 screening

仅运行 GELU、LayerNorm、Permute 各 `Hybrid × 1`、`Skills-only × 1`。两臂保持 commit、model/provider、temperature、round budgets、Planner/Generator、evaluator、device、CANN 和 case 完全一致；唯一差异是 `--knowledge-source hybrid|skills`。固定 `--generator-thinking disabled --generator-max-tokens 32768`。

```bash
stdbuf -oL -eL python -m ascendc_multi_turn \
  --op-file <GELU-or-LayerNorm-or-Permute.py> \
  --output-dir <paired-output>/<hybrid-or-skills> \
  --provider deepseek --model deepseek-v4-flash --temperature 0.2 \
  --max-bootstrap-rounds 5 --max-rounds 2 --timeout 600 \
  --soc-version Ascend910B3 --device 0 --cann-version auto \
  --knowledge-mode structured \
  --knowledge-source <hybrid-or-skills> \
  --generator-thinking disabled --generator-max-tokens 32768 \
  --planner-thinking disabled --planner-max-tokens 8192 \
  > <paired-output>/<mode>.live.log 2>&1

tail -n +1 -F <paired-output>/<mode>.live.log
```

若 Skills-only 在 correctness/frontier 上明显落后 Hybrid，Phase 1 后停止，Hybrid 保持默认；不会用额外单次重试替换失败样本。

### 7.2 Conditional Phase 2 replacement confirmation

只有 Skills-only 在 Phase 1 的 correctness 和 compile/load/execute frontier 达到或超过 Hybrid，才对进入替代决策的算子增加两个 matched-seed/order 的 paired repetitions，使每个候选 mode/operator 至少有 3 条 paired trajectories。若第一组追加 pair 已显示一致性退化，可以提前停止；不扩展算子集合，也不调整失败臂参数。

### 7.3 指标和替代门槛

报告必须分别列出 A source-valid、B compile、C load/binding、D NPU execute、E correctness、F benchmark、G optimization、H optimized correctness，以及相邻 transition rate。效率指标使用 LLM calls、tokens/call、tokens/time-to-first-stage、分段 LLM/validation/compile/NPU/benchmark latency 和 prompt 中 knowledge/source/evidence 的实际体积；未达到的阶段记为 `N/A/censored`，不能记 0。

验收顺序固定为：correctness > compile/load/execute reachability > 多 case correctness robustness > iterations-to-first-correct > tokens-to-first-correct > time-to-first-correct > performance。只有双方达到同一目标阶段时才能比较该阶段效率；没有 correct candidate 的模式不能凭 total token 或 latency 更低宣称更有效。

Skills-only replacement 的必要条件是 correctness 不低于 Hybrid，compile/load/execute 不显著退化，并且 Phase 2 重复结果支持稳定性。否则 Hybrid 保持默认，Skills-only 仅保留为实验开关。一次 paired run 不支持 replacement claim。

如果 trajectory 显示正确知识已经进入最终 prompt，但 Generator 仍构造错误 API 或算法，只记录 `knowledge delivered correctly, generation/reasoning failure remains` 并停止扩大本轮范围；不改变 thinking、Planner 架构、state machine、Agent 数量、RAG 或 retry 上限。
