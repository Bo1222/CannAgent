# AscendC 多轮算子生成与优化

`ascendc_multi_turn` 直接规划、生成、编译、验证和优化 AscendC Kernel，不经过
TileLang 或其他 DSL 转换。默认使用预先安装并校验的结构化 AscendC 知识。

## 1. 运行前准备

首次使用或官方文档、项目约束、知识 Schema、知识编译器发生变化时，安装或更新知识：

```bash
python -m ascendc_multi_turn.knowledge.build --version 8.5.0
```

这不是每次生成算子前都要执行的命令。相同输入会复用相同的
`knowledge_build_id`；只有输入或编译规则变化才生成新构建。

默认知识源为：

```text
skills/ascendc/ascendc-translator/references/AscendC_knowledge
```

默认安装目录为：

```text
knowledge_store
```

## 2. 完整执行命令

```bash
python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/1_GELU.py \
  --output-dir outputs/1_GELU \
  --provider deepseek \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --max-bootstrap-rounds 8 \
  --max-rounds 5 \
  --max-total-rounds 5 \
  --soc-version Ascend910B3 \
  --device 0
```

无需传入 `--knowledge-mode`，默认值是 `structured`。`--knowledge-mode document`
只用于直接检索原始 Markdown 的对照或诊断实验。

### 可插拔 prompt knowledge source

`--knowledge-source` 可选择原 structured knowledge、CANNBot Skills 或两者的
stage-aware 合并。它只控制 Planner/Generator 的 prompt knowledge；evaluator 的
API constraint validation、fixed rules、installed headers 和评测流程保持不变。

```bash
# 原 structured baseline（默认）
python -m ascendc_multi_turn ... --knowledge-source structured

# 只把 CANNBot Skills 注入 prompt
python -m ascendc_multi_turn ... --knowledge-source skills

# structured + CANNBot Skills
python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/1_GELU.py \
  --output-dir outputs/1_GELU_skill_adapter \
  --knowledge-source hybrid \
  --soc-version Ascend910B3
```

Skills 知识来自包内
`ascendc_multi_turn/knowledge_modules/cannbot_a08c4970_knowledge_base` 内置知识库。Runner 创建时一次性
校验 knowledge base/local manifest、逐文件 SHA256、stage allowlist 和 schema v3 mapping，随后每轮
只做内存选择。运行不需要、也不会读取 sibling `cannbot-skills` 仓库。

`--cannbot-skills-root`、`--skill-mapping` 和 `CANNBOT_SKILLS_ROOT` 已停止支持；一旦使用会
立即返回迁移错误。更新精选知识时应显式维护包内 knowledge base、manifest 和 mapping，而不是在
生产运行中替换来源。

Embedded Skill section 一旦被路由选中，在 `bounded` 和 `full-selected` 两种模式都会完整
投递；prompt 宽度由 stage、failure ownership、operator family、active profile 和 symbol
evidence 控制，不从 section 中间截断。`--knowledge-input-mode full-selected` 保留被选中
Structured 条目的完整字段，但不再取消 per-audience 总预算、runtime declaration 上限或
重复评测日志压缩。默认 `bounded` 使用紧凑字段投影。两种模式都不能取消
模型服务自身的 context-window 上限；服务拒绝完整 prompt 时不会自动降级为裁剪输入。

Debug route 区分 Host、Host/Kernel boundary 和 Kernel：pybind/include/container/stream 属于
Host；GM_ADDR、`__gm__`、descriptor transport、wrapper signature/cast 属于 boundary；精确
AscendC overload/template 属于 Kernel API；507001/507035/MTE 属于 runtime memory；广泛且
shape-dependent 的数值错误回到 kernel design。只有明确 dtype/cast/tolerance 证据才进入
precision debug。每轮 artifact 同时记录 `input_route` 和 `result_failure_route`。

旧 `--skill-adapter` 保留为 `--knowledge-source hybrid` 的兼容别名。没有新参数
时仍使用原 `structured` baseline。`skills` 只关闭旧 knowledge store 的 prompt
注入，不删除知识库；为了保持三组验证标准一致，local evaluator 仍可使用同一
structured build 执行 API constraint validation。`document` mode 不与 Skills 组合。

关键 Host API 可在正式 paired run 前只验证一次，并通过环境变量供两臂共同读取：

```bash
python -m ascendc_multi_turn.probe_knowledge \
  --runtime-version 8.5.2 \
  --soc-version Ascend910B3 \
  --output-dir /tmp/cannagent_knowledge_probes

export CANNAGENT_PROBE_MANIFEST=/tmp/cannagent_knowledge_probes/manifest.json
```

probe manifest 与 CANN/toolkit、PyTorch/torch_npu、Host C++/CANN `bisheng` compiler、
include roots、SoC、项目 build contract 和 probe contract 的 fingerprint 绑定。环境发生变化时旧结果不会继续作为 Level 3
Verified；普通生成过程只读取 manifest，不会每轮重新编译 probe。

## 3. 多轮执行流程

```text
BOOTSTRAP → PLAN → GENERATE → SOURCE VALIDATION
          → API CONSTRAINT VALIDATION → COMPILE
          → SMOKE → SHAPE → DTYPE → FULL CORRECTNESS
          → PERFORMANCE → SETTLE
          → DIAGNOSE → OPTIMIZATION
```

- `BOOTSTRAP`：在没有正确基线时生成完整可执行候选。
- `PLAN`：依据任务、当前代码、历史结果和结构化知识确定本轮修改目标。
- `GENERATE`：直接生成 AscendC Kernel、Host 绑定和 Python 调用代码。
- `SOURCE VALIDATION`：检查文件布局、Kernel launch、扩展导入和 forward 调用等结构问题。
- `API CONSTRAINT VALIDATION`：依据适用版本和调用上下文检查 API 参数约束。
- `COMPILE/CORRECTNESS/PERFORMANCE`：在真实工具链和 NPU 上编译，按固定 profile 逐级校验，
  仅在完整用例通过后测量性能；smoke、shape 或 dtype 通过都不能建立正确基线。
- `SETTLE`：更新 source、compile、runtime、correctness 和 performance frontier。
- `DIAGNOSE`：把失败转换为 `observations/ruled_out/unknowns`；证据充分时只提出一个局部修复，
  证据不足时返回零 item，并在 Generator 前以可恢复 `blocked` 停止且不消耗候选预算。
- `OPTIMIZATION`：仅在完整正确 baseline 和实际性能测量基础上提出一个局部优化并重新评估。

`--max-bootstrap-rounds` 限制建立正确基线的尝试数，`--max-rounds` 限制基线后的
性能优化轮数，`--max-total-rounds` 限制两者合计的候选评估次数。

## 4. 知识构建目录及每个文件的含义

```text
knowledge_store/cann/<version>/
├── current.json
└── builds/<knowledge_build_id>/
    ├── raw/
    ├── normalized/
    ├── facts/atomic_facts.json
    ├── cards/api_cards.json
    ├── cards/failure_cards.json
    ├── cards/pattern_cards.json
    ├── cards/project_contracts.json
    ├── indexes/symbols.json
    ├── build_manifest.json
    └── validation_report.json
```

### `current.json`

记录该 CANN 知识版本当前启用的 `knowledge_build_id`。普通运行读取这个指针；只有复现
旧实验时才需要显式传入 `--knowledge-build-id`。

### `builds/<knowledge_build_id>/`

一次通过校验的不可变知识构建。ID 由源文件哈希、CANN 版本、Schema 版本和知识编译器
版本共同决定，避免更新知识后静默改变旧实验的输入。

### `raw/`

构建时复制的原始官方 Markdown、项目指南和项目知识清单。用于来源审计、重新构建和
证据回查，不会整目录放入 Kernel Agent prompt。

### `normalized/`

每份 Markdown 对应一个 JSON，保留标题层级、段落、表格、代码候选、约束、示例、
来源 URL、文档 ID 和源文件哈希。它是确定性解析结果，不包含 LLM 总结。

### `facts/atomic_facts.json`

可校验的原子事实集合。每条事实包含：

- `subject/predicate/value`：事实主体、关系和值；
- `applicability`：API、参数结构以及可用的版本、SoC、overload、方向等上下文；
- `provenance`：文档、章节、原文证据和源文件哈希；
- `authority/schema_version`：权威等级和数据结构版本。

同名字段不会脱离 API 和调用上下文合并为全局规则。

### `cards/api_cards.json`

设计上用于按真实 API 名称组织导航卡片，连接 API、相关事实和来源文档。卡片本身不是
新的官方事实；枚举成员、常量和普通大写单词不应成为独立 API 卡片。

例如 `ARARARAR` 是 `ReducePattern` 的枚举成员，其中 A 表示 Normal 轴，R 表示
Reduce 轴；它不是 API。当前构建中的 `api_ARARARAR` 来自已确认的索引污染：索引器
把官方 Reduce 表格中压缩的枚举声明误识别成 API。本轮仅记录该问题，不修改现有结构化
输出或知识编译规则。

### `cards/failure_cards.json`

描述错误信号、所属子系统、候选原因类别、检查顺序和检查目标。它决定“应该检查什么”，
不硬编码某个 API 的具体修复值。

### `cards/pattern_cards.json`

保存可复用的 Kernel/Host 实现模式，例如 Vector、Cube、混合 C/V 和跨核同步模式，
用于 Planner 和 Generator 选择整体实现结构。

### `cards/project_contracts.json`

保存本仓库强制约束，例如 Host ABI、当前 NPU stream、允许修改的文件和 Python forward
必须调用扩展。其 authority 是 `PROJECT_CONTRACT`，不会冒充官方 API 事实。

### `indexes/symbols.json`

从候选 AscendC 符号到 `ApiCard` 的索引。运行时先完成精确符号匹配，再读取事实，避免
`DataCopy`、`DataCopyPad`、`DataCopyExt` 之间因名称相似而混用语义。当前索引仍存在
上述枚举成员污染，不能把 `symbols.json` 中的每个键都解释为官方 API。

### `build_manifest.json`

记录 CANN 知识版本、Schema/编译器版本、构建 ID、输入路径、每个源文件哈希和产物计数，
用于复现与完整性检查。

### `validation_report.json`

记录构建是否有效、证据缺失、事实冲突以及文档/事实/API 卡片数量。构建无效时不会更新
`current.json`。

## 5. 知识如何进入每一轮

每轮先从算子、阶段、CANN 版本、SoC、当前代码符号、API 调用、活动计划和上轮
`StructuredFailure` 创建 `KnowledgeContext`。`StructuredKnowledgeRouter` 随后：

1. 使用 `symbols.json` 精确解析当前 API；
2. 仅选择上下文匹配的 `AtomicFact`；
3. 根据错误信号选择 Failure Card；
4. 根据算子和阶段选择 Pattern Card；
5. 加入必须遵守的 Project Contract；
6. 生成有界的 `KnowledgeBundle`，分别投影给 Planner、Generator 和 Diagnose；
7. 由 `ApiConstraintValidator` 使用相同事实做编译前确定性检查。

默认结构化模式不调用一个额外的 LLM 去阅读和选择完整官方文档。

## 6. `output-dir` 中会生成什么

一次完整运行的输出目录大致如下。标记为“条件生成”的文件只会在对应阶段实际执行、
失败、重试或获得正确基线后出现。

```text
outputs/1_GELU/
├── model.py
├── 1_GELU.json
├── model_new_ascendc.py
├── kernel/
│   ├── pybind11.cpp
│   ├── <kernel-source>.cpp
│   ├── <optional-header>.h
│   └── build/                         # 编译期间临时存在
└── .llm_state/
    ├── trajectory.json
    ├── run_state.json
    ├── summary.json
    ├── token_usage.json
    ├── calls.jsonl
    ├── invocations.jsonl
    ├── orchestration_attempts.jsonl
    ├── environment_preflight.log
    ├── plan.json
    ├── plan.md
    ├── interface_contract.json
    ├── repair_state.json
    ├── baseline.json
    ├── best.json
    ├── DONE
    ├── frontiers/
    ├── incidents/
    ├── experience_candidates/
    ├── planner_v<NN>/ 或 diagnose_v<NN>/
    └── round_<NN>/
```

实际目录只包含已经运行到的阶段。例如首轮在 Source Validation 失败时，不会存在
`build.log`、`correctness.log` 或 `performance.json`。

### 顶层输入与最终源码

- `model.py`：从 `--op-file` 复制的原始 PyTorch 参考实现，用于定义算子语义和正确性基准，
  不是生成结果。
- `<operator>.json`，例如 `1_GELU.json`：与参考实现同名的测试 case 描述；源文件不存在时
  不生成。
- `model_new_ascendc.py`：最终保留候选的 Python wrapper。正常情况下应导入生成的 AscendC
  扩展，并在 `forward()` 中调用它。
- `kernel/pybind11.cpp`：Host 侧 PyBind/ACL 绑定、参数检查、stream 获取和 Kernel launch
  入口。
- `kernel/<kernel-source>.cpp`：包含 `__aicore__` Kernel 及相关 AscendC 计算代码。具体文件名
  由 Generator 决定。
- `kernel/<optional-header>.h|hpp`：可选的 tiling 数据结构、声明或公共辅助代码。
- `kernel/build/`：`utils/build_ascendc.py` 在评测期间产生的编译中间文件和扩展产物，内容
  依赖 CANN 工具链；它是不可编辑的临时生成目录，最终恢复最佳源码时通常会被清理。

运行结束时，顶层 `model_new_ascendc.py` 和 `kernel/` 会恢复为性能最优的正确候选；如果还
没有正确基线，则恢复到已到达的最深 frontier，不能把它们简单理解为“最后一轮输出”。

### `.llm_state/` 全局运行状态

- `trajectory.json`：整个多轮实验的主记录，包含每轮 decision、评测结果、失败指纹、
  StructuredFailure、计划项、候选路径、frontier、知识版本、A-H stage observation 和
  evaluator stage timing。
- `run_state.json`：可恢复检查点，记录当前 attempt、待执行 evaluation round 和
  `PLAN/EDIT/EVAL/DIAGNOSE` 等 pending phase。`--resume` 主要依据它继续。
- `summary.json`：本次命令结束时打印到终端的最终摘要副本，包括是否成功、停止原因、
  baseline/best round、accepted/latest attempt 身份与候选路径、被拒尝试、token 用量、最后失败及
  `stage_normalized_metrics`；证据不足阻塞时还包含 `blocked_detail`。未到达阶段的
  tokens/time-to-first 值为 `null`（censored），不是 0。
- `token_usage.json`：总 token 和 planner、generator、diagnose 等调用类型的分类统计。
- `calls.jsonl`：每次 LLM 调用一行，记录模型、耗时、finish reason、token、重试序号和
  prompt metadata；Adapter 模式还记录 selected/rejected knowledge ID、symbol evidence 类型、
  provenance/confidence 及 knowledge/source/evaluator-evidence 的渲染体积。
- `invocations.jsonl`：每次启动或 `--resume` 的配置记录，一次命令一行。
- `orchestration_attempts.jsonl`：不计入候选评估预算的编排失败，例如 LLM 传输、格式或本地
  基础设施失败；没有这类失败时可能不存在。
- `environment_preflight.log`：CANN、设备和运行环境预检查日志。
- `plan.json`：当前可执行计划及每个 item 的状态、决策、hypothesis 和 expected signal。
- `repair_state.json`：当前原子接受基线、开放/已清除错误、失败方案族和门/profile/case 进展。
- `interface_contract.json`：首个被接受的已编译候选导出的 pybind 模块、`*_do` 声明与定义、
  Kernel entry 契约；计划未显式允许接口变化时，候选必须保持一致。
- `plan.md`：`plan.json` 的人类可读版本。
- `baseline.json`：第一个通过编译、正确性和性能评测的完整源码 bundle；建立基线前不存在。
- `best.json`：当前性能最佳正确候选的完整源码 bundle。
- `DONE`：运行正常完成时创建的空标记；`paused` 或未建立基线的 `blocked` 状态不会创建。
- `knowledge_state.json`：失败指纹或 document 模式工作集等可恢复知识状态，按需要生成。
- `planning_error_<NN>.log`：Planner/Diagnose 输出无法调用或解析时的错误，条件生成。

### `frontiers/`、`incidents/` 和经验候选

- `frontiers/manifest.json`：记录当前最深 evaluation frontier 及对应 attempt。
- `frontiers/source.json`、`compile.json`、`runtime.json`、`correctness.json`、
  `performance.json`：到达相应阶段时保存的完整源码 bundle，用于失败后的稳定回滚；只有
  实际跨过该 frontier 才生成。
- `incidents/incident-*.json`：每个已评估候选的审计记录，包含修复前后 diff、活动假设、
  关联事实、评测结果和旧失败信号是否消失。
- `experience_candidates/experience-*.json`：只有修复后通过 correctness 且原失败信号消失
  才产生的已确认经验候选；它不会覆盖官方事实。

### `planner_v<NN>/` 与 `diagnose_v<NN>/`

- `prompt.txt`：该次 Planner 或 Diagnose 实际收到的完整 prompt。
- `response.txt`：模型原始文本响应；发生传输重试时还可能有 `response_retry_<NN>.txt`。
- `result.json`：从响应解析出的结构化计划。

`planner_v<NN>` 用于初始规划或常规重新规划，`diagnose_v<NN>` 用于连续失败触发的诊断规划。

### `round_<NN>/` 每轮候选目录

- `knowledge_bundle.json`：structured 模式为本轮选择的硬约束、API 语义、Cards、项目约束和
  evidence；这是分析“本轮实际拿到了什么知识”的首要文件。
- `retrieval_trace.json`：每项知识的命中、拒绝和选择原因，用于检查错检、漏检和相似 API
  污染。
- `selected_knowledge.json`：本轮最终知识选择的持久化表示；structured 模式下与 bundle
  内容接近，document 模式下记录选中的文档集合。
- `references.md`：经过长度限制、真正拼入 Planner/Generator 上下文的可读知识文本。
- `prompt.txt`：Generator 本轮实际收到的完整 prompt。
- `response.txt`：Generator 的原始输出正文。
- `retry_prompt.txt`、`response_retry_<NN>.txt`：输出被截断或调用重试时生成。
- `candidate.json`：解析并合并增量修改后的完整候选源码 bundle，也是进入 EVAL 前的恢复
  检查点；它最适合用来比较某轮是否整体替换代码。
- `bundle_validation.log`：候选缺少 wrapper、pybind 或 Kernel 源码时生成。
- `response_format.log`：Generator 输出不是合法文件 bundle 时生成。
- `source_validation.log`：AscendC 文件、include、Kernel launch、扩展连接等源码结构检查。
- `api_constraint_validation.log`：基于结构化事实执行的 API 参数约束检查。
- `resolved_api_calls.json`：API Constraint Validator 解析出的调用、overload、参数单位、对齐
  和来源事实；仅启用并执行该 Validator 后生成。
- `static_validation.log`：检查扩展导入、`forward()` Kernel 调用等项目静态契约。
- `build.log`：AscendC 编译命令的完整输出。
- `correctness.log`：NPU 正确性测试输出和设备错误信息。
- `correctness_<profile>.log`、`correctness_<profile>.json`：smoke、shape、dtype、full 各级的
  独立日志和逐 case 状态；首个失败级别之后不再继续执行。
- `interface_contract_validation.log`：候选未经计划授权修改已冻结接口时生成。
- `no_op_edit.log`：非 mock 运行中候选只改变注释/空白或完全没有源码变化时生成；该尝试不进入 evaluator。
- `performance.log`：性能脚本执行日志。
- `performance.json`：逐 case 性能与 speedup 的机器可读结果。
- `evaluation_incomplete.log`：Evaluator 返回信息不完整时的规范化错误。
- `evaluation_error.log`：Evaluator 自身抛出未处理异常时生成。
- `llm_error.log`：Knowledge Router 或 Generator 调用耗尽重试时生成。
- `knowledge_prompt.txt`、`knowledge_response.txt`、`knowledge_reuse.json`、
  `runtime_header_facts.json`：仅 `document` 模式使用的文档路由、复用和运行时头文件事实记录；
  默认 `structured` 模式通常不会生成。

使用 `skills` 或 `hybrid` source 后还会生成：

- `planner_context.json` / `generator_context.json`：分 audience 的阶段、Skill、
  引用、排除项和字符预算审计；
- `planner_references.md` / `references.md`：Planner 和 Generator 实际收到的不同知识投影；
- `skill_selection_planner.json` / `skill_selection_generator.json`：Skill 触发与引用选择；
- `knowledge_bundle_planner.json` / `knowledge_bundle_generator.json`：两次投影所基于的
  official knowledge 包；`skills` source 下为空并记录 source-policy trace；
- `runtime_header_facts_planner.json` / `runtime_header_facts_generator.json`：
  按需对当前精确符号查询的已安装 CANN/torch_npu 公共头文件事实、Level 0–3 confidence
  和 environment fingerprint。只有 fingerprint 匹配的 compile probe 才能标记 Level 3。

## 7. Hybrid / Skills-only 对照评测

当前 B/D 只比较 Hybrid 与 Skills-only。同一个 commit、模型、温度、CANN/SoC、设备、case
和轮数分别运行；固定 Generator thinking disabled、32768 max tokens：

```bash
# Skills-only
python -m ascendc_multi_turn \
  --op-file <case.py> \
  --output-dir <skills-output> \
  --knowledge-source skills \
  <共同参数>

# Hybrid
python -m ascendc_multi_turn \
  --op-file <case.py> \
  --output-dir <hybrid-output> \
  --knowledge-source hybrid \
  <共同参数>
```

Phase 1 对 GELU、LayerNorm、Permute 各做一次 pair，用于筛除 infrastructure failure 并定位
新 bottleneck，不支持稳定替代结论。只有 Skills-only 的 correctness/frontier 达到或超过
Hybrid，才进入 Phase 2：对进入决策的算子增加两个 paired repeats，使每个 mode/operator
至少有 3 条 paired trajectories；若追加 pair 已明显退化则提前停止。

优先比较 correctness，再比较 compile/load/execute reachability、多 case robustness、
iterations/tokens/time-to-first-correct，最后才是正确候选的 performance。双方没有达到相同
目标阶段时，不用 total token 或总时延宣称某模式更高效。详细路由与 replacement 门槛见
`skills_adapter.md`。

## 8. 轨迹和排障文件

每轮数据位于：

```text
<output-dir>/.llm_state/round_<NN>/
```

重点检查：

- `knowledge_bundle.json`：本轮实际提供给 Agent 的知识；
- `retrieval_trace.json`：每个候选知识为何命中、拒绝或降级；
- 生成源码和修改记录：判断是否发生整体替换；
- validation、compile、correctness 日志：确定失败阶段；
- trajectory/checkpoint：检查失败指纹、frontier 和下一待执行阶段。

如果连续多轮停留在相同 failure fingerprint，应检查检索结果是否包含对应 API/Host 合同、
Planner 是否提出新假设、Generator 是否落实局部修改，以及未推进 frontier 的候选是否正确回滚。

## 9. 更新知识后的检查

```bash
python -m unittest discover \
  -s ascendc_multi_turn/structured_knowledge/tests \
  -p 'test_*.py' -v

python -m unittest discover -s tests -p 'test_*.py' -v
```

进一步的数据流和安装说明见 `docs/structured-ascendc-knowledge.md`，总体架构见
`repo_ascendc/doc/architecture.md`。
