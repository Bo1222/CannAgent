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

## 3. 多轮执行流程

```text
BOOTSTRAP → PLAN → GENERATE → SOURCE VALIDATION
          → API CONSTRAINT VALIDATION → COMPILE
          → CORRECTNESS → PERFORMANCE → SETTLE
          → DIAGNOSE → OPTIMIZATION
```

- `BOOTSTRAP`：在没有正确基线时生成完整可执行候选。
- `PLAN`：依据任务、当前代码、历史结果和结构化知识确定本轮修改目标。
- `GENERATE`：直接生成 AscendC Kernel、Host 绑定和 Python 调用代码。
- `SOURCE VALIDATION`：检查文件布局、Kernel launch、扩展导入和 forward 调用等结构问题。
- `API CONSTRAINT VALIDATION`：依据适用版本和调用上下文检查 API 参数约束。
- `COMPILE/CORRECTNESS/PERFORMANCE`：在真实工具链和 NPU 上编译、校验正确性和测量性能。
- `SETTLE`：更新 source、compile、runtime、correctness 和 performance frontier。
- `DIAGNOSE/OPTIMIZATION`：把失败转换为结构化证据，提出局部修复并重新评估。

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

## 6. 轨迹和排障文件

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

## 7. 更新知识后的检查

```bash
python -m unittest discover \
  -s ascendc_multi_turn/structured_knowledge/tests \
  -p 'test_*.py' -v

python -m unittest discover -s tests -p 'test_*.py' -v
```

进一步的数据流和安装说明见 `docs/structured-ascendc-knowledge.md`，总体架构见
`repo_ascendc/doc/architecture.md`。
