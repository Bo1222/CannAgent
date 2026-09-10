# AscendC 直接 LLM 多轮生成

`ascendc_multi_turn` 是独立于 Claude Code/Claude Agent 的 AscendC 生成路径。当前支持 DeepSeek 和 OpenAI，两者均通过 `POST /chat/completions` 调用；多轮状态、源码修改、编译、正确性反馈、性能比较、best 版本选择和 token 统计均由本仓库的 Python 代码负责。

原有 `agents/ascend-kernel-developer.md` 和 Claude Code 配置保留为兼容入口，但本模块不会加载或调用它们。
直接流程只把 PyTorch reference、cases、AscendC/CANN 知识和真实评测结果作为规划与生成输入，不创建 `model_new_tilelang.py`，也不读取 TileLang-to-AscendC 转译指南。

## 运行

首次使用先安装配置依赖并创建本地配置：

```bash
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，选择 `LLM_PROVIDER=deepseek` 或 `openai`，并填写对应的
`DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`。命令行参数仍可覆盖 Provider、模型和地址。
`.env` 已被 Git 忽略，不应提交真实密钥。

首次运行或知识源更新后，先编译并发布当前 CANN 版本的结构化知识：

```bash
python -m ascendc_multi_turn.knowledge.build --version 8.5.0
```

普通运行默认使用 `structured` 模式并自动读取 `knowledge_store/cann/8.5.0/current.json`。
构建步骤、数据结构和运行时作用见 [structured-ascendc-knowledge.md](structured-ascendc-knowledge.md)。

DeepSeek 示例：

```bash
python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/1_GELU.py \
  --output-dir outputs/1_GELU \
  --provider deepseek \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
  --max-bootstrap-rounds 8 \
  --max-rounds 5 \
  --soc-version Ascend910B3 \
  --device 0
```

OpenAI 示例只需改为 `--provider openai`，默认读取 `OPENAI_BASE_URL`、
`OPENAI_MODEL` 和 `OPENAI_API_KEY`。两种 Provider 都使用 Chat Completions；
密钥只从 `.env`/进程环境读取，不写入轨迹文件。

没有 NPU 时可验证编排流程：

```bash
python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/1_GELU.py \
  --output-dir /tmp/ascendc-flow-check \
  --max-rounds 3 \
  --mock
```

`--mock` 不调用模型 API，也不编译代码，仅验证多轮状态机、文件协议、best 选择与日志。它不能证明生成算子的正确性。
如果尚未安装结构化知识，可在仅测试文档检索路径时显式增加
`--knowledge-mode document`。

默认情况下，程序把带时间戳的阶段进度写到 `stderr`，包括当前轮次、结构化知识
检索、生成 LLM、API 约束检查、静态检查、编译、正确性和性能阶段。耗时阶段每 15 秒输出一次
心跳，结束时显示 KEEP/DISCARD 和失败摘要。最终机器可读 JSON 单独写到
`stdout`，因此仍可安全重定向或交给 `jq`。使用 `--quiet` 可关闭所有进度输出。

`--max-bootstrap-rounds`（默认 8）限制从零生成到获得“编译、正确性、benchmark 全部
有效”的基线候选数；`--max-rounds` 只计算基线后的性能评估。可选的
`--max-total-rounds` 对两阶段已经完成的候选评测设置统一总上限，适合固定轮数的诊断实验。
这些参数在续跑时都是总上限，不是新增轮数。API、JSON 格式或环境失败不消耗预算；若候选已生成而 EVAL 暂停，
`--resume` 会直接重评该 checkpoint，不重复调用模型。bootstrap 用尽时状态为
`blocked` 且不写 DONE；总上限用尽但仍无 baseline 时同样为 `blocked`。摘要中的
`stop_reason` 区分 `max_total_rounds`、`max_bootstrap_rounds`、`max_rounds` 和暂停原因。

## LLM token 与思考模式

Triton AutoResearch 不在仓库中向 `claude --print` 设置输出 token 数量；它控制的是
真实评估轮、单任务 wall-clock 和瞬态 CLI 恢复。直接 Chat Completions 路径不能依赖
agent runtime 自动管理输出，因此使用分调用预算：

| 调用 | 默认 max tokens | DeepSeek thinking | effort |
|---|---:|---|---|
| document-mode router（仅 `document`） | 4096 | disabled | — |
| generator | 65536 | enabled | high |
| PLAN / DIAGNOSE | 8192 | disabled | low（仅在启用 thinking 时发送） |

可通过 `--router-max-tokens`、`--generator-max-tokens`、`--planner-max-tokens` 及对应
thinking/effort 参数覆盖。初始规划返回一个完整 baseline 蓝图，后续 PLAN/DIAGNOSE
返回 3–5 个结构化实验项；它们都不应继承代码生成的大输出预算。此独立默认值来自
GELU 真实短程运行中一次 PLAN 消耗 37928 token 的反馈。
`--repair-*` 保留一个兼容周期但已弃用且不再触发专用调用。旧 `--max-tokens` 和
`ASCENDC_LLM_MAX_TOKENS` 仍兼容，并在启动时打印警告。
DeepSeek thinking 请求不发送无效的 temperature，并启用 JSON Output。API 初次失败后
最多额外重试 `ASCENDC_LLM_TRANSIENT_RETRIES=3` 次。

## 每轮流程

1. 读取参考 PyTorch 模型、同名 JSON 用例、当前 best 源码及上轮评测反馈。
2. 检测运行时 CANN 版本，选择完全匹配或同主版本的本地知识版本。
3. 默认读取当前 CANN 知识版本已经发布的 `current.json`，由代码符号、SoC、阶段、当前计划和结构化失败构造 `KnowledgeContext`。
4. `StructuredKnowledgeRouter` 先精确匹配 API，再按 applicability 选择事实；FailureCard 和 PatternCard 只提供诊断与实现模式，不得用相似名称替代 API 精确匹配。
5. 把选择结果保存为 `KnowledgeBundle` 和 `retrieval_trace.json`，并在 24000 字符上限内投影给当前 Agent。仅当显式使用 `--knowledge-mode document` 时，才调用文档路由模型选择原始 Markdown。
6. 生成首个候选前先创建一个直接 AscendC baseline 计划，覆盖算法、block/tiling、内存与搬运、dtype/shape/尾块、Host ABI 和文件布局；禁止 TileLang、其他 DSL、中间实现和源码转换。
7. 首个候选评估后，根据真实结果创建 3–5 个计划项；每个后续 EDIT 只执行一个可检验假设。
8. 限制模型只能修改 `model_new_ascendc.py` 和 `kernel/` 下的源码，阻止绝对路径、目录穿越和 build 文件写入。
9. 先校验 Host ABI：pybind 只声明/调用 `extern "C" *_do`，wrapper 在 kernel 源中用 `kernel<<<blockDim, nullptr, stream>>>` 启动；禁止 `acl_rt_launch.h`、`ACLRT_LAUNCH_KERNEL` 和绝对 include。之后再执行编译、正确性和性能测试。
10. 基线前失败记 FAIL；基线后更快 KEEP、有效但不快 DISCARD、无效 FAIL。连续 3 次 FAIL 进入 DIAGNOSE，计划耗尽进入 REPLAN。
11. 只有正确且具有有效 score 的候选才能成为 baseline/best；`--resume` 校验知识版本并迁移 v2 轨迹，同时从旧知识状态中清除 DSL 补充项。

## 产物与 token 统计

输出算子目录中的 `.llm_state/` 保存：

- `trajectory.json`：每轮 KEEP/DISCARD、编译/正确性/性能结果；
- `calls.jsonl`：每次请求的 requested/served model、thinking、effort、token cap、reasoning token、finish reason、重试编号与 usage；原始 reasoning 正文不写入聚合日志；
- `invocations.jsonl`：每次首次运行/续跑的非密钥有效配置；
- `orchestration_attempts.jsonl`：未进入候选评测、因而不消耗 round 的失败 checkpoint；
- `run_state.json`：当前 pending evaluation round 与稳定 attempt id；
- `token_usage.json`：总 token以及按 `knowledge_router`/`planner`/`diagnose`/`generator` 分类的汇总；
- `round_NN/{knowledge_bundle.json,retrieval_trace.json}`：默认结构化检索的知识结果和选择依据；
- `knowledge_state.json`、`knowledge_prompt.txt`、`knowledge_response.txt`：仅用于显式的 `document` 模式；
- `round_NN/{selected_knowledge.json,references.md,runtime_header_facts.json}`：`domain=ascendc` 的选择结果、实际注入内容和运行时头文件证据；
- `round_NN/{prompt.txt,response.txt,candidate.json}`：可复现的生成输入、输出和源码快照；
- `round_NN/{static_validation.log,source_validation.log,build.log,correctness.log,performance.log}`：各个已执行评测阶段的完整输出；
- `plan.json`、`plan.md` 和 `planner_vNN/`/`diagnose_vNN/`：当前计划、可读进度和规划证据；
- `baseline.json`、`best.json` 与 `summary.json`：sticky baseline、最佳实现及双预算汇总。

token 数以 API 返回的 `usage` 为准，并单独汇总 API 提供的 reasoning token。即使
最终 `content` 为空，也会先记录 served model、usage、reasoning 是否存在和
`finish_reason`，再分类为 `llm_output_exhausted` 或 `llm_empty_content`。若某个兼容
服务不返回 usage，相应字段为空，不使用字符数伪造真实账单 token。

失败时，终端和 `summary.json.failure` 都会给出 `round`、`stage`、稳定的
`code`、精简错误 `excerpt` 和 `details_path`。例如编译失败会标记为
`stage=ascendc_build` / `code=ascendc_build_failed`，完整编译输出保存在该轮的
`build.log`。编译错误摘要从完整日志提取，不再只看末尾输出。`trajectory.json`
每个候选只有一次 `evaluation`；失败由下一 plan item 的独立 EDIT/EVAL 处理。
