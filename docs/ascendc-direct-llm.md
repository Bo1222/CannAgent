# AscendC 直接 LLM 多轮生成

`ascendc_multi_turn` 是独立于 Claude Code/Claude Agent 的 AscendC 生成路径。当前支持 DeepSeek 和 OpenAI，两者均通过 `POST /chat/completions` 调用；多轮状态、源码修改、编译、正确性反馈、性能比较、best 版本选择和 token 统计均由本仓库的 Python 代码负责。

原有 `agents/ascend-kernel-developer.md` 和 Claude Code 配置保留为兼容入口，但本模块不会加载或调用它们。

## 运行

首次使用先安装配置依赖并创建本地配置：

```bash
pip install -r requirements.txt
cp .env.example .env
```

编辑 `.env`，选择 `LLM_PROVIDER=deepseek` 或 `openai`，并填写对应的
`DEEPSEEK_API_KEY` 或 `OPENAI_API_KEY`。命令行参数仍可覆盖 Provider、模型和地址。
`.env` 已被 Git 忽略，不应提交真实密钥。

DeepSeek 示例：

```bash
python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/1_GELU.py \
  --output-dir outputs/1_GELU \
  --provider deepseek \
  --model deepseek-v4-flash \
  --base-url https://api.deepseek.com \
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

默认情况下，程序把带时间戳的阶段进度写到 `stderr`，包括当前轮次、知识路由
LLM、生成 LLM、静态检查、编译、正确性和性能阶段。耗时阶段每 15 秒输出一次
心跳，结束时显示 KEEP/DISCARD 和失败摘要。最终机器可读 JSON 单独写到
`stdout`，因此仍可安全重定向或交给 `jq`。使用 `--quiet` 可关闭所有进度输出。

续跑时，`--max-rounds` 表示总轮数上限，不是新增轮数。例如已有 7 轮时，
`--resume --max-rounds 8` 只执行第 8 轮，并显示
`evaluated=7, target=8, remaining=1`。这里的轮数只统计已经进入静态检查、编译、
正确性或性能评测的候选。知识路由、API、JSON 格式或本地环境失败不会消耗评估轮，
而是保留当前 checkpoint 供下次 `--resume` 使用。旧轨迹会保留原始 attempt 编号，
并从历史记录推导真实评估轮数。

## LLM token 与思考模式

Triton AutoResearch 不在仓库中向 `claude --print` 设置输出 token 数量；它控制的是
真实评估轮、单任务 wall-clock 和瞬态 CLI 恢复。直接 Chat Completions 路径不能依赖
agent runtime 自动管理输出，因此使用分调用预算：

| 调用 | 默认 max tokens | DeepSeek thinking | effort |
|---|---:|---|---|
| knowledge router | 4096 | disabled | — |
| generator | 65536 | enabled | high |
| compile repair | 65536 | enabled | max |

可通过 `--router-max-tokens`、`--generator-max-tokens`、`--repair-max-tokens`、
对应的 `--*-thinking` / `--*-reasoning-effort` 覆盖。旧 `--max-tokens` 和
`ASCENDC_LLM_MAX_TOKENS` 仍兼容，但会统一覆盖三类预算并在启动时打印警告。
DeepSeek thinking 请求不发送无效的 temperature，并启用 JSON Output。API 初次失败后
最多额外重试 `ASCENDC_LLM_TRANSIENT_RETRIES=3` 次。

## 每轮流程

1. 读取参考 PyTorch 模型、同名 JSON 用例、当前 best 源码及上轮评测反馈。
2. 检测运行时 CANN 版本，选择完全匹配或同主版本的本地知识版本。
3. 首轮调用知识路由器查看带唯一 `doc_id` 的完整 API 清单，建立最多 8 篇文档的任务级工作集；稳定的后续轮次直接复用，不再调用路由模型。
4. 若源码或编译诊断出现工作集之外的新 API，先进行确定性符号匹配；有歧义时只把最多 5 个候选交给知识路由器，并最多新增 2 篇。完整清单只在任务首次路由时读取一次，重复错误也不会重新注入全量索引。
5. 从工作集中选最多 5 篇当前相关 API 页，按照编译诊断出现顺序优先注入当前安装 CANN 的公共头文件声明、冲突提示和精简核心规则。`rg` 仅用于加速，缺失时自动使用 Python 扫描。默认知识上下文上限为 24000 字符。
6. 调用生成器返回完整文件内容或文件增量。
7. 限制模型只能修改 `model_new_ascendc.py` 和 `kernel/` 下的源码，阻止绝对路径、目录穿越和 build 文件写入。
8. 依次执行 wrapper 退化检查、保守的 AscendC C++ 源码检查、AscendC 编译、正确性验证和性能测试。源码检查会捕获明确错误的 `TPipe::EnQue` 一类 API 归属和重复文件级常量。若源码检查或真实编译器失败，同轮额外执行至多一次针对诊断的修复 LLM 调用并重新评测。
9. 只有正确性通过、性能阶段成功且具有有效 score 的候选才能成为 best；之后只有性能分数严格提升才 KEEP，否则回滚到 best。
10. 将本轮结果和知识状态反馈给下一轮；`--resume` 会校验知识版本一致性并兼容旧轨迹迁移。

## 产物与 token 统计

输出算子目录中的 `.llm_state/` 保存：

- `trajectory.json`：每轮 KEEP/DISCARD、编译/正确性/性能结果；
- `calls.jsonl`：每次请求的 requested/served model、thinking、effort、token cap、reasoning token、finish reason、重试编号与 usage；原始 reasoning 正文不写入聚合日志；
- `invocations.jsonl`：每次首次运行/续跑的非密钥有效配置；
- `orchestration_attempts.jsonl`：未进入候选评测、因而不消耗 round 的失败 checkpoint；
- `run_state.json`：当前 pending evaluation round 与稳定 attempt id；
- `token_usage.json`：总 token 以及按 `knowledge_router`/`generator`/`compile_repair` 分类的汇总；
- `knowledge_state.json`：任务级文档工作集、已知 API 符号、路由次数和失败指纹；
- `round_NN/{knowledge_prompt.txt,knowledge_response.txt}`：仅在该轮实际调用知识路由 LLM 时存在；复用轮写 `knowledge_reuse.json`；
- `round_NN/{selected_knowledge.json,references.md,runtime_header_facts.json}`：选择结果、实际注入内容和运行时头文件证据；
- `round_NN/{prompt.txt,response.txt,candidate.json}`：可复现的生成输入、输出和源码快照；
- `round_NN/{static_validation.log,source_validation.log,build.log,correctness.log,performance.log}`：各个已执行评测阶段的完整输出；
- `round_NN/repair_01/`：发生编译失败时的定向修复 prompt、response、知识、候选和第二次评测日志；
- `best.json` 与 `summary.json`：最终最佳实现及汇总。

token 数以 API 返回的 `usage` 为准，并单独汇总 API 提供的 reasoning token。即使
最终 `content` 为空，也会先记录 served model、usage、reasoning 是否存在和
`finish_reason`，再分类为 `llm_output_exhausted` 或 `llm_empty_content`。若某个兼容
服务不返回 usage，相应字段为空，不使用字符数伪造真实账单 token。

失败时，终端和 `summary.json.failure` 都会给出 `round`、`stage`、稳定的
`code`、精简错误 `excerpt` 和 `details_path`。例如编译失败会标记为
`stage=ascendc_build` / `code=ascendc_build_failed`，完整编译输出保存在该轮的
`build.log`。编译错误摘要从完整日志提取，不再只看末尾输出。`trajectory.json`
除最终 `evaluation` 外，还通过 `evaluation_attempts` 保留修复前后的两次结果。
