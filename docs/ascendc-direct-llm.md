# AscendC 直接 LLM 多轮生成

`ascendc_multi_turn` 是独立于 Claude Code/Claude Agent 的 AscendC 生成路径。它只要求模型服务提供 OpenAI 兼容的 `POST /chat/completions` 接口；多轮状态、源码修改、编译、正确性反馈、性能比较、best 版本选择和 token 统计均由本仓库的 Python 代码负责。

原有 `agents/ascend-kernel-developer.md` 和 Claude Code 配置保留为兼容入口，但本模块不会加载或调用它们。

## 运行

DeepSeek 示例：

```bash
export DEEPSEEK_API_KEY=你的密钥
python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/1_GELU.py \
  --output-dir outputs/1_GELU \
  --model deepseek-chat \
  --base-url https://api.deepseek.com \
  --api-key-env DEEPSEEK_API_KEY \
  --max-rounds 5 \
  --soc-version Ascend910B3 \
  --device 0
```

GPT 或其他 OpenAI 兼容服务只需替换 `--model`、`--base-url` 和 `--api-key-env`。密钥仅从环境变量读取，不写入轨迹文件。

没有 NPU 时可验证编排流程：

```bash
python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/1_GELU.py \
  --output-dir /tmp/ascendc-flow-check \
  --max-rounds 3 \
  --mock
```

`--mock` 不调用模型 API，也不编译代码，仅验证多轮状态机、文件协议、best 选择与日志。它不能证明生成算子的正确性。

## 每轮流程

1. 读取参考 PyTorch 模型、同名 JSON 用例、当前 best 源码及上轮评测反馈。
2. 调用一次 LLM，要求返回结构化的完整文件内容或文件增量。
3. 限制模型只能修改 `model_new_ascendc.py` 和 `kernel/` 下的源码，阻止绝对路径、目录穿越和 build 文件写入。
4. 依次执行静态检查、AscendC 编译、正确性验证和性能测试。
5. 首个正确候选成为 best；之后只有性能分数严格提升才 KEEP，否则回滚到 best。
6. 将本轮结果反馈给下一轮。运行中断后使用相同参数并加 `--resume` 继续。

## 产物与 token 统计

输出算子目录中的 `.llm_state/` 保存：

- `trajectory.json`：每轮 KEEP/DISCARD、编译/正确性/性能结果；
- `calls.jsonl`：每次调用的模型、耗时和服务端返回的 usage；
- `token_usage.json`：整个算子的 prompt/completion/total token 汇总；
- `round_NN/{prompt.txt,response.txt,candidate.json}`：可复现的逐轮输入、输出和源码快照；
- `best.json` 与 `summary.json`：最终最佳实现及汇总。

token 数以 API 返回的 `usage` 为准。若某个兼容服务不返回 usage，相应字段为空，不使用字符数伪造真实账单 token。
