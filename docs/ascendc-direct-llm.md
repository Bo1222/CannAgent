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
  --model deepseek-chat \
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

## 每轮流程

1. 读取参考 PyTorch 模型、同名 JSON 用例、当前 best 源码及上轮评测反馈。
2. 检测运行时 CANN 版本，选择完全匹配或同主版本的本地知识版本。
3. 调用知识路由器，基于 `ascendc-translator` Skill 选择相关专项指南和具体 CANN API 页面。
4. 受控加载选中文档，再调用生成器返回完整文件内容或文件增量。
5. 限制模型只能修改 `model_new_ascendc.py` 和 `kernel/` 下的源码，阻止绝对路径、目录穿越和 build 文件写入。
6. 依次执行静态检查、AscendC 编译、正确性验证和性能测试。
7. 首个正确候选成为 best；之后只有性能分数严格提升才 KEEP，否则回滚到 best。
8. 将本轮结果和知识选择反馈给下一轮；`--resume` 会校验知识版本一致性。

## 产物与 token 统计

输出算子目录中的 `.llm_state/` 保存：

- `trajectory.json`：每轮 KEEP/DISCARD、编译/正确性/性能结果；
- `calls.jsonl`：知识路由和生成调用的模型、耗时、调用类型与 usage；
- `token_usage.json`：总 token 以及按 `knowledge_router`/`generator` 分类的汇总；
- `round_NN/{knowledge_prompt.txt,knowledge_response.txt,selected_knowledge.json,references.md}`：知识选择及实际注入内容；
- `round_NN/{prompt.txt,response.txt,candidate.json}`：可复现的生成输入、输出和源码快照；
- `best.json` 与 `summary.json`：最终最佳实现及汇总。

token 数以 API 返回的 `usage` 为准。若某个兼容服务不返回 usage，相应字段为空，不使用字符数伪造真实账单 token。
