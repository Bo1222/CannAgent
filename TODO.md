# AscendC Agent E2E-03 reasoning 诊断与复跑 TODO

状态：E2E-03A、E2E-03B **已解决**；reasoning 审计、长度耗尽处理和安全默认值已通过真实 API
诊断。关闭 Generator thinking 的端到端复跑已越过原截断点，但在 E2E-04 Diagnose 证据投影缺陷处
提前停止，全流程仍**部分解决**。保留
`outputs/e2e_add_e2e02f_20260918` 作为只读历史现场，**不得使用 `--resume`**。

## 本轮已完成

- 默认以 `0600` 独立保存每次响应的 `reasoning*.txt`；可用 `--reasoning-log metadata` 只保存摘要。
- Generator 默认关闭 thinking，新配置默认模型名为 `deepseek-flash`。
- 空正文的 `finish_reason=length` 不再按 transport 故障盲重试；部分正文只进行一次语义续写。
- E2E-03B 已修复：Generator 失败不再访问空 `knowledge_state`，可落盘 `summary.json` 并正确标记
  `failure_stage=llm_generator`。
- 分层 API 诊断成功复现 thinking/high 非收敛，完整 reasoning 已以 `0600` 保存；相同 Prompt 关闭
  thinking 后正常生成 11 文件 bundle。
- 新 E2E 的 7 次调用全部正常结束且无 reasoning；3 个候选进入 source validation 后失败，随后 Diagnose
  因证据投影不足在 3/10 候选处 BLOCKED。
- 9 项专项测试与 78 项全量测试通过。

## 已完成的门禁与验证

- Asc DevKit 健康检查通过：版本 `9.1.0`，commit
  `c785b5f76f23c9dc0ddb553174463d0497e59288`。
- 环境为 `torch_npu 2.7.1.post4`、1×Ascend910B3；DeepSeek 请求模型为
  `deepseek-v4-flash`，服务端记录模型为 `deepseek-flash`。
- `python -m unittest discover -s tests -v`：69 项全部通过。
- 历史 planner 响应重放通过：`planner_v01` 零变化；`planner_v02` 剔除
  `kernel/add_alpha.cpp`、`python/model_new.py`；`planner_v03` 剔除 `model_new.py`；bundle
  最终校验仍拒绝 `model_new.py`。
- Mock 一轮成功建立 Mock baseline，确认预算、状态机、工程布局与 DevKit provenance；临时目录已删除，
  Mock 不作为真实正确性或性能证据。

## 真实运行结果

- 输出目录：`outputs/e2e_add_e2e02f_20260918`。
- 初始 planner 成功，计划中的六个 `target_files` 均符合白名单。
- Generator 初次调用及三次配置内重试全部返回 `finish_reason=length`；每次 completion 均约
  65,536 tokens，且几乎全部为 reasoning，没有最终正文，四个 `response*.txt` 均为空。
- 重试耗尽后，`runner.py` 的 `except LLMCallFailure` 路径读取
  `knowledge_state.initialized`；当前架构中 `knowledge_state` 被固定为 `None`，因此触发
  `AttributeError: 'NoneType' object has no attribute 'initialized'`。
- 没有生成 `candidate.json`，没有执行 source validation、构建、正确性或 benchmark；
  `evaluations_completed` 实际为 0。异常退出前未生成 `summary.json`，`trajectory.json` 仍错误地保留
  `status=running`、`phase=EDIT`。

## 待处理问题

### E2E-03A：Generator reasoning 耗尽输出预算

状态：**已解决**。

- 真实诊断证明 `/models`、最小 thinking 与 non-thinking 均正常；复杂 Prompt 的 thinking/high 在整个
  token 预算内反复权衡广播实现而没有进入正文，相同 Prompt 关闭 thinking 后正常返回完整 bundle。
- 新运行统一使用 canonical `deepseek-flash` 并默认关闭 Generator thinking；历史空响应不计为候选失败。
- 原始 reasoning 位于
  `outputs/deepseek_diagnostic_e2e03a_20260918/real_prompt_thinking_high/reasoning.txt`，权限 `0600`。

### E2E-03B：LLM 失败处理访问空 knowledge_state

状态：**已解决**。

- `_prepare_candidate()` 的 `LLMCallFailure` 已固定分类为 `llm_generator`，不再访问
  `knowledge_state.initialized`。
- 回归测试已验证空正文长度耗尽时不抛未捕获异常、评测数为 0、状态为暂停、失败码为
  `llm_output_exhausted`，并生成 `summary.json` 与 reasoning 审计文件。

### E2E-04：Diagnose 未获得 source validation 原始证据

状态：**部分解决**。source-validation 原始错误与 repair base 已修复，真实运行已推进到 build；完整
10 轮正确性验收尚未通过。

- 已支持 `error[rule]` 解析，并把有界的 `failure_stage/failure_code/error_excerpt/details_path/candidate`
  投影到 Diagnose；hash 不再是唯一证据。
- source-validation 候选现作为 run-local repair base 保留；真实复跑的候选 2 已清除两个 source 错误并
  进入 build，证明修复有效。
- 新复跑在 evaluation 8 后因缺少 ASC CMake 模块实际路径和被拒候选 diff 而由 Diagnose 正确返回
  `evidence_status=insufficient`。曾追加的 CMake toolchain 自动注入及 candidate diff 投影已按用户要求
  回退，后续不得在未确认方案边界前重新引入。

## 下一次复跑约束

1. 保留 `outputs/e2e_add_diagnosefix_20260918` 为只读证据，不恢复其 BLOCKED 现场。
2. 明确 CMake 构建环境问题的处理边界后，再使用另一个全新输出目录重跑；显式传入 `--model deepseek-flash`、
   `--generator-thinking disabled` 与 `--reasoning-log full`。
3. 保持 50-case Add、最多 10 个已评测候选、真实 CANN 9.1.0 与 Ascend910B3 的原验收口径。
4. 只有候选通过 `smoke/shape/dtype/full`、覆盖 case `0..49`，并产生有效
   `torch.npu.Event` benchmark 与正数 baseline score，才可判定完整流程通过。
5. 修复和复跑结果必须继续关联问题编号，并在 `CHANGELOG.md` 中记录“未解决、部分解决、已解决或
   不采纳”的状态与实际验证证据。
