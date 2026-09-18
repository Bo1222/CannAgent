# AscendC Agent Architecture Analysis Report 与 9.1.0 执行方案

## 决策摘要

根因权重：Prompt 30%，RAG 30%，Memory 10%，Workflow 30%。四者相互放大：旧 Prompt
用 case contract 弥补检索和验证缺口，检索又依赖冗余结构化卡片，Memory 会保留失败过程，
Workflow 缺少固定证据门和完整工程验证，最终形成“失败 case → 增加规则 → 新冲突”的循环。

本方案不继续增加 case contract。目标知识层只有两部分：固定提交的 CANNBot 文本技能和固定
commit 的 Asc DevKit 9.1.0 实时检索。原 930 条 atomic fact、341 张 API card、2 张 failure
card、5 张 pattern card、3 个 project contract 及其构建、路由、校验和模式开关全部删除，
且没有兼容回退。

## CannAgent 与 CANNBot 直调差异及统一目标

旧 CannAgent 生成 `model_new_ascendc.py + kernel/pybind11.cpp + kernel/*.cpp`，通过自定义
`*_do` ABI 和 PyBind 模块启动 Kernel；CANNBot 生成独立 ASC/C++ CMake 工程，通过
`op_kernel/op_host/op_extension` 分层、PyTorch dispatcher、PrivateUse1 和 Meta 注册完成调用。
二者此前并不一致。

统一后 CannAgent 采用 CANNBot 工程形态，同时保留评测系统要求的 `ModelNew` Python 外壳。
外壳只负责加载共享库和调用 `torch.ops`，核心计算仍由 AscendC Kernel 完成。

## 版本与供应链

- Asc DevKit repository：`https://gitcode.com/cann/asc-devkit.git`
- Asc DevKit ref：`9.1.0`
- Asc DevKit commit：`c785b5f76f23c9dc0ddb553174463d0497e59288`
- DevKit 文档根：`docs/api/`，不是旧版 `docs/zh/api/`
- CANNBot commit：`a08c49706e35a400d7c77e0875bc7c72a3a79012`
- 运行时要求：CANN 9.1.0 和匹配的 torch_npu

DevKit 由显式 `init` 命令写入 XDG cache。初始化使用锁、临时目录、固定 commit checkout、
健康检查和原子替换；普通 Agent 运行完全离线。显式目录也必须通过版本、commit 以及
`docs/api`、`examples`、`include`、`impl` 路径检查。

## 目标架构

### System Layer

Role 只定义职责与边界：生成 CANN 9.1.0 AscendC 直调工程，不枚举具体失败 case 的修复。

### Knowledge Layer

1. CANNBot 固定文本子集提供工程流程、设计、调试、精度和性能实践。
2. Asc DevKit 提供当前版本的 API 文档、示例、声明和实现证据。
3. installed headers 和 compile probe 只作为环境绑定的直接事实，不提升为跨环境知识。
4. 图片、二进制、结构化卡片和推测性经验不进入 Prompt。

检索顺序固定为：

```text
提取精确符号与变体
  → docs/api/**/*.md
  → examples/**/*
  → include/**/*
  → impl/**/*
```

每个 excerpt 携带 `evidence_id/source_kind/source_path/version/commit/score`，按 audience 字符预算
注入。没有找到证据时明确标记未知，不以旧知识库兜底。

### Workflow Layer

```text
Analyze
  → Retrieve
  → Design
  → Implement
  → Static Validate
  → Project CMake Build
  → Correctness Profiles
  → Performance
  → Evidence-driven Repair
```

Repair 只能针对当前 open diagnostics；候选必须重新经过后续 gate。通过的 source/compile/
runtime/correctness/performance frontier 可作为回退点，但失败尝试不得成为稳定知识。

### Tool Layer

- DevKit 管理：`python -m ascendc_multi_turn.devkit init|status`
- 离线检索：直接扫描固定 checkout，忽略图片，仅返回有界文本 excerpt
- 构建：候选目录自身 `cmake -S ... -B build` 与 `cmake --build`
- 静态契约：目录角色、CMake ASC、dispatcher 注册、ModelNew/torch.ops、旧 ABI 禁止项
- 验证：correctness profile 继续使用独立进程；performance 由包内 benchmark 使用
  `torch.npu.Event` 评分并对最慢 case 补充 profiler 证据，二者均从 `<task>/build` 加载候选

### Memory Layer

持久化：用户目标、CANN/SoC/DevKit fingerprint、直接诊断、已通过 gate、accepted frontier、最终结论。
不持久化为知识：临时 workaround、未验证假设、单次失败规则、旧卡片 ID。运行内 attempt ledger
可用于防止重复尝试，但作用域仅限当前任务。

## 分阶段实施与结果

### Phase 1：Prompt 和产物契约

- 删除运行时 case-contract 块。
- Prompt 改为 Role/Capability/Workflow/Tool/Memory 五层。
- 首轮产物改成完整 CANNBot 工程，拒绝 PyBind/`*_do` 历史 ABI。
- 状态：已实施；需在真实 9.1.0 环境继续做生成质量回归。

### Phase 2：RAG

- 删除旧结构化官方知识的源码、数据和 CLI 开关。
- 固定 CANNBot 十个技能的文本子集并校验 hash。
- 增加 9.1.0 DevKit 管理、健康检查、四级检索、证据 provenance 和图片排除。
- 状态：已实施；在线搜索不属于普通运行路径，后续如启用必须单独加来源与版本审计。

### Phase 3：Agent Workflow

- 构建改为候选工程 CMake。
- 保留静态校验、正确性 profile、包内 performance benchmark 和验证前沿；执行链不调用仓库级
  `skills/`。
- 接口不变量改为 dispatcher/Python op/kernel entry。
- 状态：已实施；本机已检测到 CANN 9.1.0，但托管 DevKit 尚未执行显式 `init`，且本次没有
  真实生成候选，因此 NPU build/run acceptance 尚未执行。

### Phase 4：Memory

- `StructuredFailure` 改为 `FailureEvidence`。
- generic diagnostics、attempt 和 frontier 移出旧知识包。
- 停止生成和持久化 confirmed-experience 知识。
- 状态：已实施；任务内 repair ledger 保留，作用域不扩展到其他任务。

## 验收门槛

1. CLI 不再出现旧 knowledge mode/source/store/adapter 参数。
2. 仓库不存在旧 structured knowledge 构建和路由代码，也不存在 `knowledge_store`。
3. DevKit 非 9.1.0 或 commit 不一致时 fail closed。
4. 检索优先级、provenance 和图片排除有单元测试。
5. CANNBot 产物角色、dispatcher、Meta、ModelNew/torch.ops 与旧 ABI 拒绝有单元测试。
6. 在 CANN 9.1.0 + 匹配 torch_npu 的 NPU 环境完成真实 CMake build、correctness、performance。

第 6 项是部署环境验收；当前仅确认 runtime 版本为 9.1.0，不把单元测试或 Mock 结果冒充
真实 NPU 工程验收。
