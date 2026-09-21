# AscendC Multi-turn Agent（CANN 9.1.0）

本模块使用固定版本 Asc DevKit 官方资料和仓库内固定提交的 CANNBot 文本技能，生成、构建、
验证并优化 AscendC 直调算子工程。不存在 document/structured/hybrid 模式，也不存在旧知识库回退。

## 固定依赖

- Asc DevKit：GitCode `cann/asc-devkit` 的 `9.1.0` 分支，固定 commit
  `c785b5f76f23c9dc0ddb553174463d0497e59288`
- CANN runtime：`9.1.0`
- CANNBot 技能：commit `a08c49706e35a400d7c77e0875bc7c72a3a79012` 的受控文本子集

初始化和检查 DevKit：

```bash
python -m ascendc_multi_turn.devkit init
python -m ascendc_multi_turn.devkit status
```

普通运行不访问网络，只读托管缓存；也可传入经过相同版本和 commit 健康检查的
`--asc-devkit-dir` 或 `ASC_DEVKIT_DIR`。

```bash
python -m ascendc_multi_turn \
  --op-name my_operator \
  --op-file /path/to/model.py \
  --op-json /path/to/cases.json \
  --output-dir /path/to/output \
  --asc-devkit-dir /path/to/asc-devkit \
  --reasoning-log full
```

Generator 默认关闭 thinking，把输出预算保留给五个算子逻辑文件。每次 API 调用若返回
`reasoning_content`，默认在对应轮次保存 `reasoning*.txt`，文件权限为 `0600`；
`calls.jsonl` 记录其相对路径、长度与 SHA256，而不复制原文。可用
`--reasoning-log metadata` 禁止原文落盘。`finish_reason=length` 属输出预算耗尽，不会按 transport
错误重复相同请求。

## 工作流

`Analyze → Retrieve → Design → Implement → Static Validate → Build → Correctness → Performance → Repair`

检索按 `docs/api → examples → include → impl` 执行，所有证据记录版本、commit 和相对路径；
图片不会进入上下文。Memory 只保留目标、环境、直接证据、验证前沿和最终结论，不会把临时
workaround 或一次失败规则提升为知识。

性能阶段由包内 `python -m ascendc_multi_turn.benchmark` 执行，不读取或调用仓库根目录
`skills/`。全部正确性 profile 通过后，benchmark 使用 `torch.npu.Event` 对同一组输入分别测量
reference 与 AscendC 实现，以各 case speedup 的几何平均作为分数；最慢的少量 case 再执行
profiler 诊断，为下一轮局部优化提供 operator 和 latency 证据。Profiler 诊断不是主评分来源。

## 产物契约

新任务先由 `create_ascendc_project.py` 复制固定模板并从 `Model.forward` 确定性生成 ABI：

```text
model_new_ascendc.py
CMakeLists.txt
op_kernel/<op>_tiling.h
op_kernel/<op>_kernel.asc
op_host/<op>.asc
op_host/data_utils.h
op_extension/<op>_torch.cpp
op_extension/register.cpp
op_extension/ops.h
scripts/
model.py
<cases>.json
```

其中 `scripts/` 为空目录，不生成 benchmark 或测试脚本；`model.py` 与 JSON 原样复制。
固定模板负责 `CMakeLists.txt`、`ops.h`、`register.cpp` 和 `data_utils.h`，LLM 只补全
kernel、tiling、host、torch 接入实现与 `model_new_ascendc.py`，不得修改工程骨架。
项目自己的 CMake 负责编译 ASC Kernel 和共享库；扩展使用 `TORCH_LIBRARY` 注册
PrivateUse1 与 Meta，`ModelNew` 加载共享库并通过 `torch.ops` 调用。旧
`kernel/pybind11.cpp`、`PYBIND11_MODULE` 和 `*_do` ABI 会被拒绝。

也可以只生成未实现算法的确定性工程骨架：

```bash
python create_ascendc_project.py \
  --op-name add_alpha \
  --op-file benchmarks/NPUKernelBench/level1/3_Add.py \
  --op-json benchmarks/NPUKernelBench/level1/3_Add.json \
  --output outputs/add_alpha_template_example
```
