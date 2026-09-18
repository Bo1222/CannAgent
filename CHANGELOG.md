# 项目变更日志

仓库中的代码、配置、测试或文档修改都必须记录在这里。日期使用 `YYYY-MM-DD`，不得记录秘密信息。
代码标识符、文件路径、命令、模型名、产品名和原始错误信息可以保留原文，其余叙述统一使用中文。
对应有TODO的修改项的时候使用二级标题作为修改的部分，并且放在文件首，目的是避免日期的累积就遗忘对应的TODO列表。

## 2026-09-18

### 方案：当前进度 Git 归档

状态：**已解决**。

- 上传前补充忽略规则，排除 `asc_profile_case*/` 性能分析运行产物和
  `ascendc_multi_turn/asc-devkit-*/` 本地 DevKit 检出，避免将可再生成的运行数据与第三方源码纳入仓库。
- 本次归档保留当前工作区中的架构重构、推理审计、Diagnose 证据链修复及 CMake 扩展回退结果；不提交
  `.env`、API 密钥、访问令牌或本地实验输出。

### 验证

- 上传前执行暂存区完整性检查与高置信度敏感信息扫描，未发现凭据特征；`.env`、运行输出和本地 DevKit
  均保持忽略状态。完整单元测试恢复固定版本 CANNBot 文件原始内容后共 80 项通过。
- `git diff --check` 仅报告固定版本 CANNBot 文件
  `ascendc-performance-best-practices/SKILL.md` 的一处上游行尾空格；清理该空格会破坏
  `vendor_manifest.json` 哈希并导致 7 项测试失败，因此保持供应商内容原样。提交和远端推送结果以 Git
  命令的实际返回为准。

### 方案：E2E-04 Diagnose 证据断链修复与 CMake 扩展回退

状态：Diagnose 的原始 source-validation 证据投影与 repair-base 保留**已解决**；真实复跑已证明流程能
从 source validation 推进到 build，但 10 轮正确性目标仍**部分解决**。按用户要求，随后针对 CMake
toolchain 自动注入所做的扩展已完整回退，不属于当前保留变更。

- 根因一：`diagnostics.py` 原先只解析 `error:`，不能解析 source validator 使用的
  `error[missing_asc_language]:` 格式，导致具体规则、文件和原文被降级为
  `stage_error:unknown:<hash>`。解析器现同时识别带规则码的错误，并以规则码作为稳定 category。
- 根因二：首个候选停在 source validation 时被 `INITIAL_REJECT`，runner 将 `current/previous` 恢复为空，
  下一轮只能重新生成整个工程。现在 source-validation 候选可成为 run-local repair base；同阶段清除
  部分直接诊断时使用 `PARTIAL_KEEP` 累积修复，不把它误认为正确 baseline。
- 根因三：Attempt 历史原先只投影 error ID、cleared/new 和 gate 进度。现在每条记录额外携带有界的
  `failure_stage`、`failure_code`、`error_excerpt`、`details_path` 与 candidate 路径，并明确 hash 只用于
  稳定关联、不能替代原始错误。
- 在全新目录 `outputs/e2e_add_diagnosefix_20260918` 使用 `deepseek-flash`、Generator no-thinking 执行
  真实复跑。候选 1 的两个 CMake source 错误被保留；候选 2 精确修复后 source validation 通过并进入
  CMake build，证明原 E2E-04 的“具体错误丢失、空工程重生”断链已解除。
- 运行最终完成 8 个候选评测、17 次 LLM 调用、512,991 tokens；6 个候选达到 source-valid，尚无候选
  编译成功。第 9 次 Diagnose 能准确区分 ASC module 与 Torch CMake 定位历史，并因缺少已安装模块真实
  路径及被拒候选 diff 返回 `evidence_status=insufficient`；最终
  `stop_reason=diagnosis_evidence_insufficient`，没有伪造修复项，也未恢复该现场。

### 分析

- no-thinking 模型并非看不懂明确诊断：修复投影后，它在下一候选中同时清除了
  `missing_asc_package` 与 `missing_asc_language`，真实推进一个评测阶段。此前连续重建错误工程主要是
  状态机丢弃候选与证据格式不兼容造成的。
- 当前未完成目标的后续瓶颈已转为构建环境证据边界，不再是最初的 hash-only Diagnose。曾实现
  Evaluator 自动解析并注入 Torch `CMAKE_PREFIX_PATH`、ASC `CMAKE_MODULE_PATH`，以及向 Diagnose
  投影候选 diff；用户要求回退“针对 CMake 问题做的操作”后，上述代码、preflight、日志增强和测试均已
  删除，保留现有 Evaluator CMake 调用语义。
- 回退只作用于尚未提交的 CMake 扩展，不回退已经通过真实 E2E 验证的 `error[rule]` 解析、source
  repair-base 与原始评测 evidence 投影；历史输出目录保持只读，未执行 `--resume` 或修改候选文件。

### 验证

- `python -m unittest tests.test_progressive_evaluation tests.test_repair_state -v`：14 项全部通过，覆盖
  带规则码的 source 诊断、source candidate 初始保留、部分修复累积与 Diagnose 原文投影。
- CMake 回退后静态扫描确认不存在 `_torch_cmake_prefix`、`_asc_cmake_module_path`、toolchain 自动注入、
  `diagnostic_candidate_diff` 或对应测试；相关 `py_compile`、`ruff check --select I` 与
  `git diff --check` 通过。
- 真实复跑 `summary.json`：`evaluations_completed=8`、`accepted_attempt_id=5`、首次 source-valid 为
  evaluation 2、`baseline_round=null`、0 次 correctness/benchmark；因此不得表述为 10 轮内正确算子
  已生成。

### 方案：E2E-03 reasoning 可审计化、截断处理与分层 API 诊断

状态：E2E-03A、E2E-03B 与 reasoning 落盘、默认配置、长度耗尽处理均**已解决**；关闭 Generator
thinking 后的真实端到端复跑已越过原截断点，但暴露新的 E2E-04 Diagnose 证据投影缺陷，全流程验收
仍为**部分解决**。

- 新增 `ASCENDC_REASONING_LOG_MODE` 与 `--reasoning-log {full,metadata}`，新运行默认使用 `full`。
  每次成功收到 API 响应后，在解析正文前原子写入 `response*.txt`；存在 reasoning 时同步写入对应的
  `reasoning*.txt`，文件权限固定为 `0600`。`metadata` 模式只记录长度、SHA256 和路径元数据，不保存
  reasoning 正文。
- `calls.jsonl` 新增正文与 reasoning 文件路径、字符数、UTF-8 字节数、SHA256、请求模型与返回模型
  是否不一致等审计字段，但不把 reasoning 正文嵌入 JSONL。`.gitignore` 排除运行目录中的
  `reasoning*.txt`，文档明确其敏感性与本地保存边界。
- 新配置默认请求规范模型名 `deepseek-flash`，Generator thinking 默认从 `enabled` 改为 `disabled`；
  Planner 保持原有策略。现有显式环境变量仍优先，真实诊断和复跑命令将显式传参，避免本机旧 `.env`
  影响结论。
- 重构 `_call_llm()` 的失败语义：网络或 provider 异常仍使用 transient retry；已收到响应且
  `finish_reason=length`、正文为空时立即返回 `llm_output_exhausted`，不再把同一确定性截断重复调用三次；
  有部分正文的长度截断只允许一次语义续写，第二次仍截断则暂停。reasoning 落盘失败使用
  `reasoning_persistence_failed`，避免产生不可审计的运行。
- 修复 E2E-03B：Generator 的 `LLMCallFailure` 始终分类为 `llm_generator`，不再读取已废弃且为
  `None` 的 `knowledge_state.initialized`；失败运行现在可靠落盘 `summary.json`，候选评测数保持为零。
- 新增 `python -m ascendc_multi_turn.llm_diagnostic` 分层诊断：先查询 `/models`，再分别执行最小
  non-thinking、最小 thinking/low、真实 Generator Prompt thinking/high 有界观察，以及真实 Prompt
  non-thinking 完整 bundle 校验。最小探针使用独立 `diagnostic` 系统角色，避免与 Generator 的
  file-bundle 契约冲突；响应、reasoning、模型目录与 manifest 以 `0600` 保存且不记录密钥或
  Authorization header。
- 经明确授权后执行真实诊断，输出目录为 `outputs/deepseek_diagnostic_e2e03a_20260918`。`/models`
  返回 `deepseek-flash` 与 `deepseek-v4-pro`；本次请求和返回均为 `deepseek-flash`，没有路由不一致。
- 诊断通过后在全新目录 `outputs/e2e_add_reasoningfix_20260918` 启动 Add 50-case、最多 10 候选实验，
  显式使用 `--model deepseek-flash --generator-thinking disabled --reasoning-log full`。运行完成 3 个候选
  评测和 7 次 LLM 调用后由 workflow 自行进入 `BLOCKED`，未执行恢复。

### 分析

- E2E-03A 的历史四次 reasoning 因旧实现未落盘而无法反向恢复，但同一 65,236 字符 Prompt 的受控复现
  已给出直接证据：thinking/high 在 8,192 token 上限内产生 8,192 reasoning tokens、29,010 字符
  reasoning、0 字符正文并以 `finish_reason=length` 结束；关闭 thinking 后使用 5,406 completion tokens
  生成 15,406 字符、11 文件的有效 bundle，并以 `finish_reason=stop` 结束。
- 原始 reasoning 不是代码正文，而是在广播 stride、UB 标量广播、`DataCopyParams` 与 case 49 性能之间
  反复选择和推翻方案；文本中 `Actually` 出现 38 次、`Hmm` 32 次、`Let me reconsider` 13 次，末尾在
  未完成句子中截断。最小 thinking/low 探针只使用 78 reasoning tokens 并正常返回正文，说明 thinking
  接口本身可用，非收敛由复杂真实 Prompt 与 thinking/high 的组合触发。
- `/models` 和四个探针均证明当前 canonical 模型为 `deepseek-flash`。历史请求
  `deepseek-v4-flash`、返回 `deepseek-flash` 更符合旧别名被路由到 canonical 模型，不能据此认定 API
  故障；当前请求 canonical 名时不存在路由差异。
- `finish_reason=length` 是服务已完成的一次响应，不是 transport 故障。原先将它纳入三次 blind retry
  既额外消耗约 196,608 reasoning tokens，也没有增加信息；新的分流保留一次“部分正文续写”机会，
  同时阻止空正文的确定性重复。
- 新 E2E 的 7 次调用全部 `finish_reason=stop`、reasoning 字符数为 0、请求和返回模型一致，证明关闭
  Generator thinking 能消除原 E2E-03A 截断并恢复候选生成。三个候选仍都未越过 source validation：
  第 1、3 个缺少 `find_package(ASC...)` 与 `LANGUAGES ASC`，第 2 个另外缺少 shared library load。
- 新登记 E2E-04，状态为**未解决**：`source_validation.log` 已含具体文件、行号和规则名，但 Diagnose
  Prompt 只收到 `stage_error:unknown:<hash>`，因此错误声明“未提供任何可定位的具体失败原因”，在
  3/10 候选时以 `diagnosis_evidence_insufficient` 提前停止。该问题属于 workflow 证据投影，不是模型
  reasoning、API、CANN 或 NPU 故障；本轮没有擅自恢复或绕过状态机。

### 验证

- `python -m unittest tests.test_llm_reasoning_audit -v`：9 项全部通过，覆盖 `0600` 完整落盘、
  metadata 模式、不盲重试空正文长度耗尽、部分正文单次语义续写、transport retry、落盘失败、
  E2E-03B 暂停状态、分层诊断 manifest 与新默认值。
- `python -m unittest discover -s tests -v`：78 项全部通过。
- `python -m py_compile ...`、新增诊断与测试文件的 `ruff check`、相关文件的
  `ruff check --select I`：全部通过。对已有大型模块执行全规则检查仍会报告本次改动前已存在的 lint，
  因而不将其误记为全仓库 lint 通过。
- 真实 API 分层诊断 `manifest.json` 为 `success=true`：最小 non-thinking、最小 thinking/low 和真实
  Prompt non-thinking 三个必需门均通过；thinking/high 受控复现长度耗尽。完整 reasoning 保存于
  `real_prompt_thinking_high/reasoning.txt`，权限 `0600`、大小 29,167 bytes、SHA256
  `eaad7d147d3b623d16cfc33ff7e44de7ffe01fa345e6a780d047d6b110900cbf`。
- 新 E2E 共 7 次 LLM 调用、183,104 tokens，完成 3 个候选评测；全部 Generator 正常返回正文，但
  3 个候选都在 source validation 失败，0 次构建、0 个 correctness profile、0 次 benchmark，最终
  `status=blocked`、`stop_reason=diagnosis_evidence_insufficient`。因此 E2E-03A 的配置修复有真实证据，
  但不得把本轮表述为全流程或算子能力通过。

### 方案：E2E-03 E2E-02F Add 算子端到端受控复跑

状态：路径修复的确定性验证**已解决**；真实端到端执行因 Generator 输出长度与 runner 异常处理缺陷
**未解决**。本次属于 workflow 回归导致预算未完成，不能评价 10 轮算子生成能力。

- 使用 `benchmarks/NPUKernelBench/level1/3_Add.py` 的 50 个 case，在全新目录
  `outputs/e2e_add_e2e02f_20260918` 执行真实 DeepSeek → Planner → Generator → 静态校验 → 构建 →
  正确性 → benchmark 实验，预算上限为 10 个已评测候选。
- 真实执行参数保持与 E2E-02 一致：请求模型 `deepseek-v4-flash`、Generator thinking enabled/high、
  65,536 输出 token、Planner thinking disabled/low、单阶段超时 1800 秒、Ascend910B3、CANN 9.1.0
  与固定 GitCode DevKit。

```bash
python -m ascendc_multi_turn \
  --op-file benchmarks/NPUKernelBench/level1/3_Add.py \
  --output-dir outputs/e2e_add_e2e02f_20260918 \
  --provider deepseek --model deepseek-v4-flash --temperature 0.2 \
  --generator-max-tokens 65536 --planner-max-tokens 8192 \
  --generator-thinking enabled --planner-thinking disabled \
  --generator-reasoning-effort high --planner-reasoning-effort low \
  --llm-transient-retries 3 --max-bootstrap-rounds 10 --max-rounds 1 \
  --max-total-rounds 10 --timeout 1800 --device 0 \
  --soc-version Ascend910B3 --cann-version 9.1.0 \
  --asc-devkit-dir /mnt/workspace/CannAgent/ascendc_multi_turn/asc-devkit-9.1.0
```

- 实验期间未修改 Prompt、workflow 或候选源码；旧目录
  `outputs/e2e_add_gitcode_91_20260918` 只用于历史 planner 响应重放，没有复用为新任务。
- 初始 Planner 成功且六个 `target_files` 全部位于白名单内。Generator 初次调用及三次配置内重试均
  返回 `finish_reason=length`，没有产生最终正文；重试耗尽后 runner 在处理 `LLMCallFailure` 时触发
  `AttributeError: 'NoneType' object has no attribute 'initialized'` 并异常退出。
- 实际为 0 个候选完成评测、0 次 source validation、0 次构建、0 个正确性 profile、0 次 benchmark；
  因而没有首次正确轮、50/50 case 证据、baseline 或性能分数。

### 分析

- 关联 E2E-02F：历史 `planner_v01/v02/v03` 重放证明非法路径会在进入 Generator 前剔除，真实初始
  Planner 也只生成合法路径；本次没有再次出现 `model_new.py` bundle 拒绝。由于 Generator 未返回
  bundle，真实执行没有到达能够再次触发旧故障的阶段，所以 E2E-02F 的代码级结论保持**已解决**，
  全流程验收仍只能标记为**部分解决**。
- 新登记 E2E-03A，状态为**未解决**：四次 Generator completion 分别为 65,535、65,536、65,536、
  65,536 tokens，reasoning tokens 合计 262,143；四个 `response*.txt` 均为空。请求模型为
  `deepseek-v4-flash`，服务端四次均记录为 `deepseek-flash`。这是稳定的输出预算耗尽，不是 transport
  断连，也不能计为候选失败。
- 新登记 E2E-03B，状态为**未解决**：`runner.py` 当前将 `knowledge_state` 固定为 `None`，但
  `_prepare_candidate()` 的 `LLMCallFailure` 异常分支仍通过 `knowledge_state.initialized` 判断失败
  阶段。该陈旧引用使本应记录为 `llm_generator` 的暂停变成未捕获异常，并阻止生成 `summary.json`。
- 输出现场不可安全恢复：`run_state.json` 停在 `pending_phase=EDIT`，`trajectory.json` 仍为
  `status=running`，且没有候选或评测结果。按照实验约束不执行 `--resume`，待两个问题修复后使用全新
  目录重跑。

### 验证

- 环境门禁：DevKit `healthy=true`、版本 `9.1.0`、commit
  `c785b5f76f23c9dc0ddb553174463d0497e59288`；`torch_npu 2.7.1.post4`；1×Ascend910B3；DeepSeek
  地址与密钥已配置但未写入日志。
- `python -m unittest discover -s tests -v`：69 项全部通过；此前登记的无真实 `rg` 回退失败本次未
  复现。
- E2E-02F 历史响应重放：`planner_v01` 合法 target 零变化、零告警；`planner_v02` 剔除
  `kernel/add_alpha.cpp` 与 `python/model_new.py`；`planner_v03` 剔除 `model_new.py`；直接向
  `parse_file_bundle()` 输入 `model_new.py` 仍得到白名单拒绝。
- Mock 冒烟：1 个候选建立 Mock baseline，`evaluations_completed=1`、`baseline_round=1`，知识源 commit
  正确；Mock 临时目录已删除，不作为真实性结论。
- 真实运行约 18 分 28 秒，共 5 次 LLM 调用、385,325 tokens，其中 Planner 1 次正常完成，Generator
  4 次全部长度截断；`calls.jsonl`、`round_01/llm_error.log`、空的四个响应文件、`run_state.json` 与
  `trajectory.json` 保存了直接证据。

### 后续事项

- E2E-03A：定位并解决 Generator thinking 连续耗尽 65,536 token 且无最终正文的问题。
- E2E-03B：修复空 `knowledge_state` 的异常处理并补充“全部 Generator 重试截断”的状态落盘回归测试。
- 两项均验证后用全新目录重跑同一 50-case、10 候选实验；不得恢复本次异常现场。

### 方案：E2E-02F 修复 planner → generator 的越界路径指令链

状态：**已解决**（代码改动 3 处，重放验证通过；端到端复跑已执行，因独立的 E2E-03A、E2E-03B
未到达 bundle 评测阶段，结果见 E2E-03）。

- `prompts.py` 的 `_PLAN_BODY` 中 planner 输出契约示例由 `["kernel/<name>.cpp"]` 修正为与
  `INITIAL_PLAN_OUTPUT_CONTRACT` 一致的合法命名，消除"示例教模型写 legacy `kernel/` 布局"的自相
  矛盾。
- `parse_plan()` 新增 `_sanitize_target_files()`：对 `target_files` 逐项复用
  `bundle.validate_relative_path` 的白名单，越界项在进入 plan 前剔除并以 `warnings.warn` 告警；
  不采用抛错，避免经 `runner.py` 的 planning 失败路径直接转为 `planning_response_invalid` 暂停。
- 生成器输出契约（`OUTPUT_CONTRACT`）补充白名单与冲突裁决：只允许写入根文件
  `model_new_ascendc.py`、`CMakeLists.txt` 以及 `op_kernel/`、`op_host/`、`op_extension/`、
  `scripts/` 下的源码；活动 plan item 的 `target_files` 与该白名单冲突时以白名单为准，不得创建
  白名单外的文件。

### 分析

- 触发链（E2E-02 候选 3）：planner 契约示例示范越界自由命名 → planner 产出越界 `target_files`
  （`model_new.py`）→ `parse_plan()` 无白名单校验 → plan item 被原样注入生成器提示词 → 生成器按
  指令写出越界文件 → bundle 白名单整包拒绝并暂停。
- 修复后"契约示例、planner 输出、生成器指令"三处都不再允许越界路径；最后一道闸（`bundle.py`
  的解析校验）保持不变，仍作为原子兜底。

### 验证

- 真实轨迹重放：`planner_v03/response.txt` → 剔除 `model_new.py` 并告警、保留合法 target；
  `planner_v02/response.txt` → 剔除 `kernel/add_alpha.cpp`、`python/model_new.py`；
  `planner_v01/response.txt`（原本合法）→ 输出零变化、零告警。
- 单元测试 69 项：68 通过；唯一失败为改动前已存在的
  `test_declaration_ranking_rejects_comment_hits_and_keeps_count_overload`（本机无真实 `rg`
  二进制时的 rglob 回退缺陷，与本次改动无关，见 E2E-02 门禁记录）。

### 方案：E2E-02 AscendC Agent 10 轮端到端验收（按 TODO.md 重跑）

状态：环境门禁与 Mock 冒烟**已通过**；真实 10 轮执行**未完成预算**，在候选 3 处以非法 bundle
暂停，最终分类为第 3 类（Agent 能力失败）的限定情形，验收结论**未解决**。

- 执行命令（Phase 3，与 `TODO.md` 一致）：

```bash
python -m ascendc_multi_turn --op-file benchmarks/NPUKernelBench/level1/3_Add.py \
  --output-dir outputs/e2e_add_gitcode_91_20260918 --provider deepseek --model deepseek-v4-flash \
  --temperature 0.2 --generator-max-tokens 65536 --planner-max-tokens 8192 \
  --generator-thinking enabled --planner-thinking disabled \
  --generator-reasoning-effort high --planner-reasoning-effort low --llm-transient-retries 3 \
  --max-bootstrap-rounds 10 --max-rounds 1 --max-total-rounds 10 --timeout 1800 --device 0 \
  --soc-version Ascend910B3 --cann-version 9.1.0 \
  --asc-devkit-dir /mnt/workspace/CannAgent/ascendc_multi_turn/asc-devkit-9.1.0
```

- 环境指纹：GitCode DevKit `9.1.0`、commit `c785b5f76f23c9dc0ddb553174463d0497e59288`；CANN
  `9.1.0`（`ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.1.0`）；`torch_npu 2.7.1.post4`；1×
  Ascend910B3；provider `deepseek`、请求模型 `deepseek-v4-flash`；知识源
  `cannbot_skills+asc_devkit_9.1.0`，summary 中 `knowledge.status=exact`。
- 环境门禁：69 项单元测试 68 通过、1 失败
  （`test_declaration_ranking_rejects_comment_hits_and_keeps_count_overload`）。失败根因已定位：
  本机没有真实的 `rg` 二进制（shell 中 `rg` 是 Claude Code 注入的函数，`shutil.which("rg")` 为
  `None`），`runtime_knowledge` 走 rglob 回退分支而该分支不过滤注释行；用真实 CANN 9.1.0 头文件
  验证 Add 的 runtime facts 渲染无注释污染，不阻塞本次执行；该缺陷在 HEAD 中已存在，非本次引入。
- 实际候选轮数：2 个候选完成评测（`evaluations_completed=2`、`attempts_completed=3`、
  `discarded_attempts=[1,2]`）；候选 3 的 bundle 被拒，未消耗评测轮。
- 逐轮失败：候选 1、候选 2 均为 `ascendc_source_validation_failed`（`kind=candidate`），细节为
  `CMakeLists.txt:1: error[missing_asc_language]: project CMake must contain LANGUAGES ASC`
  （LLM 生成 `project(... LANGUAGES CXX)`）；候选 3 为 `response_format_failed`
  （`kind=orchestration`）：`invalid LLM file bundle: path is outside the CANNBot AscendC project
  sources: 'model_new.py'`。
- 首次正确轮：无。50/50 case 证据：无（未产出 `correctness=true` 候选，未进入
  smoke/shape/dtype/full 任一 profile）。
- benchmark 结果：未建立 baseline（`baseline_round`、`baseline_score` 为 `null`），无
  `torch.npu.Event` 性能数据。
- 最高 frontier：`stage_normalized_metrics.milestones` 全部为 `censored`，未越过 source validation。
- 预算与耗时：运行约 18 分钟（02:25:05–02:43:17），8 次 LLM 调用、454,920 tokens（其中 reasoning
  236,484）；无 transport、NPU、CANN 或文件系统故障。
- 停止与恢复：`stop_reason=response_format_failed`、`phase=PAUSED`、`pending_phase=EDIT`。按
  `TODO.md` 的恢复约束（只有明确的 infrastructure/transport 暂停才可用 `--resume`），非法 bundle
  属框架失败，未执行 `--resume`，暂停现场原样保留供审计。

### 分析

- 最终分类：最接近第 3 类（Agent 能力失败）——环境正常、无基础设施阻塞；但必须限定：只评测了 2
  个候选、第 3 个候选在 bundle 校验阶段被拒，未用满 10 轮预算，不能表述为"10 个候选均未通过
  full correctness"。
- 候选 1、2 的 `missing_asc_language` 是单次候选证据：生成器连续两次未按 CANNBot 约定声明 CMake
  的 `LANGUAGES ASC`；不作为新的 case contract。
- 候选 3 非法 bundle 的根因已定位到 planner 侧：后续轮使用的 `PLAN_OUTPUT_CONTRACT` 示例
  （`prompts.py:81`，`"target_files": ["kernel/<name>.cpp"]`）指向阶段契约明令禁止的 legacy
  `kernel/` 布局；planner（v02、v03）据此产出越界 `target_files`（`python/model_new.py`、
  `model_new.py`），而 `parse_plan()`（`prompts.py:686`）只校验 target_files 是字符串列表、未按
  bundle 白名单校验，越界路径经 `build_prompt()`（`prompts.py:508`）原样注入生成器提示词；生成器
  按其输出契约同时写出 `model_new_ascendc.py` 与 plan 要求的 `model_new.py`，整包被
  `validate_relative_path`（`bundle.py:45`）拒绝并暂停。首轮 planner 使用
  `INITIAL_PLAN_OUTPUT_CONTRACT`（示例为合法命名）时 target_files 完全合规，可佐证两套契约示例
  不一致即为诱因。越界 bundle 按设计未计入候选轮。
- 观察（待核实）：`calls.jsonl` 中服务端返回的模型名为 `deepseek-flash`，与请求的
  `deepseek-v4-flash` 不一致；8 次调用全部 `finish_reason=stop`，属接入层命名/路由观察，不影响
  本次分级结论。

### 验证

- `summary.json`：`evaluations_completed=2`（≤10）；`accepted_attempt_id`、`accepted_candidate`、
  `best_round` 均为 `null`——未把失败候选保留为 best。
- `round_01/source_validation.log`、`round_02/source_validation.log` 与
  `round_03/response_format.log` 分别保存了 `missing_asc_language` 与非法 bundle 的原始错误。
- 输出目录 `outputs/e2e_add_gitcode_91_20260918` 为全新目录，未复用；Mock 临时目录已删除；执行
  期间未修改 Prompt、case、候选源码，未新增 case contract。

### 方案：E2E-01 AscendC Agent 10 轮端到端验收

状态：本次执行被用户主动终止，验收结论**未解决**；完整重跑方案已保存到 `TODO.md`。

- 环境门禁通过：GitCode DevKit 为 `9.1.0`、commit
  `c785b5f76f23c9dc0ddb553174463d0497e59288`，Ascend910B3、`torch_npu 2.7.1.post4`、
  DeepSeek 配置可用，仓库 69 项单元测试通过。
- Mock 一轮成功建立 baseline，验证新工程布局、状态机、预算与 provenance；Mock 结果不作为真实
  NPU 正确性或性能结论。
- 真实 Add 测试第 1 个候选在一次截断恢复后进入 source validation，因 `CMakeLists.txt` 缺少
  `project(... LANGUAGES ASC ...)` 失败，未进入编译；第 2 轮 Generator retry 期间收到停止指令。
- 已向运行会话发送中断并确认 `KeyboardInterrupt` 退出；删除
  `outputs/e2e_add_gitcode_91_20260918` 与对应 Mock 临时目录，未保留可 resume 状态或中间候选。
- 先清空旧 `TODO.md`，再写入新的环境门禁、Mock、真实 10 轮命令、50-case 验收口径、证据审计、
  停止/恢复规则和四级结果分类。重新执行时必须使用全新输出，不得把本次中断当作失败样本。

### 分析

- 本次只完成 1 个真实候选评测且第 2 个候选未生成完成，不能判断框架能否在 10 轮内生成正确算子。
- 第 1 轮 source failure 是单次候选证据，不应提升为新的 case contract；后续重跑必须保持既定框架，
  通过相同的 source、build、correctness 和 benchmark 门自然验证修复能力。

### 验证

- 进程会话在第 2 轮 Generator retry 中断并返回 `KeyboardInterrupt`；进程扫描未发现残留的
  `python -m ascendc_multi_turn` 任务。
- 删除后检查两个精确目录均不存在；`TODO.md` 已替换为当前待执行方案。

### 方案：ARCH-93 Asc DevKit 供应源迁移到 GitCode

状态：配置与回归测试**已解决**；托管缓存初始化与真实端到端验收**部分解决**。

- 按限定范围只查询 GitCode，确认官方仓库为 `https://gitcode.com/cann/asc-devkit.git`，
  `9.1.0` 分支当前指向 commit `c785b5f76f23c9dc0ddb553174463d0497e59288`。
- `ascendc_multi_turn/devkit.py` 将仓库、ref 和固定 commit 同步迁移到上述 GitCode 来源；健康检查
  继续要求版本、commit 和 `docs/api`、`examples`、`include`、`impl` 全部精确匹配。
- Prompt、运行摘要、模块文档、架构方案和内置 docs-search 文本统一使用新的 provenance；运行摘要
  改为读取同一常量，避免以后出现配置与审计字段漂移。
- 内置 docs-search 属于 manifest 明示的 9.1.0 本地适配；本次同步更新其 SHA256 与适配说明，
  CANNBot 上游 commit 保持不变，启动时仍执行完整 hash 校验。
- 新增供应链回归断言，明确拒绝把旧仓库 URL、`v9.1.0` ref 或旧 commit 重新带回当前配置。

### 分析

- GitCode 正式版本引用名为 `9.1.0`，不是原配置中的 `v9.1.0`；仅替换域名而保留旧 ref/commit
  会导致初始化或健康检查继续失败。
- 固定完整 commit 而不是只跟随可移动分支，保证离线检索证据、Prompt 声明和运行摘要可复现。
- 第一次 GitCode 临时浅克隆响应过慢并已中止；该网络现象不属于 Agent 候选失败，也没有消耗
  端到端测试轮次。配置验证完成后仍需重新初始化托管缓存，才能继续真实 NPU 端到端验收。

### 验证

- GitCode API `repos/cann/asc-devkit/branches/9.1.0` 返回完整 commit
  `c785b5f76f23c9dc0ddb553174463d0497e59288`。
- `python -m unittest tests.test_ascendc_91_architecture -v`：9 项通过，覆盖 GitCode URL、ref、
  commit、clone/checkout 命令、健康检查、检索优先级、图片排除、manifest hash 与 Mock 工作流。
- `python -m unittest discover -s tests -v`：69 项通过；仅出现既有 `torch_npu` owner 与无设备沙箱
  warning，无测试失败。
- `ruff check --select I ...`、`python -m py_compile ascendc_multi_turn/*.py
  tests/test_ascendc_91_architecture.py` 与 `git diff --check`：通过。
- `python -m ascendc_multi_turn.devkit init` 已确认改走 GitCode 且不再出现 Gitee 鉴权错误，但 Git
  clone 数分钟没有 pack 进度或错误，人工中止；finally 已清理临时目录和锁。随后 `devkit status`
  指向新 commit 缓存目录并按预期报告未初始化。真实 NPU 端到端测试尚未启动、未消耗候选轮次。

### 方案：ARCH-92 AscendC benchmark 脱离仓库级 Skills

状态：代码实施**已解决**；真实 NPU 性能验收**部分解决**。

- 新增 `ascendc_multi_turn/benchmark.py`，由 `python -m ascendc_multi_turn.benchmark` 在独立
  子进程中执行；`LocalAscendEvaluator` 不再调用
  `skills/ascendc/performance-analyzer/references/performance.py`。
- 新 benchmark 与正确性流程共用 `model.py` 的 `get_input_groups/get_inputs` 语义、候选优先的
  `get_init_inputs` 和 `<task>/build` 加载路径；仅接受 NPU，禁止 CPU/CUDA 性能回退。
- 主评分改用 `torch.npu.Event` 的逐次样本，中位数计算单 case speedup，所有 case 的几何平均
  作为总分。报告 schema 记录 mean、median、p90、标准差、变异系数、内存和测量顺序。
- 所有 case 完成 Event 测量后先原子保存有效报告，再只对最差的最多三个 case 执行 profiler
  诊断。Profiler 只提供 operator 证据，不参与主评分；补充诊断失败不会覆盖已经完成的有效分数。
- Evaluator 校验报告版本、NPU Event 测量方式、有限正数时延、唯一 case 索引、与 full
  correctness 完全一致的 case 集，以及报告总分是否等于逐 case 几何平均。
- 优化 Prompt 的评测摘要现在携带有界的 measurement、diagnostics 和 bottleneck/operator 证据，
  使性能修改基于实际慢 case，而不是只看到 overall speedup。
- 保留 `utils/verification_ascendc.py` 和 compile probe 对 `utils/build_ascendc.py` 的使用：二者不
  依赖仓库级 `skills/`。旧 `utils/performance.py` 与根目录 performance-analyzer 仅供旧流程，
  multi-turn 不导入、不检索也不执行它们。

### 分析

- 旧 benchmark 并非通过知识路由命中，而是 evaluator 硬编码执行 skill 脚本；因此删除结构化
  知识路由不会解除这个运行时耦合，必须在执行层替换入口。
- 直接改调 `utils/performance.py` 只能把耦合从 `skills/` 移到另一个遗留脚本，并继续继承
  CPU/CUDA 回退、错误退出码、算术平均与 evaluator 几何平均不一致、全量 profiler 开销等问题。
  将测量实现收归 `ascendc_multi_turn` 才能让报告 schema、验收门和优化反馈由同一模块维护。
- 包内 `knowledge_modules/.../skills` 是固定 CANNBot 文本来源，不是根目录 `CannAgent/skills`
  的可执行依赖；性能最佳实践仍可作为优化知识，但不再承担 benchmark 执行。

### 验证

- `pytest -q`：67 项通过；仅有既有 `torch_npu` 文件 owner warning 和无设备测试环境的
  device-count warning，无失败。
- `ruff check ascendc_multi_turn/benchmark.py ascendc_multi_turn/evaluator.py
  ascendc_multi_turn/diagnostics.py tests/test_ascendc_benchmark.py`：通过。
- `python -m py_compile ascendc_multi_turn/*.py utils/verification_ascendc.py` 与
  `git diff --check`：通过。
- 静态扫描确认 `ascendc_multi_turn/*.py` 不含根目录 performance-analyzer、
  `skills/ascendc/` 或 `REPO_ROOT / "skills"` 引用；命令构造回归测试确认 evaluator 只调用
  `python -m ascendc_multi_turn.benchmark`。
- 本轮未用真实候选执行 NPU Event/profiler 集成测试，因此真实 NPU 性能验收仍为**部分解决**，
  不把纯函数、Mock 或 CLI help 结果记录成真实性能通过。


## 2026-09-17

### 方案：ARCH-91 AscendC Agent 9.1.0 架构重构

状态：框架实施**已解决**；真实 NPU 候选验收**部分解决**。

- 将 `CannAgent/ascendc_multi_turn` 从“结构化官方知识 + case contract + 旧 PyBind ABI”重构为
  “固定 CANNBot 文本技能 + 固定 Asc DevKit 9.1.0 证据 + 分阶段验证工作流”。完整方案与实际
  结果保存在 `docs/ascendc-agent-architecture-plan.md`。
- 删除旧 `structured_knowledge` 构建、抽取、路由、API constraint validator、experience promotion、
  相关测试、迁移报告和 `knowledge` document 模式；删除本地未跟踪的 `knowledge_store`，不存在
  document/structured/skills/hybrid 回退。
- CLI 删除 `--knowledge-mode`、`--knowledge-source`、`--knowledge-input-mode`、
  `--knowledge-store`、`--knowledge-build-id`、`--skill-adapter` 及外部 skill 路径参数，只保留
  `--asc-devkit-dir` 作为经过健康检查的显式覆盖。
- 新增 `ascendc_multi_turn/devkit.py`：固定 `v9.1.0` 与 commit
  `e5451c7feba11f9e259f4fbc19c4bf82f8c0496c`，提供 XDG cache、锁、临时 checkout、健康检查、
  原子替换以及 `python -m ascendc_multi_turn.devkit init|status`。
- 新增 `devkit_retrieval.py`，离线按 `docs/api → examples → include → impl` 检索精确符号与变体；
  证据记录 `source_kind/source_path/version/commit/score`，图片及其他非文本文件不进入上下文。
- 在 `ascendc_multi_turn/knowledge_modules/cannbot_a08c4970_knowledge_base` 固定 CANNBot commit
  `a08c49706e35a400d7c77e0875bc7c72a3a79012` 的十个文本技能，新增 hash 清单；docs-search
  副本适配 CANN 9.1.0 的 `docs/api` 布局。运行路由不读取仓库级 `skills/` 目录。
- 删除 CannAgent 本地 knowledge modules 及 manifest；`skill_mapping.yaml` 只引用固定 CANNBot
  文本模块。曾误改的仓库级 `skills/`、相关 Agent 文档和测试均已恢复到本次工作开始前的内容，
  不属于本次新路由变更；评测器虽仍把 performance analyzer 当执行工具调用，但知识路由不读取它。
- Prompt 删除运行时 case-contract 大块，改为 Role、Capability、Workflow、Tool、Memory 五层；
  固定 `Analyze → Retrieve → Design → Implement → Static Validate → Build → Correctness →
  Performance → Repair`，禁止把临时 workaround 或一次失败规则提升为长期知识。
- 产物契约改为完整 CANNBot 工程：`model_new_ascendc.py`、`CMakeLists.txt`、`op_kernel/`、
  `op_host/`、`op_extension/`、`scripts/`；要求 `TORCH_LIBRARY`、PrivateUse1、Meta、共享库加载和
  `torch.ops` 调用，拒绝 `kernel/pybind11.cpp`、`PYBIND11_MODULE` 与 `*_do` 历史 ABI。
- Evaluator 改用候选项目自身 CMake 配置和构建；正确性与性能加载路径统一为 `<task>/build`。
  接口不变量改为 dispatcher registration、Python op 和 Kernel entry。
- 将通用诊断、attempt 与 frontier 类型移出旧知识包；`StructuredFailure` 更名为
  `FailureEvidence`，停止把 incident 提升为跨任务 confirmed experience。

### 分析

- case contract 无限增长的根因不是单个规则缺失，而是知识、工作流和验证边界错位：旧系统把
  API/ABI 知识硬编码进 Prompt，用规则匹配代替版本化检索，并以不完整的旧工程 ABI 执行；每个
  新失败只能继续补规则，规则之间随后发生冲突。
- CannAgent 旧直调产物与 CANNBot 并不一致。前者是 `kernel/pybind11.cpp + *_do`，后者是
  ASC/C++ CMake 工程和 PyTorch dispatcher。统一工程边界后，Prompt 不再需要维护旧 ABI 的
  case contract。
- 结构化卡片的冗余、弱 provenance 和图片副本不适合作为当前版本真值。9.1.0 API 的权威来源
  改为固定 DevKit checkout；CANNBot 只提供工程和调试方法，不替代精确 API 声明。
- Memory 的职责是保存目标、环境、直接证据、验证前沿和最终结论；失败尝试仅保留在任务内
  repair ledger，不能污染跨任务知识。
- `EvidenceBundle` 和上下文选择器已直接使用 DevKit evidence，未保留 `api_cards`、
  `failure_cards`、`project_contracts` 等旧结构化知识兼容空壳，避免旧抽象继续成为隐式接口。

### 验证

- `pytest -q`：56 项通过，1 条既有 `torch_npu` 文件 owner warning；无失败。
- `python -m py_compile ascendc_multi_turn/*.py utils/verification_ascendc.py
  skills/ascendc/performance-analyzer/references/performance.py`：通过。
- `python -m ascendc_multi_turn --help`：仅出现 `--asc-devkit-dir`，旧 knowledge 参数均不存在。
- `python -m ascendc_multi_turn.devkit --help`：通过；包入口改为 lazy import，避免模块预加载警告。
- 当前 `ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.1.0`，runtime 探测结果为
  `runtime_version=9.1.0`、`status=exact`。
- 托管 DevKit cache 当前未初始化，`devkit status` 按预期返回 unhealthy 并列出缺失路径；
  正常非 Mock 运行会 fail closed。本次未执行网络初始化，也未把 Mock/单元测试结果记为真实
  NPU build、correctness 或 performance 通过；该部署验收状态为**部分解决**。


## (✅)TODO-22：性能基准无法 import 已编译扩展（建不起 baseline 的最后一堵墙）（2026-09-17）

状态：已实施并**直接复现验证**。这是唯一一堵与契约无关、纯属基础设施的墙。

### 分析

- TODO-19 那一轮的**第 9 轮已经产出正确算子**：
  `compiled: true`、`correctness: true`、`passed_profiles = ['smoke', 'shape', 'dtype', 'full']`
  ——全部正确性 profile 通过。但 `score: null`，`baseline_round: null`。
- 失败原因与模型、契约均无关：
  ```
  performance.py → _load_impl → 加载 model_new_ascendc.py
  ModuleNotFoundError: No module named 'add_ext'
  ```
- 对照两份脚本：`utils/verification_ascendc.py`（`verification_ascendc.py:554-568`）在加载候选前
  会把 `WORKDIR` 与 `kernel/build` 插入 `sys.path`，因此正确性验证可以 import 扩展；
  而 `skills/ascendc/performance-analyzer/references/performance.py` 的 `_load_impl`
  直接 `_load_module`，**从不设置 sys.path**。
- 后果：一个已经通过全部正确性验证的算子，在紧接着的基准测试里连 import 都失败
  → `performance_score_missing` → `valid=False` → 永远建立不了 baseline。
  因为两条路径的加载方式不一致，该缺陷不会在正确性阶段暴露。

### 实施方案与结果

- `skills/ascendc/performance-analyzer/references/performance.py`：
  新增 `_ensure_import_paths(output_dir)`，把 `<task_dir>` 与 `<task_dir>/kernel/build`
  中**存在的**目录插入 `sys.path`；在 `_load_impl` 开头调用。
  与 `verification_ascendc.py` 的做法对齐，使「通过了正确性验证」与「能被基准测试加载」
  使用同一套 import 约定。
- 该改动只影响 import 路径设置，不改动测量逻辑、重复次数、profiler 使用或输出格式。

### 验证（直接复现，非跑批推断）

1. 用 `outputs/nonthinking_add_oscfix_10round_20260916/.llm_state/round_09/candidate.json`
   还原第 9 轮的候选，`utils/build_ascendc.py` 重新编译成功，
   产出 `add_ext.cpython-311-aarch64-linux-gnu.so`。
2. **修复前**（`git stash push` 还原 `performance.py`）运行基准：
   `exit=1`，`ModuleNotFoundError: No module named 'add_ext'`，`perf json 未产生`
   —— 与跑批中的失败逐字一致。
3. **修复后**运行基准：通过 import 阶段并进入测量，profiler 逐 case 解析至 case47/50
   （修复前在 import 处立即失败）。**import 根因确认修复。**
4. 该次验证被我自己设的 `timeout 900` 在第 47 个 case 处终止（`exit=124`），
   因此**没有取到完整 `performance.json`**——这是验证脚本的超时，不是被测流程的超时。
- **尚未验证，且已列为下一个疑点**：评测器调用基准时固定使用
  `--warmup 10 --repeats 50`（`evaluator.py:495-505`），超时为 `self.timeout`（默认 600s）。
  而上述较轻的 `--warmup 2 --repeats 5` 已耗时超过 900s。
  因此**即使 import 修好，性能阶段仍可能因 600s 超时而失败**。
  已启动一次按评测器原参数（`--warmup 10 --repeats 50`）的计时测量来确认，
  结论未出前不宣称性能阶段已通。
- 全量测试与其它流程未受影响：本轮只改 `sys.path` 设置，未触及被测脚本的测量逻辑。

### 这一轮的意义

- 在此之前，`TODO-11` ~ `TODO-21` 都是在**契约层**排除障碍；而这一条是**执行层**的缺陷，
  且是「已经正确」到「被判定为正确」之间的最后一步。
- 记录一条方法论结论：**当轨迹卡在「模型已产出正确结果但流程判定失败」时，
  应当先核对测量/加载路径的一致性，而不是继续补契约文本。**
  本轮之前的多轮迭代都属于后者，边际收益递减。


## (⏳)TODO-21：标量数据通路导致向量核超时（507034）（2026-09-17）

状态：契约已实施；**确定性校验经评估后放弃**（理由见下）。端到端复跑中。

### 分析

- TODO-20 之后 `507035` 不再出现，但稳定卡在 `507034`（`Vector core execution timed out`，
  `subsystem = MTE`），失败固定在 `dtype` profile 的 case 3（float16, 128×128）。
  每轮以 600s 超时收场，是当前最大的单轮时间开销。
- 定位到候选 Kernel 的实现方式：`ProcessTileTyped` 的输入搬运与输出写回**完全使用逐元素标量访问**——
  ```cpp
  for (int64_t i = 0; i < count; ++i) {
      int64_t xOff = InputOffset(...);
      xLocal.SetValue(i, xTyped.GetValue(xOff));
  }
  ...
  for (int64_t i = 0; i < count; ++i) outTyped.SetValue(base + i, outLocal.GetValue(i));
  ```
  代码注释写着“Gather inputs into UB using DataCopy for contiguous spans; fall back to scalar
  SetValue only for broadcast”，但实现是无条件走标量路径。128 个元素（smoke、fp32）能跑完，
  16384 个元素（fp16 128×128）即超时。
- 契约里只有“优先使用向量化 AscendC 操作、对齐搬运和有界 UB 用量”这样一句**偏好**，
  没有说明「逐元素读写 `GlobalTensor` 作为主数据通路会超时」这一硬后果。

### 实施方案与结果

- **契约**（`ascendc_multi_turn/prompts.py`，`RULES`）：在向量化那条之后新增一条，写明
  「禁止用逐元素 `GetValue`/`SetValue` 读写 `GlobalTensor` 充当数据搬运」，
  给出后果（`507034 Vector core execution timed out`，且编译与静态校验都不预警），
  并给出正确做法：`DataCopy`/`DataCopyPad`（对齐用 `DataCopyParams`，
  非对齐用 `DataCopyExtParams + DataCopyPadExtParams<T>`），
  广播维用标量读一个值后在 UB 内 `Duplicate` 扩展，其余连续段仍走 `DataCopy`；
  逐元素 `SetValue` 仅允许用于构造极小的常量 mask/padding。
- **确定性校验：评估后放弃。**曾实现 `scalar_data_path` 规则——扫描运行期边界的
  循环体内的 `GetValue`/`SetValue`。该规则对目标候选的判定完全正确
  （精确命中数据循环第 134、162 行，未误报第 47/49/51 行的 `kMaxRank` 常量上限循环），
  但**在 8 个已知可用的归档样例上误报 2 个**：
  `gather_elements_v2`（`for (col = 0; col < tiling_.IG; ++col)`，gather 语义本身就需要逐元素）
  与 `reshape_matmul_rowwise_quant_int8`。
  即「运行期循环里的标量访问」并不能推出「本可以用 `DataCopy`」——
  gather/scatter/per-row quantize 等算子的算法语义就要求逐元素。
  该前提不成立，规则无法在不牺牲正确性的前提下收窄，**故移除**，只保留契约说明。
  归档样例的零 issue 底线已复核恢复。

### 验证

- `python -m py_compile` 通过；Ruff `F/I` 通过。
- 移除规则后复核：`archive_tasks/` 全部 8 个样例回到 **0 issue**。
- 全量测试：165 项中 164 项通过。唯一失败项仍为该环境既有失败，与本轮无关。
- 本轮未新增测试：被放弃的规则不应留下断言其行为的测试。


## (⏳)TODO-20：设备错误码族路由（507034 未被识别）（2026-09-17）

状态：已实施并验证。

### 分析

- TODO-18/19 之后，`507035` 崩溃不再出现，但真实跑批暴露了**同族的另一个码**：
  `runtime_code = 507034`，`Vector core execution timed out`，`subsystem = MTE`，
  `reason = MTE failure`，失败在 `dtype` profile 的 case 3（float16, 128×128）。
- `context_selector._RUNTIME_MARKERS` 逐码枚举了 `507035` 与 `507001`，
  **没有 `507034`**。`diagnostics.py` 提取 `runtime_code` 时用的是
  `(50|56)\d{4}` 的口径，两处口径不一致，导致标记表存在盲区。
- 命中该盲区时，设备故障会被继续按非运行时失败判断，
  路由偏离设备故障应有的 `runtime_debug`——而 `runtime_debug` 正是
  `ascendc-runtime-debug` 知识与 507xxx 排查流程的入口。
- 这是「逐项枚举」而非「按类判定」造成的缺陷，与 TODO-17 的 torch 白名单同源：
  枚举表会在新增成员时静默失效。

### 实施方案与结果

- `ascendc_multi_turn/context_selector.py`：
  - 新增 `_DEVICE_ERROR_CODE = re.compile(r"\b(?:50|56)\d{4}\b")`，
    与 `diagnostics.py` 的 `runtime_code` 口径对齐，按**族**而非逐码匹配。
  - 从 `_RUNTIME_MARKERS` 中移除逐码枚举的 `507035`、`507001`，
    改由该正则覆盖；`assign_to_runtime` 的判据增加 `or _DEVICE_ERROR_CODE.search(failure_text)`。
- 说明：排查中一度怀疑 `_RUNTIME_MARKERS` 的小写条目（`mte`、`plog`）无法匹配大写文本，
  核对后确认 `failure_text` 在构造时已统一 `.lower()`（`context_selector.py:163`），
  该怀疑不成立，未做改动。

### 验证

- 新增 `test_device_error_code_family_routes_to_runtime_debug`：
  对 `507034`、`507035`、`507001`、`561234`、`500001` 断言路由到 `runtime_debug`，
  并断言非该族的数字（`12345`）不会被误判为设备错误码。
  用 `git stash push -- ascendc_multi_turn/context_selector.py` 还原后该测试**失败**，
  恢复后通过。
- 全量测试：165 项中 164 项通过。唯一失败项仍为该环境既有失败，与本轮无关。
  Ruff `F/I` 通过。


## (⏳)TODO-19：宿主描述符振荡（507035 反复回归）（2026-09-16）

状态：已实施并验证，端到端复跑中。

### 分析

- TODO-18 的契约修复经真实运行验证**部分生效**：第 4 轮 `507035` 消失，首次产出真实数值对比
  （`MERE=7.55848`、`rel_mismatch_ratio=1`，即"数值全错"而非"设备崩溃"）。
  说明规范写法确实被模型接受过一次。
- 但模型随即**振荡**：r5、r6、r7 又改回 Host 指针写法，`507035` 连续回归三轮；
  r8 恢复正常但数值仍为 `MERE=0.999999`，r9 再次失败。
  10 轮中有 **4 轮**重复消耗在同一个已经给过规范写法的错误上。
- 这类失败的特征决定了它必须用确定性校验而非契约文字来解决：
  通过 `source_validation`、通过编译，只在设备侧以 `507035` 暴露，
  每轮烧掉一次完整 NPU 评测与一轮生成预算。仅靠 RULES 中的文字说明无法阻止回退。

### 实施方案与结果

- **确定性校验**（`ascendc_multi_turn/source_validation.py`）：新增 issue code
  `host_pointer_descriptor`。收集 `kernel/pybind11.cpp` 中以 `std::vector<...>` /
  `std::array<...>` 声明的局部变量，检查 `*_do(...)` 调用第 2 位起（跳过 `blockDim` 与 `stream`）
  的实参；实参形如 `<该变量>.data()` 即报错，并给出 `.to(at::kPrivateUse1)` 的正确做法。
  该规则与 TODO-18 的契约说明是**互补**关系：契约告诉模型怎么做，规则阻止它改回去。

### 验证

- **回归底线**：`archive_tasks/` 全部 8 个含 `pybind11.cpp` 的样例仍为 **0 issue**。
- 对真实候选验证：`nonthinking_add_allowlist_10round_20260916` 的 `round_05`
  （`507035` 崩溃候选）由 `host_pointer_descriptor` 命中，
  而 `nonthinking_add_descfix_10round_20260916` 的 `round_04`（已采用规范写法的候选）不报错。
- 新增 3 项测试，其中 `test_host_container_used_for_its_own_purpose_is_not_flagged`
  专门守住「Host vector 仅用于本地计算时不得误报」。
- 全量测试：164 项中 163 项通过。唯一失败项仍为该环境既有失败，与本轮无关。

### 真实运行轨迹（`outputs/nonthinking_add_descfix_10round_20260916`）

| 轮 | decision | failure_stage | MERE / 状态 |
|---:|---|---|---|
| 1 | FAIL | `ascendc_source_validation` | `missing_local_include: 'tiling/op_tiling.h'` |
| 2 | FAIL | `ascendc_build` | Bisheng `not support bf16 type cast` |
| 3 | PARTIAL_KEEP | `ascendc_build` | 编译诊断减少 |
| 4 | PARTIAL_KEEP | `correctness` | **507035 消失**；MERE 7.55848，100% 元素错 |
| 5、6、7 | FAIL | `correctness` | **507035 回归**（改回 Host 指针） |
| 8 | PARTIAL_KEEP | `correctness` | MERE 0.999999 |
| 9 | FAIL | `correctness` | 再次失败 |
| 10 | — | — | — |

- 结果 `success=false`、`status=blocked`、`stop_reason=max_total_rounds`、10 次评测、
  484,703 tokens（prompt 406,049 / completion 78,654）。**未建立 baseline。**
- **阶段结论**：`source_validation` 与 `ascendc_build` 已非主阻塞（10 轮中 8 轮进入 `correctness`）。
  当前唯一阻塞是模型在「Host 指针」与「NPU 可见存储」之间来回改写，
  已由 `host_pointer_descriptor` 转为首轮确定性拦截。


## (⏳)TODO-18：描述符必须落在 NPU 可见存储（507035 的根因）（2026-09-16）

状态：已实施并验证，端到端复跑中。

### 分析

- TODO-17 的三项修复经真实运行验证**全部生效**：`torch.broadcast_shapes` 不再触发
  `static_validation` 失败（r2 起直接进入 `correctness`），`is_npu` 成员写法不再出现，
  DIAGNOSE 的 `falsifies` 缺口未再触发规划失败。**首轮即到达 correctness（此前是第 7 轮）。**
- 但 r2、r3、r4、r5、r7、r8、r9、r10 **连续 8 轮**全部以同一个运行时错误失败：
  `RuntimeError: ACL stream synchronize failed, error code:507035`
  （向量核异常 `vector core exception`）。模型 8 轮内未能定位。
- 定位根因：候选 `pybind11.cpp` 中
  ```cpp
  std::vector<int64_t> xStride(rank, 0);
  std::vector<int64_t> yStride(rank, 0);
  ...
  add_do(..., rank, outShape.data(), xStride.data(), yStride.data(), ...);
  ```
  **把 `std::vector::data()`（Host 地址）当作 device 描述符传给 Kernel**。
  Host 内存对 NPU 不可见，kernel 读到的是无效地址，直接触发向量核异常。
- 该规则**确实写在 `direct_launch_abi.md`**，而且点名的就是 `std::vector::data()`：
  “A Host stack address, `std::vector::data()`, or ordinary `const void*` is not converted
  into device-visible storage by a C++ cast.”
  但这两句在 `round_05/prompt.txt` 中出现 **0 次**——与该模块的 `__gm__` 一句同样没有进入提示词。
  属于同一缺陷类：**要求文档化在仓库里，但没有出现在模型当期实际读到的契约中**。
- 对照仓库内已知可用的样例（`archive_tasks/avg_pool3_d`），规范做法是三步：
  在 CPU 上按结构体填 `tiling` → `tilingCpu.to(at::kPrivateUse1)` 拷到 NPU → 传
  `tilingNpu.storage().data()`。模型从未被这样教导过。

### 实施方案与结果

- **契约**（`ascendc_multi_turn/prompts.py`，`RULES`）：新增一条，写明
  「传给 `_do` 的指针/描述符参数必须是 NPU 可见存储」，
  点名 `std::vector::data()`、栈上数组、CPU tensor 存储都是 Host 地址、不会因转换而 device 可见，
  并给出后果（`507035 vector core exception`，编译器不预警）与**三步规范做法**
  （CPU 结构体填充 → `.to(at::kPrivateUse1)` → 传 `tilingNpu.storage().data()` 并保证存活），
  同时要求把形状/stride 合并进一个描述符结构而不是各传一个 Host 指针。
  三步写法直接取自 `archive_tasks/avg_pool3_d/kernel/pybind11.cpp` 的实际实现，非人工归纳。
- **确定性校验**（`ascendc_multi_turn/source_validation.py`）：新增 issue code
  `host_pointer_descriptor`。在 `kernel/pybind11.cpp` 中收集以 `std::vector<...>` /
  `std::array<...>` 声明的局部变量，再检查 `*_do(...)` 调用中第 2 位起（跳过 `blockDim`
  与 `stream`）的实参；若实参形如 `<该变量>.data()`，报错并给出 `.to(at::kPrivateUse1)`
  的正确做法。该失败此前只能在设备侧以 `507035` 暴露，每轮消耗一次完整 NPU 评测；
  现在首轮即可拦截。

### 验证

- `python -m py_compile` 通过；Ruff `F/I` 对 `prompts.py`、`source_validation.py` 通过。
- **回归底线**：`archive_tasks/` 全部 8 个含 `pybind11.cpp` 的样例仍为 **0 issue**，无假阳性。
- 对真实候选验证：`nonthinking_add_allowlist_10round_20260916` 的 `round_05`
  （即 507035 崩溃的候选）由 `host_pointer_descriptor` 命中。
- 新增 3 项测试：`test_rejects_host_container_address_as_a_descriptor`、
  `test_accepts_an_npu_resident_descriptor`、
  `test_host_container_used_for_its_own_purpose_is_not_flagged`
  （最后一项守住「Host vector 仅用于本地计算时不得误报」）。
- 全量测试：164 项中 163 项通过。唯一失败项仍为该环境既有失败，与本轮无关。

### 真实运行轨迹（`outputs/nonthinking_add_allowlist_10round_20260916`）

| 轮 | decision | failure_stage | 说明 |
|---:|---|---|---|
| 1 | FAIL | `ascendc_build` | `platform_ascendc` 未声明（缺 include），r2 已修复 |
| 2 | PARTIAL_KEEP | `correctness` | **首轮到达 correctness**；此后全部为 `507035` |
| 3、4、5、7、8、9、10 | FAIL | `correctness` | `RuntimeError: ... error code:507035` |

- 结果 `success=false`、`status=blocked`、`stop_reason=max_total_rounds`、10 次评测、
  467,005 tokens（prompt 418,532 / completion 48,473）。**未建立 baseline。**
- **阶段结论**：契约层的障碍已基本清空——`static_validation`、`ascendc_build` 不再是主阻塞，
  首轮即进入 `correctness`。当前唯一阻塞是 Host 指针当 device 描述符（本轮修复尚未加载），
  属于单点、可定位、有规范写法的问题。


## (⏳)TODO-17：forward() 的 torch 闭集白名单与 is_npu 写法（2026-09-16）

状态：已实施并验证，端到端复跑中。

### 分析

- TODO-16 的 `__gm__` 修复经真实运行验证**部分生效**：`ascendc_build` 阶段的
  `expecting parameter ... to have memory qualifier __gm__` 不再出现。
  该轮 3 次评测后仍 `paused`，暴露两个新阻塞。
- **阻塞一（真实模型写法错误）**：`round_03` 的编译错误为
  `'const class at::Tensor' has no member named 'is_npu'`。
  `cannagent_host_abi.md` 写明了正确形式
  `torch_npu::utils::is_npu(const at::Tensor&)`（自由函数），
  该句在本轮提示词中出现 1 次，但契约**从未警示成员写法 `tensor.is_npu()` 不存在**；
  更值得注意的是校验器自己的 `ALLOWED_TENSOR_METHODS` **包含 `is_npu`**，
  对模型是反向暗示。模型因此写了成员形式。
- **阻塞二（真实模型写法错误）**：`round_01`、`round_02` 在 `static_validation` 失败——
  `forward()` 第 17 行 `torch.broadcast_shapes` 被判为
  “部分计算仍使用 PyTorch，需全部移入 AscendC kernel”。
- 阻塞二的根因仍是同一缺陷类：`validate_ascendc_impl.py` 用**闭集白名单**判定
  `forward()` 中的 torch 调用，白名单外一律视为计算；而契约只给了
  “不得用 torch/ATen 执行核心计算”“可以创建或 reshape tensor”这样的原则，
  **从未给出白名单边界**。模型无从知道 `torch.broadcast_shapes` 越界而
  `torch.empty` 合法——两者看起来都是“形状操作”。
- **阻塞三（TODO-17 的直接触发点）**：第 4 轮 planning 以
  `diagnosis plan item 1 must identify a falsified prior approach` 终止，整个 run `paused`。
  经核对，DIAGNOSE 响应中 `items[0].falsifies` 为 `[]`，而 `evidence_refs` 完好。
  `parse_plan(require_evidence=True)` 强制 `falsifies` 非空，但契约从未声明该要求。
  **这正是此前 G5 审计发现、随后按指令回退掉的那一条缺口**——回退后立刻以真实运行复现，
  代价是整个 run 在第 4 轮终止。

### 实施方案与结果

- **`forward()` torch 白名单写进契约**（`ascendc_multi_turn/prompts.py`，`RULES`）：
  按校验器的实际闭集列出允许的 13 个 `torch.*` 函数与 51 个 tensor 方法（按形状/元信息、
  布局、dtype/设备、标量取值、buffer 分配、原地标记分组），
  并点名典型越界写法 `torch.broadcast_shapes`、`torch.max/min/sum/stack/cat`、
  `.sum()/.mean()/.mul()/.add()`；
  明确要求「需要广播后的输出形状时不要在 Python 里用 torch 推导，放到 Host C++
  （`pybind11.cpp`）或 Kernel 侧按右对齐规则自行计算」。
- **`is_npu` 写法消歧**（`RULES`）：明确 NPU tensor 校验用已安装头文件的自由函数形式
  `torch_npu::utils::is_npu(tensor)`，并写明 `tensor.is_npu()` 成员写法不存在的诊断原文。
- **DIAGNOSE 的两项强制要求重新写进契约**（`DIAGNOSE_OUTPUT_CONTRACT` 尾段）：
  `items[].evidence_refs` 非空且 `line_excerpt` 为原文摘录；
  `items[].falsifies` 必须非空。并注明缺失会被判为无效响应、整轮终止。
  该条系 G5 审计发现、曾被回退，现因真实运行复现而重新实施。

### 验证

- `python -m py_compile` 通过；Ruff `F/I` 对 `prompts.py` 通过。
- 全量测试：161 项中 160 项通过。唯一失败项仍为该环境既有失败，与本轮无关。
- 白名单内容取自 `validate_ascendc_impl.py` 的 `ALLOWED_TORCH_FUNCS` 与
  `ALLOWED_TENSOR_METHODS` 实际取值，非人工归纳。

### 真实运行轨迹（`outputs/nonthinking_add_gmfix_10round_20260916`）

| 轮 | decision | failure_stage | 说明 |
|---:|---|---|---|
| 1 | FAIL | `static_validation` | `torch.broadcast_shapes` 越界 |
| 2 | FAIL | `static_validation` | 同上 |
| 3 | FAIL | `ascendc_build` | `'const class at::Tensor' has no member named 'is_npu'` |
| 4 | — | `llm_planning` | DIAGNOSE `falsifies` 为空 → 整轮 `paused` |

- 结果 `success=false`、`status=paused`、`stop_reason=planning_response_invalid`、
  3 次评测、123,656 tokens。**未建立 baseline。**
- **正面结论**：`__gm__` 振荡不再出现，`ascendc_build` 阶段的首要阻塞已解除。


## (⏳)TODO-16：Kernel 入口描述符形参必须带 __gm__（2026-09-16）

状态：已实施并验证。端到端复跑仍无正确算子，但数值首次真正收敛，见下方轨迹。

### 分析

- TODO-15 的契约修复（stream 必须取自运行时）经真实运行验证**生效**：
  第 7 轮起候选写出
  `void* stream = c10_npu::getCurrentNPUStream().stream(false);` 并附
  `TORCH_CHECK(stream != nullptr, ...)`，输出不再是全 0。
- 同一轮暴露出下一个阻塞：`__global__ __aicore__` Kernel 入口的描述符形参不合法。
  该阻塞占用了 **10 轮中的 6 轮**（r1–r6 全部 `ascendc_build` 失败），是最贵的一项。
- 失败形态是一次**振荡**：模型在两种错误写法之间来回改写——
  - `const int64_t (&outShape)[kMaxRank]`（引用数组）→
    `__global__ function parameter is not allowed to be a reference type`
  - `int64_t outShape[kMaxRank]`（缺 `__gm__`）→
    `expecting parameter outShape to have memory qualifier __gm__`
  两种写法各自只差一个限定符，但模型的 repair 记录把它们都标为 REPRESSION，
  于是反复重试同一族方案直到预算耗尽。
- 该要求确实写在 `direct_launch_abi.md`（"Kernel helpers that consume global-memory
  descriptors must preserve the `__gm__` address-space qualifier in their parameters"），
  但该句在 `round_06/prompt.txt` 中出现 **0 次**——模块的注入不保证覆盖到需要它的阶段。
  与 TODO-11~15 同类：**要求没有出现在模型当期实际读到的契约里**。

### 实施方案与结果

- **契约**（`ascendc_multi_turn/prompts.py`）：`RULES` 新增一条，给出**唯一正确形式**与两种错误写法的
  对应诊断，直接终止振荡：
  - 合法：`__global__ __aicore__ void k(GM_ADDR x, GM_ADDR out, int64_t rank, __gm__ int64_t* outShape, ...)`；
  - 明写「指针/数组形参若指向 global memory 必须带 `__gm__`，且不得写成引用」，
    并同时引用两条诊断原文，使模型能识别自己正处于振荡中。
- **确定性校验**（`ascendc_multi_turn/source_validation.py`）：新增 issue code
  `kernel_descriptor_missing_gm`。定位非 pybind11 源码中的 `__global__ __aicore__` 入口，
  解析其形参，对「指针 / 引用 / 数组」且既非 `GM_ADDR` 也无 `__gm__` 的形参报错，
  消息区分「引用」与「缺 `__gm__`」两种成因并给出正确写法。
  该阻塞此前只能在编译阶段暴露、每轮消耗一次完整编译。

### 验证

- 新增 3 项测试：`test_rejects_unqualified_kernel_descriptor_parameter`、
  `test_rejects_reference_kernel_descriptor_parameter`（振荡的另一半）、
  `test_accepts_a_gm_qualified_kernel_descriptor`。
- **回归底线**：`archive_tasks/` 全部 8 个含 `pybind11.cpp` 的样例仍为 **0 issue**，无假阳性。
- 对真实候选验证：`round_01`、`round_06` 由 `kernel_descriptor_missing_gm` 命中。
- 全量测试：161 项中 160 项通过（该数已扣除本轮回退掉的 G5 一致性测试）。
  唯一失败项仍为该环境既有失败，与本轮无关。Ruff `F/I` 对修改文件通过。

### 真实运行轨迹（`outputs/nonthinking_add_streamfix_10round_20260916`）

| 轮 | decision | failure_stage | 说明 |
|---:|---|---|---|
| 1–6 | FAIL / PARTIAL_KEEP | `ascendc_build` | 全部为 `__gm__` 振荡（本轮未含 TODO-16 修复） |
| 7 | PARTIAL_KEEP | `correctness` | **stream 修复生效**；MERE 0.999999 → **0.1836**，rel_mismatch_ratio 9.4% |
| 8 | PARTIAL_KEEP | `correctness` | — |
| 9 | PARTIAL_KEEP | `correctness` | 回退到 0.7031（改写引入回归） |
| 10 | PARTIAL_KEEP | `correctness` | **0.1101，rel_mismatch_ratio 2.3%——历次最好** |

- 结果 `success=false`、`status=blocked`、`stop_reason=max_total_rounds`、10 次评测、
  443,634 tokens（prompt 388,643 / completion 54,991）。**未建立 baseline。**
- 结论：**契约路径已被证明有效**。`stream` 契约修复把「输出全 0」变成「2.3% 元素不对」，
  数值第一次真正收敛；剩余阻塞是 6 轮被 `__gm__` 振荡吃掉，而该修复本轮尚未加载。
- 按用户指令，本轮**回退了 G5（契约↔执行一致性对照）本体**，仅保留 `__gm__` 相关改动。
  G5 审计过程发现的三条契约补漏（`absolute_include`、Host 指针解引用、DIAGNOSE `falsifies` 非空）
  一并回退，**未实施**；这三条是已知的契约缺口，后续如需可重新补上。


## (⏳)TODO-15：launch stream 必须取自运行时（禁止 nullptr）（2026-09-16）

状态：已实施并验证，端到端复跑中。

### 分析

- TODO-14 的定位反馈一经上线立刻见效：`round_04` 与 `round_06` 的评测字符串变为
  `rel_mismatch_ratio=1, first_mismatch(index=(0,), ref=-2.00922, cand=0)`，
  **输出恒为 0**——这是只靠 `MERE=0.999999` 永远看不出来的事实。
- 顺该线索检查候选：`kernel/pybind11.cpp` 第 117 行为
  `void* stream = nullptr;`，随后 `add_do(blockDim, stream, ...)`。
  Kernel 传入了空 stream。
- 对照仓库内**已知可用的** `archive_tasks/`：8 个含 `pybind11.cpp` 的样例**全部**使用
  `auto aclStream = c10_npu::getCurrentNPUStream().stream(false);`
  ——这是一个既有约定，但从未写进契约。
- 提示词不仅未声明，还主动埋了陷阱：`RULES` 与阶段契约给出的标准形式是
  `kernel<<<blockDim, nullptr, stream>>>`，其中**字面就含 `nullptr`**；
  模型若没有真实 stream，最自然的写法就是把它也填成 `nullptr`。
  实测提示词中要求取得真实 stream 的语句数为 **0**。
- 该失败形态的特征是「能编译、能启动、不报错、输出全 0」，
  编译器与确定性校验都无法发现，是本轮迭代中代价最高的一类问题。

### 实施方案与结果

- **G6a｜契约**（`ascendc_multi_turn/prompts.py`）：
  - `RULES` 新增：launch 的第三个 `<<<>>>` 参数必须是运行时取得的当前 NPU stream
    （例如 `c10_npu::getCurrentNPUStream().stream(false)`，以带匹配 runtime fact 的声明为准），
    **不得传 `nullptr`/`NULL`/`0`**；并明确
    `kernel<<<blockDim, nullptr, stream>>>` 中的 `nullptr` 是**第二个槽位**（类型/配置位），不是 stream。
  - 同时说明后果：stream 为 `nullptr` 时 Kernel 仍能编译启动但不写入输出，实测输出恒为 0。
  - `bootstrap_generation` 阶段契约补同一约束。
- **G6b｜确定性校验**（`ascendc_multi_turn/source_validation.py`）：
  新增 issue code `host_null_stream`。在 `kernel/pybind11.cpp` 中定位 `*_do(...)` 调用，
  按顶层逗号拆分实参（括号/尖括号/花括号感知），检查第 2 个实参：
  若是 `nullptr`/`NULL`/`0` 字面量，或是同文件内被赋值为空字面量的变量，则报错，
  并在消息中给出正确做法与后果。该检查让这类「静默全 0」在首轮即被拦截，
  而不是等到 NPU 跑完精度对比才发现。

### 验证

- 新增 3 项测试：`test_rejects_null_stream_passed_as_a_variable`、
  `test_rejects_a_null_literal_stream_argument`、
  `test_accepts_a_real_stream_and_ignores_the_launch_template_nullptr`
  （后者同时守住「`kernel<<<..., nullptr, stream>>>` 里的 nullptr 不得误报」）。
- 对两个真实候选验证：`round_04`、`round_06` 均由 `host_null_stream` 命中，
  且是该候选**唯一**的 source issue。
- 对 `archive_tasks/` 全部 8 个含 pybind11.cpp 的样例验证：**0 个报 issue**，无回归。
- 全量测试：158 项中 157 项通过。唯一失败项仍为该环境既有失败，与本轮无关。

### 真实运行轨迹（TODO-14 那一轮：`outputs/nonthinking_add_locfix_10round_20260916`）

| 轮 | decision | failure_stage |
|---:|---|---|
| 1 | FAIL | `ascendc_source_validation` |
| 2 | FAIL | `ascendc_build` |
| 3 | PARTIAL_KEEP | `ascendc_build` |
| 4 | PARTIAL_KEEP | `correctness` |
| 5 | FAIL | `ascendc_build` |
| 6 | PARTIAL_KEEP | `correctness` |
| 7 | PARTIAL_KEEP | `correctness` |
| 8 | FAIL | `ascendc_build` |
| 9 | PARTIAL_KEEP | `correctness` |
| 10 | FAIL | `ascendc_build` |

- 结果 `status=blocked`、`stop_reason=max_total_rounds`、10 次评测、478,784 tokens。
- **重要正面结论**：该轮**跑满了全部 10 轮预算而不是中途 `paused`**，
  说明 TODO-13 的 `parse_plan` 重标注修复生效——诊断轮不再因为标签写错而整轮终止。
- 负面结论：所有进入 correctness 的轮次 `MERE` 恒为 0.999999、`MARE=1`，
  模型始终没有修好「输出全 0」。结合 G6 的定位，根因即空 stream。


## (⏳)TODO-14：浮点精度失配的定位反馈（2026-09-16）

状态：已实施并验证，端到端复跑中。

### 分析

- TODO-13 的运行已经把算子做到**编译通过并在 NPU 上执行**（r4、r5、r7、r8 进入 correctness），
  但数值正确性停滞：`MERE` 从 0.999999 改善到 0.65625 后不再变化，`max_abs_diff` 恒为 4.19074。
- 检查反馈质量时发现关键缺口：`utils/verification_ascendc.py` 的
  `_tensor_diff_summary` 存在**路径不对称**——
  - **整数路径**一直报告 `first_mismatch(index=..., ref=..., cand=...)`，
  - **浮点路径**（Add 走这条）只报告
    `max_abs_diff / mean_abs_diff / MERE / MARE / threshold`，**没有任何索引或取值**。
- 后果：模型只知道“整体误差是多少”，不知道“错在哪”。
  `ascendc-precision-debug` 知识模块教的是 `binary-search-debug` 与 `data-comparison`，
  但这些方法都需要逐元素取值才能用，而 prompt 里没有。
- 该缺口可由现有数据量化：`MERE=0.65625 = 84/128`。
  用“128 个元素里前 84 个偏移 4.19”复现，得到的 `rel_mismatch_ratio` 正是 0.6562，
  说明真实失败是**少数元素正确、多数元素整体偏移**，属于可用定位信息直接指向的类型
  （tiling/分核区间或 tail 处理），而不是需要盲猜的精度问题。

### 实施方案与结果

- `utils/verification_ascendc.py`：
  - 新增 `_linear_index_to_coord(linear_index, shape)`，把扁平下标还原为多维坐标（含 0 维）。
  - 浮点路径在原有聚合统计之后追加有界定位信息：
    以与 MERE 相同的逐元素相对误差判据
    （`diff / (|ref| + 1e-12) > threshold`）计算命中比例与首个失配点，
    追加 `rel_mismatch_ratio=...` 与
    `first_mismatch(index=..., ref=..., cand=...)`。
  - 仅在存在失配时追加，全对时不产生额外输出；整数路径输出格式保持不变。

### 验证

- 新增 `tests/test_verification_diff_summary.py`（6 项）覆盖：
  浮点失配给出索引与 ref/cand 取值、全对不产生定位输出、0 维张量用空坐标、
  多维张量定位到正确坐标、同号 inf 相减产生的 nan 不破坏定位、整数路径输出不变。
- 用 `git stash push -- utils/verification_ascendc.py` 还原到修改前，
  6 项中 **4 项失败**（两个纯守卫项修改前后均通过），恢复后 6/6 通过。
- 全量测试：155 项中 154 项通过。唯一失败项仍为该环境既有失败，与本轮无关。
  Ruff `F/I` 对修改文件通过。


## (⏳)TODO-13：Host 编译单元边界（禁止 device 头进入 pybind11.cpp）（2026-09-16）

状态：已实施并验证。端到端仍未产出正确算子，但**首次做到编译通过并在 NPU 上执行**；
剩余阻塞为数值正确性，见下方轨迹。

### 分析

- TODO-11 修复后重跑（`outputs/nonthinking_add_contractfix_10round_20260916`），
  第 1 轮即产出正确的 `extern "C" void add_do(uint32_t blockDim, void* stream, ...)`
  与 `<<<blockDim, nullptr, stream>>>`，source validation 首次通过并进入编译，
  评测数从 1 提升到 4。**G1 由此得到真实运行验证。**
- 第 2–3 轮的编译失败暴露下一个「未声明要求」：
  `tikcpp/tikcfw/kernel_operator.h:18:10: fatal error: kernel_tpipe_impl.h: No such file or directory`。
- 该文件确实存在于 `asc/impl/basic_api/` 与 `ascendc/include/basic_api/impl/`，
  但**不在 tikcpp 目录下**——它需要 AscendC 的 device include 路径。
- `utils/build_ascendc.py` 证实了边界：`kernels`（device 目标）通过
  `ascendc_include_directories(...)` 获得 AscendC 路径；
  `pybind11_lib`（host 目标，第 119、141 行）只得到 `{kernel_dir}`、`TORCH_NPU_PATH/include`、
  `TORCH_PATH/include` 等，**没有任何 device include 路径**。
- 三轮候选的 `kernel/pybind11.cpp` 都写了 `#include "kernel_operator.h"`，
  动机是使用 `GM_ADDR`：声明为
  `extern "C" void add_do(uint32_t blockDim, void* stream, GM_ADDR x, GM_ADDR y, GM_ADDR out, ...)`。
  而 `GM_ADDR` 定义在 kernel 头中，于是 host 单元被动引入 device 头并必然编译失败。
- 提示词不仅未声明该约束，还**主动误导**：runtime facts 段以
  `### GM_ADDR [Level 2 Installed] ... Required include: \`#include "kernel_operator.h"\``
  的形式注入，模型照做即掉入陷阱。
- 仓库自身的正确写法在测试夹具中：`extern "C" void test_do(uint32_t blockDim, void *stream, uint8_t *x);`
  ——host 侧用 `uint8_t*` 而非 `GM_ADDR`。`archive_tasks/` 全部 10 个样例的 `pybind11.cpp`
  也都不包含任何 device 头，说明这是既有约定，只是从未写进契约。

### 实施方案与结果

- **G4a｜契约**（`ascendc_multi_turn/prompts.py`）：
  - `RULES` 新增：`kernel/pybind11.cpp` 是**纯 Host 编译单元**，不得 include
    `kernel_operator.h`、`kernel_tiling/...` 或任何 AscendC device 头；
    `GM_ADDR`、`__gm__`、`GlobalTensor`、`TPipe` 只允许出现在 kernel .cpp 中。
  - `RULES` 明确 host 侧 `_do` 指针参数写 `uint8_t*`/`const void*`/`void*`，
    调用点用 `reinterpret_cast<uint8_t*>(tensor.data_ptr())` 传入；device 侧定义可按需用 `GM_ADDR`，
    两侧以个数/顺序/宽度一致为准。
  - `bootstrap_generation` 阶段契约补同一约束。
- **G4b｜确定性校验**（`ascendc_multi_turn/source_validation.py`）：
  新增 `_DEVICE_ONLY_HEADERS` 与 issue code `host_includes_device_header`，
  仅对 `kernel/pybind11.cpp` 生效；报错信息同时给出正确做法（`uint8_t*`）。
  该约束此前只能在编译阶段暴露，现在首轮即可确定性拦截。
- **G4c｜DIAGNOSE 判定语义**（`ascendc_multi_turn/prompts.py`）：
  - `DIAGNOSE_OUTPUT_CONTRACT` 尾段改写，明确 `evidence_status` 描述的是
    **能否在不猜测的前提下给出可执行修复项**，而非对失败原因有多确定。
  - `parse_plan` 增加重标注：`allow_insufficient` 且 `evidence_status="sufficient"`
    但 `items` 为空且 `unknowns` 非空时，按 `insufficient` 处理，
    使 Runner 记录可恢复的 `blocked` 而不是硬 `paused`。
    仅在 DIAGNOSE 路径生效；Planner 路径保持严格。

### 验证

- 新增 4 项测试：`test_host_translation_unit_must_not_include_device_headers`、
  `test_kernel_unit_may_include_device_headers`、`test_diagnose_zero_items_with_unknowns_is_relabelled_insufficient`
  及契约守卫 `test_every_plan_contract_shows_the_element_shape_of_structured_fields`、
  `test_diagnose_contract_is_not_derived_from_the_planner_tail`。
- 用三个真实候选验证 G4b：全部由 `host_includes_device_header` 在评测前确定性拦截。
- 用真实失败响应验证 G4c：`diagnose_v05`（标注 `sufficient`、`items=[]`、`unknowns=3`）
  在新代码下解析为 `insufficient`；同一响应在 `allow_insufficient=False` 下仍被拒。
- 全量测试：149 项中 148 项通过。唯一失败项
  `test_declaration_ranking_rejects_comment_hits_and_keeps_count_overload`
  为该环境既有失败，与本轮无关。

### 真实运行轨迹（`outputs/nonthinking_add_hostfix_10round_20260916`）

| 轮 | decision | failure_stage | 说明 |
|---:|---|---|---|
| 1 | FAIL | `ascendc_source_validation` | `cuda_tensor_check`：用了 `.is_cuda()`（`cannagent_host_abi.md` 已列为负面事实，属真实模型错误） |
| 2 | FAIL | `ascendc_build` | 编译错误 |
| 3 | PARTIAL_KEEP | `ascendc_build` | `at::Tensor` 无 `is_npu` 成员（应为 `torch_npu::utils::is_npu(tensor)`），编译诊断减少 |
| 4 | PARTIAL_KEEP | `correctness` | **首次编译通过并在 NPU 执行**；smoke MERE=0.999999 |
| 5 | PARTIAL_KEEP | `correctness` | MERE 0.999999 → 0.65625 |
| 6 | FAIL | `ascendc_build` | `-Wc++11-narrowing`：`unsigned long` → `uint32_t` |
| 7 | FAIL | `correctness` | 修回编译，MERE 与 r5 相同（0.65625） |
| 8 | FAIL | `correctness` | 未改善 |
| 9 | FAIL | `ascendc_build` | 编译错误 |
| 10 | — | `llm_planning` | **TODO-12/G4c 已修复的标签歧义**：模型返回 `sufficient` + 零 item + 2 条 unknowns，旧代码抛错 `planning response must contain 1 items`，整轮 `paused` |

- 全程 389,355 tokens（prompt 349,715 / completion 39,640）。
- 结论：**source validation 与编译阶段的问题已由契约修复清除**；
  `GM_ADDR` 陷阱不再复现（`host_includes_device_header` 从未在第 1 轮触发）。
  剩余阻塞是纯粹的数值正确性，且已停滞：MERE 在 0.65625 不动，`max_abs_diff` 恒为 4.19074。
- 第 10 轮的死因已定位为标签歧义——该缺陷在本次运行**启动之后**才修复，进程内未加载，
  已用真实响应离线验证新代码会将其转为可恢复的 `blocked`。


## (⏳)TODO-12：规划契约元素 schema 全覆盖与执行反馈可用性（2026-09-16）

状态：已实施并验证。

### 分析

- TODO-10 修复后重跑（`outputs/nonthinking_add_contractfix_10round_20260916` 的前身
  `outputs/nonthinking_add_zh_prompt_g1g3_10round_20260916`）：第 1 轮 planning 成功、
  4 轮评测完成，第 2 轮 planning 以
  `planning response unknowns must contain complete unknown objects` 失败。
- 根因与 TODO-10 缺陷 A **同类且只差一个字段**：`PLAN_OUTPUT_CONTRACT` 与
  `DIAGNOSE_OUTPUT_CONTRACT` 仍把 `unknowns` 渲染为裸空数组 `[]`，不展示元素形状
  （要求 `{question, required_evidence}`）；TODO-10 只修了 `INITIAL_PLAN_OUTPUT_CONTRACT` 一份。
- 该机制得到一次干净的自然实验印证：同一轮 `PLAN_OUTPUT_CONTRACT` 中
  `observations`、`ruled_out` 展示了解释 schema，模型输出即为合法 dict；
  `unknowns` 未展示 schema，模型即输出字符串数组并被拒。
  **契约把结构化字段渲染成 `[]` 的地方，模型就会写错元素形状。**
- 同时消除 `DIAGNOSE_OUTPUT_CONTRACT` 由 `PLAN_OUTPUT_CONTRACT.replace()` 派生的脆弱性
  （TODO-11 的 P1）：正文措辞一旦改动，replace 静默失配会把 Planner 尾段发给 DIAGNOSE。

### 实施方案与结果

- 新增 `_PLAN_BODY` 共享正文与两个独立尾段，`PLAN_OUTPUT_CONTRACT` 与
  `DIAGNOSE_OUTPUT_CONTRACT` 由显式拼接生成，移除 `str.replace()` 派生。
- `_PLAN_BODY` 为 `unknowns` 补上完整的 `{question, required_evidence}` 对象示例，
  并追加与 INITIAL 契约一致的禁令：结构化字段的元素必须是对象、不得写成字符串数组、
  无内容时可留空 `[]`。
- `INITIAL_PLAN_OUTPUT_CONTRACT` 补充说明 `items[].falsifies` 是字符串数组，
  bootstrap 阶段无已否定方案时保持 `[]`。

### 验证

- 新增守卫测试 `test_every_plan_contract_shows_the_element_shape_of_structured_fields`：
  解析三个契约的 JSON 示例，断言 `observations`/`ruled_out`/`unknowns`/`items[].evidence_refs`
  都必须给出非空的对象示例。对修复前的契约文本运行该守卫会失败
  （`unknowns 渲染为 [] 或非对象示例`），对三个修复后契约全部通过。
- 新增 `test_diagnose_contract_is_not_derived_from_the_planner_tail`：
  断言两个契约的尾段互不出现对方独有措辞。
- 该守卫是**针对整类缺陷**的，而不是针对某个字段：后续任何契约新增结构化字段时，
  只要渲染成 `[]` 就会被测试拦住。


## (⏳)TODO-11：Host wrapper ABI 契约对齐与校验器诊断可用性（2026-09-16）

状态：G1、G2、G3 已实施并验证。端到端仍未产出正确算子，**未修复项已定位并在下方记录**
（`unknowns` 元素 schema 缺失），按指令停止继续修改。

### 分析

- 真实运行 `benchmarks/NPUKernelBench/level1/3_Add.py` 的第 1–3 轮全部在
  `ascendc_source_validation` 失败，长期被归因为“生成质量问题”。逐候选核对后确认：
  **主要原因是第三个「校验器强制、契约从未声明」缺陷**，与前两个
  （TODO-10 的 `observations` 元素 schema、diagnose 契约错配）同类。
- `source_validation.py` 的 `_DO_DECLARATION` 与 `_DO_DEFINITION` 硬性要求返回类型为
  `void`：`r'extern\s+"C"\s+void\s+([A-Za-z_]\w*_do(?:_\w+)?)...'`。
  而提示词与所有知识模块一律使用通配写法 `extern "C" *_do`，
  `extern "C" void` 在 `round_03/prompt.txt` 中出现 **0 次**。唯一写明 `void` 的地方是
  测试夹具 `tests/test_ascendc_multi_turn.py` 的 `_write_valid_launch_fixture`，它不是提示词。
- 三轮候选实际写出的都是 `extern "C" int <name>_do(...)`
  （`add_do` / `add_alpha_do` / `add_do`），全部合理但全部不匹配正则，
  因此 `declarations` 为空，报出的是
  `missing_host_wrapper_declaration: pybind11.cpp must declare and call at least one extern "C" *_do host launch wrapper`
  —— 而候选**确实声明并调用了**该 wrapper。报错描述的失败原因与真实原因无关。
- 单点验证：把 `round_03` 候选**仅**把 `int` 改成 `void`（其余一字不动），
  `validate_source_tree` 从 1 个 issue 变为 **0 个 issue**。该候选结构完整，此前是纯粹的误杀。
- 遮蔽效应：`missing_host_wrapper_declaration` 只在 `declarations` 为空时触发，
  而定义、签名一致性、Kernel launch、指针解引用、是否被调用这 5 项检查全部写在
  `for name in declarations.items():` 循环内。`declarations` 为空 → 这 5 项一次都不执行。
  实测：仅修 `void` 后，第 1、2 轮才**新出现** `missing_kernel_launch`。
  这是循环不收敛的直接原因——每轮只暴露一个被遮蔽的问题，且暴露的是假问题。
- 需要区分真实模型错误：第 2 轮的 wrapper 在 Host 侧直接实例化 Kernel 类
  （`AddAlphaKernel<float,float> kernel; kernel.Init(...); kernel.Process();`）
  并用 `(void)blockDim; (void)stream;` 丢弃 launch 参数，全部文件 `<<<` 出现 0 次，
  Kernel 不会在 NPU 执行。这类错误是真实的，只是此前被假错误遮蔽。
- 对照证据：`aclrtGetCurrentStream` 作为负面事实写在 `cannagent_host_abi.md` 中，
  模型第 3 轮即改对。说明「写进契约 → 模型遵守」链路有效，缺的是覆盖率。

### 实施方案与结果

- **G1｜把精确 ABI 写进契约**（`ascendc_multi_turn/prompts.py`）：
  - `RULES` 改为给出精确形式
    `extern "C" void <name>_do(uint32_t blockDim, void* stream, <其余参数...>)`，
    并明写「返回类型必须是 void，不得返回状态码或错误码」、
    「wrapper 定义与 __aicore__ Kernel 同文件，参数个数/顺序/类型必须与声明完全一致」。
  - `bootstrap_generation` 阶段契约补充同一形式，并显式禁止第 2 轮暴露的失败模式：
    「wrapper 是 Host 函数：不得在 Host 上直接实例化 Kernel 类、调用其 Init/Process 方法，
    也不得用 `(void)blockDim; (void)stream;` 丢弃 launch 参数」。
- **G2｜让报错自解释**（`ascendc_multi_turn/source_validation.py`）：
  - 新增 issue code `host_wrapper_return_type`。当存在 `*_do` wrapper 但返回类型不是 `void` 时，
    报出实际签名与期望签名，例如
    ``add_do is declared as `extern "C" int add_do(...)`; the project ABI requires `extern "C" void add_do(...)` — the wrapper return type must be void``。
  - `missing_host_wrapper_declaration` 文案补充期望形式
    ``extern "C" void <name>_do(uint32_t blockDim, void* stream, ...)``，
    不再只说 “必须声明并调用”。
- **G3｜解除遮蔽**（`ascendc_multi_turn/source_validation.py`）：
  - 新增仅接受任意返回类型的宽松正则 `_LOOSE_DO_DECLARATION` / `_LOOSE_DO_DEFINITION`，
    **只在严格匹配失败时启用**，参数部分用 `[^{}]*?` 以避免跨越函数体误匹配。
  - 该兜底路径下仍运行定义、签名一致性、Kernel launch、指针解引用与调用检查，
    使一轮反馈即包含全部问题，而不是逐轮挤牙膏。
  - 严格路径行为完全不变：现有通过用例（含 `archive_tasks` 全量零 issue 校验）不受影响。

### 验证

- 三个真实候选重新校验，反馈一次到位：

  | 轮 | 修改前 | 修改后 |
  |---|---|---|
  | 1 | `unsupported_stream_api`, `missing_host_wrapper_declaration`（误导） | 同上 + `host_wrapper_return_type` + `missing_kernel_launch` |
  | 2 | 同上 | 同上 + `host_wrapper_return_type` + `missing_kernel_launch` |
  | 3 | `missing_host_wrapper_declaration`（误导） | 仅 `host_wrapper_return_type`，并给出期望签名 |

- 新增 3 项测试于 `tests/test_ascendc_source_validation.py`：
  `test_names_the_return_type_instead_of_claiming_no_wrapper_was_declared`、
  `test_wrong_return_type_does_not_hide_the_remaining_wrapper_checks`、
  `test_accepts_the_documented_wrapper_signature`。
  用 `git stash push -- ascendc_multi_turn/source_validation.py` 还原到修改前，
  前两项**失败**（`AssertionError: 'host_wrapper_return_type' not found in {'missing_host_wrapper_declaration'}`），
  第三项为正向守卫，修改前后均通过。恢复后 13 项全部通过。
- 全量测试：144 项中 143 项通过。唯一失败项
  `test_declaration_ranking_rejects_comment_hits_and_keeps_count_overload`
  是该环境下的既有失败，与本轮改动无关（已验证：只导入 `knowledge_probe` 与
  `runtime_knowledge`，不导入被改文件）。结构知识测试 25 项全部通过。
  Ruff `F/I` 对修改文件通过。
- **G1 由真实运行验证有效**：以同等端到端命令重跑
  （`outputs/nonthinking_add_zh_prompt_g1g3_10round_20260916`，参数与前一 run 一致），
  第 1 轮生成物中已出现
  `extern "C" void add_do(uint32_t blockDim, void* stream, ...)` 与
  `<kernel><<<blockDim, nullptr, stream>>>(...)`，且该轮提示词中出现精确 ABI 要求 2 次。
  `void` 缺陷不再复现。

### 未修复项与停止原因

- 该次端到端运行仍**未产出正确算子**：1 轮评测后即
  `stop_reason=planning_response_invalid`，`status=paused`，未建立 baseline，
  累计 52,570 tokens（prompt 43,623 / completion 8,947，`prompt_cache_hit_tokens=0`）。
- **第 1 轮**失败原因是**真实模型错误**，与本次改动无关：
  `kernel/pybind11.cpp:10: error[missing_local_include]: local include
  'tiling/platform/platform_ascendc.h' does not exist`。
  候选引用了当前 CANN 构建 include 根下不存在的平台头文件路径。
- **第 2 轮**新阻塞点：
  `planning failed: planning response unknowns must contain complete unknown objects`。
  根因与 TODO-10 的缺陷 A **完全同类，只差一个字段**：
  `PLAN_OUTPUT_CONTRACT` 与 `DIAGNOSE_OUTPUT_CONTRACT` 仍把 `unknowns` 渲染为裸空数组 `[]`，
  不展示元素形状（要求 `{question, required_evidence}`）；TODO-10 只修复了
  `INITIAL_PLAN_OUTPUT_CONTRACT` 一份。
- 该机制得到一次干净的自然实验印证：本轮 `PLAN_OUTPUT_CONTRACT` 中
  `observations`、`ruled_out` 有元素 schema，模型输出正确（均为 dict）；
  `unknowns` 无 schema，模型即输出字符串数组并被拒。
  **契约渲染为 `[]` 的字段，模型就会写错形状**——三个契约中凡如此者均需修复。
- 按指令停止继续修改，未在本次改动 `unknowns` schema，也未再次重跑。
- 说明：本轮按要求只更新 `CHANGELOG.md`，**未同步更新 `/mnt/workspace/流程.md`**，
  该文档中 1.5.1（`RULES` 原文）、1.4.4 与 2.5 节的内容已与当前代码不一致。


## (⏳)TODO-10：bootstrap / diagnose 规划契约与校验器对齐（2026-09-16）

状态：缺陷 A、缺陷 B 均已修复，并各自有回归测试与「修复前必失败」验证。

### 分析

- 真实运行 `benchmarks/NPUKernelBench/level1/3_Add.py`（10 轮 bootstrap，
  `--knowledge-source hybrid --knowledge-input-mode full-selected --generator-thinking disabled`）
  连续暴露两个「输出契约与 `parse_plan` 校验器不一致」缺陷，二者均非本轮工作区改动引入。
- 缺陷 A：`INITIAL_PLAN_OUTPUT_CONTRACT` 把 `observations`、`ruled_out`、`unknowns`、
  `items[].evidence_refs` 渲染为裸空数组 `[]`，不展示元素形状；而 `parse_plan` 对这四个字段的
  元素校验是无条件执行的，普通 Planner 路径同样适用。相比之下 `PLAN_OUTPUT_CONTRACT`
  给出了完整对象 schema，只有 bootstrap 使用的这一份缺失。
- 缺陷 A 的后果：模型返回字符串数组，17 处违规（`observations` 7/7、`ruled_out` 5/5、
  `unknowns` 2/2、`evidence_refs` 3/3），`parse_plan` 只报第一个即抛错。模型给出的内容
  在语义上完整可用，只是 JSON 形状不符；把字符串按语义包成对象后校验立即通过。
- 缺陷 B：`runner.py:722` 用 `initial = current is None and previous is None` 判断
  「是否首次规划」，把「工作区没有实现」等同于「首次规划」。bootstrap 连续 FAIL 时
  SETTLE 走 FAIL 分支（`runner.py:1744-1746`）把 `current` 置回空的 base bundle、
  `previous` 置回 `None`，因此从未产生被接受候选时 `initial` 恒为 `True`。
- 缺陷 B 的后果：连续 3 次 FAIL 触发 DIAGNOSE 后，`call_type` 是 `diagnose`、
  system prompt 是失败诊断器，但 user 仍收到 bootstrap 蓝图措辞和
  `INITIAL_PLAN_OUTPUT_CONTRACT`；同时 `require_evidence=True` 施加 diagnose 专属校验
  （`prompts.py:726-727` 要求 `falsifies` 非空），而契约示例写的是 `"falsifies": []`。
  三条消息互相冲突，模型正确遵守契约反而被判无效。
- 规划失败会直接终止整轮运行（`runner.py:1520-1536` → `paused` + `break`）。
  `--llm-transient-retries` 只覆盖传输异常与空内容，`parse_plan` 在 `_call_llm`
  重试循环之外调用，schema 失败不重试；`--resume` 因 `response.txt` 解析失败会重新调用 LLM，
  只是重新采样，不是修复。

### 实施方案与结果

- 补全 `INITIAL_PLAN_OUTPUT_CONTRACT`（`ascendc_multi_turn/prompts.py:84-125`）：
  `observations`、`ruled_out`、`unknowns`、`items[].evidence_refs` 改为展示完整对象 schema；
  追加显式禁令「元素必须是对象，键名与上面完全一致；不得写成字符串数组，缺键或空字符串
  都会被判定为无效响应」；保留并明确「bootstrap 阶段尚无评测证据时这些数组可以留空 `[]`」。
- 为保留已落盘的高质量 bootstrap 蓝图，未重新调用 LLM：将
  `outputs/nonthinking_add_hybrid_zh_prompt_10round_20260916/.llm_state/planner_v01/response.txt`
  中四组字符串按语义重塑为对象，模型原文一律保留在 `line_excerpt`／`hypothesis`／`question`，
  仅补出契约要求的容器键；`ruled_out[].reason` 与 `unknowns[].required_evidence` 是按内容
  合成的容器字段，`response.txt` 原文备份为同目录 `response.original.txt`。
- 以完全相同的参数加 `--resume` 续跑；续跑复用已落盘响应，未产生新的 LLM 调用。
- 缺陷 B 通过**解耦规划模式与运行状态**修复，不再用「工作区是否为空」推断规划类型：
  - `ascendc_multi_turn/prompts.py` 新增三个显式模式常量 `PLANNING_MODE_BLUEPRINT`、
    `PLANNING_MODE_PLAN`、`PLANNING_MODE_DIAGNOSE` 与 `PLANNING_MODES`。
  - `build_plan_prompt` 的 `initial: bool` 与 `diagnosis_required: bool` 两个入参被
    单个 `planning_mode: str` 取代；未知模式直接 `ValueError`。三种模式的
    `purpose`、`execution_rules`、`output_contract` 一一对应，契约与前言不可能再错配。
  - `runner.MultiTurnRunner` 新增 `_planning_mode(diagnose_required, evaluations_completed)`
    静态方法：DIAGNOSE 优先；其次依据本轮之前的累计评测次数判断——`evaluations_completed == 0`
    才是蓝图模式，否则为证据驱动的 PLAN。判据只取编排计数器，不读工作区或历史结果。
  - `_create_plan` 改为只接收 `planning_mode`，并由它派生 `diagnosis_required`
    （`planning_mode == PLANNING_MODE_DIAGNOSE`），使 `call_type`、`require_evidence`、
    `allow_insufficient`、输出契约、`_save_plan` 的 `origin` 全部来自同一个判断。
  - 规划提示的 `repair_state` 注入条件由 `not initial` 改为
    `planning_mode != PLANNING_MODE_BLUEPRINT`，DIAGNOSE 现在能收到开放错误与失败方案族。
- 新增回归测试 `RunnerTests.test_diagnose_after_repeated_bootstrap_failures_gets_the_diagnose_contract`
  与测试替身 `RecordingDiagnosisProvider`、`RepeatingSourceValidationFailureEvaluator`。
  后者复刻真实 Add 运行的失败形状：`ascendc_source_validation` 在 `_stage_rank` 中排位 0，
  首轮即 `INITIAL_REJECT`，于是每轮都回滚到空 base，`current`/`previous` 恒为空——
  正是触发缺陷 B 的状态。测试断言 DIAGNOSE 收到的提示必须含 DIAGNOSE 契约与相应前言、
  不得含蓝图契约或蓝图前言，同时第 1 轮 planner 仍必须是蓝图。
- 本节同时补充了 `/mnt/workspace/流程.md`：新增 2.5 节记录两个缺陷的完整证据链与验证，
  更新 1.4.4、2.3.5、2.4 与第四部分问题清单（新增 P11、P12，并修正 P10 的行号引用）。

### 验证

- `python -m py_compile ascendc_multi_turn/prompts.py`：修改前通过；缺陷 B 修复后对
  `prompts.py` 与 `runner.py` 一并通过。Ruff `F/I` 检查对
  `prompts.py`、`runner.py`、`tests/test_ascendc_multi_turn.py` 全部通过。
- 缺陷 B 的关键验证是**修复前必失败**：用
  `git stash push -- ascendc_multi_turn/prompts.py ascendc_multi_turn/runner.py`
  临时还原到修复前，新回归测试以如下方式失败——
  `AssertionError: '返回恰好一个完整 baseline item' unexpectedly found in '# AscendC bootstrap 规划\n\n在生成任何源码前，用 AscendC 术语给出一个完整实现蓝图。…'`，
  即 DIAGNOSE 调用收到的是 bootstrap 蓝图前言与含 `"falsifies": []` 的 INITIAL 契约，
  与生产环境第 4 轮的终止原因一致。恢复修复后同一测试通过。
- 针对 `parse_plan` 分别验证三种输入：符合对象 schema 且非空 → 通过；四个数组全空 → 通过
  （新契约明确允许）；字符串数组（本次事故形状）→ 仍按预期拒绝。确认修复只放宽了
  契约表述，未放宽校验强度。
- 重塑后的 `planner_v01/response.txt` 经 `parse_plan(min_items=1, max_items=1,
  require_evidence=False, allow_insufficient=False)` 通过，
  解析出 `evidence_status=sufficient`、`items=[bootstrap-1]`、
  7 条 `observations`／5 条 `ruled_out`／2 条 `unknowns`。
- 续跑证据：`planner_v01/result.json` 写入时间晚于 `response.txt`，且目录内无新增
  `response_retry_*.txt`；`response.original.txt` 保持原始 `20:59:47` 时间戳，
  证明规划未重新调用 LLM 而是复用了落盘响应。
- 续跑结果：第 1 轮规划被成功复用，流水线进入 EVAL 并连续完成 3 轮评测，
  缺陷 A 的修复已由真实运行验证；第 4 轮因**缺陷 B** 终止，
  `stop_reason=planning_response_invalid`，
  `message=planning failed: diagnosis plan item 1 must identify a falsified prior approach`。
  本轮未建立 baseline，不能据此宣称编译率或正确率改善。
- 第 1–3 轮失败原因与本次修复无关：三轮均在 `ascendc_source_validation` 失败，
  第 1–2 轮为 `unsupported_stream_api`（`aclrtGetCurrentStream`）叠加
  `missing_host_wrapper_declaration`，第 3 轮流 API 已修正但仍缺
  `extern "C" *_do` host wrapper。这属于生成侧收敛问题，修复规划契约不会改变该结果。
- 运行仓库测试套件（`python -m unittest discover -s tests -p 'test_*.py'`）：141 项中 140 项通过
  （缺陷 B 修复前为 140 项中 139 项通过，新增 1 项即本次回归测试）。
  唯一失败项 `test_declaration_ranking_rejects_comment_hits_and_keeps_count_overload`
  与本轮改动无关：它只导入 `knowledge_probe` 与 `runtime_knowledge`，不导入 `prompts.py`；
  且用 `git stash push -- ascendc_multi_turn/prompts.py` 临时还原到 HEAD 后重跑，
  该测试以完全相同的方式失败，属该环境下的既有失败（断言
  `Formula: Tanh` 注释未被过滤，与真实 CANN 头文件内容相关）。
- 本轮改动仅涉及一个 Prompt 常量：`git diff --stat` 为 1 个文件、13 行新增、4 行删除，
  未触及校验逻辑、状态机、路由或评测流程。
- 未在修复后重新调用真实 LLM 验证缺陷 A 的修复是否稳定降低 `planning_response_invalid`
  发生率；当前只有一次续跑证据（第 1 轮规划成功复用并进入 EVAL）。
  测试中的 mock 计划响应始终按 `falsifies: [] if initial else [...]` 构造，
  不会同时出现 `initial=True` 与 `diagnosis_required=True`，因此现有测试无法覆盖缺陷 B，
  修复缺陷 B 时应补充该组合的回归测试。


### 分析

- 当前核心缺陷不是 Prompt 太短，而是 Planner、DIAGNOSE、Generator、knowledge router 共用同一句 system prompt，
  导致角色边界只存在于 user prompt；关联 P-21，角色分离的代码问题状态为“已解决”。
- `repair_policy` 已能区分初始生成、编译修复、运行时修复、正确性修复和优化，但此前只有编译约束完整进入 Generator，
  其他阶段仍使用泛化的修复或优化指令；阶段约束已接通，尚无真实模型轨迹验证服从效果，问题状态为“部分解决”。
- DIAGNOSE 此前无论证据是否充分都必须返回一个 item，真实轨迹因此会在只确认 case 启动或进程信号时猜测同步、别名等源码根因；
  零 item 阻塞已由状态机测试覆盖，但尚无真实轨迹复验，问题状态为“部分解决”。
- 知识只按 `knowledge_module_id + section_id` 去重，同一正文可通过不同 ID 重复进入 Prompt；正文哈希去重已由回归测试覆盖，
  但尚未重跑真实 Add 轨迹统计 token 降幅，关联 P-26，问题状态为“部分解决”。
- `/mnt/workspace/prompt modify.md` 只作为调研输入，本次未修改该文件。Generator 仍返回完整文件 bundle，
  `analysis` 只说明目标证据、实际改动和保留不变量；diff 继续由 Runner 在候选生成后计算并留档。

### 实施方案与结果

- 为 `planner`、`diagnose`、`generator` 和 `knowledge_router` 建立独立中文 system prompt 与稳定 ID；
  `generator_retry` 复用 Generator，未知 call type 直接失败。Planner、Generator、知识路由、截断重试和会渲染到 Prompt 的
  项目 mapping 控制文字均改为中文；JSON key、API 名、源码、诊断、`doc_id` 和知识原文保持不变。
- 计划响应扩展为 `evidence_status`、`observations`、`ruled_out`、`unknowns`、`diagnosis`、`items`。
  Planner 必须给出一个 item；DIAGNOSE 证据充分时给出一个 item，证据不足时必须给出非空 `unknowns` 和零 item。
- DIAGNOSE 零 item 时 Runner 在 Generator 和 evaluator 前进入可恢复 `blocked`，记录
  `stop_reason=diagnosis_evidence_insufficient`、`pending_phase=DIAGNOSE` 和 `blocked_detail`；不生成候选、
  不增加候选评测次数，也不消耗 Generator 或候选预算，`--resume` 使用新的 plan version 继续诊断。
- Generator 根据 `repair_policy` 注入 `bootstrap_generation`、`compile_repair`、`runtime_repair`、
  `correctness_repair`、`performance_tuning` 或 `optimization` 契约，不再使用泛化的 “Repair or optimize”。
  紧凑评测增加首个失败或未完成 case、case 数和有界性能字段。
- 删除两个 builder 中无效的 `full_input` 参数；`parse_plan` 默认改为单 item。输入模式继续只由 selector 和 budget 控制。
- 知识 section 提取后对规范化正文计算 SHA-256；重复正文只保留首个高优先级来源，其他 ID 以 `alias_of` 写入 selection trace。
  `bounded` 按完整知识模块原子跳过，`full_selected` 保留完整选择；`rendered_skill_ids` 只记录实际进入 Prompt 的模块。
- 复用现有 operator-family 路由，确定性增加 `shape_regime` 和 `risk_flags`，没有新增 LLM 调用。
  当前 structured pattern 缺少硬件验证和 shape 元数据，因此不作为 exemplar 注入。

### 验证

- 全部仓库测试通过：165 项。新增覆盖角色 system prompt、计划 schema 交叉约束、DIAGNOSE 零 item 阻塞及恢复、
  Generator 六类阶段契约、紧凑 case/性能证据、正文哈希去重、原子预算、实际渲染 ID、确定性 shape/risk
  路由和未验证 exemplar 拒绝。
- 修改的 Python 文件通过 `python -m py_compile`；Ruff `F/I` 检查通过；`skill_mapping.yaml` 通过 JSON 解析。
- 本轮未调用真实 LLM 或 NPU，不能据此宣称首次编译率、完整正确率、token 或 wall time 已改善。
- `git diff --check` 对本轮文件通过；工作树中用户已有的 `AGENTS.md` 末尾空行仍会被完整工作树检查报告，未擅自覆盖该修改。

### 暂未采纳但正确的后续 TODO

- TODO-14：将现有知识类型和阶段 mapping 迁移为显式 L0～L5 分类；前置条件是迁移不会重复注入或削弱当前确定性路由。
- TODO-15：评估 2～4 个结构不同的初始 seed 或 population；必须单独定义额外模型成本、候选预算和公平选择规则。
- TODO-16：把当前 run-local repair experience 扩展为跨任务、经硬件验证的性能经验库，记录适用条件、禁止条件、
  hardware、CANN version、cases 和超过噪声阈值的性能证据。
- TODO-17：补充 core utilization、搬运与计算占比、UB 使用和热点等稳定 profiler 信号；在此之前性能 Prompt
  只能使用实际已有的 latency、operator 和 memory 数据。
- TODO-18：建设带 operator family、shape regime 和真实硬件验证元数据的 L5 exemplar 集；不得把普通教程、
  structured pattern 描述或未验证候选升级为 exemplar。
- 不新增模式识别 LLM：现有规则路由可以直接扩展；不增加冗长通用自检清单，因为其不可验证且会扩大 token；
  不采用 DSL 加四次 lowering，因为这属于新的生成架构而非当前已证实缺陷的修复。

## 2026-09-16

### 第一项：Prompt 中文化评估与执行决策

- 本轮没有直接把 Planner、DIAGNOSE 和 Generator Prompt 全量翻译成中文。现有真实轨迹不能证明语言是 Add
  失败根因；AscendC API、C++ 类型、编译诊断和上游技术材料以英文原文出现，机械翻译会增加术语漂移并削弱精确匹配。
- 本轮先修改 Prompt 的信息结构和约束执行：加入评测门、profile/case 进展、接受基线、接口契约、结构化失败方法、
  单项假设及验伪条件，同时继续完整保留源码、benchmark 和用例。
- 关联问题：Prompt 中文化收益未知。验证证据：本轮真实 Add 已在英文 Prompt 下暴露出可定位的 Host 指针非法解引用，
  但未形成中英对照数据。问题状态：暂缓，不能宣称中文或英文更优。
- 原 TODO-09 曾计划做严格配对的中英 Prompt A/B；该决策已被同日后续的“TODO-09：全角色中文 Prompt 控制层与证据闭环”覆盖，
  不再执行中英 A/B，也不增加 `en-v2`、`mixed-v2` 或语言选择分支。

### 本轮实施内容

- 将原有含义不清的知识命名统一为“知识模块”和“知识库”：生产字段使用 `knowledge_module_id`、
  `knowledge_modules`、`knowledge_base_id`，目录使用 `knowledge_modules/cannbot_a08c4970_knowledge_base`，
  元数据使用 `knowledge_module_manifest.json` 和 `knowledge_base_manifest.json`。相关生产代码、配置、测试和文档中不再使用旧术语。
- 内置并校验固定版本的 CANNBot 精选知识库；Adapter 在 Runner 初始化时一次性校验路径、SHA256、stage 和引用关系，
  随后使用不可变内存 registry。旧外部 root、mapping 参数或环境变量被使用时立即返回迁移错误。
- mapping 显式列出叶子知识；选择以 `knowledge_module_id + section_id` 去重，按受众和路由执行确定性预算，
  单个已选知识模块完整投递且不在中间截断。Hybrid 与 Skills-only 仍只在知识来源组合上不同。
- 路由增加 `ascendc_api`、`host_abi`、`launch_abi`、`operator_semantic`、`project_local`、
  `compiler_noise` 六类 symbol domain，并综合 source file、profile、case、wrapper/kernel 参数、runtime code 和 correctness 证据。
- 每轮分别记录输入路由和结果失败路由；普通 `Tensor`、`Traceback`、`COMPILER`、`PYBIND11_MODULE` 与项目函数名
  不再触发 installed-header API 缺失查询。
- 接通 `smoke → shape → dtype → full → benchmark` 渐进评测。每个 case 在 reference、candidate 调用和比较前后原子写入 JSON；
  任一级失败立即停止，只有 `full` 全部通过后才允许 benchmark 和正确 baseline。
- 统一评测门为 `source_valid → compiled → loaded → kernel_started → comparison_completed → full_correct → benchmarked`，
  并把 profile、已通过 case 和门级差量纳入接受判断及 Prompt。
- 增加候选语义哈希；仅改注释或空白的候选不再进入 evaluator 或消耗候选评测预算，连续空转会触发 DIAGNOSE。
- 扩展单计划项为 `hypothesis_id`、`target_files`、`edit_scope`、`allow_interface_change`、
  `evidence_refs`、`expected_signal` 和 `falsifies`；DIAGNOSE 必须给出证据引用和可验伪条件。
- 保存最近八次与开放错误相关的结构化尝试摘要；同一 `error_id + approach_family` 连续失败两次后升级 DIAGNOSE。
- 从已编译候选生成 `InterfaceContract`，在未授权时阻断 pybind module、Host wrapper 声明/定义及 Kernel entry 签名变化。
- 新增高置信 Host 指针检查：`*_do` 的设备地址参数只能转发给 Kernel launch，禁止在 Host wrapper 中解引用或通过别名解引用。
- 对重复编译诊断和设备输出做紧凑化；structured failure 优先服从 evaluator stage，并补充 core/block、PC、serial、扩展错误字段及归因状态。
- verification 子进程因负返回码退出时显式记录信号编号和名称。`SIGSEGV` 会进入 runtime 路由，但不再被错误当作
  “NPU Kernel 已启动”的证据。
- summary 同时记录 accepted/latest attempt ID、候选路径和丢弃尝试；文档同步说明渐进 profile、门语义、知识选择预算和接口保护。

### 已登记问题的实施状态

| 关联问题 | 修改与验证证据 | 问题状态 | 剩余风险 |
|---|---|---|---|
| P-01、P-05 | 已实现门/profile/case 进展向量；单测证明新增已通过 case 可 `PARTIAL_KEEP`，真实 Round 3 保留了编译诊断减少的进展。 | 部分解决 | 编译期旧诊断消失但出现新诊断时仍可能被接受，尚需更严格的诊断偏序。 |
| P-03 | Generator Prompt 已呈现接受基线、门进展、开放错误和不可退化项。 | 部分解决 | 真实实验未进入 shape 或完整正确性，尚未验证 post-compile 反馈能推动语义修复。 |
| P-04 | 已生成并执行签名级 `InterfaceContract`；真实产物保存于 `.llm_state/interface_contract.json`。 | 部分解决 | descriptor 布局、launch body 和跨文件结构体字段尚未完全冻结。 |
| P-06 | 重复方法指纹触发 DIAGNOSE；真实 Round 8～10 生成了带证据和验伪条件的诊断方案。 | 部分解决 | 模型仍连续保留同一 Host 非法解引用，说明升级存在但约束服从不足。 |
| P-07～P-09 | Add 最小算子语义与广播约束常驻，其他架构、ABI、runtime、precision 和 performance 内容按证据选择。 | 部分解决 | 尚未用达到 shape profile 的真实轨迹验证长期语义常驻是否足够。 |
| P-10、P-26 | 恢复来源/受众确定性预算、条目去重与紧凑日志；不再让 `full_selected` 关闭全部知识预算。 | 部分解决 | 本次 21 次 LLM 调用仍使用 497,858 token，Prompt 总字符数仍较高。 |
| P-12 | 精确 symbol domain、installed-header fallback 和未验证 API trace 已接通。 | 部分解决 | 主要由单元及轨迹回放验证，尚缺新 API 真实编译失败的完整降级样本。 |
| P-13 | 磁盘保留完整 ledger，Prompt 投递最近八次结构化相关尝试。 | 部分解决 | 相关性摘要仍可能丢失更早但关键的失败方案。 |
| P-14 | case 在调用前后即时原子落盘；真实 Round 5 保留 case 0、`[128] + [128]`、float32 和 `alpha=1.0`。 | 已解决 | 旧实验只记录到 `started`；新增细分 checkpoint 需下一次真实运行验证。 |
| P-15 | 编译诊断重复行已折叠并显示次数，原始日志完整保留。 | 部分解决 | 设备 dump 折叠尚未获得新的真实异常复验。 |
| P-16 | 编译 stage 优先于文本 marker；core/block/PC/serial 字段通过回归测试。 | 部分解决 | 新设备异常字段只由 fixture 验证。 |
| P-17 | 错误摘要按根因 marker 排序；负返回码新增 `SIGSEGV` 行并路由 runtime。 | 部分解决 | 信号修复发生在本次真实实验后，目前只有回归测试证据。 |
| P-20 | 渐进 profile 已进入真实 evaluator；本次 smoke 失败后未误跑 shape/dtype/full/benchmark。 | 部分解决 | 尚无 profile 晋级到 full 的真实成功轨迹。 |
| P-21、P-24 | DIAGNOSE 强制 evidence/falsifies，计划目标文件和接口授权被生产代码消费。 | 部分解决 | 真实 DIAGNOSE 仍未消除重复非法实现。 |
| P-22 | 签名保护和 Host 设备指针非法解引用成为确定性 source validation。 | 部分解决 | 通用 broadcast/DataCopy 正确性仍没有语义 IR 或 CPU oracle。 |
| P-23 | 语义空改不评测、不计候选预算，并有回归测试。 | 部分解决 | mock 模式有意关闭该拦截；本次真实 Add 未出现语义空改，缺少真实触发证据。 |
| P-27 | summary 真实记录 accepted attempt 5、latest attempt 10 和丢弃尝试 1、6～10。 | 已解决 | accepted 工作目录语义保持不变，使用者需读取 summary 区分最新失败尝试。 |
| P-02、P-18、P-19、P-25 | 本轮按审核决定未实施逐文件合并、多计划项、默认 patch/AST 或独立 repair 调用。 | 暂缓 | 继续遵守下述 TODO 的前置条件。 |

### `torch.npu` 与真实 Add 验证

- 环境复验结果：`torch=2.7.1+cpu`、`torch_npu=2.7.1.post4`、`torch.npu.is_available()=True`、
  `torch.npu.device_count()=1`、设备 0 为 `Ascend910B3`。DeepSeek provider、`deepseek-v4-flash`、base URL 与 API key 均已配置；
  日志中的 `libop_plugin_atb` owner mismatch 是警告，未据此判定设备不可用。
- 按计划只执行了一次真实 Add Hybrid 实验，目录为
  `outputs/nonthinking_add_hybrid_progressive_10round_20260916/`。配置为 temperature 0.2、Generator thinking disabled、
  max tokens 32768、progressive bootstrap、最多 10 个 candidate、CANN 8.5.2、Ascend910B3、device 0、Hybrid，未传外部知识路径。
- Planner 第 6 版曾返回不合法 JSON；使用 `--resume` 继续同一实验，没有新建第二个实验。最终完成 10 个 candidate、21 次 LLM 调用，
  使用 431,629 输入 token、66,229 输出 token、总计 497,858 token，因 `max_total_rounds` 停止。
- 分门结果：source validation 在 Round 2 首次通过；编译在 Round 5 首次通过；load 与 NPU Kernel 启动没有可靠证据；
  smoke case 0 在 candidate 返回前以 `rc=-11` 崩溃；shape、dtype、full 50-case correctness 和 benchmark 均未到达。
- Round 10 又退化为 `std::min<int64_t>` 不受支持的编译错误。因此本次 Add 没有正确 baseline，不能称为生成成功。
- 对 Round 5～10 候选的事后检查发现 Host `*_do` wrapper 将 NPU 上的 tiling 地址强转为普通 Host 结构体指针并读取
  `dtypeCode`，这是 `SIGSEGV` 的高置信根因。新增 source validator 能在评测前报告
  `host_dereferences_device_pointer`，但该拦截是在真实实验结束后加入，只能记为代码和回归验证，不能倒推本次实验已通过。
- 结论：相关 Host/launch 知识已选中并投递，模型仍重复非法实现；当前瓶颈是生成推理或确定性约束执行，继续增加文档不是优先方案。

### 验证

- 全部仓库测试通过：157 项。
- 修改模块通过 `python -m py_compile`；变更范围通过 Ruff `F/I` 检查和 `git diff --check`。
- 轨迹回放覆盖 include、`__gm__`、Host `.vec()`、507035/507001、非法 cast、wrapper mismatch、广播 stride=0 和数值不匹配路由。
- 知识库在没有 sibling CANNBot 仓库时可加载；manifest 文件与 SHA256、内部 ID 引用、去重和完整渲染均由测试验证。

### 暂缓事项（后续 TODO）

- TODO-01～TODO-08：继续保留 2026-09-15 已登记的逐文件接受、多计划项、patch/AST、独立 repair 调用、自动换基、
  通用 stride/DataCopy 静态证明、verified skeleton 和跨调用 SHA 占位方案及各自前置条件。
- TODO-09（已解决）：取消中英 Prompt 配对 A/B，生产链路统一使用全角色中文控制层；不保留英文或 mixed Prompt 分支。
- TODO-10：扩展 `InterfaceContract` 到 descriptor 布局、共享 tiling 结构体和 launch body，并为授权变更设计明确迁移流程。
- TODO-11：为 broadcast/index/DataCopy 建立小型语义计划 IR 与 CPU oracle，先离线验证 offset、stride=0、source span 和 tail。
- TODO-12：修正编译期 `PARTIAL_KEEP` 的诊断偏序，禁止用一个新的同级根因换掉旧根因后被误判为单调进展。
- TODO-13：下一次真实实验复验 `reference_passed → candidate_started → candidate_returned` checkpoint、`SIGSEGV` 路由和
  `host_dereferences_device_pointer` 前置拦截；不得把这项复验计作本轮已完成。



## 2026-09-15

### Claude 方案问题审核台账

- 审核对象：`/mnt/workspace/.claude/plans/current_plan.md` 中的 P-01～P-27、S-01～S-27。
- 代码证据来自当前工作树；真实运行证据来自
  `outputs/nonthinking_add_knowledge_base_hybrid_10round_20260915/.llm_state/summary.json`、
  各轮 Prompt、知识引用、候选源码和评测日志。
- 该次 Add 轨迹共完成 8 次候选评测；第 9 轮因模型服务连接被拒而暂停。第 4 轮首次达到
  编译、加载和 NPU Kernel 启动，随后第 4～8 轮稳定失败于 `507035/MTE`。
- 状态含义：`已确认` 表示有直接证据；`部分确认` 表示现象存在但原归因或方案过度；
  `不采纳` 表示事实不成立或不应按缺陷修复；`暂缓` 表示当前证据不足以承担对应架构风险。

| 编号 | 审核状态 | 准确的问题定义与证据 | 对原解决方案的修订 |
|---|---|---|---|
| P-01 | 已确认，未解决 | 编译阶段已有诊断集合的 `PARTIAL_KEEP`，但编译通过后的运行与正确性阶段仍缺少可接受的单调进展信号。 | 实现按评测门和用例集合定义的单调进展，不把“尚未全正确”一律等同于零进展。 |
| P-02 | 不采纳 | 候选以完整 Host/Kernel bundle 原子接受和回滚是事实，但这是维持跨文件 ABI 一致性的安全属性，不是已证实根因。 | 保留原子 bundle；不实施逐文件拼接，除非未来先有可证明的跨文件接口与来源一致性检查。 |
| P-03 | 已确认，未解决 | frontier 写入产物，但生成 Prompt 没有清楚说明已通过的门、当前接受基线和不可退化条件。 | 注入紧凑的已通过门、当前基线 ID、开放错误和禁止退化项，不注入整份 frontier 日志。 |
| P-04 | 已确认，未解决 | `repair_policy.py` 目前只渲染允许路径和保护说明，没有在候选接受前执行 ABI/保护区校验。 | 建立确定性的 `InterfaceContract`，强制检查三方签名、参数顺序、descriptor 布局和受保护区域。 |
| P-05 | 已确认，未解决 | 正确性异常时没有持久化已通过用例、首个失败用例和能力增量，无法区分部分进展与零进展。 | 保存门级和用例级进展向量，并以相对当前接受基线的差量决定是否接受。 |
| P-06 | 已确认，未解决 | 相同错误可连续采用同族方案，现有三轮重规划阈值不按 `error_id + approach_family` 升级。 | 增加重复方案指纹；同一错误的同族方案重复时强制给出新假设或停止，不自动换基。 |
| P-07 | 部分确认，未解决 | 路由会让与当前主路由无关的语义材料退出 Prompt；但把全部语义知识永久常驻会重新制造噪声。 | 常驻当前算子的最小语义和接口契约，其他知识仍按证据增量选择。 |
| P-08 | 部分确认，未解决 | 后续轮次不再投递初始硬件材料是事实，但没有证据支持所有 NPU 架构资料必须常驻。 | 仅在 UB、对齐、核间同步或容量证据出现时注入精确事实；稳定的已验证数值可进入最小约束集。 |
| P-09 | 部分确认，未解决 | runtime/correctness 轮可能不再看到完整 ABI/compiler 教程；真实需要的是稳定接口约束，而非整套教程。 | 常驻 `InterfaceContract` 和禁止模式；编译/API 说明仍由当前错误路由。 |
| P-10 | 不采纳原结论 | Round 5 Planner Prompt 实际含多个已安装声明，包括 `DataCopyExtParams` 和 `DataCopyPad`；“只有一个 header”不成立。 | 将问题改为“宽泛符号匹配带来无关声明和超大引用块”，按精确声明、来源优先级和 overload 数量治理。 |
| P-11 | 不采纳 | Planner Prompt 实际包含 API 卡片和 Level 2 已安装声明，“Planner 拿不到 API facts”与产物不符。 | 不新增重复 API 注入；改进 Planner 对已有精确事实的组织和引用。 |
| P-12 | 部分确认，未解决 | Structured 精确查询会拒绝未收录类型，但 installed-header fallback 已提供声明；问题是两条来源的回退与追踪不统一。 | 保留精确拒绝，随后按已安装头文件→项目源码→明确未验证假设三级降级，并记录来源和置信度。 |
| P-13 | 部分确认，未解决 | 原始 history 不进入 Prompt 是有意控制宽度；真正问题是失败方法摘要最多 600 字符，可能丢失关键结构。 | 保留磁盘完整 ledger；Prompt 只投递与开放错误相关、结构化去重的假设、证据、结果和禁止重复项。 |
| P-14 | 已确认，未解决 | 验证器只在完整 case 循环结束后记录输入和比较；异常返回会使 case 信息为空。 | 每个 case 执行前后立即落盘索引、profile、shape、dtype、期望/实际摘要和通过状态。 |
| P-15 | 已确认，未解决 | 多轮 Prompt 中重复设备 dump 和相同错误块，占用大量输入且没有新增证据。 | 对日志做结构化解析、内容哈希和重复计数；Prompt 保留首个实例、变化段和计数，原始日志仍完整落盘。 |
| P-16 | 已确认，未解决 | `parse_structured_failure` 在编译阶段判断之前匹配 `mte/aicore`，可能误分 stage，关键字段也常为空。 | 先按 evaluator stage 和 source file 分类，再解析错误码、core/block、API、地址和 case；空符号保持空并增加归因状态。 |
| P-17 | 部分确认，未解决 | excerpt 并非始终简单取尾部，但现有优先行策略会选择低信息量 Traceback/设备噪声。 | 按阶段、源码文件、首个根因、错误码和调用栈位置排序，保留首个根因及少量上下文。 |
| P-18 | 暂缓 | 单计划项是可追踪的单假设限制，一个假设允许修改多个必要文件；没有证据证明 item 数量导致本次失败。 | 暂不放开多 item；先补全单 item 的目标文件、假设和验收字段。 |
| P-19 | 暂缓 | Generator 输出完整文件是事实，但尚无证据证明它是 Add 失败主因；patch 会引入应用失败和上下文漂移。 | 先用受保护区域和候选 diff 审计约束整文件输出；patch/AST 编辑列入后续实验。 |
| P-20 | 已确认，未解决 | `case_profiles.py` 引用了不存在的 `EvaluationProfile` 且没有生产调用者，渐进 profile 尚未接入评测。 | 统一 profile 模型并接入 evaluator/runner，按 smoke→broadcast→dtype→full 递进，smoke 通过不得称为正确 baseline。 |
| P-21 | 已确认，未解决 | Planner 与 DIAGNOSE 使用同一 system prompt，输入规模和输出契约高度相似，诊断没有强制证据闭环。 | DIAGNOSE 强制引用可核验 evidence、列出被排除假设和新假设；不强制开启模型 thinking。 |
| P-22 | 已确认，未解决 | 关键语义/ABI 约束主要是 Prompt 文本，Generator 违反后缺乏确定性拦截。 | 先实现高置信 ABI、禁止 Host descriptor 强转和静态越界模式检查；一般广播索引验证需语义 IR。 |
| P-23 | 已确认，未解决 | 真实轨迹存在只改注释而消耗候选轮次的空转。 | 比较去注释后的语义哈希和 bundle diff；空改动不进入 evaluator、不消耗评测预算，并触发重新规划。 |
| P-24 | 已确认，未解决 | plan item 缺少 `target_files`、假设 ID、允许的接口变化和验收信号。 | 扩展单 item schema，并让 repair policy 和 verifier 实际消费这些字段。 |
| P-25 | 部分确认，未解决 | `repair_*` 配置和 call type 基本未进入状态机是事实；但无 reasoning 是显式 `generator_thinking=disabled` 所致。 | 暂不新增昂贵调用；先让 DIAGNOSE 和现有 Generator 形成证据闭环，再做独立 repair call 的 A/B。 |
| P-26 | 已确认，未解决 | `full_selected` 关闭多层预算，Planner Prompt 已增长到约 27 万字符。 | 恢复按来源和证据的确定性预算、去重和稳定摘要；不使用跨独立 LLM 调用的 SHA 占位符代替正文。 |
| P-27 | 已确认，未解决 | 任务根目录主要反映 accepted candidate，最后失败尝试需进入 round/candidate 产物才能辨认。 | summary 明确记录 accepted/latest attempt ID、路径和丢弃原因，不改变安全的 accepted 输出语义。 |

### 审核结论

- Claude 方案准确抓住了四个主要问题簇：评测反馈缺少粒度、post-compile 接受策略缺少单调进展、
  Prompt 含重复或弱相关材料、关键契约只有文本而缺少执行检查。
- 整 bundle、单计划项和完整文件输出目前都没有直接根因证据，不能与 P-01、P-14、P-16 等已证实问题等量处理。
- S-10、S-11 与实际 Prompt 不符；S-02 会破坏 Host/Kernel ABI 原子性；S-26 的跨调用哈希省略不适用于无共享上下文的请求；
  S-22 对一般 DataCopy/broadcast 范围做正则证明的可信度不足。
- 当前真实瓶颈已推进到 Kernel 启动后的 `507035/MTE`。后续应先改善 case、地址、descriptor 生命周期和 DataCopy 范围证据，
  不能仅继续增加知识文本。

### 具体执行方案

#### 批次 A：先修观测，不改变候选选择语义

1. 建立统一门模型：`source_valid → compiled → loaded → kernel_started → comparison_completed → full_correct → benchmarked`。
2. 修复 structured failure 的 stage 优先级并提取首个根因；为空的字段增加归因状态。
3. correctness evaluator 按 case 即时记录 profile、shape、dtype、索引、通过状态和异常摘要。
4. 折叠重复 device dump；summary 同时公开 accepted/latest attempt 身份和路径。
5. 验收：真实 507035 必须保留首个失败 case、Kernel 已启动事实和地址类证据。

#### 批次 B：建立单调进展和保护边界

1. 定义进展向量：已通过门、profile、case 集、开放诊断和新引入诊断。
2. bundle 仍原子接受；仅当候选严格增加能力且不破坏已通过门、不重现 cleared error 时接受。
3. 从最后一个已编译 frontier 生成 `InterfaceContract`，冻结 pybind 声明、`*_do` 定义、Kernel entry、参数顺序、
   descriptor 布局和 launch 形式；显式授权时才允许变化。
4. 扩展单 plan item：`hypothesis_id`、`target_files`、`allow_interface_change`、`evidence_refs`、
   `expected_signal` 和 `falsifies`。
5. 增加空转检测与 `error_id + approach_family` 重复升级。

#### 批次 C：收紧知识与 Prompt

1. 常驻当前算子的最小语义契约和 `InterfaceContract`；其他 Host、Kernel API、runtime、precision、performance 知识按证据选择。
2. installed declaration 按源码、精确 symbol、声明而非调用点和 overload 数量排序；Structured 未命中时显式降级并记录 trace。
3. 失败历史改成与开放错误关联的结构化摘要；原始 ledger 只留在磁盘。
4. 各来源使用确定性预算并按知识条目原子投递；不截断条目中部，也不用跨调用 SHA 占位符。

#### 批次 D：接通渐进评测和高置信检查

1. 修复并接入 profile 模型，固定顺序为 smoke、broadcast shape、dtype、完整 50 case、benchmark。
2. profile 只在当前级全部通过后推进；完整正确性前不得建立 baseline 或进入 optimization。
3. 编译前执行 Host/Kernel 三方 ABI、一致签名、禁止 descriptor 非法强转和受保护区域检查。
4. Broadcast/DataCopy 通用验证先作为告警；只有真实用例证明低误报后才升级为阻断。

#### 批次 E：分层验证

1. 运行受影响单元测试、全部仓库测试、静态检查和 Add 轨迹回放。
2. 在真实 NPU 环境只运行一次同配置 Add Hybrid，最多 10 个候选，并实时保留日志。
3. 分别记录七个 gate，不以 smoke 或 Kernel 启动代替完整正确性。
4. 若明确约束已完整投递且相同非法 ABI 或 stride=0 方案仍连续重复，则停止扩充文档，记录为
   `知识已可用且已投递；剩余问题属于生成推理或约束执行`。

### 后续修改的验收规则

- 每项修复必须写明：`关联问题`、`修改内容`、`验证证据`、`问题状态`、`剩余风险`。
- 只有直接覆盖失败路径的测试或真实轨迹证据，才能改为“已解决”；仅有代码或普通单元测试时最多记为“部分解决”。
- 必须分别验证知识“选中、完整渲染、候选遵守、评测通过”，不能用 Prompt 中出现知识替代结果验证。
- 不得把 Kernel 启动、单个 smoke case或部分 profile 通过写成完整 Add 正确性成功。

### 暂缓事项（后续 TODO）

- TODO-01：逐文件接受或合并。前置条件是可机器验证的跨文件 InterfaceContract 和来源一致性；当前保留原子 bundle。
- TODO-02：一次 Planner 输出多个 plan item。先验证单 item 的结构化假设、验收和累积进展有效。
- TODO-03：默认 patch/AST 编辑。先量化完整文件输出造成的实际退化，再做带失败回退的对照实验。
- TODO-04：独立 `repair` 模型调用及 reasoning 配置。先消除 DIAGNOSE 与 Planner 同质化，再做成本/收益 A/B。
- TODO-05：自动换基或回滚到更早 frontier。必须保证不会丢弃刚获得的单调进展。
- TODO-06：通用 stride/DataCopy 静态证明。优先设计小型语义计划 IR 和 CPU oracle，不用高误报正则作为硬门。
- TODO-07：verified skeleton、typed patch 或 AST 级编辑。仅在知识完整投递但模型仍重复同一错误时启动。
- TODO-08：跨调用只发送 SHA/未变段占位符。独立 LLM 调用没有共享正文上下文，当前明确不实施。

### 未按中文格式记录的原因与修正

- 原 `AGENTS.md` 只明确规定“方案必须使用中文”，没有要求整个 `CHANGELOG.md` 的历史变更、验证和分析都用中文；
  先前代理沿用了已有英文格式。这是规则解释过窄，不是工具或仓库限制。
- 已把 `AGENTS.md` 改为中文并新增强制规则；本文件历史叙述也统一为中文，原先缺失标题的 2026-09-12 实验已单独归档。
- `.gitignore` 原先排除了 `AGENTS.md`。这不会阻止当前工作区中的代理读取它，但会阻止规则随仓库版本传播；
  现已解除该忽略项，使中文记录和问题验收规则可以纳入版本管理。
- 关联问题：中文记录规范。验证证据：规则文件和本文件全文检查。问题状态：已解决。

### 知识命名、内置知识和路由修订

- 统一使用“知识模块”表示可路由内容，“知识库”表示固定版本的 CANNBot 文档集合；字段改为
  `knowledge_module_id`、`knowledge_modules` 和 `knowledge_base_id`。
- 包内目录改为 `knowledge_modules/cannbot_a08c4970_knowledge_base`，元数据改为
  `knowledge_module_manifest.json` 和 `knowledge_base_manifest.json`。
- 建立包内只读 CANNBot 知识库，保存上游 commit、逐文件 SHA256、许可证、来源、stage 和冲突规则。
- Adapter 初始化时校验并缓存 manifest 和全部文档；mapping 只引用内部 ID，按模块与 section 去重并完整投递。
- 增加六类 symbol domain，避免普通 `Tensor`、`COMPILER`、`Traceback`、`PYBIND11_MODULE` 和项目函数触发 API 缺失查询。
- 新增 Broadcast、直调 ABI、Host C++ 和 Kernel 内存知识；分别记录 `input_route` 与 `result_failure_route`。
- 真实设备复验：`torch.npu.is_available()` 为 true，设备数为 1，设备为 `Ascend910B3`。沙箱内 507899 不能推断物理设备不可用。

### 验证

- 仓库测试通过 123 项，Structured Knowledge 测试通过 25 项。
- 修改模块通过 Python 编译、针对性 Ruff 和 `git diff --check`。
- 内置 registry 加载 67 个文档，其中 60 个来自 CANNBot、7 个来自 CannAgent；Add 回放覆盖九类真实失败。

### 分析

- 知识回放已证明相关内容可用且完整投递；真实运行若仍重复非法 ABI/cast 或 stride=0 连续搬运，应转向
  InterfaceContract、语义计划 IR 或 verified skeleton，而不是继续堆叠文档。

## 2026-09-14

### 内置 CANNBot 知识库与边界感知路由

- 计划并实现将外部 CANNBot 精选文档内置为只读知识库，取消运行时 sibling 仓库依赖。
- 搬入 60 个上游 Markdown 及许可证，新增 7 个 CannAgent 本地知识模块；未搬入 workflow、脚本、评测和 CMake。
- mapping 升级为 schema v3，叶子章节显式映射；Adapter 改为不可变内存 registry，并实施 manifest/hash/path/stage 校验。
- Runner 只构造包内 Adapter；旧外部参数和环境变量被使用时立即失败。
- 接入 typed symbol domain、失败归属、profile/case 字段、Prompt 新顺序和双 route 观测。
- 新增 Add 去敏 fixture，覆盖 include、`__gm__`、`.vec()`、507035、非法 cast、507001、wrapper mismatch 和广播错误。

## 2026-09-13

### 编译修复与分层轨迹记忆

- 将完整 Attempt、持久 repair state 和 Prompt 摘要分层；稳定知识与单次失败方法分开存储。
- 新增 compiler error ID、attempt ledger、repair state、base/latest rejected 身份和恢复语义。
- 同一编译 frontier 支持 `PARTIAL_KEEP`；修复路由污染并在 compile repair Prompt 中加入模板实例化约束。
- Planner/DIAGNOSE 每次只生成一个垂直步骤；回归覆盖 `{AddTiling, Muls} → {Muls}` 的累积修复。
- 仓库测试通过 122 项，Structured Knowledge 测试通过 25 项，并通过 Python 编译和 `git diff --check`。

### 全量选中知识输入与可观测性

- 增加实验性 `knowledge_input_mode=full_selected` 和 Prompt 字节数、SHA256、selected/rendered ID、来源字符数及截断观测。
- 增加环境指纹运行时事实和 Host/AscendC probe；只有环境匹配且 probe 成功的精确事实可提升到 Level 3。
- 改进声明排序、注释过滤、overload 保留、负面事实条件和 failure/source/planned symbol 优先级。
- 移除会灾难性回溯的重复声明正则，保留确定性的 CUDA、Host ABI、include、queue-owner 和 wrapper 检查。

### Add 实验

- 三轮全量输入实验的五次 LLM 请求均接收完整 Prompt，最大输入 38,922 token；未产生正确候选，最终失败于
  `Muls<bfloat16_t>`，而对应限制已完整投递，因此属于生成推理失败。
- 十轮能力实验保存在 `outputs/nonthinking_add_full_selected_10round_20260913/`：10 轮 source/API/static 均通过，
  10 轮编译均失败，诊断在 `AddTiling` GM copy 构造和 `Muls` overload 间反复。
- 十轮实验共 15 次 LLM 调用，无输入截断；总用量为 1,053,495 输入 token 与 66,205 输出 token。
- 实验未进入 load、NPU execution、correctness、baseline 或 optimization；证明移除输入预算不足以单独解决生成问题。
- Host/Kernel probe 在 CANN 8.5.2 / Ascend910B3 上通过，环境指纹为 `06b8e945d4cd781bd1b4`，提升 11 个事实到 Level 3。
- 后续回归通过 108 项；修改文件通过 Python 编译、Ruff 和 `git diff --check`。

## 2026-09-12

### B/D 知识源消融实验

- 在真实 Ascend910B3 上对 GELU、LayerNorm、Permute 依次运行 B Hybrid 与 D Skills-only；每组使用相同 Agent、
  DeepSeek `deepseek-v4-flash`、temperature 0.2、5 次 bootstrap、2 次 baseline 后优化和真实 evaluator。
- 六组产物与日志保存在 `outputs/ablation_bd_final_20260912/`。

| 算子 | 模式 | 编译候选 | 正确候选 | 首次编译轮次 | 总 token | 端到端秒数 |
|---|---|---:|---:|---:|---:|---:|
| GELU | B Hybrid | 0/5 | 0/5 | — | 94,357 | 100 |
| GELU | D Skills-only | 0/5 | 0/5 | — | 114,186 | 216 |
| LayerNorm | B Hybrid | 0/5 | 0/5 | — | 133,123 | 130 |
| LayerNorm | D Skills-only | 0/5 | 0/5 | — | 162,541 | 203 |
| Permute | B Hybrid | 0/5 | 0/5 | — | 165,079 | 273 |
| Permute | D Skills-only | 3/5 | 0/5 | 2 | 169,397 | 554 |

### 分析与验证

- B 算子级编译率为 0/3，D 为 1/3；正确率均为 0/3，不能证明 Skills-only 可替代 Structured Knowledge。
- Permute D 第 3 轮实际执行 149 个 NPU case，但后续 FP16/FP32 case 大范围错误，说明索引或搬运实现错误。
- 审计确认 D 没有回退 Structured Knowledge；48 次 LLM 调用均为 `finish_reason=stop`，六条轨迹均为非 mock evaluator。
- 没有候选建立正确 baseline，因此所有组都未进入 optimization。

## 2026-09-11

### 变更

- 开始为 CANNBot 中的 AscendC 生成知识制作 Adapter，以改善 CannAgent 知识内容薄弱、模型难以利用的问题。

## 2026-09-10

### 变更

- 用带权威等级的 Host、Vector、Cube、C/V 和跨核项目指南替换八份未使用或过时的 TileLang 转换文档。
- 增加项目契约、失败卡、模式卡和来源追踪；明确 `document`、`structured` 与 knowledge build 的含义。
- 将 `structured` 设为默认模式，发布校验后的 `current.json`；重命名旧 `knowledge_v2`、router 和 checker。
- 增加多轮 README，覆盖运行、阶段、知识构建、逐轮路由、轨迹分析和输出目录。

### 分析与验证

- 主要缺口是 API corpus 没有系统 Host binding/launch contract，而不是物理文档数量。
- 完整 8.5.0 构建生成 121 个文档、930 个原子事实、341 个 API 卡、3 个项目契约、2 个失败卡和 5 个模式卡。
- `ARARARAR` 是 Reduce 表中的合法轴模式，但暴露了 API identity 提取问题。
- Structured/仓库测试先后通过 24/79 和 25/80；默认 structured 的 GELU mock 流程成功。

## 2026-09-09

### 变更

- 用 `ascendc_multi_turn/knowledge/` package 替换旧单文件，并提供离线构建入口。
- 完成 M0～M5：不可变知识构建、语义路由、StructuredFailure、调用解析、源码约束、incident/experience/frontier/rollback。
- 将 direct runner 从 TileLang Prompt 耦合中解开，首次生成前增加单项 AscendC baseline 方案。
- 增加 `--max-total-rounds`，让 baseline 前失败也可按总候选数终止。

### 分析与验证

- frontier 只在候选证明更深评测能力时前进；confirmed experience 必须通过正确性并消除目标 failure signature。
- API identity 必须先于相似检索；`DataCopy`、`DataCopyPad`、`DataCopyExt` 不得互相污染。
- direct runner 没有 TileLang import、编译或评测阶段；旧耦合仅来自知识 supplement 和命名。
- 仓库测试先后通过 74 和 76 项，并通过 Python 编译、Ruff、`git diff --check` 和两轮 mock。

## 2026-09-08

### 变更

- 将同轮 compiler repair 改为显式 bootstrap、PLAN、EDIT、EVAL、SETTLE、DIAGNOSE/REPLAN 状态机。
- baseline 默认 8 次尝试，`--max-rounds` 只计 baseline 后性能候选；新增 schema v3 checkpoint 和 v2 migration。
- 增加 Host launch ABI guard：pybind 必须调用 Kernel 源中的 `extern "C" *_do` 并采用规定 launch 形式。
- Planner 默认关闭 thinking、输出上限 8192，与 Generator 配置分离。

### 分析与验证

- 当时 GELU 失败共享错误 Host ABI 假设：使用不存在的 `ACLRT_LAUNCH_KERNEL` 和 `acl/acl_rt_launch.h`。
- 状态机不能保证有限预算内生成正确 Kernel，但可隔离预算、拒绝无效接口并提供可恢复 blocked 状态。
- 通过 73 项测试、Python 编译、Ruff、空白检查和八个归档 Host launch 布局检查。

## 2026-09-07

### 变更

- 汇总 direct AscendC 多轮实现：provider 控制、进度、评测预算、持久知识路由、installed header 证据、
  structured diagnostics、保守源码校验和同轮 compiler repair。
- 编辑器数据库缓存不进入功能变更。

### 分析与验证

- 当时 runner 是 `knowledge route → generate → evaluate → select`，尚不是显式阶段状态机；恢复也不能从精确子阶段继续。
- 通过 65 项测试、`git diff --check` 和秘密模式扫描，未发现秘密信息。

## 2026-09-06

### 变更

- 评测轮只在候选进入 evaluator 后计数；路由、传输、空响应、知识和格式失败暂停 checkpoint 供 `--resume`。
- 设置 router 4096、Generator 65536、compiler repair 65536 的独立输出预算，默认模型改为 `deepseek-v4-flash`。
- 记录请求/服务模型、thinking、token、finish reason、编排 attempt 和 pending run，不复制 reasoning 正文。
- 使用紧凑 API manifest、首次完整路由、后续增量路由、精确符号匹配和 installed-header fallback。
- 增加保守源码检查、一次定向同轮修复和回归保护。

### 分析与验证

- installed CANN public header 是 API owner、overload 和参数拼写的权威来源；文档负责语义但可能与 patch release 不同。
- 生成 115 文档 API manifest；通过 65 项测试、Python 编译、空白检查和 mock smoke。

## 2026-09-05

### 变更

- 增加 DeepSeek/OpenAI provider、独立环境凭据、CANN 版本检测和同 major 文档 fallback。
- 每轮生成前增加知识路由，保存 Prompt、响应、选择、引用、版本和分类 token。
- 增加 source-aware wrapper 校验、精确 pybind module 绑定、进度/心跳和分 stage 日志。
- 增加结构化失败字段及旧 trajectory 迁移；知识路由改为持久 working set，并加入 installed-header 冲突和同轮 repair。
- 修正 CANN 8.5.2 `DataCopyPadExtParams::paddingValue`，限制 `CopyTiling` 适用范围。

### 分析与验证

- 内置 API 文档基于 CANN 8.5.0；索引声明 114 页，实际有 115 个 Markdown、115 个标题和 89 张图片。
- 115 页只有 81 个不同短 API 名，短名不是稳定路由键。
- 8.5.2 header 使用 `paddingValue`，8.5.0 文档写 `padValue`，同 major fallback 仍需运行时校准。
- 本日测试随增量从 15、22、34 扩展到 54；Python/Bash 检查和 mock 流程通过。

## 2026-09-04

### 变更

- 增加根目录 `.env`、`.env.example`、`python-dotenv`、集中环境读取和共享 shell loader。
- 为 Triton 批量生成、AscendC Claude 批量生成和 AutoResearch 增加可配置模型默认值并保留命令行覆盖。
- 增加启动校验、忽略规则、配置测试和 direct AscendC 使用文档。

### 分析

- `docs/non-claude-execution-analysis.md` 记录非 Claude AscendC 执行循环、状态/记忆、固定引用、产物、恢复和评测边界。
