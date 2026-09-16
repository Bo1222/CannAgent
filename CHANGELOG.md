# 项目变更日志

仓库中的代码、配置、测试或文档修改都必须记录在这里。日期使用 `YYYY-MM-DD`，不得记录秘密信息。
代码标识符、文件路径、命令、模型名、产品名和原始错误信息可以保留原文，其余叙述统一使用中文。

## 2026-09-16

### 第一项：Prompt 中文化评估与执行决策

- 本轮没有直接把 Planner、DIAGNOSE 和 Generator Prompt 全量翻译成中文。现有真实轨迹不能证明语言是 Add
  失败根因；AscendC API、C++ 类型、编译诊断和上游技术材料以英文原文出现，机械翻译会增加术语漂移并削弱精确匹配。
- 本轮先修改 Prompt 的信息结构和约束执行：加入评测门、profile/case 进展、接受基线、接口契约、结构化失败方法、
  单项假设及验伪条件，同时继续完整保留源码、benchmark 和用例。
- 关联问题：Prompt 中文化收益未知。验证证据：本轮真实 Add 已在英文 Prompt 下暴露出可定位的 Host 指针非法解引用，
  但未形成中英对照数据。问题状态：暂缓，不能宣称中文或英文更优。
- TODO-09：做严格配对的中英 Prompt A/B；只翻译自然语言指令和字段说明，API 名、代码、诊断、JSON key、上游原文保持不变。
  使用相同模型、temperature、知识输入、用例顺序和候选预算，比较首次编译、首次完整正确、总 token、重复错误率和 wall time。

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
- TODO-09：执行上述中英 Prompt 配对 A/B；在结果前不全量翻译生产 Prompt。
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
