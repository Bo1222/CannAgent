from __future__ import annotations

import json
import re
from typing import Any

from .diagnostics import compact_evaluation
from .models import EvalResult, FileBundle

OUTPUT_CONTRACT = r"""
只返回一个 JSON 对象，不要使用 Markdown 代码围栏：
{
  "analysis": "目标证据：...；实际改动：...；保留的不变量：...",
  "files": [
    {"path": "model_new_ascendc.py", "content": "修改后文件的完整内容"},
    {"path": "kernel/pybind11.cpp", "content": "修改后文件的完整内容"},
    {"path": "kernel/<name>.cpp", "content": "修改后文件的完整内容"}
  ],
  "delete": ["kernel/obsolete_file.cpp"]
}
`files` 中的 `content` 必须是修改后文件的完整内容，不是 diff 或 patch。未出现在 `files`/`delete`
中的路径保持不变。首轮必须提供自包含的完整实现，包括 model_new_ascendc.py、kernel/pybind11.cpp、
至少一个非 pybind 的 kernel .cpp 及所有必需头文件。禁止写入构建产物。
`analysis` 只需简短说明目标证据、实际改动和保留的不变量，不要输出通用自检清单。
""".strip()


RULES = """
- 使用 AscendC 完整实现 reference 语义；不得用 torch/ATen 执行核心计算。
- model_new_ascendc.py 可以创建或 reshape tensor 并调用编译扩展，但不得用 torch 算子作为 fallback。
- pybind11.cpp 只负责校验、输出或 workspace 分配、tiling 参数和 Kernel launch。
- pybind11.cpp 必须声明并调用 extern "C" *_do Host wrapper；wrapper 与其 __aicore__ Kernel 放在一起，
  并使用 kernel<<<blockDim, nullptr, stream>>> 启动。本项目禁止包含 acl/acl_rt_launch.h 或使用 ACLRT_LAUNCH_KERNEL。
- PYBIND11_MODULE 的字面模块名必须与 model_new_ascendc.py 导入的模块一致。
- 覆盖测试用例中的全部 dtype、shape 和 attribute。
- 优先使用向量化 AscendC 操作、对齐搬运和有界 UB 用量。
- 返回前核对每个 API 的 owner（例如 EnQue 属于 TQue 而不是 TPipe）以及本地 helper 的参数数量。
- 不得虚构 AscendC 字段或 overload；提供 installed runtime header 声明时，其优先级高于示例和其他版本文档。
""".strip()


PLAN_OUTPUT_CONTRACT = r"""
只返回一个 JSON 对象，不要使用 Markdown 代码围栏：
{
  "evidence_status": "sufficient",
  "observations": [
    {"source": "evaluation", "line_excerpt": "原始证据摘录", "interpretation": "该证据直接支持的事实"}
  ],
  "ruled_out": [
    {"hypothesis": "已排除假设", "reason": "排除理由", "evidence_refs": [{"source": "evaluation", "line_excerpt": "原始证据摘录"}]}
  ],
  "unknowns": [],
  "diagnosis": "基于证据说明当前瓶颈或失败",
  "items": [
    {
      "id": "p1",
      "kind": "correctness",
      "hypothesis": "一个可验证假设",
      "change": "一项具体源码修改",
      "expected_signal": "能够验证该假设的评测结果",
      "target_files": ["kernel/<name>.cpp"],
      "edit_scope": "文件或区域",
      "allow_interface_change": false,
      "evidence_refs": [{"source": "evaluation", "line_excerpt": "准确证据摘录"}],
      "falsifies": ["被当前证据否定的历史假设或方案 ID"],
      "order": 1
    }
  ]
}
普通 Planner 必须返回 `evidence_status="sufficient"` 和恰好一个基于当前 accepted implementation
及证据的 item。不要在此响应中返回源码文件。
""".strip()


DIAGNOSE_OUTPUT_CONTRACT = PLAN_OUTPUT_CONTRACT.replace(
    "普通 Planner 必须返回 `evidence_status=\"sufficient\"` 和恰好一个基于当前 accepted implementation\n"
    "及证据的 item。不要在此响应中返回源码文件。",
    "DIAGNOSE 证据充分时必须返回 `evidence_status=\"sufficient\"` 和恰好一个 item；"
    "证据不足时必须返回 `evidence_status=\"insufficient\"`、`items=[]`，并在 `unknowns` 中说明缺少什么证据。"
    "不得用猜测填补未知项。不要在此响应中返回源码文件。",
)


INITIAL_PLAN_OUTPUT_CONTRACT = r"""
只返回一个 JSON 对象，不要使用 Markdown 代码围栏：
{
  "evidence_status": "sufficient",
  "observations": [],
  "ruled_out": [],
  "unknowns": [],
  "diagnosis": "直接概括 reference 语义和实现约束",
  "items": [
    {
      "id": "bootstrap-1",
      "kind": "correctness",
      "hypothesis": "一个可验证的端到端 AscendC 实现假设",
      "change": "覆盖算法、block/tiling、内存和数据搬运、dtype/shape/tail、Host ABI 及源码布局的完整蓝图",
      "expected_signal": "静态校验、编译、全部正确性用例和有效 benchmark score 均成功",
      "target_files": ["model_new_ascendc.py", "kernel/pybind11.cpp", "kernel/<name>.cpp"],
      "edit_scope": "file",
      "allow_interface_change": true,
      "evidence_refs": [],
      "falsifies": [],
      "order": 1
    }
  ]
}
返回恰好一个完整 baseline item，不要在此响应中返回源码文件。
不得使用或建议 TileLang、其他 DSL、中间实现或 source-to-source conversion。
""".strip()


COMPILATION_REPAIR_CONTRACT = """
- 当前实现是 accepted repair base，修复必须在其上累积。
- 只解决列出的 open compiler errors，并保留 cleared error IDs 代表的修复。
- 不得重新引入已清除的诊断，不得做性能重构或无关 ABI 修改。
- 普通 runtime `if` 不能阻止 C++ template instantiation。不支持的 API/dtype 组合必须在编译期隔离，
  或使用当前 toolchain 支持的语法放入类型专用实现。
- installed declarations 和当前 compiler diagnostics 是权威证据；不得虚构 overload 或用不受支持的 cast 掩盖类型错误。
""".strip()


STAGE_CONTRACTS = {
    "bootstrap_generation": """
- 生成覆盖 reference 和全部测试用例约束的完整端到端实现。
- 明确 block/tiling、数据搬运、dtype、shape、tail、Host ABI 和文件布局，不得只生成局部脚手架。
""".strip(),
    "compile_repair": COMPILATION_REPAIR_CONTRACT,
    "runtime_repair": """
- 只修复已有 runtime code、signal、地址、descriptor 生命周期、GM/UB 范围或 DataCopy span 证据指向的问题。
- 不得把“Kernel 是否启动”等 unknown gate 当作已确认事实；保留已通过的编译、Host ABI 和无关算术路径。
- 修改必须对应一个可由同一 failing case 验证的运行时根因。
""".strip(),
    "correctness_repair": """
- 以首个 failing case、首个 mismatch、reference 语义和当前 tiling 为依据，只修复一个根因相关区域。
- 先区分 formula、index/offset、tail、layout、dtype accumulation、同步或 Host/Kernel contract，不得无证据改 ABI。
- 保留全部已通过 case、编译修复、模块接口和 Kernel launch。
""".strip(),
    "performance_tuning": """
- 仅在 full correctness 已通过且存在真实性能测量时优化；没有测量时不得猜测瓶颈。
- 只提出并执行一个可证伪的局部优化，说明目标 latency/counter 和回退条件。
- 禁止改变 API、shape 覆盖、数值语义和已通过的完整正确性。
""".strip(),
    "optimization": """
- 当前 accepted baseline 已通过完整正确性。只根据实际 latency、operator、memory 或 profiler 证据执行一个局部优化。
- 保持 API、shape 覆盖和数值语义；候选必须重新通过完整正确性，并由同一 case set 比较性能。
""".strip(),
}


def render_repair_state(repair_state: dict[str, Any] | None) -> str:
    """Render only the run-local state relevant to currently open errors."""

    if not repair_state:
        return "（没有活动的轨迹修复状态）"
    lines = [
        f"已接受 attempt：{repair_state.get('accepted_attempt_id') or 'none'}",
        f"最近拒绝 attempt：{repair_state.get('latest_rejected_attempt_id') or 'none'}",
        "",
        "开放错误：",
    ]
    open_errors = repair_state.get("open_errors", [])
    if open_errors:
        for item in open_errors:
            lines.append(
                "- "
                + str(item.get("error_id", "unknown"))
                + f"; symbol={item.get('symbol') or 'unknown'}; category={item.get('category') or 'unknown'}"
                + f"; evidence={item.get('normalized_message') or item.get('excerpt') or ''}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "不得回归的已清除错误："])
    cleared = repair_state.get("cleared_error_ids", [])
    lines.extend([f"- {item}" for item in cleared] or ["- none"])
    lines.extend(["", "与当前开放错误相关的历史失败方案："])
    relevant = repair_state.get("failed_approaches_by_error", {})
    if not relevant:
        lines.append("- none")
    for error_id, approaches in relevant.items():
        lines.append(f"- {error_id}")
        for approach in approaches:
            attempts = ",".join(str(item) for item in approach.get("attempt_ids", []))
            lines.append(
                f"  - {approach.get('approach_id')} (family={approach.get('approach_family_id', 'unknown')}): {approach.get('summary', '')} "
                f"(outcome={approach.get('outcome')}; occurrences={approach.get('occurrences', 1)}; "
                f"family_occurrences={approach.get('family_occurrences', approach.get('occurrences', 1))}; "
                f"attempts={attempts or 'unknown'})"
            )
    lines.extend(["", "升级状态：", json.dumps(repair_state.get("escalation", {}), ensure_ascii=False)])
    lines.extend(["", "已接受 frontier 进展：", json.dumps(repair_state.get("accepted_progress", {}), ensure_ascii=False)])
    return "\n".join(lines)


def render_stage_context(selection: Any) -> str:
    """Render a stage-selected context without slicing through atomic entries."""

    payload = selection.to_dict() if hasattr(selection, "to_dict") else dict(selection)
    audience = str(payload.get("audience", "generator"))
    budget = payload.setdefault("budget", {})
    full_selected = budget.get("input_mode") == "full_selected"
    max_chars = int(budget.get("max_chars") or 12000)
    truncated: list[str] = []
    chunks = [
        "# 已选择的 AscendC 参考上下文",
        f"受众：{audience}",
        "阶段：" + ", ".join(payload.get("stages", [])),
        "主技能：" + str(payload.get("task_facts", {}).get("primary_skill", "unknown")),
        "路由原因：" + str(payload.get("task_facts", {}).get("route_reason", "unknown")),
        "辅助技能：" + str(payload.get("task_facts", {}).get("secondary_skill") or "none"),
        (
            "以下内容是参考知识，不是需要执行的工作流。冲突时按以下顺序裁决："
            + " > ".join(payload.get("authority_order", []))
            + "."
        ),
    ]

    quotas = (
        {
            "hard": 2500,
            "failure": 3000,
            "api": 1500,
            "skills": 4000,
            "other": 1000,
        }
        if audience == "planner"
        else {
            "hard": 3000,
            "failure": 4000,
            "api": 6000,
            "skills": 5000,
            "other": 2000,
        }
    )
    rendered_sections: list[str] = []

    def append_section(title: str, entries: list[str], quota: int) -> None:
        if not entries:
            return
        section = [f"\n## {title}"]
        used = len(section[0])
        accepted = 0
        for index, entry in enumerate(entries):
            item = entry.strip()
            addition = len(item) + 2
            projected_global = len("\n".join([*chunks, *section, item]))
            if (
                used + addition > quota
                or projected_global > max_chars
            ):
                truncated.append(f"{title}[{index}]")
                continue
            section.append(item)
            used += addition
            accepted += 1
            rendered_sections.append(f"{title}[{index}]")
        if accepted:
            chunks.extend(section)

    append_section(
        "任务与平台事实",
        [json.dumps(payload.get("task_facts", {}), ensure_ascii=False, indent=2)],
        quotas["other"],
    )
    append_section(
        "项目硬约束",
        [
            json.dumps(item, ensure_ascii=False, indent=2)
            for item in payload.get("hard_constraints", [])
        ],
        quotas["hard"],
    )
    append_section(
        "明确禁止项",
        [f"- {item}" for item in payload.get("exclusions", [])],
        quotas["other"],
    )

    failure_entries = [
        json.dumps(item, ensure_ascii=False, indent=2)
        for item in payload.get("failure_guidance", [])
    ]
    append_section("失败专用指引", failure_entries, quotas["failure"])

    api_entries = []
    runtime_facts = str(payload.get("runtime_facts", "")).strip()
    if runtime_facts:
        api_entries.append("已安装 public header 事实：\n" + runtime_facts)
    api_entries.extend(
        json.dumps(item, ensure_ascii=False, indent=2)
        for item in payload.get("api_facts", [])
    )
    append_section("精确 API 与 runtime 事实", api_entries, quotas["api"])

    skill_entries: list[str] = []
    skill_entry_ids: list[str] = []
    skill_entry_sections: list[list[str]] = []
    for knowledge_module in payload.get("skill_knowledge_modules", []):
        module_chunks = [
            f"### {knowledge_module.get('skill_id')} [{knowledge_module.get('stage')}; role={knowledge_module.get('role', 'support')}]",
            f"用途：{knowledge_module.get('purpose', '')}",
            f"触发原因：{knowledge_module.get('trigger_reason', '')}",
            f"预期产物：{knowledge_module.get('expected_artifact', '')}",
        ]
        section_ids: list[str] = []
        for excerpt in knowledge_module.get("excerpts", []):
            headings = ", ".join(excerpt.get("headings", [])) or "完整知识模块"
            module_chunks.append(
                f"#### {knowledge_module.get('skill_id')} 来源：{excerpt.get('source')}（{headings}）\n"
                f"来源：{excerpt.get('origin', 'cannbot')}；confidence=Level {excerpt.get('confidence_level', 1)}；provenance={excerpt.get('provenance', 'documented_skill')}\n"
                + str(excerpt.get("text", ""))
            )
            if excerpt.get("knowledge_module_id") and excerpt.get("section_id"):
                section_ids.append(
                    f"{excerpt.get('knowledge_module_id')}#{excerpt.get('section_id')}"
                )
        skill_entries.append("\n".join(module_chunks))
        skill_entry_ids.append(str(knowledge_module.get("skill_id", "")))
        skill_entry_sections.append(section_ids)
    # 知识模块以完整 section 为原子单位；bounded 模式只跳过整项，不截断正文。
    rendered_knowledge_sections: list[str] = []
    rendered_skill_ids: list[str] = []
    if skill_entries:
        heading = "\n## 相关内置知识模块"
        accepted: list[str] = []
        for index, item in enumerate(skill_entries):
            normalized = item.strip()
            projected_global = len("\n".join([*chunks, heading, *accepted, normalized]))
            if not full_selected and projected_global > max_chars:
                truncated.append(f"相关内置知识模块[{index}]")
                continue
            accepted.append(normalized)
            rendered_sections.append(f"相关内置知识模块[{index}]")
            if skill_entry_ids[index]:
                rendered_skill_ids.append(skill_entry_ids[index])
            rendered_knowledge_sections.extend(skill_entry_sections[index])
        if accepted:
            chunks.append(heading)
            chunks.extend(accepted)

    other_entries = [
        f"- {item}" for item in payload.get("design_patterns", [])
    ]
    append_section("已选设计模式", other_entries, quotas["other"])
    append_section(
        "紧凑 provenance",
        [
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in payload.get("provenance", [])
        ],
        quotas["other"],
    )
    text = "\n".join(chunks)
    budget["used_chars"] = len(text)
    budget["estimated_tokens"] = (len(text) + 3) // 4
    budget["truncated_sections"] = truncated
    metadata = payload.setdefault("selection_metadata", {})
    metadata["rendered_sections"] = rendered_sections
    metadata["truncated_sections"] = truncated
    metadata["rendered_chars"] = len(text)
    metadata["rendered_estimated_tokens"] = (len(text) + 3) // 4
    metadata["input_mode"] = "full_selected" if full_selected else "bounded"
    metadata["input_truncated"] = bool(truncated)
    metadata["rendered_skill_ids"] = rendered_skill_ids
    metadata["rendered_knowledge_module_sections"] = rendered_knowledge_sections
    if full_selected:
        metadata["rendered_structured_ids"] = [
            str(item.get("fact_id") or item.get("card_id"))
            for item in payload.get("api_facts", [])
            if item.get("fact_id") or item.get("card_id")
        ]
        metadata["rendered_runtime_fact_ids"] = list(
            metadata.get("runtime_fact_ids", [])
        )
    if hasattr(selection, "budget"):
        selection.budget = budget
    if hasattr(selection, "selection_metadata"):
        selection.selection_metadata = metadata
    return text


def _bundle_text(bundle: FileBundle | None) -> str:
    if bundle is None or not bundle.files:
        return "（当前还没有实现）"
    chunks = []
    for path, content in sorted(bundle.files.items()):
        chunks.append(f"### {path}\n```\n{content}\n```")
    return "\n\n".join(chunks)


def build_prompt(
    *,
    reference_code: str,
    cases_text: str,
    current: FileBundle | None,
    previous_result: EvalResult | None,
    round_num: int,
    knowledge_context: str,
    phase: str = "BOOTSTRAP",
    plan_item: dict | None = None,
    failure_fingerprints: list[str] | None = None,
    repair_state: dict[str, Any] | None = None,
    knowledge_selection: Any = None,
    protected_regions: dict[str, Any] | None = None,
) -> str:
    if previous_result is None:
        feedback = "没有上一轮评测；生成初始实现。"
    else:
        feedback = json.dumps(
            compact_evaluation(previous_result),
            ensure_ascii=False,
            indent=2,
        )
    protected_regions = protected_regions or {
        "stage": "bootstrap_generation",
        "protected_region_ids": [],
        "protected_snippets": {},
    }
    edit_stage = str(protected_regions.get("stage") or "bootstrap_generation")
    if edit_stage not in STAGE_CONTRACTS:
        edit_stage = "bootstrap_generation"
    if previous_result and previous_result.failure_stage == "performance":
        edit_stage = "performance_tuning"
    action = {
        "bootstrap_generation": "生成完整的初始 AscendC 实现。",
        "compile_repair": "只修复当前开放的编译或静态校验错误。",
        "runtime_repair": "只修复当前运行时证据指向的一个根因。",
        "correctness_repair": "只修复当前正确性证据指向的一个根因。",
        "performance_tuning": "仅根据已有测量修复 benchmark 或优化性能。",
        "optimization": "在完整正确 baseline 上执行一个局部性能优化。",
    }[edit_stage]
    plan_text = (
        json.dumps(plan_item, ensure_ascii=False, indent=2)
        if plan_item
        else "（没有活动 plan item；仅历史 EVAL checkpoint 可出现）"
    )
    selection_payload = (
        knowledge_selection.to_dict()
        if hasattr(knowledge_selection, "to_dict")
        else dict(knowledge_selection or {})
    )
    task_facts = selection_payload.get("task_facts", {})
    ownership = str(task_facts.get("failure_ownership") or "Kernel")
    active_profile = str(task_facts.get("active_profile") or "完整评测或未指定")
    profile_details = json.dumps(
        {
            "case_indices": task_facts.get("profile_case_indices", []),
            "features": task_facts.get("profile_features", []),
            "diagnostic_source_files": task_facts.get("diagnostic_source_files", []),
            "input_route": task_facts.get("input_route", {}),
        },
        ensure_ascii=False,
        indent=2,
    )
    protected_text = json.dumps(protected_regions, ensure_ascii=False, indent=2)
    forbidden = list(dict.fromkeys(selection_payload.get("exclusions", [])))
    forbidden_text = "\n".join(f"- {item}" for item in forbidden) or "- 遵守下述强制规则中的禁止项。"
    repair_section = f"## 开放错误、已清除错误和相关失败方案\n{render_repair_state(repair_state)}\n\n"
    return f"""# AscendC {phase.lower()} 编辑 attempt {round_num}

## 当前目标
{action}

活动 plan item：
```json
{plan_text}
```

## 失败归属
{ownership}

## 当前评测 profile
{active_profile}

## 失败或新引入的 case 特征
```json
{profile_details}
```

## 受保护区域
```json
{protected_text}
```

{repair_section}## 当前阶段契约
阶段：{edit_stage}
{STAGE_CONTRACTS[edit_stage]}

## 必须满足的语义和 ABI 契约
{RULES}

## 精确 installed/verified 事实和相关知识模块
{knowledge_context}

## 明确禁止模式
{forbidden_text}

## Reference PyTorch model 与 benchmark 语义（只读）
```python
{reference_code}
```

## 测试用例
```jsonl
{cases_text}
```

## 当前实现
{_bundle_text(current)}

## 上一轮评测
```json
{feedback}
```

## 输出契约
{OUTPUT_CONTRACT}
"""


def build_plan_prompt(
    *,
    reference_code: str,
    cases_text: str,
    current: FileBundle | None,
    result: EvalResult | None,
    mode: str,
    history: list[dict],
    diagnosis_required: bool = False,
    knowledge_context: str = "",
    initial: bool = False,
    repair_state: dict[str, Any] | None = None,
) -> str:
    feedback = (
        json.dumps(
            compact_evaluation(result),
            ensure_ascii=False,
            indent=2,
        )
        if result is not None
        else "尚无实现完成评测。"
    )
    ledger_lines: list[str] = []
    for record in history[-8:]:
        repair = record.get("repair_attempt", {}) if isinstance(record, dict) else {}
        item = record.get("plan_item", {}) if isinstance(record, dict) else {}
        ledger_lines.append(
            " | ".join(
                (
                    f"attempt={record.get('attempt_id', record.get('round'))}",
                    f"hypothesis={str(item.get('hypothesis', ''))[:240]}",
                    f"outcome={repair.get('outcome') or record.get('decision')}",
                    f"cleared={repair.get('cleared_error_ids', [])}",
                    f"new={repair.get('new_error_ids', [])}",
                    f"progress={repair.get('progress', {})}",
                )
            )
        )
    ledger = "\n".join(ledger_lines) or "（没有已完成 attempt）"
    if initial:
        purpose = (
            "在生成任何源码前，用 AscendC 术语给出一个完整实现蓝图。"
        )
        execution_rules = (
            "唯一的 plan item 必须描述完整候选，而不是局部文件或分阶段半成品。"
            "只能依据 reference 语义、测试用例、CANN 约束和 AscendC 执行模型推理。"
        )
        output_contract = INITIAL_PLAN_OUTPUT_CONTRACT
    else:
        purpose = (
            "诊断上一方案重复失败的原因；只有证据充分时才给出实质不同的下一方案。"
            if diagnosis_required
            else "创建下一项证据驱动的实施计划。"
        )
        execution_rules = (
            "针对当前 accepted implementation 返回一个纵向完整的下一步。一个主假设需要时可以修改多个文件，\n"
            "但不得换名重复失败方案。DIAGNOSE 证据不足时返回零 item，不得猜测源码根因。"
        )
        output_contract = (
            DIAGNOSE_OUTPUT_CONTRACT if diagnosis_required else PLAN_OUTPUT_CONTRACT
        )
    repair_section = (
        f"## 当前轨迹修复状态\n{render_repair_state(repair_state)}\n\n"
        if repair_state and mode == "bootstrap" and not initial
        else ""
    )
    return f"""# AscendC {mode} 规划

{purpose}
{execution_rules}

## 强制规则
{RULES}

项目的 Host launch ABI 固定：pybind 声明并调用 extern "C" *_do wrapper；每个 wrapper 与其
__aicore__ Kernel 放在一起，并使用 kernel<<<blockDim, nullptr, stream>>>。
不支持 acl/acl_rt_launch.h 和 ACLRT_LAUNCH_KERNEL。

## Reference PyTorch model（只读）
```python
{reference_code}
```

## 测试用例
```jsonl
{cases_text}
```

## 当前实现
{_bundle_text(current)}

## 最新评测证据
```json
{feedback}
```

## Attempt 记录
{ledger}

## 按阶段选择的 AscendC 参考
{knowledge_context or "（未选择额外参考材料）"}

{repair_section}## 输出契约
{output_contract}
"""


def parse_plan(
    text: str,
    *,
    min_items: int = 1,
    max_items: int = 1,
    require_evidence: bool = False,
    allow_insufficient: bool = False,
) -> dict:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("planning response does not contain a JSON object")
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError("planning response must contain an items list")
    evidence_status = str(payload.get("evidence_status", "")).strip()
    if evidence_status not in {"sufficient", "insufficient"}:
        raise ValueError("planning response evidence_status must be sufficient or insufficient")
    observations = payload.get("observations")
    ruled_out = payload.get("ruled_out")
    unknowns = payload.get("unknowns")
    if not isinstance(observations, list) or not all(
        isinstance(item, dict)
        and str(item.get("source", "")).strip()
        and str(item.get("line_excerpt", "")).strip()
        and str(item.get("interpretation", "")).strip()
        for item in observations
    ):
        raise ValueError("planning response observations must contain complete evidence objects")
    if not isinstance(ruled_out, list) or not all(
        isinstance(item, dict)
        and str(item.get("hypothesis", "")).strip()
        and str(item.get("reason", "")).strip()
        and isinstance(item.get("evidence_refs", []), list)
        and all(
            isinstance(ref, dict)
            and str(ref.get("source", "")).strip()
            and str(ref.get("line_excerpt", "")).strip()
            for ref in item.get("evidence_refs", [])
        )
        for item in ruled_out
    ):
        raise ValueError("planning response ruled_out must contain complete hypothesis objects")
    if not isinstance(unknowns, list) or not all(
        isinstance(item, dict)
        and str(item.get("question", "")).strip()
        and str(item.get("required_evidence", "")).strip()
        for item in unknowns
    ):
        raise ValueError("planning response unknowns must contain complete unknown objects")
    if evidence_status == "insufficient":
        if not allow_insufficient:
            raise ValueError("only DIAGNOSE may return insufficient evidence")
        if payload["items"]:
            raise ValueError("insufficient evidence must return zero plan items")
        if not unknowns:
            raise ValueError("insufficient evidence must list required unknowns")
    elif not min_items <= len(payload["items"]) <= max_items:
        expected = str(min_items) if min_items == max_items else f"{min_items} to {max_items}"
        raise ValueError(f"planning response must contain {expected} items")
    if require_evidence and evidence_status == "sufficient" and not observations:
        raise ValueError("diagnosis with sufficient evidence must list observations")
    normalized = []
    required = ("id", "kind", "hypothesis", "change")
    for index, item in enumerate(payload["items"], 1):
        if not isinstance(item, dict) or any(not str(item.get(key, "")).strip() for key in required):
            raise ValueError(f"plan item {index} is incomplete")
        expected_signal = str(item.get("expected_signal", "")).strip()
        change = str(item["change"]).strip()
        if not expected_signal:
            embedded = re.search(
                r"(?ims)^\s*\d+[.)]\s*expected signal\s*$\n(?P<value>.+)\Z",
                change,
            )
            if embedded:
                expected_signal = embedded.group("value").strip()
                change = change[: embedded.start()].rstrip()
        if not expected_signal:
            raise ValueError(f"plan item {index} is incomplete")
        target_files = item.get("target_files", [])
        evidence_refs = item.get("evidence_refs", [])
        falsifies = item.get("falsifies", [])
        if not isinstance(target_files, list) or not all(isinstance(path, str) for path in target_files):
            raise ValueError(f"plan item {index} target_files must be a string list")
        if not isinstance(evidence_refs, list) or not all(isinstance(ref, dict) for ref in evidence_refs):
            raise ValueError(f"plan item {index} evidence_refs must be an object list")
        if not isinstance(falsifies, list) or not all(isinstance(value, str) for value in falsifies):
            raise ValueError(f"plan item {index} falsifies must be a string list")
        if require_evidence and (
            not evidence_refs
            or any(not str(ref.get("line_excerpt", "")).strip() for ref in evidence_refs)
        ):
            raise ValueError(f"diagnosis plan item {index} must cite exact evidence excerpts")
        if require_evidence and not falsifies:
            raise ValueError(f"diagnosis plan item {index} must identify a falsified prior approach")
        normalized.append(
            {
                **{key: str(item[key]).strip() for key in required if key != "change"},
                "change": change,
                "expected_signal": expected_signal,
                "target_files": [path.strip() for path in target_files if path.strip()],
                "edit_scope": str(item.get("edit_scope", "file")).strip() or "file",
                "allow_interface_change": bool(
                    item.get("allow_interface_change", item.get("allow_abi_change", False))
                ),
                "evidence_refs": evidence_refs,
                "falsifies": [value.strip() for value in falsifies if value.strip()],
                "order": int(item.get("order", index)),
            }
        )
    diagnosis = str(payload.get("diagnosis", "")).strip()
    if not diagnosis:
        raise ValueError("planning response diagnosis must be non-empty")
    return {
        "evidence_status": evidence_status,
        "observations": observations,
        "ruled_out": ruled_out,
        "unknowns": unknowns,
        "diagnosis": diagnosis,
        "items": normalized,
    }
