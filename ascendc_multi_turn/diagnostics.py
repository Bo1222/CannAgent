from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import EvalResult
from .state_schema import DiagnosticRecord, FailureEvidence

ERROR_LINE = re.compile(
    r"error(?:\[[A-Za-z0-9_.-]+\])?:|fatal(?: error)?:|traceback|calledprocesserror|timed out|\[fail(?:ed)?\]",
    re.IGNORECASE,
)
QUOTED_SYMBOL = re.compile(r"'(?:AscendC::)?([A-Za-z_][A-Za-z0-9_:<>]*)'")
SOURCE_SYMBOL = re.compile(r"\b(?:AscendC::)?([A-Z][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?\s*\(")
TYPE_SYMBOL = re.compile(r"\b(?:AscendC::)?([A-Z][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?")
BOUNDARY_SYMBOL = re.compile(
    r"\b(GM_ADDR|__gm__|PYBIND11_MODULE|getCurrentNPUStream|is_npu|[A-Za-z_]\w*_do)\b"
)
SEMANTIC_SYMBOL = re.compile(r"\b(broadcast|stride|shape|tail|dtype|descriptor)\b", re.IGNORECASE)
_SYMBOL_NOISE = {
    "CMake",
    "COMPILER",
    "Compiler",
    "Error",
    "Exception",
    "Failed",
    "Kernel",
    "Model",
    "PYBIND11_MODULE",
    "Tensor",
    "TraceBack",
    "Traceback",
}
_ASCENDC_API_SYMBOLS = {
    "Abs", "Add", "Adds", "Cast", "Compare", "Copy", "DataCopy",
    "DataCopyExtParams", "DataCopyPad", "DataCopyPadExtParams", "Div", "Erf",
    "Exp", "FreeTensor", "Gather", "GatherMask", "GlobalTensor", "InitBuffer",
    "LocalTensor", "Log", "Max", "Min", "Mul", "Muls", "PipeBarrier",
    "ReduceMax", "ReduceMean", "ReduceMin", "ReduceSum", "SetFlag", "Sqrt",
    "Sub", "TBuf", "TBufPool", "TEventID", "TPipe", "TQue", "TQueBind",
    "Tanh", "TransDataTo5HD", "WaitFlag",
}
_HOST_ABI_SYMBOLS = {"getCurrentNPUStream", "is_npu"}
_LOCATED_ERROR = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^:\n]+):\d+(?::\d+)?:\s*"
    r"(?P<severity>fatal\s+error|error)(?:\[(?P<rule>[A-Za-z0-9_.-]+)\])?:\s*"
    r"(?P<message>.+)$",
    re.IGNORECASE,
)
_PLAIN_ERROR = re.compile(
    r"(?P<severity>fatal\s+error|error)(?:\[(?P<rule>[A-Za-z0-9_.-]+)\])?:\s*"
    r"(?P<message>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SymbolEvidence:
    symbol: str
    kinds: tuple[str, ...]
    sources: tuple[str, ...]
    domains: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnostic_lines(output: str, *, max_lines: int = 30, max_chars: int = 6000) -> list[str]:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    selected: list[str] = []
    seen: set[str] = set()
    for line in lines:
        if not ERROR_LINE.search(line):
            continue
        # Parallel builds may concatenate two absolute paths. Keep the useful
        # compiler suffix while deduplicating exact repeats.
        marker = line.rfind("/mnt/")
        if marker > 0:
            line = line[marker:]
        if line in seen:
            continue
        seen.add(line)
        if sum(len(item) + 1 for item in selected) + len(line) > max_chars:
            break
        selected.append(line)
        if len(selected) >= max_lines:
            break
    if selected:
        return selected
    return lines[-min(max_lines, len(lines)) :]


def compact_diagnostics(output: str, *, max_lines: int = 30, max_chars: int = 6000) -> str:
    return "\n".join(
        diagnostic_lines(
            collapse_device_dump(output), max_lines=max_lines, max_chars=max_chars
        )
    )


def collapse_device_dump(output: str) -> str:
    """Collapse repeated per-core device diagnostics into an auditable summary."""

    lines = output.splitlines()
    device_lines = [
        line.strip()
        for line in lines
        if re.search(r"the error from device|the extend info", line, re.IGNORECASE)
    ]
    if len(device_lines) < 3:
        return output

    def integers(pattern: str) -> list[int]:
        return sorted(
            {
                int(value)
                for value in re.findall(pattern, output, flags=re.IGNORECASE)
            }
        )

    def tokens(pattern: str) -> list[str]:
        return list(
            dict.fromkeys(
                value.strip().rstrip(",;)")
                for value in re.findall(pattern, output, flags=re.IGNORECASE)
                if value.strip()
            )
        )

    summary = {
        "repeated_device_records": len(device_lines),
        "cores": integers(r"core[_ ]?id\s*[:=]\s*(\d+)"),
        "blocks": integers(r"(?:block|blk)[_ ]?id\s*[:=]\s*(\d+)"),
        "pc_start": tokens(r"(?:pc\s*start|start\s*pc)\s*[:=]\s*(0x[0-9a-f]+)"),
        "pc_current": tokens(r"(?:pc\s*current|current\s*pc)\s*[:=]\s*(0x[0-9a-f]+)"),
        "serial": tokens(r"serial(?:\s*number)?\s*[:=]\s*([^,;\s]+)"),
        "extend_error_str": tokens(r"(?:errorstr|error[_ ]str)\s*[:=]\s*([^,;\n]+)"),
    }
    retained: list[str] = []
    kept_template = False
    for line in lines:
        if re.search(r"the error from device|the extend info", line, re.IGNORECASE):
            if not kept_template:
                retained.append(line)
                kept_template = True
            continue
        retained.append(line)
    retained.append(
        "[设备诊断折叠] " + json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    )
    return "\n".join(retained)


def diagnostic_fingerprint(output: str) -> str:
    normalized = []
    for line in diagnostic_lines(output):
        line = re.sub(r"/[^ :]+/", "/…/", line)
        line = re.sub(r":\d+(?::\d+)?", ":N", line)
        normalized.append(line)
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()[:16]


def _diagnostic_source_file(path: str | None) -> str | None:
    if not path:
        return None
    normalized = path.strip().replace("\\", "/")
    kernel_marker = normalized.rfind("/kernel/")
    if kernel_marker >= 0:
        return normalized[kernel_marker + 1 :]
    name = Path(normalized).name
    return name or None


def _diagnostic_symbol(message: str) -> str | None:
    for pattern in (
        r"(?:call to|initialization of)\s+'(?:AscendC::)?([A-Za-z_]\w*)'",
        r"(?:dtype|check dtype)\s+in\s+([A-Za-z_]\w*)",
        r"function template specialization\s+'(?:AscendC::)?([A-Za-z_]\w*)",
        r"undeclared identifier\s+'(?:AscendC::)?([A-Za-z_]\w*)'",
        r"undefined (?:reference|symbol).*?'(?:AscendC::)?([A-Za-z_]\w*)'",
    ):
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
    symbols = extract_api_symbols(message)
    return symbols[0] if symbols else None


def _diagnostic_category(stage: str, message: str, source_file: str | None) -> str:
    lowered = message.lower()
    if any(marker in lowered for marker in ("gm_addr", "__gm__", "address space", "descriptor", "const void")):
        return "launch_abi"
    if any(
        marker in lowered
        for marker in (
            "is_cuda",
            "cuda tensor",
            "at::cuda",
            "c10::cuda",
            "getcurrentcudastream",
        )
    ):
        return "host_abi"
    if source_file and Path(source_file).name == "pybind11.cpp":
        if any(marker in lowered for marker in (".vec", "std::vector", "no member named", "include file", "file not found", "no such file")):
            return "host_cpp"
        return "host_abi"
    if "static assertion failed" in lowered and "dtype" in lowered:
        return "kernel_api_dtype"
    if "no matching function" in lowered:
        return "kernel_api_overload"
    if "no matching constructor" in lowered:
        return "cpp_type_construction"
    if "undeclared identifier" in lowered or "was not declared" in lowered:
        return "missing_symbol"
    if "undefined reference" in lowered or "undefined symbol" in lowered:
        return "linker"
    if "template" in lowered or "instantiation" in lowered:
        return "kernel_api_template"
    if stage in {"ascendc_build", "compile"}:
        return "kernel_compile"
    return stage or "unknown"


def _normalize_diagnostic_message(message: str) -> str:
    normalized = re.sub(r"/[^\s:'\"]+", "/…", message)
    normalized = re.sub(r":\d+(?::\d+)?", ":N", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def extract_diagnostic_records(*, stage: str, output: str) -> list[DiagnosticRecord]:
    """Normalize direct tool diagnostics for comparison; never validates source code."""

    records: list[DiagnosticRecord] = []
    seen: set[str] = set()
    for raw_line in output.splitlines():
        line = raw_line.strip()
        located = _LOCATED_ERROR.search(line)
        plain = located or _PLAIN_ERROR.search(line)
        if plain is None:
            continue
        message = plain.group("message").strip()
        source_file = _diagnostic_source_file(
            located.group("path") if located is not None else None
        )
        symbol = _diagnostic_symbol(message)
        rule = plain.groupdict().get("rule")
        category = rule.lower() if rule else _diagnostic_category(stage, message, source_file)
        normalized = _normalize_diagnostic_message(message)
        identity = "|".join(
            (stage or "unknown", category, symbol or "unknown", normalized)
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
        error_id = f"{category}:{symbol or 'unknown'}:{digest}"
        if error_id in seen:
            continue
        seen.add(error_id)
        records.append(
            DiagnosticRecord(
                error_id=error_id,
                stage=stage or "unknown",
                category=category,
                symbol=symbol,
                source_file=source_file,
                normalized_message=normalized,
                evidence_origin="compiler" if stage in {"ascendc_build", "compile"} else "evaluator",
                excerpt=line[:2000],
            )
        )
    if len(records) > 1:
        records = [
            item
            for item in records
            if not (
                item.category == "kernel_compile"
                and item.symbol and item.symbol.lower() == "cmake"
                and "returned non-zero exit status" in item.normalized_message.lower()
            )
        ]
    return records


def diagnostics_for_result(result: EvalResult | None) -> list[DiagnosticRecord]:
    if result is None or not result.error:
        return []
    output = read_result_log(result)
    records = extract_diagnostic_records(
        stage=result.failure_stage or "unknown",
        output=output,
    )
    if records:
        return records
    message = _normalize_diagnostic_message(
        result.error_excerpt or result.error or result.failure_code or "unknown failure"
    )
    identity = "|".join(
        (result.failure_stage or "unknown", "stage_error", "unknown", message)
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return [
        DiagnosticRecord(
            error_id=f"stage_error:unknown:{digest}",
            stage=result.failure_stage or "unknown",
            category="stage_error",
            symbol=None,
            source_file=None,
            normalized_message=message,
            evidence_origin="evaluator",
            excerpt=(result.error_excerpt or result.error)[:1000],
        )
    ]


def diagnostic_delta(
    before: list[DiagnosticRecord],
    after: list[DiagnosticRecord],
    *,
    historically_cleared: list[str] | None = None,
) -> dict[str, list[str]]:
    before_ids = {item.error_id for item in before}
    after_ids = {item.error_id for item in after}
    historical = set(historically_cleared or [])
    return {
        "before": sorted(before_ids),
        "after": sorted(after_ids),
        "cleared": sorted(before_ids - after_ids),
        "new": sorted(after_ids - before_ids),
        "reintroduced": sorted(after_ids & historical),
    }


def classify_symbol_domain(symbol: str) -> str:
    base = symbol.split("::")[-1].split("<", 1)[0]
    if base in _SYMBOL_NOISE or base.lower() in {"compiler", "traceback"}:
        return "compiler_noise"
    if base in _HOST_ABI_SYMBOLS:
        return "host_abi"
    if base in {"GM_ADDR", "__gm__"} or base.endswith("_do") or base.lower() == "descriptor":
        return "launch_abi"
    if base.lower() in {"broadcast", "stride", "shape", "tail", "dtype"}:
        return "operator_semantic"
    if base in _ASCENDC_API_SYMBOLS or base.startswith(("DataCopy", "Reduce")):
        return "ascendc_api"
    return "project_local"


def _extract_symbols(*texts: str) -> list[tuple[str, str]]:
    symbols: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        symbol = token.split("::")[-1].split("<", 1)[0]
        if len(symbol) >= 3 and symbol not in seen:
            seen.add(symbol)
            symbols.append((symbol, classify_symbol_domain(symbol)))

    for text in texts:
        for match in QUOTED_SYMBOL.finditer(text):
            add(match.group(1))
        for token in SOURCE_SYMBOL.findall(text):
            add(token)
        for match in TYPE_SYMBOL.finditer(text):
            token = match.group(1)
            if token.startswith(("DataCopy", "GlobalTensor", "LocalTensor", "TQue", "TPipe", "TBuf")):
                add(token)
        for token in BOUNDARY_SYMBOL.findall(text):
            add(token)
        for token in SEMANTIC_SYMBOL.findall(text):
            add(token.lower())
    return symbols


def extract_api_symbols(*texts: str) -> list[str]:
    """Return only symbols eligible for installed-header/API lookup."""

    return [
        symbol
        for symbol, domain in _extract_symbols(*texts)
        if domain in {"ascendc_api", "host_abi", "launch_abi"}
    ]


def extract_symbol_evidence(
    *,
    failure: str = "",
    source: str = "",
    planned: str = "",
) -> list[SymbolEvidence]:
    """Extract API symbols while retaining every deterministic evidence origin."""

    ordered: list[str] = []
    evidence: dict[str, dict[str, list[str]]] = {}
    for kind, label, text in (
        ("failure", "evaluation", failure),
        ("source", "candidate", source),
        ("planned", "active_plan", planned),
    ):
        for symbol, domain in _extract_symbols(text):
            if symbol not in evidence:
                ordered.append(symbol)
                evidence[symbol] = {"kinds": [], "sources": [], "domains": []}
            if kind not in evidence[symbol]["kinds"]:
                evidence[symbol]["kinds"].append(kind)
            if label not in evidence[symbol]["sources"]:
                evidence[symbol]["sources"].append(label)
            if domain not in evidence[symbol]["domains"]:
                evidence[symbol]["domains"].append(domain)
    return [
        SymbolEvidence(
            symbol=symbol,
            kinds=tuple(evidence[symbol]["kinds"]),
            sources=tuple(evidence[symbol]["sources"]),
            domains=tuple(evidence[symbol]["domains"]),
        )
        for symbol in ordered
    ]


def observe_evaluation_stages(
    result: EvalResult,
    *,
    workflow_phase: str = "bootstrap",
) -> dict[str, dict[str, str]]:
    """Build an observation-only A-H ledger without changing evaluator decisions."""

    stage = result.failure_stage or ""
    validation_failures = {
        "bundle_validation",
        "ascendc_source_validation",
        "api_constraint_validation",
        "static_validation",
        "interface_contract_validation",
        "response_format",
    }
    observed: dict[str, dict[str, str]] = {}

    def put(name: str, status: str, reason: str) -> None:
        observed[name] = {"status": status, "reason": reason}

    if stage in validation_failures:
        put("A_source_valid", "fail", f"failure_stage={stage}")
    else:
        put("A_source_valid", "pass", "evaluation reached compiler or later")

    if result.compiled:
        put("B_compile", "pass", "EvalResult.compiled=true")
    elif stage in validation_failures:
        put("B_compile", "not_reached", "source/static validation failed")
    else:
        put("B_compile", "fail", f"failure_stage={stage or 'unknown'}")

    verification = f"{result.verify_output}\n{result.error}\n{result.error_excerpt}".lower()
    load_failure = any(
        marker in verification
        for marker in (
            "modulenotfounderror",
            "importerror",
            "undefined symbol",
            "cannot open shared object",
            "failed to load",
            "input must be a cuda tensor",
            "input must be an npu tensor",
        )
    )
    comparison = (
        "comparison" in verification and "case[" in verification
    ) or any(
        isinstance(item, dict) and item.get("status") in {"passed", "failed"}
        for item in result.case_results
    )
    signal_failure = any(
        marker in verification for marker in ("sigsegv", "terminated by signal")
    )
    runtime_failure = signal_failure or any(
        marker in verification
        for marker in (
            "aicore exception",
            "device exception",
            "acl_error",
            "mte",
            "kernel not found",
        )
    )
    case_started = any(isinstance(item, dict) for item in result.case_results)
    candidate_started = any(
        isinstance(item, dict)
        and item.get("status") in {"candidate_started", "candidate_returned", "passed", "failed"}
        for item in result.case_results
    )
    if not result.compiled:
        put("C_load", "not_reached", "compile did not pass")
        put("D_execute", "not_reached", "load was not attempted")
    elif result.correctness or comparison:
        put("C_load", "pass", "verification produced candidate comparisons")
        put("D_execute", "pass", "verification produced candidate comparisons")
    elif load_failure:
        put("C_load", "fail", "verification reported module/binding failure")
        put("D_execute", "not_reached", "binding failed")
    elif signal_failure:
        put(
            "C_load",
            "pass" if case_started else "unknown",
            "active case proves module/model loading" if case_started else "native signal does not prove load status",
        )
        put("D_execute", "unknown", "native signal does not prove NPU kernel entry")
    elif runtime_failure:
        put("C_load", "pass", "runtime failure occurred after binding")
        put("D_execute", "fail", "device/runtime failure")
    elif case_started:
        put("C_load", "pass", "verification persisted an active case after module/model loading")
        put(
            "D_execute",
            "pass" if candidate_started else "unknown",
            "candidate invocation started" if candidate_started else "active case does not prove candidate invocation",
        )
    else:
        put("C_load", "unknown", "verification log does not prove load status")
        put("D_execute", "unknown", "verification log does not prove execution status")

    if result.correctness:
        put("E_correct", "pass", "EvalResult.correctness=true")
    elif not result.compiled or load_failure or runtime_failure:
        put("E_correct", "not_reached", "candidate did not reach completed comparison")
    elif comparison:
        put("E_correct", "fail", "comparison completed with mismatch")
    else:
        put("E_correct", "unknown", "correctness comparison status is ambiguous")

    benchmarked = result.correctness and isinstance(result.score, (int, float)) and result.score > 0
    benchmark_status = "pass" if benchmarked else "fail" if result.correctness else "not_reached"
    put("F_benchmark", benchmark_status, "valid positive score" if benchmarked else "correct candidate lacks a valid positive benchmark" if result.correctness else "no correct benchmark baseline")
    optimizing = workflow_phase == "optimization"
    put("G_optimization", "pass" if optimizing else "not_reached", "optimization phase" if optimizing else "bootstrap phase")
    retained = optimizing and result.correctness
    put(
        "H_optimized_correct",
        "pass" if retained else "fail" if optimizing else "not_reached",
        "optimized candidate remained correct" if retained else "optimized candidate lost correctness" if optimizing else "optimization was not attempted",
    )
    return observed


def evaluation_gates(result: EvalResult) -> dict[str, str]:
    """Return canonical, non-overlapping evaluation gates for acceptance and audit."""

    observed = observe_evaluation_stages(result)
    a = observed["A_source_valid"]["status"]
    b = observed["B_compile"]["status"]
    c = observed["C_load"]["status"]
    d = observed["D_execute"]["status"]
    e = observed["E_correct"]["status"]
    comparison_completed = (
        "pass"
        if e in {"pass", "fail"}
        else "not_reached" if e == "not_reached" else "unknown"
    )
    return {
        "source_valid": a,
        "compiled": b,
        "loaded": c,
        "kernel_started": "pass" if d in {"pass", "fail"} else d,
        "comparison_completed": comparison_completed,
        "full_correct": "pass" if result.correctness else (
            "not_reached" if comparison_completed == "not_reached" else "fail"
        ),
        "benchmarked": observed["F_benchmark"]["status"],
    }


def parse_failure_evidence(
    *, stage: str, output: str, case_info: dict[str, Any] | None = None
) -> FailureEvidence:
    lowered = output.lower()
    compile_stage = stage in {
        "ascendc_build",
        "compile",
        "static_validation",
        "ascendc_source_validation",
        "api_constraint_validation",
        "bundle_validation",
        "interface_contract_validation",
        "response_format",
    }
    runtime_match = None if compile_stage else re.search(
        r"\b(ACL_ERROR_[A-Z0-9_]+|[A-Z]+-\d{3,}|(?:50|56)\d{4})\b",
        output,
    )
    core_match = re.search(r"\bcore[_ ]?id\s*[:=]\s*(\d+)", output, re.IGNORECASE)
    block_match = re.search(r"\bblock[_ ]?id\s*[:=]\s*(\d+)", output, re.IGNORECASE)
    sub_error_match = re.search(r"\bsub[_ ]?error(?:[_ ]?type)?\s*[:=]\s*([^,;\n]+)", output, re.IGNORECASE)
    if compile_stage:
        subsystem = "COMPILER" if stage in {"ascendc_build", "compile"} else "SOURCE"
        reason = compact_diagnostics(output, max_lines=1, max_chars=1000) or "build validation failure"
    elif "mte" in lowered:
        subsystem = "MTE"
        reason = "illegal configuration" if "illegal configuration" in lowered else "MTE failure"
    elif "aicore" in lowered or "ai core" in lowered:
        subsystem = "AICORE"
        reason = "AI Core exception"
    elif "acl_error" in lowered or stage.startswith("acl"):
        subsystem = "ACL"
        reason = "ACL runtime error"
    else:
        subsystem = "RUNTIME" if stage == "correctness" else stage.upper()
        reason = compact_diagnostics(output, max_lines=1, max_chars=1000) or "evaluation failure"
    device_line = next(
        (
            line.strip()
            for line in output.splitlines()
            if re.search(r"exception|illegal configuration", line, re.IGNORECASE)
        ),
        None,
    )
    related_symbols = extract_api_symbols(output)
    faulting_cores = sorted(
        {int(value) for value in re.findall(r"core[_ ]?id\s*[:=]\s*(\d+)", output, re.IGNORECASE)}
    )
    faulting_blocks = sorted(
        {int(value) for value in re.findall(r"(?:block|blk)[_ ]?id\s*[:=]\s*(\d+)", output, re.IGNORECASE)}
    )

    def unique_tokens(pattern: str) -> list[str]:
        return list(
            dict.fromkeys(
                value.strip().rstrip(",;)")
                for value in re.findall(pattern, output, flags=re.IGNORECASE)
                if value.strip()
            )
        )

    pc_start = unique_tokens(r"(?:pc\s*start|start\s*pc)\s*[:=]\s*(0x[0-9a-f]+)")
    pc_current = unique_tokens(r"(?:pc\s*current|current\s*pc)\s*[:=]\s*(0x[0-9a-f]+)")
    serial = unique_tokens(r"serial(?:\s*number)?\s*[:=]\s*([^,;\s]+)")
    extend_error_str = unique_tokens(r"(?:errorstr|error[_ ]str)\s*[:=]\s*([^,;\n]+)")
    attribution_status = "attributed" if any(
        (
            runtime_match,
            core_match,
            block_match,
            sub_error_match,
            related_symbols,
            faulting_cores,
            faulting_blocks,
            pc_current,
            (case_info or {}).get("failed_case_index") is not None,
        )
    ) else "stage_only"
    return FailureEvidence(
        stage=stage,
        runtime_code=runtime_match.group(1) if runtime_match else None,
        subsystem=subsystem,
        device_exception=device_line,
        reason=reason,
        core_id=int(core_match.group(1)) if core_match else None,
        block_id=int(block_match.group(1)) if block_match else None,
        sub_error_type=sub_error_match.group(1).strip() if sub_error_match else None,
        case_info=case_info or {},
        related_symbols=related_symbols,
        attribution_status=attribution_status,
        faulting_cores=faulting_cores,
        faulting_blocks=faulting_blocks,
        pc_start=pc_start,
        pc_current=pc_current,
        serial=serial,
        extend_error_str=extend_error_str,
    )


def compact_evaluation(result: EvalResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    output = result.error_excerpt
    if not output:
        output = result.verify_output if result.failure_stage == "correctness" else result.compile_output
    first_incomplete_or_failed = next(
        (
            item
            for item in result.case_results
            if isinstance(item, dict)
            and item.get("status") not in {"passed", "reference_passed", "candidate_returned"}
        ),
        None,
    )
    if first_incomplete_or_failed is None and not result.correctness and result.case_results:
        first_incomplete_or_failed = next(
            (item for item in reversed(result.case_results) if isinstance(item, dict)),
            None,
        )
    case_keys = (
        "case_index",
        "index",
        "profile",
        "status",
        "checkpoint",
        "shape",
        "shapes",
        "dtype",
        "dtypes",
        "attrs",
        "error",
        "expected_summary",
        "actual_summary",
        "max_abs_diff",
        "max_rel_diff",
    )
    first_case = (
        {
            key: first_incomplete_or_failed[key]
            for key in case_keys
            if key in first_incomplete_or_failed
        }
        if first_incomplete_or_failed
        else None
    )
    performance = {
        key: value
        for key, value in result.performance.items()
        if key
        in {
            "schema_version",
            "status",
            "score",
            "overall_speedup",
            "measurement",
            "diagnostics",
            "framework",
            "implementation",
            "mock",
        }
    }
    per_case = result.performance.get("per_case_speedup", [])
    if isinstance(per_case, list):
        performance["per_case_speedup"] = per_case[:8]
        if len(per_case) > 8:
            performance["per_case_speedup_omitted"] = len(per_case) - 8
    bottlenecks = result.performance.get("bottlenecks", [])
    if isinstance(bottlenecks, list):
        compact_bottlenecks = []
        for item in bottlenecks[:3]:
            if not isinstance(item, dict):
                continue
            compact_item = {
                key: item[key]
                for key in (
                    "index",
                    "inputs",
                    "reference_ms",
                    "ascendc_ms",
                    "speedup",
                    "reference_cv",
                    "ascendc_cv",
                )
                if key in item
            }
            profiling = item.get("profiling", {})
            if isinstance(profiling, dict):
                compact_item["profiling"] = {
                    implementation: {
                        "status": profile.get("status"),
                        "reason": profile.get("reason"),
                        "operators": profile.get("operators", [])[:5],
                    }
                    for implementation, profile in profiling.items()
                    if isinstance(profile, dict)
                }
            compact_bottlenecks.append(compact_item)
        performance["bottlenecks"] = compact_bottlenecks
    return {
        "compiled": result.compiled,
        "correctness": result.correctness,
        "score": result.score,
        "failure_stage": result.failure_stage,
        "failure_code": result.failure_code,
        "message": result.error,
        "diagnostics": compact_diagnostics(output),
        "failure_evidence": result.failure_evidence,
        "active_profile": result.active_profile,
        "passed_profiles": result.passed_profiles,
        "passed_case_indices": result.passed_case_indices,
        "first_incomplete_or_failed_case": first_case,
        "case_result_count": len(result.case_results),
        "performance": performance,
        "evaluation_gates": evaluation_gates(result),
    }


def read_result_log(result: EvalResult) -> str:
    if result.details_path:
        path = Path(result.details_path)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    if result.failure_stage == "correctness":
        return result.verify_output
    return result.compile_output or result.error
