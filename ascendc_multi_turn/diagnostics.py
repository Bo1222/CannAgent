from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .models import EvalResult
from .structured_knowledge.schema import DiagnosticRecord, StructuredFailure

ERROR_LINE = re.compile(
    r"error:|fatal(?: error)?:|traceback|calledprocesserror|timed out|\[fail(?:ed)?\]",
    re.IGNORECASE,
)
QUOTED_SYMBOL = re.compile(r"'(?:AscendC::)?([A-Za-z_][A-Za-z0-9_:<>]*)'")
SOURCE_SYMBOL = re.compile(r"\b(?:AscendC::)?([A-Z][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?\s*\(")
TYPE_SYMBOL = re.compile(r"\b(?:AscendC::)?([A-Z][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?")
_SYMBOL_NOISE = {
    "CMake",
    "Error",
    "Exception",
    "Failed",
    "Kernel",
    "Model",
    "Traceback",
}
_LOCATED_ERROR = re.compile(
    r"(?P<path>(?:[A-Za-z]:)?[^:\n]+):\d+(?::\d+)?:\s*"
    r"(?P<severity>fatal\s+error|error):\s*(?P<message>.+)$",
    re.IGNORECASE,
)
_PLAIN_ERROR = re.compile(
    r"(?P<severity>fatal\s+error|error):\s*(?P<message>.+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SymbolEvidence:
    symbol: str
    kinds: tuple[str, ...]
    sources: tuple[str, ...]

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
    return "\n".join(diagnostic_lines(output, max_lines=max_lines, max_chars=max_chars))


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
        category = _diagnostic_category(stage, message, source_file)
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


def extract_api_symbols(*texts: str) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        symbol = token.split("::")[-1].split("<", 1)[0]
        if len(symbol) >= 3 and symbol not in _SYMBOL_NOISE and symbol not in seen:
            seen.add(symbol)
            symbols.append(symbol)

    for text in texts:
        for match in QUOTED_SYMBOL.finditer(text):
            add(match.group(1))
        for token in SOURCE_SYMBOL.findall(text):
            add(token)
        for match in TYPE_SYMBOL.finditer(text):
            token = match.group(1)
            if token.startswith(("DataCopy", "GlobalTensor", "LocalTensor", "TQue", "TPipe", "TBuf")):
                add(token)
    return symbols


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
        for symbol in extract_api_symbols(text):
            if symbol not in evidence:
                ordered.append(symbol)
                evidence[symbol] = {"kinds": [], "sources": []}
            if kind not in evidence[symbol]["kinds"]:
                evidence[symbol]["kinds"].append(kind)
            if label not in evidence[symbol]["sources"]:
                evidence[symbol]["sources"].append(label)
    return [
        SymbolEvidence(
            symbol=symbol,
            kinds=tuple(evidence[symbol]["kinds"]),
            sources=tuple(evidence[symbol]["sources"]),
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
    comparison = "comparison" in verification and "case[" in verification
    runtime_failure = any(
        marker in verification
        for marker in (
            "aicore exception",
            "device exception",
            "acl_error",
            "mte",
            "kernel not found",
        )
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
    elif runtime_failure:
        put("C_load", "pass", "runtime failure occurred after binding")
        put("D_execute", "fail", "device/runtime failure")
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


def parse_structured_failure(
    *, stage: str, output: str, case_info: dict[str, Any] | None = None
) -> StructuredFailure:
    lowered = output.lower()
    runtime_match = re.search(r"\b(ACL_ERROR_[A-Z0-9_]+|[A-Z]+-\d{3,}|(?:50|56)\d{4})\b", output)
    core_match = re.search(r"\bcore[_ ]?id\s*[:=]\s*(\d+)", output, re.IGNORECASE)
    block_match = re.search(r"\bblock[_ ]?id\s*[:=]\s*(\d+)", output, re.IGNORECASE)
    sub_error_match = re.search(r"\bsub[_ ]?error(?:[_ ]?type)?\s*[:=]\s*([^,;\n]+)", output, re.IGNORECASE)
    if "mte" in lowered:
        subsystem = "MTE"
        reason = "illegal configuration" if "illegal configuration" in lowered else "MTE failure"
    elif "aicore" in lowered or "ai core" in lowered:
        subsystem = "AICORE"
        reason = "AI Core exception"
    elif "acl_error" in lowered or stage.startswith("acl"):
        subsystem = "ACL"
        reason = "ACL runtime error"
    elif stage in {"ascendc_build", "compile", "static_validation", "ascendc_source_validation"}:
        subsystem = "COMPILER" if stage in {"ascendc_build", "compile"} else "SOURCE"
        reason = compact_diagnostics(output, max_lines=1, max_chars=1000) or "build validation failure"
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
    return StructuredFailure(
        stage=stage,
        runtime_code=runtime_match.group(1) if runtime_match else None,
        subsystem=subsystem,
        device_exception=device_line,
        reason=reason,
        core_id=int(core_match.group(1)) if core_match else None,
        block_id=int(block_match.group(1)) if block_match else None,
        sub_error_type=sub_error_match.group(1).strip() if sub_error_match else None,
        case_info=case_info or {},
        related_symbols=extract_api_symbols(output),
    )


def compact_evaluation(result: EvalResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    output = result.error_excerpt
    if not output:
        output = result.verify_output if result.failure_stage == "correctness" else result.compile_output
    return {
        "compiled": result.compiled,
        "correctness": result.correctness,
        "score": result.score,
        "failure_stage": result.failure_stage,
        "failure_code": result.failure_code,
        "message": result.error,
        "diagnostics": compact_diagnostics(output),
        "structured_failure": result.structured_failure,
    }


def read_result_log(result: EvalResult) -> str:
    if result.details_path:
        path = Path(result.details_path)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    if result.failure_stage == "correctness":
        return result.verify_output
    return result.compile_output or result.error
