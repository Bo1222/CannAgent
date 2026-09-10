from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .models import EvalResult
from .structured_knowledge.schema import StructuredFailure

ERROR_LINE = re.compile(
    r"error:|fatal(?: error)?:|traceback|calledprocesserror|timed out|\[fail(?:ed)?\]",
    re.IGNORECASE,
)
QUOTED_SYMBOL = re.compile(r"'(?:AscendC::)?([A-Za-z_][A-Za-z0-9_:<>]*)'")
SOURCE_SYMBOL = re.compile(r"\b(?:AscendC::)?([A-Z][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?\s*\(")
TYPE_SYMBOL = re.compile(r"\b(?:AscendC::)?([A-Z][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?")


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


def extract_api_symbols(*texts: str) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        symbol = token.split("::")[-1].split("<", 1)[0]
        if len(symbol) >= 3 and symbol not in seen:
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
        (line.strip() for line in output.splitlines() if re.search(r"exception|illegal configuration", line, re.I)),
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
