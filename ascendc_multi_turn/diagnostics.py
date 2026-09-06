from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from .models import EvalResult


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
        "details_path": result.details_path,
    }


def read_result_log(result: EvalResult) -> str:
    if result.details_path:
        path = Path(result.details_path)
        if path.is_file():
            return path.read_text(encoding="utf-8", errors="replace")
    if result.failure_stage == "correctness":
        return result.verify_output
    return result.compile_output or result.error
