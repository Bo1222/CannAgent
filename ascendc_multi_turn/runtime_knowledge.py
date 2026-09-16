from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .knowledge_probe import (
    collect_environment_fingerprint,
    load_verified_probe_facts,
    torch_include_roots_without_import,
    torch_npu_include_roots,
)


@dataclass
class RuntimeFacts:
    symbols: list[str]
    text: str
    conflicts: list[str]
    include_root: str | None
    facts: list[dict[str, object]]
    environment_fingerprint: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _cann_root(runtime_version: str) -> Path | None:
    candidates = []
    for name in ("ASCEND_HOME_PATH", "ASCEND_INSTALL_PATH"):
        if os.getenv(name):
            candidates.append(Path(os.environ[name]).expanduser())
    candidates.extend(
        [
            Path.home() / "Ascend" / f"cann-{runtime_version}",
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path("/usr/local/Ascend/ascend-toolkit"),
        ]
    )
    for candidate in candidates:
        if (candidate / "aarch64-linux/asc/include").is_dir():
            return candidate.resolve()
        if (candidate / "asc/include").is_dir():
            return candidate.resolve()
    return None


def _include_roots(cann_root: Path) -> list[Path]:
    base = cann_root / "aarch64-linux/asc/include"
    if not base.is_dir():
        base = cann_root / "asc/include"
    return [path for path in (base / "basic_api", base / "adv_api") if path.is_dir()]


def _installed_cann_version(cann_root: Path | None, requested_version: str) -> str:
    """Identify the toolkit on disk without treating a caller label as environment identity."""

    if cann_root is None:
        return requested_version
    for path in (
        cann_root / "compiler/version.info",
        cann_root / "opp/version.info",
        cann_root / "version.info",
    ):
        if not path.is_file():
            continue
        match = re.search(
            r"(?<!\d)(\d+\.\d+(?:\.\d+)?)(?!\d)",
            path.read_text(encoding="utf-8", errors="replace"),
        )
        if match:
            return match.group(1)
    match = re.search(r"cann[-_](\d+\.\d+(?:\.\d+)?)", cann_root.name, re.IGNORECASE)
    return match.group(1) if match else requested_version


def _symbol_priority(symbol: str) -> tuple[int, str]:
    noise = {"CMake", "Error", "Exception", "Failed", "Kernel", "Model", "Traceback"}
    return (1 if symbol in noise or symbol.lower() in {"cmake", "traceback"} else 0, symbol)


def _matching_locations(symbol: str, roots: list[Path]) -> list[tuple[Path, int]]:
    if not roots:
        return []
    matches: list[tuple[Path, int]] = []
    identifier = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])")
    rg = shutil.which("rg")
    if rg:
        command = [
            rg,
            "-n",
            "--glob",
            "*.h",
            "--fixed-strings",
            "--word-regexp",
            symbol,
            *map(str, roots),
        ]
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        for line in completed.stdout.splitlines():
            match = re.match(r"(.+?):(\d+):", line)
            if not match:
                continue
            item = (Path(match.group(1)), int(match.group(2)))
            try:
                matched_line = item[0].read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[item[1] - 1]
            except (OSError, IndexError):
                matched_line = ""
            if matched_line.lstrip().startswith(("//", "/*", "*")):
                continue
            if item not in matches:
                matches.append(item)
            if len(matches) >= 512:
                break
    else:
        for root in roots:
            for path in sorted(root.rglob("*.h")):
                try:
                    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                for line_number, line in enumerate(lines, 1):
                    if identifier.search(line):
                        matches.append((path, line_number))
                        if len(matches) >= 512:
                            break
                if len(matches) >= 512:
                    break
            if len(matches) >= 512:
                break

    def priority(item: tuple[Path, int]) -> tuple[int, str, int]:
        name = str(item[0])
        try:
            source_lines = item[0].read_text(encoding="utf-8", errors="replace").splitlines()
            line = source_lines[item[1] - 1]
            nearby = " ".join(source_lines[item[1] - 1 : item[1] + 3])
        except (OSError, IndexError):
            line = ""
            nearby = ""
        score = 50
        if line.lstrip().startswith(("//", "/*", "*")):
            score += 80
        if re.search(rf"\b(?:struct|class)\s+{re.escape(symbol)}\b", line):
            score -= 40
        elif re.search(rf"\b(?:void|inline)\s+{re.escape(symbol)}\s*(?:<[^>]*>)?\s*\(", line):
            score -= 30
        if re.search(r"\b(?:count|calCount)\b", nearby):
            score -= 8
        if "op_frame" in name or name.endswith("_impl.h"):
            score += 20
        if "/torch_npu/csrc/core/npu/" in name:
            score -= 30
        if "/third_party/" in name:
            score += 30
        if "/basic_api/" in name:
            score -= 10
        if "kernel_struct" in name and symbol.endswith(("Params", "Tiling")):
            score -= 20
        if name.endswith(("_intf.h", "kernel_tpipe.h")):
            score -= 10
        return score, name, item[1]

    return sorted(matches, key=priority)[:5]


def _excerpt(
    path: Path,
    line_number: int,
    symbol: str,
    max_chars: int | None = 3500,
) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, line_number - 1)
    while start > 0 and lines[start - 1].lstrip().startswith(("template", "#if", "#ifdef", "#elif")):
        start -= 1
    end = min(len(lines), line_number + 1)
    # Include a complete nearby struct when practical.
    for index in range(max(0, line_number - 5), min(len(lines), line_number + 2)):
        if re.search(rf"\b(?:struct|class)\s+{re.escape(symbol)}\b", lines[index]):
            start = index
            depth = 0
            for cursor in range(index, min(len(lines), index + 80)):
                depth += lines[cursor].count("{") - lines[cursor].count("}")
                end = cursor + 1
                if depth <= 0 and cursor > index:
                    break
            break
    else:
        parens = braces = angles = 0
        saw_brace = False
        for cursor in range(start, min(len(lines), start + 80)):
            line = lines[cursor]
            parens += line.count("(") - line.count(")")
            braces += line.count("{") - line.count("}")
            saw_brace = saw_brace or "{" in line
            angles += line.count("<") - line.count(">")
            end = cursor + 1
            if saw_brace and cursor >= line_number - 1 and braces <= 0:
                break
            if not saw_brace and cursor >= line_number - 1 and parens <= 0 and angles <= 0 and ";" in line:
                break
    text = "\n".join(lines[start:end])
    return text if max_chars is None else text[:max_chars]


def collect_runtime_facts(
    symbols: list[str],
    *,
    runtime_version: str,
    cache_path: Path | None = None,
    max_chars: int | None = 8000,
    soc_version: str = "unknown",
    project_root: Path | None = None,
    probe_manifest: Path | None = None,
    failure_evidence: str = "",
) -> RuntimeFacts:
    normalized = sorted(dict.fromkeys(symbols), key=_symbol_priority)
    if max_chars is not None:
        normalized = normalized[:24]
    root = _cann_root(runtime_version)
    cann_roots = _include_roots(root) if root else []
    host_roots = torch_npu_include_roots()
    all_roots = [*cann_roots, *host_roots]
    torch_roots, _ = torch_include_roots_without_import()
    # The CLI/runtime label may be 8.5.0 while the resolved toolkit is 8.5.2.
    # Verification follows the concrete toolkit, headers and compilers on disk;
    # the caller label alone must not invalidate an otherwise identical probe.
    installed_version = _installed_cann_version(root, runtime_version)
    fingerprint = collect_environment_fingerprint(
        runtime_version=installed_version,
        soc_version=soc_version,
        toolkit_root=root,
        include_roots=[*all_roots, *torch_roots],
        project_root=project_root,
    )
    verified = load_verified_probe_facts(
        probe_manifest, fingerprint_id=fingerprint.fingerprint_id
    )
    # A bounded cache is not valid evidence that the selected declarations were
    # rendered in full. Rebuild unbounded runtime text on full-selected calls.
    if max_chars is not None and cache_path and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("symbols") == normalized
                and cached.get("environment_fingerprint", {}).get("fingerprint_id")
                == fingerprint.fingerprint_id
            ):
                return RuntimeFacts(
                    symbols=normalized,
                    text=str(cached.get("text", "")),
                    conflicts=list(cached.get("conflicts", [])),
                    include_root=cached.get("include_root"),
                    facts=list(cached.get("facts", [])),
                    environment_fingerprint=dict(cached.get("environment_fingerprint", {})),
                )
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    if not all_roots:
        facts = RuntimeFacts(
            normalized,
            "Installed CANN and torch_npu public headers were not found.",
            [],
            None,
            [],
            fingerprint.to_dict(),
        )
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"runtime_version": runtime_version, **facts.to_dict()}
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return facts
    chunks = [
        f"Environment fingerprint: {fingerprint.fingerprint_id}",
        f"Installed CANN {installed_version} and torch_npu public-header declarations:",
        "Confidence order: Level 3 Verified > Level 2 Installed > Level 1 Documented >> Level 0 Inferred.",
    ]
    found: set[str] = set()
    fact_records: list[dict[str, object]] = []
    per_symbol_chars = (
        None
        if max_chars is None
        else max(800, min(3500, max_chars // max(1, min(len(normalized), 8))))
    )
    for symbol in normalized:
        locations = _matching_locations(symbol, all_roots)
        if not locations:
            continue
        found.add(symbol)
        path, line_number = locations[0]
        excerpts: list[str] = []
        for candidate_path, candidate_line in locations:
            candidate_excerpt = _excerpt(
                candidate_path,
                candidate_line,
                symbol,
                max_chars=(
                    None
                    if per_symbol_chars is None
                    else max(600, per_symbol_chars - sum(len(item) for item in excerpts))
                ),
            )
            if candidate_excerpt and candidate_excerpt not in excerpts:
                excerpts.append(candidate_excerpt)
            if (
                per_symbol_chars is not None
                and sum(len(item) for item in excerpts) >= per_symbol_chars
            ):
                break
        excerpt = "\n\n// Additional installed overload/declaration\n".join(excerpts)
        if per_symbol_chars is not None:
            excerpt = excerpt[:per_symbol_chars]
        owner = root if root and path.is_relative_to(root) else next(
            (candidate for candidate in host_roots if path.is_relative_to(candidate)), path.parent
        )
        relative = path.relative_to(owner)
        if "/basic_api/" in str(path):
            required_include = '#include "kernel_operator.h"'
        elif "/adv_api/" in str(path):
            required_include = f'#include "{path.relative_to(next(item for item in cann_roots if path.is_relative_to(item)))}"'
        elif any(path.is_relative_to(candidate) for candidate in host_roots):
            host_owner = next(candidate for candidate in host_roots if path.is_relative_to(candidate))
            required_include = f'#include "{path.relative_to(host_owner)}"'
        else:
            required_include = None
        fact_id = f"runtime:{symbol}"
        probe = verified.get(fact_id)
        level = 3 if probe else 2
        provenance = "compile_probe" if probe else "installed_header"
        fact_records.append(
            {
                "fact_id": fact_id,
                "symbol": symbol,
                "confidence_level": level,
                "confidence_name": "Verified" if level == 3 else "Installed",
                "provenance": provenance,
                "source_path": str(path),
                "source_line": line_number,
                "required_include": required_include,
                "environment_fingerprint": fingerprint.fingerprint_id,
                "probe_id": probe.get("probe_id") if probe else None,
                "probe_status": probe.get("probe_status") if probe else "not_run",
            }
        )
        chunk = (
            f"\n### {symbol} [Level {level} {'Verified' if level == 3 else 'Installed'}]"
            f"\nFact ID: {fact_id}\nSource: {relative}:{line_number}"
            + (f"\nRequired include: `{required_include}`" if required_include else "")
            + f"\n```cpp\n{excerpt}\n```"
        )
        remaining = (
            None if max_chars is None else max_chars - sum(len(item) for item in chunks)
        )
        if remaining is None or remaining >= 200:
            if remaining is None or len(chunk) <= remaining:
                chunks.append(chunk)
    missing = [symbol for symbol in normalized if symbol not in found]
    lowered_failure = failure_evidence.lower()
    for symbol in missing:
        direct_failure = symbol.lower() in lowered_failure and any(
            marker in lowered_failure
            for marker in ("not declared", "no matching", "not found", "unsupported", "not supported")
        )
        fact_records.append(
            {
                "fact_id": f"runtime-missing:{symbol}",
                "symbol": symbol,
                "confidence_level": 2 if direct_failure else 0,
                "confidence_name": "Installed negative" if direct_failure else "Inferred missing",
                "provenance": "installed_header_and_compiler" if direct_failure else "header_scan_only",
                "source_path": None,
                "source_line": None,
                "environment_fingerprint": fingerprint.fingerprint_id,
                "probe_id": None,
                "probe_status": "not_run",
                "negative": True,
            }
        )
        if direct_failure:
            chunks.append(
                f"\n### {symbol} [Level 2 Installed negative]\n"
                "The symbol was absent from the scanned public API headers and the current compiler diagnostic directly rejects it. "
                "Do not use it as an AICore API unless a matching compile probe supersedes this fact."
            )
    text = "\n".join(chunks)
    if max_chars is not None:
        text = text[:max_chars]
    conflicts: list[str] = []
    if "DataCopyPadExtParams" in normalized and "paddingValue" in text:
        conflicts.append(
            "Runtime DataCopyPadExtParams uses paddingValue; ignore fallback documentation that says padValue."
        )
    if "CopyTiling" in normalized:
        if "CopyTiling" in missing:
            conflicts.append(
                "CopyTiling was not found in the scanned public API headers; do not use it for an arbitrary tiling struct."
            )
        elif "adv_api/matmul" in text:
            conflicts.append(
                "The located CopyTiling declaration belongs to the Matmul API; it is not a generic copier for arbitrary tiling structs."
            )
    facts = RuntimeFacts(
        normalized,
        text,
        conflicts,
        str(root) if root else None,
        fact_records,
        fingerprint.to_dict(),
    )
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"runtime_version": runtime_version, **facts.to_dict()}
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return facts
