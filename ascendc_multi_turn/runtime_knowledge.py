from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class RuntimeFacts:
    symbols: list[str]
    text: str
    conflicts: list[str]
    include_root: str | None

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
            if item not in matches:
                matches.append(item)
            if len(matches) >= 64:
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
                        if len(matches) >= 64:
                            break
                if len(matches) >= 64:
                    break
            if len(matches) >= 64:
                break

    def priority(item: tuple[Path, int]) -> tuple[int, str, int]:
        name = str(item[0])
        try:
            line = item[0].read_text(encoding="utf-8", errors="replace").splitlines()[item[1] - 1]
        except (OSError, IndexError):
            line = ""
        score = 50
        if re.search(rf"\b(?:struct|class)\s+{re.escape(symbol)}\b", line):
            score -= 40
        elif re.search(rf"\b{re.escape(symbol)}\s*(?:<[^>]*>)?\s*\(", line):
            score -= 30
        if "op_frame" in name or name.endswith("_impl.h"):
            score += 20
        if "kernel_struct" in name and symbol.endswith(("Params", "Tiling")):
            score -= 20
        if name.endswith("_intf.h") or name.endswith("kernel_tpipe.h"):
            score -= 10
        return score, name, item[1]

    return sorted(matches, key=priority)[:3]


def _excerpt(path: Path, line_number: int, symbol: str, max_chars: int = 3500) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, line_number - 6)
    end = min(len(lines), line_number + 45)
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
    text = "\n".join(lines[start:end])
    return text[:max_chars]


def collect_runtime_facts(
    symbols: list[str],
    *,
    runtime_version: str,
    cache_path: Path | None = None,
    max_chars: int = 8000,
) -> RuntimeFacts:
    normalized = list(dict.fromkeys(symbols))[:24]
    if cache_path and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("runtime_version") == runtime_version and cached.get("symbols") == normalized:
                return RuntimeFacts(
                    symbols=normalized,
                    text=str(cached.get("text", "")),
                    conflicts=list(cached.get("conflicts", [])),
                    include_root=cached.get("include_root"),
                )
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    root = _cann_root(runtime_version)
    if root is None:
        facts = RuntimeFacts(normalized, "Installed CANN public headers were not found.", [], None)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"runtime_version": runtime_version, **facts.to_dict()}
            cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return facts
    roots = _include_roots(root)
    chunks = [f"Installed CANN {runtime_version} public-header declarations (authoritative):"]
    found: set[str] = set()
    per_symbol_chars = max(800, min(3500, max_chars // max(1, min(len(normalized), 8))))
    for symbol in normalized:
        locations = _matching_locations(symbol, roots)
        if not locations:
            continue
        found.add(symbol)
        path, line_number = locations[0]
        excerpt = _excerpt(path, line_number, symbol, max_chars=per_symbol_chars)
        relative = path.relative_to(root)
        chunk = f"\n### {symbol}\nSource: {relative}:{line_number}\n```cpp\n{excerpt}\n```"
        remaining = max_chars - sum(len(item) for item in chunks)
        if remaining < 200:
            break
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
    missing = [symbol for symbol in normalized if symbol not in found]
    text = "\n".join(chunks)[:max_chars]
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
    facts = RuntimeFacts(normalized, text, conflicts, str(root))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"runtime_version": runtime_version, **facts.to_dict()}
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return facts
