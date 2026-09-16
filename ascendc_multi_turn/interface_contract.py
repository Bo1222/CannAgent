from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import FileBundle

_MODULE = re.compile(r"PYBIND11_MODULE\s*\(\s*([A-Za-z_]\w*)\s*,")
_HOST_SIGNATURE = re.compile(
    r'extern\s+"C"\s+void\s+([A-Za-z_]\w*_do)\s*\((.*?)\)\s*([;{])',
    re.DOTALL,
)
_KERNEL_SIGNATURE = re.compile(
    r'extern\s+"C"\s+(?:__global__\s+)?__aicore__\s+void\s+'
    r"([A-Za-z_]\w*)\s*\((.*?)\)\s*\{",
    re.DOTALL,
)


def _normalize_signature(parameters: str) -> str:
    return re.sub(r"\s+", " ", parameters).strip()


def capture_interface_contract(bundle: FileBundle) -> dict[str, Any]:
    """Capture the stable Python/pybind/host-wrapper/kernel interface surface."""

    modules: list[dict[str, str]] = []
    host_declarations: list[dict[str, str]] = []
    host_definitions: list[dict[str, str]] = []
    kernel_entries: list[dict[str, str]] = []
    for path, source in sorted(bundle.files.items()):
        for match in _MODULE.finditer(source):
            modules.append({"path": path, "name": match.group(1)})
        for match in _HOST_SIGNATURE.finditer(source):
            item = {
                "path": path,
                "name": match.group(1),
                "parameters": _normalize_signature(match.group(2)),
            }
            (host_declarations if match.group(3) == ";" else host_definitions).append(item)
        for match in _KERNEL_SIGNATURE.finditer(source):
            kernel_entries.append(
                {
                    "path": path,
                    "name": match.group(1),
                    "parameters": _normalize_signature(match.group(2)),
                }
            )
    return {
        "schema_version": 1,
        "pybind_modules": modules,
        "host_wrapper_declarations": host_declarations,
        "host_wrapper_definitions": host_definitions,
        "kernel_entries": kernel_entries,
    }


def compare_interface_contract(
    expected: dict[str, Any], candidate: FileBundle
) -> list[str]:
    actual = capture_interface_contract(candidate)
    issues: list[str] = []
    for key in (
        "pybind_modules",
        "host_wrapper_declarations",
        "host_wrapper_definitions",
        "kernel_entries",
    ):
        if expected.get(key, []) != actual.get(key, []):
            issues.append(
                f"{key} changed: expected={expected.get(key, [])!r}; "
                f"actual={actual.get(key, [])!r}"
            )
    return issues


def load_interface_contract(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def save_interface_contract(path: Path, contract: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
