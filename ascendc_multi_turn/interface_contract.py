from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import FileBundle

_TORCH_LIBRARY = re.compile(
    r"TORCH_LIBRARY(?P<kind>_IMPL|_FRAGMENT)?\s*\(\s*(?P<namespace>[A-Za-z_]\w*)"
    r"(?:\s*,\s*(?P<dispatch>[A-Za-z0-9_:]+))?\s*,"
)
_TORCH_OP = re.compile(r"torch\.ops\.([A-Za-z_]\w*)\.([A-Za-z_]\w*)")
_KERNEL_SIGNATURE = re.compile(
    r'extern\s+"C"\s+(?:__global__\s+)?__aicore__\s+void\s+'
    r"([A-Za-z_]\w*)\s*\((.*?)\)\s*\{",
    re.DOTALL,
)


def _normalize_signature(parameters: str) -> str:
    return re.sub(r"\s+", " ", parameters).strip()


def capture_interface_contract(bundle: FileBundle) -> dict[str, Any]:
    """Capture the stable dispatcher, Python wrapper, and kernel entry surface."""

    registrations: list[dict[str, str]] = []
    python_ops: list[dict[str, str]] = []
    kernel_entries: list[dict[str, str]] = []
    for path, source in sorted(bundle.files.items()):
        for match in _TORCH_LIBRARY.finditer(source):
            registrations.append(
                {
                    "path": path,
                    "kind": match.group("kind") or "DEF",
                    "namespace": match.group("namespace"),
                    "dispatch": match.group("dispatch") or "",
                }
            )
        for match in _TORCH_OP.finditer(source):
            python_ops.append({"path": path, "namespace": match.group(1), "operator": match.group(2)})
        for match in _KERNEL_SIGNATURE.finditer(source):
            kernel_entries.append(
                {
                    "path": path,
                    "name": match.group(1),
                    "parameters": _normalize_signature(match.group(2)),
                }
            )
    return {
        "schema_version": 2,
        "torch_registrations": registrations,
        "python_ops": python_ops,
        "kernel_entries": kernel_entries,
    }


def compare_interface_contract(
    expected: dict[str, Any], candidate: FileBundle
) -> list[str]:
    actual = capture_interface_contract(candidate)
    issues: list[str] = []
    for key in (
        "torch_registrations",
        "python_ops",
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
