from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

from .models import FileBundle


_ALLOWED_ROOT_FILE = "model_new_ascendc.py"
_ALLOWED_SUFFIXES = {".py", ".cpp", ".cc", ".cxx", ".h", ".hpp"}


def _json_object(text: str) -> dict:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("LLM response does not contain a JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")
    return value


def validate_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("file path must be a non-empty string")
    normalized = value.replace("\\", "/").strip()
    path = Path(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe file path: {value!r}")
    if normalized == _ALLOWED_ROOT_FILE:
        return normalized
    if not normalized.startswith("kernel/") or path.suffix not in _ALLOWED_SUFFIXES:
        raise ValueError(f"path is outside the editable AscendC sources: {value!r}")
    if normalized.startswith("kernel/build/"):
        raise ValueError("kernel/build is generated and cannot be edited")
    return normalized


def parse_file_bundle(text: str) -> FileBundle:
    payload = _json_object(text)
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("response.files must be a non-empty list")
    files: dict[str, str] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            raise ValueError("each files entry must be an object")
        path = validate_relative_path(item.get("path", ""))
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"file {path!r} has empty content")
        files[path] = content

    raw_delete = payload.get("delete", [])
    if not isinstance(raw_delete, list):
        raise ValueError("response.delete must be a list")
    delete = [validate_relative_path(path) for path in raw_delete]
    overlap = set(files).intersection(delete)
    if overlap:
        raise ValueError(f"paths cannot be both written and deleted: {sorted(overlap)}")
    return FileBundle(files=files, delete=delete, analysis=str(payload.get("analysis", "")))


def apply_bundle(task_dir: Path, bundle: FileBundle) -> None:
    task_dir = task_dir.resolve()
    for relative in bundle.delete:
        target = (task_dir / relative).resolve()
        target.relative_to(task_dir)
        if target.is_file():
            target.unlink()
    for relative, content in bundle.files.items():
        target = (task_dir / relative).resolve()
        target.relative_to(task_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, target)


def capture_bundle(task_dir: Path) -> FileBundle:
    files: dict[str, str] = {}
    wrapper = task_dir / _ALLOWED_ROOT_FILE
    if wrapper.is_file():
        files[_ALLOWED_ROOT_FILE] = wrapper.read_text(encoding="utf-8")
    kernel_dir = task_dir / "kernel"
    if kernel_dir.is_dir():
        for path in sorted(kernel_dir.rglob("*")):
            if not path.is_file() or "build" in path.relative_to(kernel_dir).parts:
                continue
            if path.suffix in _ALLOWED_SUFFIXES:
                files[path.relative_to(task_dir).as_posix()] = path.read_text(encoding="utf-8")
    return FileBundle(files=files)


def restore_bundle(task_dir: Path, bundle: FileBundle) -> None:
    wrapper = task_dir / _ALLOWED_ROOT_FILE
    if wrapper.exists() and _ALLOWED_ROOT_FILE not in bundle.files:
        wrapper.unlink()
    kernel_dir = task_dir / "kernel"
    if kernel_dir.exists():
        for path in sorted(kernel_dir.rglob("*"), reverse=True):
            if path.is_file() and "build" not in path.relative_to(kernel_dir).parts:
                path.unlink()
        build = kernel_dir / "build"
        if build.exists():
            shutil.rmtree(build)
    apply_bundle(task_dir, bundle)


def validate_initial_bundle(bundle: FileBundle) -> None:
    paths = set(bundle.files)
    if _ALLOWED_ROOT_FILE not in paths:
        raise ValueError(f"initial generation must include {_ALLOWED_ROOT_FILE}")
    if "kernel/pybind11.cpp" not in paths:
        raise ValueError("initial generation must include kernel/pybind11.cpp")
    kernel_cpp = [p for p in paths if p.startswith("kernel/") and p.endswith(".cpp") and p != "kernel/pybind11.cpp"]
    if not kernel_cpp:
        raise ValueError("initial generation must include at least one AscendC kernel .cpp file")
