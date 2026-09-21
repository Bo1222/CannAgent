from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tokenize
from pathlib import Path

from .models import FileBundle

_ALLOWED_ROOT_FILES = {"model_new_ascendc.py", "CMakeLists.txt"}
_ALLOWED_DIRECTORIES = {"op_kernel", "op_host", "op_extension", "scripts"}
_ALLOWED_SUFFIXES = {".py", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".asc", ".cmake"}


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
    if normalized in _ALLOWED_ROOT_FILES:
        return normalized
    if not path.parts or path.parts[0] not in _ALLOWED_DIRECTORIES or path.suffix not in _ALLOWED_SUFFIXES:
        raise ValueError(f"path is outside the CANNBot AscendC project sources: {value!r}")
    if "build" in path.parts:
        raise ValueError("build output is generated and cannot be edited")
    return normalized


def parse_file_bundle(text: str, *, allowed_paths: set[str] | None = None) -> FileBundle:
    payload = _json_object(text)
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("response.files must be a non-empty list")
    files: dict[str, str] = {}
    for item in raw_files:
        if not isinstance(item, dict):
            raise ValueError("each files entry must be an object")
        path = validate_relative_path(item.get("path", ""))
        if allowed_paths is not None and path not in allowed_paths:
            raise ValueError(f"path is protected or outside the active operator logic: {path!r}")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"file {path!r} has empty content")
        files[path] = content

    raw_delete = payload.get("delete", [])
    if not isinstance(raw_delete, list):
        raise ValueError("response.delete must be a list")
    delete = [validate_relative_path(path) for path in raw_delete]
    if allowed_paths is not None:
        protected_delete = sorted(set(delete) - allowed_paths)
        if protected_delete:
            raise ValueError(
                f"protected paths cannot be deleted: {protected_delete}"
            )
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
    for root_file in sorted(_ALLOWED_ROOT_FILES):
        path = task_dir / root_file
        if path.is_file():
            files[root_file] = path.read_text(encoding="utf-8")
    for directory in sorted(_ALLOWED_DIRECTORIES):
        source_dir = task_dir / directory
        if not source_dir.is_dir():
            continue
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and "build" not in path.relative_to(source_dir).parts and path.suffix in _ALLOWED_SUFFIXES:
                files[path.relative_to(task_dir).as_posix()] = path.read_text(encoding="utf-8")
    return FileBundle(files=files)


def restore_bundle(task_dir: Path, bundle: FileBundle) -> None:
    existing = capture_bundle(task_dir)
    for relative in set(existing.files) - set(bundle.files):
        (task_dir / relative).unlink(missing_ok=True)
    apply_bundle(task_dir, bundle)


def validate_initial_bundle(bundle: FileBundle) -> None:
    paths = set(bundle.files)
    required_exact = {
        "model_new_ascendc.py",
        "CMakeLists.txt",
        "op_host/data_utils.h",
        "op_extension/register.cpp",
        "op_extension/ops.h",
    }
    missing = sorted(required_exact - paths)
    if missing:
        raise ValueError(f"initial generation is missing required CANNBot project files: {missing}")
    required_patterns = {
        "op_kernel/<op>_tiling.h": any(p.startswith("op_kernel/") and p.endswith("_tiling.h") for p in paths),
        "op_kernel/<op>_kernel.asc": any(p.startswith("op_kernel/") and p.endswith("_kernel.asc") for p in paths),
        "op_host/<op>.asc": any(p.startswith("op_host/") and p.endswith(".asc") for p in paths),
        "op_extension/<op>_torch.cpp": any(p.startswith("op_extension/") and p.endswith("_torch.cpp") for p in paths),
    }
    absent_roles = [name for name, present in required_patterns.items() if not present]
    if absent_roles:
        raise ValueError(f"initial generation is missing required CANNBot project roles: {absent_roles}")
    legacy = sorted(path for path in paths if path == "kernel/pybind11.cpp" or path.startswith("kernel/"))
    if legacy:
        raise ValueError(f"legacy pybind/_do project layout is not accepted: {legacy}")
    script_files = sorted(path for path in paths if path.startswith("scripts/"))
    if script_files:
        raise ValueError(f"scripts directory must remain empty: {script_files}")
    unfinished = sorted(
        path for path, content in bundle.files.items() if "<LLM-TODO:" in content
    )
    if unfinished:
        raise ValueError(f"candidate still contains LLM TODO markers: {unfinished}")


def validate_logic_bundle(bundle: FileBundle) -> None:
    """Validate a generator response before it is merged into a fixed scaffold."""

    paths = set(bundle.files)
    roles = {
        "model_new_ascendc.py": "model_new_ascendc.py" in paths,
        "op_kernel/<op>_tiling.h": any(
            path.startswith("op_kernel/") and path.endswith("_tiling.h") for path in paths
        ),
        "op_kernel/<op>_kernel.asc": any(
            path.startswith("op_kernel/") and path.endswith("_kernel.asc") for path in paths
        ),
        "op_host/<op>.asc": any(
            path.startswith("op_host/") and path.endswith(".asc") for path in paths
        ),
        "op_extension/<op>_torch.cpp": any(
            path.startswith("op_extension/") and path.endswith("_torch.cpp") for path in paths
        ),
    }
    missing = sorted(role for role, present in roles.items() if not present)
    if missing:
        raise ValueError(f"generator response is missing operator logic files: {missing}")
    if len(paths) != len(roles):
        raise ValueError(f"generator response must contain exactly five operator logic files: {sorted(paths)}")


def _semantic_source(path: str, source: str) -> str:
    if path.endswith(".py"):
        try:
            tokens = tokenize.generate_tokens(io.StringIO(source).readline)
            source = tokenize.untokenize(
                token for token in tokens if token.type != tokenize.COMMENT
            )
        except (tokenize.TokenError, IndentationError):
            pass
    else:
        source = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
        source = re.sub(r"//[^\n]*", " ", source)
    return re.sub(r"\s+", "", source)


def semantic_bundle_hash(bundle: FileBundle | None) -> str | None:
    if bundle is None or not bundle.files:
        return None
    digest = hashlib.sha256()
    for path, source in sorted(bundle.files.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_semantic_source(path, source).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()
