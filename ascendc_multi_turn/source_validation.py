from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]')


@dataclass(frozen=True)
class SourceIssue:
    path: str
    line: int
    code: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: error[{self.code}]: {self.message}"


def _without_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", " ", text)


def validate_source_tree(task_dir: Path) -> list[SourceIssue]:
    """Validate invariant parts of the CANNBot direct-invocation project.

    The pinned CANN 9.1.0 compiler remains authoritative for API overloads,
    templates, and device-language semantics.
    """

    issues: list[SourceIssue] = []

    def add(path: str, code: str, message: str, line: int = 1) -> None:
        issues.append(SourceIssue(path, line, code, message))

    if (task_dir / "kernel" / "pybind11.cpp").exists():
        add("kernel/pybind11.cpp", "legacy_pybind_layout", "legacy pybind11/_do ABI is not accepted")

    wrapper_path = task_dir / "model_new_ascendc.py"
    wrapper = wrapper_path.read_text(encoding="utf-8", errors="replace") if wrapper_path.is_file() else ""
    if "class ModelNew" not in wrapper:
        add("model_new_ascendc.py", "missing_model_new", "wrapper must define class ModelNew")
    if "torch.ops" not in wrapper:
        add("model_new_ascendc.py", "missing_torch_ops", "ModelNew must call the registered operator through torch.ops")
    if not re.search(r"(?:torch\.ops\.load_library|torch\.classes\.load_library)", wrapper):
        add("model_new_ascendc.py", "missing_library_load", "wrapper must load the built shared library")

    cmake_path = task_dir / "CMakeLists.txt"
    cmake = cmake_path.read_text(encoding="utf-8", errors="replace") if cmake_path.is_file() else ""
    for marker, code in (("find_package(ASC", "missing_asc_package"), ("LANGUAGES ASC", "missing_asc_language")):
        if marker not in cmake:
            add("CMakeLists.txt", code, f"project CMake must contain {marker}")
    if "SHARED" not in cmake:
        add("CMakeLists.txt", "missing_shared_library", "project CMake must build a shared torch operator library")

    extension_dir = task_dir / "op_extension"
    extension_files = sorted(extension_dir.glob("*.cpp")) if extension_dir.is_dir() else []
    extension = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in extension_files)
    for marker, code, message in (
        ("TORCH_LIBRARY", "missing_torch_registration", "op_extension must use TORCH_LIBRARY registration"),
        ("PrivateUse1", "missing_privateuse1", "operator must register PrivateUse1"),
        ("Meta", "missing_meta", "operator must register a Meta implementation"),
    ):
        if marker not in extension:
            add("op_extension/register.cpp", code, message)

    for root_name in ("op_kernel", "op_host", "op_extension"):
        root = task_dir / root_name
        for path in root.rglob("*") if root.is_dir() else []:
            if not path.is_file() or path.suffix not in {".asc", ".cpp", ".cc", ".cxx", ".h", ".hpp"}:
                continue
            relative = path.relative_to(task_dir).as_posix()
            source = path.read_text(encoding="utf-8", errors="replace")
            masked = _without_comments(source)
            if re.search(r"\b[A-Za-z_]\w*_do\s*\(", masked):
                add(relative, "legacy_do_abi", "legacy *_do launch wrappers are not accepted")
            if "PYBIND11_MODULE" in masked:
                add(relative, "legacy_pybind_module", "PYBIND11_MODULE is replaced by torch dispatcher registration")
            for line_number, line in enumerate(source.splitlines(), 1):
                include = _INCLUDE.match(line)
                if include and Path(include.group(1)).is_absolute():
                    add(relative, "absolute_include", "generated sources must not use absolute include paths", line_number)
    return issues


def render_issues(issues: list[SourceIssue]) -> str:
    return "\n".join(issue.render() for issue in issues)
