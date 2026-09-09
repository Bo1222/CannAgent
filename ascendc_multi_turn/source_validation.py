from __future__ import annotations

import re
import os
from dataclasses import dataclass
from pathlib import Path


_PIPE_DECLARATION = re.compile(r"\b(?:AscendC::)?TPipe\s*(?:[&*]\s*)?([A-Za-z_][A-Za-z0-9_]*)")
_INVALID_PIPE_METHODS = ("EnQue", "DeQue", "AllocTensor", "FreeTensor")
_FILE_VALUE = re.compile(
    r"^(?:inline\s+|static\s+|constexpr\s+|const\s+)*"
    r"(?:auto|bool|char|float|double|half|bfloat16_t|u?int(?:8|16|32|64)_t)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|\{)"
)
_INCLUDE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]')
_DO_DECLARATION = re.compile(
    r'extern\s+"C"\s+void\s+([A-Za-z_]\w*_do(?:_\w+)?)\s*\((.*?)\)\s*;',
    re.DOTALL,
)
_DO_DEFINITION = re.compile(
    r'extern\s+"C"\s+void\s+([A-Za-z_]\w*_do(?:_\w+)?)\s*\((.*?)\)\s*\{(.*?)^\s*\}',
    re.DOTALL | re.MULTILINE,
)
_KERNEL_LAUNCH = re.compile(r"\b[A-Za-z_]\w*\s*<<<\s*[^>]+>>>")
_EXTERNAL_QUOTED_HEADERS = {
    "kernel_operator.h",
    "kernel_tiling/kernel_tiling.h",
}
_EXTERNAL_QUOTED_PREFIXES = ("acl/", "torch/", "torch_npu/", "pybind11/", "ATen/", "c10/")


def _cann_include_roots() -> list[Path]:
    roots: list[Path] = []
    for name in ("ASCEND_INSTALL_PATH", "ASCEND_HOME_PATH"):
        value = os.environ.get(name)
        if value:
            roots.append(Path(value).expanduser() / "include")
    roots.extend(
        [
            Path.home() / "Ascend" / "ascend-toolkit" / "latest" / "include",
            Path("/usr/local/Ascend/ascend-toolkit/latest/include"),
        ]
    )
    return [root.resolve() for root in roots if root.is_dir()]


def _parameter_shape(parameters: str) -> tuple[str, ...]:
    values: list[str] = []
    for raw in parameters.split(","):
        value = re.sub(r"/\*.*?\*/", "", raw).strip()
        if not value or value == "void":
            continue
        value = value.split("=", 1)[0].strip()
        value = re.sub(r"\s+[A-Za-z_]\w*\s*$", "", value)
        value = re.sub(r"([*&])\s*[A-Za-z_]\w*\s*$", r"\1", value)
        value = re.sub(r"\s+([*&])", r"\1", value)
        value = re.sub(r"([*&])\s+", r"\1", value)
        values.append(re.sub(r"\s+", " ", value))
    return tuple(values)


@dataclass(frozen=True)
class SourceIssue:
    path: str
    line: int
    code: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: error[{self.code}]: {self.message}"


def validate_source_tree(task_dir: Path) -> list[SourceIssue]:
    """Catch a small set of unambiguous generated AscendC regressions.

    This intentionally does not try to parse C++. The real compiler remains
    authoritative for overload resolution, templates, and helper arity.
    """
    issues: list[SourceIssue] = []
    kernel_dir = task_dir / "kernel"
    source_text: dict[Path, str] = {}
    for path in sorted(kernel_dir.rglob("*")) if kernel_dir.is_dir() else []:
        if path.suffix not in {".cpp", ".cc", ".cxx", ".h", ".hpp"} or not path.is_file():
            continue
        relative = str(path.relative_to(task_dir))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        text = "\n".join(lines)
        source_text[path] = text
        for line_number, line in enumerate(lines, 1):
            include = _INCLUDE.match(line)
            if include:
                delimiter, header = include.groups()
                if Path(header).is_absolute():
                    issues.append(
                        SourceIssue(relative, line_number, "absolute_include", "generated sources must not use absolute include paths")
                    )
                elif header == "acl/acl_rt_launch.h":
                    issues.append(
                        SourceIssue(
                            relative,
                            line_number,
                            "unsupported_launch_header",
                            "acl/acl_rt_launch.h is not part of this build contract; declare a *_do host wrapper in an AscendC kernel .cpp instead",
                        )
                    )
                elif delimiter == '"':
                    candidates = (
                        path.parent / header,
                        kernel_dir / header,
                        task_dir / header,
                        kernel_dir / "catlass" / "include" / header,
                        task_dir / "catlass" / "include" / header,
                    )
                    external = header in _EXTERNAL_QUOTED_HEADERS or header.startswith(
                        _EXTERNAL_QUOTED_PREFIXES
                    )
                    if not external and not any(candidate.is_file() for candidate in candidates):
                        issues.append(
                            SourceIssue(relative, line_number, "missing_local_include", f"local include {header!r} does not exist")
                        )
                elif header.startswith("acl/"):
                    include_roots = _cann_include_roots()
                    if include_roots and not any((root / header).is_file() for root in include_roots):
                        issues.append(
                            SourceIssue(relative, line_number, "missing_cann_include", f"CANN include {header!r} does not exist in the active installation")
                        )
            if "ACLRT_LAUNCH_KERNEL" in line:
                issues.append(
                    SourceIssue(
                        relative,
                        line_number,
                        "unsupported_launch_macro",
                        "pybind/direct ACL launch macros are unsupported; define extern \"C\" *_do beside the __aicore__ kernel and launch it with kernel<<<blockDim, nullptr, stream>>>",
                    )
                )
        pipe_names = set(_PIPE_DECLARATION.findall(text))
        for pipe_name in pipe_names:
            method_pattern = re.compile(
                rf"\b{re.escape(pipe_name)}\s*(?:->|\.)\s*({'|'.join(_INVALID_PIPE_METHODS)})\s*\("
            )
            for line_number, line in enumerate(lines, 1):
                match = method_pattern.search(line)
                if match:
                    issues.append(
                        SourceIssue(
                            relative,
                            line_number,
                            "invalid_tpipe_owner",
                            f"TPipe does not own {match.group(1)}; queue operations belong to TQue",
                        )
                    )

        depth = 0
        declarations: dict[tuple[int, str], int] = {}
        for line_number, line in enumerate(lines, 1):
            stripped = re.sub(r"//.*$", "", line).strip()
            if depth <= 1:
                match = _FILE_VALUE.match(stripped)
                if match:
                    key = (depth, match.group(1))
                    previous = declarations.get(key)
                    if previous is not None:
                        issues.append(
                            SourceIssue(
                                relative,
                                line_number,
                                "duplicate_file_symbol",
                                f"{match.group(1)!r} duplicates a declaration at line {previous}",
                            )
                        )
                    else:
                        declarations[key] = line_number
            depth += stripped.count("{") - stripped.count("}")
            depth = max(0, depth)

    pybind_path = kernel_dir / "pybind11.cpp"
    pybind_text = source_text.get(pybind_path, "")
    declarations = {
        name: (parameters, pybind_text[:match.start()].count("\n") + 1)
        for match in _DO_DECLARATION.finditer(pybind_text)
        for name, parameters in [match.groups()]
    }
    definitions: dict[str, tuple[str, str, Path, int]] = {}
    for path, text in source_text.items():
        if path == pybind_path:
            continue
        for match in _DO_DEFINITION.finditer(text):
            name, parameters, body = match.groups()
            definitions[name] = (parameters, body, path, text[:match.start()].count("\n") + 1)

    if pybind_text and "PYBIND11_MODULE" in pybind_text and not declarations:
        issues.append(
            SourceIssue(
                "kernel/pybind11.cpp",
                1,
                "missing_host_wrapper_declaration",
                "pybind11.cpp must declare and call at least one extern \"C\" *_do host launch wrapper",
            )
        )
    for name, (decl_parameters, line_number) in declarations.items():
        definition = definitions.get(name)
        if definition is None:
            issues.append(
                SourceIssue(
                    "kernel/pybind11.cpp",
                    line_number,
                    "missing_host_wrapper_definition",
                    f"{name} is declared in pybind11.cpp but not defined in an AscendC kernel .cpp",
                )
            )
            continue
        def_parameters, body, def_path, def_line = definition
        if _parameter_shape(decl_parameters) != _parameter_shape(def_parameters):
            issues.append(
                SourceIssue(
                    str(def_path.relative_to(task_dir)),
                    def_line,
                    "host_wrapper_signature_mismatch",
                    f"{name} definition does not match its pybind11.cpp declaration",
                )
            )
        if not _KERNEL_LAUNCH.search(body):
            issues.append(
                SourceIssue(
                    str(def_path.relative_to(task_dir)),
                    def_line,
                    "missing_kernel_launch",
                    f"{name} must launch an __aicore__ kernel with kernel<<<blockDim, nullptr, stream>>>",
                )
            )
        if len(re.findall(rf"\b{re.escape(name)}\b", pybind_text)) < 2:
            issues.append(
                SourceIssue(
                    "kernel/pybind11.cpp",
                    line_number,
                    "unused_host_wrapper",
                    f"{name} is declared but never called by pybind11.cpp",
                )
            )
    return issues


def render_issues(issues: list[SourceIssue]) -> str:
    return "\n".join(issue.render() for issue in issues)
