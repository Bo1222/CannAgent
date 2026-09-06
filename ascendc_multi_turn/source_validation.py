from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_PIPE_DECLARATION = re.compile(r"\b(?:AscendC::)?TPipe\s*(?:[&*]\s*)?([A-Za-z_][A-Za-z0-9_]*)")
_INVALID_PIPE_METHODS = ("EnQue", "DeQue", "AllocTensor", "FreeTensor")
_FILE_VALUE = re.compile(
    r"^(?:inline\s+|static\s+|constexpr\s+|const\s+)*"
    r"(?:auto|bool|char|float|double|half|bfloat16_t|u?int(?:8|16|32|64)_t)\s+"
    r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:=|\{)"
)


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
    for path in sorted(kernel_dir.rglob("*")) if kernel_dir.is_dir() else []:
        if path.suffix not in {".cpp", ".cc", ".cxx", ".h", ".hpp"} or not path.is_file():
            continue
        relative = str(path.relative_to(task_dir))
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        text = "\n".join(lines)
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
    return issues


def render_issues(issues: list[SourceIssue]) -> str:
    return "\n".join(issue.render() for issue in issues)
