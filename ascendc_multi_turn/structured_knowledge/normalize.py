from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .schema import CodeCandidate, HeadingNode, NormalizedDocument, Paragraph, TableNode

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PAGE_ID = re.compile(r"^\*\*页面ID:\*\*\s*(.+?)\s*$")
_SOURCE = re.compile(r"^\*\*来源:\*\*\s*(\S+)\s*$")
_TABLE_SEPARATOR = re.compile(r"^:?-{3,}:?$")
_CODE_MARKERS = ("#include", "__aicore__", "template<", "AscendC::", "PYBIND11_MODULE")


def _cells(line: str) -> list[str]:
    return [item.strip() for item in line.strip().strip("|").split("|")]


class MarkdownNormalizer:
    """Parse Markdown structure without interpreting hardware semantics."""

    parser_version = "1"

    def parse_path(self, path: Path) -> NormalizedDocument:
        return self.parse(path.read_text(encoding="utf-8", errors="replace"), source_path=str(path))

    def parse(self, text: str, *, source_path: str) -> NormalizedDocument:
        source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document_id = Path(source_path).stem
        source_url: str | None = None
        headings: list[HeadingNode] = []
        paragraphs: list[Paragraph] = []
        tables: list[TableNode] = []
        code_candidates: list[CodeCandidate] = []
        constraints: list[Paragraph] = []
        examples: list[Paragraph] = []
        stack: list[HeadingNode] = []
        current_section = f"{document_id}::root"
        current_title = document_id
        lines = text.splitlines()
        index = 0

        while index < len(lines):
            line = lines[index].rstrip()
            page_id = _PAGE_ID.match(line)
            if page_id:
                document_id = page_id.group(1).strip()
                index += 1
                continue
            source = _SOURCE.match(line)
            if source:
                source_url = source.group(1)
                index += 1
                continue
            heading = _HEADING.match(line)
            if heading:
                level = len(heading.group(1))
                title = heading.group(2).strip()
                while stack and stack[-1].level >= level:
                    stack.pop()
                node = HeadingNode(
                    section_id=f"{document_id}::h{len(headings) + 1:03d}",
                    level=level,
                    title=title,
                    parent_id=stack[-1].section_id if stack else None,
                )
                headings.append(node)
                stack.append(node)
                current_section = node.section_id
                if len(headings) == 1:
                    current_title = title
                index += 1
                continue
            if line.strip().startswith("|") and index + 1 < len(lines):
                header = _cells(line)
                separator = _cells(lines[index + 1])
                if len(header) == len(separator) and all(_TABLE_SEPARATOR.match(cell) for cell in separator):
                    rows: list[list[str]] = []
                    index += 2
                    while index < len(lines) and lines[index].strip().startswith("|"):
                        row = _cells(lines[index])
                        if len(row) < len(header):
                            row.extend([""] * (len(header) - len(row)))
                        rows.append(row[: len(header)])
                        for cell in row:
                            if any(marker in cell for marker in _CODE_MARKERS):
                                code_candidates.append(CodeCandidate(current_section, cell, "table_cell"))
                        index += 1
                    tables.append(
                        TableNode(
                            table_id=f"{document_id}::t{len(tables) + 1:03d}",
                            section_id=current_section,
                            headers=header,
                            rows=rows,
                        )
                    )
                    continue
            if line.strip() and line.strip() != "---":
                paragraph = Paragraph(current_section, line.strip())
                paragraphs.append(paragraph)
                section_title = stack[-1].title if stack else ""
                if "约束" in section_title:
                    constraints.append(paragraph)
                if "示例" in section_title:
                    examples.append(paragraph)
                if any(marker in line for marker in _CODE_MARKERS):
                    code_candidates.append(CodeCandidate(current_section, line.strip(), "paragraph"))
            index += 1

        return NormalizedDocument(
            document_id=document_id,
            source_path=source_path,
            source_hash=source_hash,
            title=current_title,
            source_url=source_url,
            headings=headings,
            paragraphs=paragraphs,
            tables=tables,
            code_candidates=code_candidates,
            constraints=constraints,
            examples=examples,
        )
