from __future__ import annotations

from typing import Protocol

from .schema import AtomicFact, NormalizedDocument, Provenance


class FactExtractor(Protocol):
    def extract(self, document: NormalizedDocument) -> list[AtomicFact]: ...


class ParameterTableFactExtractor:
    """MVP extractor for explicit parameter-description table rows.

    Semantic extraction is deliberately separate from Markdown parsing. This
    implementation only emits facts for rows whose headers explicitly identify
    a parameter name and description.
    """

    def extract(self, document: NormalizedDocument) -> list[AtomicFact]:
        api = document.title.split("-", 1)[0].split("(", 1)[0].strip()
        facts: list[AtomicFact] = []
        for table in document.tables:
            headers = [item.replace(" ", "") for item in table.headers]
            parameter_index = next(
                (index for index, name in enumerate(headers) if name in {"参数名", "参数名称"}),
                None,
            )
            description_index = next(
                (index for index, name in enumerate(headers) if name in {"描述", "含义"}),
                None,
            )
            if parameter_index is None or description_index is None:
                continue
            for row in table.rows:
                parameter = row[parameter_index].strip()
                description = row[description_index].strip()
                if not parameter or not description:
                    continue
                evidence = f"{parameter} | {description}"
                facts.append(
                    AtomicFact(
                        subject=f"{api}.{parameter}",
                        predicate="parameter_semantics",
                        value={"parameter": parameter, "description": description},
                        applicability={"api": api},
                        provenance=Provenance(
                            document_id=document.document_id,
                            source_path=document.source_path,
                            source_hash=document.source_hash,
                            section_id=table.section_id,
                            evidence_text=evidence,
                        ),
                    )
                )
        return facts
