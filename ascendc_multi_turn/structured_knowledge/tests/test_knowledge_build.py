from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ascendc_multi_turn.structured_knowledge.extract import ParameterTableFactExtractor
from ascendc_multi_turn.structured_knowledge.knowledge_build import (
    build_knowledge,
    load_knowledge_build,
)
from ascendc_multi_turn.structured_knowledge.normalize import MarkdownNormalizer
from ascendc_multi_turn.structured_knowledge.schema import AtomicFact, Provenance
from ascendc_multi_turn.structured_knowledge.validate import (
    ConflictResolver,
    FactValidator,
)

DOC = """# DataCopyPad-API-CANN\n**页面ID:** datacopy_pad\n**来源:** https://example.test/pad\n\n#### 参数说明\n\n| 参数名称 | 含义 |\n| --- | --- |\n| blockLen | 单位为字节。AscendC::DataCopyPad(...); AscendC::DataCopy(...); AscendC::TPipepipe; AscendC::TQue<...> |\n"""


class KnowledgeBuildTests(unittest.TestCase):
    def test_public_build_cli_is_available(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "ascendc_multi_turn.knowledge.build", "--help"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--source", completed.stdout)

    def test_build_is_loadable_published_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "pad.md").write_text(DOC, encoding="utf-8")
            output = root / "store"

            first = build_knowledge(source=source, output=output, version="8.5.0")
            second = build_knowledge(source=source, output=output, version="8.5.0")

            self.assertEqual(first, second)
            manifest = load_knowledge_build(first)
            self.assertEqual(manifest["version"], "8.5.0")
            current = json.loads((output / "cann/8.5.0/current.json").read_text())
            self.assertEqual(current["knowledge_build_id"], first.name)
            self.assertTrue((first / "raw/pad.md").is_file())
            self.assertTrue((first / "normalized/datacopy_pad.json").is_file())
            symbols = json.loads((first / "indexes/symbols.json").read_text())
            for symbol in ("DataCopy", "DataCopyPad", "TPipe", "TQue"):
                self.assertIn(symbol, symbols)

    def test_validator_requires_evidence_applicability_and_source_hash(self) -> None:
        document = MarkdownNormalizer().parse(DOC, source_path="pad.md")
        fact = ParameterTableFactExtractor().extract(document)[0]
        self.assertEqual(FactValidator().validate(fact, {document.document_id: document}), [])

        invalid = AtomicFact(
            subject=fact.subject,
            predicate=fact.predicate,
            value=fact.value,
            applicability={},
            provenance=Provenance(document.document_id, "pad.md", "", fact.provenance.section_id, ""),
        )
        codes = {item.code for item in FactValidator().validate(invalid, {document.document_id: document})}
        self.assertTrue({"missing_evidence", "missing_applicability", "missing_source_hash"} <= codes)

    def test_context_difference_is_not_a_conflict(self) -> None:
        document = MarkdownNormalizer().parse(DOC, source_path="pad.md")
        base = ParameterTableFactExtractor().extract(document)[0]
        other_context = AtomicFact(
            subject=base.subject,
            predicate=base.predicate,
            value={"unit": "data_block"},
            applicability={"api": "DataCopyPad", "overload": "LocalToGlobal"},
            provenance=base.provenance,
        )
        same_context = AtomicFact(
            subject=base.subject,
            predicate=base.predicate,
            value={"unit": "element"},
            applicability=base.applicability,
            provenance=base.provenance,
        )

        self.assertEqual(ConflictResolver().find_conflicts([base, other_context]), [])
        self.assertEqual(len(ConflictResolver().find_conflicts([base, same_context])), 1)


if __name__ == "__main__":
    unittest.main()
