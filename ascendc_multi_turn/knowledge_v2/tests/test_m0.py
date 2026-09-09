from __future__ import annotations

import json
import unittest
from pathlib import Path

from ascendc_multi_turn.knowledge_v2 import (
    AtomicFact,
    FactQuery,
    MarkdownNormalizer,
    NormalizedDocument,
    ParameterTableFactExtractor,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DATACOPY_PAD = (
    REPO_ROOT
    / "skills/ascendc/ascendc-translator/references/AscendC_knowledge/api_reference/pages"
    / "atlasascendc_api_07_0265.md"
)


class KnowledgeCompilerMvpTests(unittest.TestCase):
    def test_normalized_document_round_trip_preserves_markdown_structure(self) -> None:
        document = MarkdownNormalizer().parse_path(DATACOPY_PAD)
        restored = NormalizedDocument.from_dict(json.loads(json.dumps(document.to_dict())))

        self.assertEqual(restored, document)
        self.assertEqual(document.document_id, "atlasascendc_api_07_0265")
        self.assertTrue(any(item.title == "参数说明" for item in document.headings))
        self.assertTrue(any("blockLen" in row for table in document.tables for row in table.rows))
        self.assertTrue(document.code_candidates)
        self.assertTrue(document.source_url and document.source_url.startswith("https://"))

    def test_datacopy_pad_block_len_fact_is_queryable_with_provenance(self) -> None:
        document = MarkdownNormalizer().parse_path(DATACOPY_PAD)
        facts = ParameterTableFactExtractor().extract(document)

        matches = FactQuery(facts).find(api="DataCopyPad", parameter="blockLen")

        self.assertGreaterEqual(len(matches), 2)
        fact = matches[0]
        self.assertEqual(fact.subject, "DataCopyPad.blockLen")
        self.assertEqual(fact.predicate, "parameter_semantics")
        self.assertIn("单位为字节", fact.value["description"])
        self.assertEqual(fact.provenance.document_id, document.document_id)
        self.assertIn("blockLen", fact.provenance.evidence_text)
        self.assertTrue(fact.provenance.source_hash)
        self.assertEqual(AtomicFact.from_dict(fact.to_dict()), fact)


if __name__ == "__main__":
    unittest.main()
