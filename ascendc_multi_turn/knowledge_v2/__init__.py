"""Offline semantic knowledge compiler primitives.

This package is intentionally independent from the runtime knowledge router
until the migration reaches the semantic-router phase.
"""

from .extract import FactExtractor, ParameterTableFactExtractor
from .normalize import MarkdownNormalizer
from .query import FactQuery
from .schema import (
    AtomicFact,
    CodeCandidate,
    HeadingNode,
    NormalizedDocument,
    Paragraph,
    Provenance,
    TableNode,
)

__all__ = [
    "AtomicFact",
    "CodeCandidate",
    "FactExtractor",
    "FactQuery",
    "HeadingNode",
    "MarkdownNormalizer",
    "NormalizedDocument",
    "Paragraph",
    "ParameterTableFactExtractor",
    "Provenance",
    "TableNode",
]
