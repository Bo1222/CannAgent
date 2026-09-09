"""Offline semantic knowledge compiler primitives.

This package is intentionally independent from the runtime knowledge router
until the migration reaches the semantic-router phase.
"""

from .extract import FactExtractor, ParameterTableFactExtractor
from .normalize import MarkdownNormalizer
from .query import FactQuery
from .schema import (
    ApiCard,
    AtomicFact,
    CodeCandidate,
    FailureCard,
    HeadingNode,
    NormalizedDocument,
    Paragraph,
    PatternCard,
    Provenance,
    TableNode,
)

__all__ = [
    "ApiCard",
    "AtomicFact",
    "CodeCandidate",
    "FactExtractor",
    "FactQuery",
    "FailureCard",
    "HeadingNode",
    "MarkdownNormalizer",
    "NormalizedDocument",
    "Paragraph",
    "ParameterTableFactExtractor",
    "PatternCard",
    "Provenance",
    "TableNode",
]
