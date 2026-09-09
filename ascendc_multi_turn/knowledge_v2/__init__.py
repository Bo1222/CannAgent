"""Offline semantic knowledge compiler primitives.

This package is intentionally independent from the runtime knowledge router
until the migration reaches the semantic-router phase.
"""

from .extract import FactExtractor, ParameterTableFactExtractor
from .normalize import MarkdownNormalizer
from .query import FactQuery
from .router import KnowledgeRouterV2, SnapshotView, locate_snapshot, render_bundle
from .schema import (
    ApiCard,
    AtomicFact,
    CodeCandidate,
    FailureCard,
    HeadingNode,
    KnowledgeBundle,
    KnowledgeContext,
    NormalizedDocument,
    Paragraph,
    PatternCard,
    ProjectContract,
    Provenance,
    RetrievalTraceEntry,
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
    "KnowledgeBundle",
    "KnowledgeContext",
    "KnowledgeRouterV2",
    "MarkdownNormalizer",
    "NormalizedDocument",
    "Paragraph",
    "ParameterTableFactExtractor",
    "PatternCard",
    "ProjectContract",
    "Provenance",
    "RetrievalTraceEntry",
    "SnapshotView",
    "TableNode",
    "locate_snapshot",
    "render_bundle",
]
