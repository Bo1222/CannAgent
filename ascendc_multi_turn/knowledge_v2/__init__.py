"""Offline semantic knowledge compiler primitives.

This package is intentionally independent from the runtime knowledge router
until the migration reaches the semantic-router phase.
"""

from .extract import FactExtractor, ParameterTableFactExtractor
from .normalize import MarkdownNormalizer
from .query import FactQuery
from .router import KnowledgeRouterV2, SnapshotView, locate_snapshot, render_bundle
from .call_semantics import CallSemanticsResolver
from .semantic_validator import SemanticIssue, SemanticValidator
from .experience import build_incident, persist_incident, promote_confirmed_experience
from .frontier import FrontierDecision, FrontierManager, reached_frontier
from .schema import (
    ApiCard,
    AtomicFact,
    CodeCandidate,
    FailureCard,
    ConfirmedExperience,
    HeadingNode,
    KnowledgeBundle,
    KnowledgeContext,
    IncidentRecord,
    NormalizedDocument,
    Paragraph,
    PatternCard,
    ProjectContract,
    Provenance,
    ResolvedApiCall,
    RetrievalTraceEntry,
    StructuredFailure,
    TableNode,
)

__all__ = [
    "ApiCard",
    "AtomicFact",
    "CodeCandidate",
    "CallSemanticsResolver",
    "FactExtractor",
    "FactQuery",
    "FailureCard",
    "ConfirmedExperience",
    "FrontierDecision",
    "FrontierManager",
    "HeadingNode",
    "KnowledgeBundle",
    "KnowledgeContext",
    "IncidentRecord",
    "KnowledgeRouterV2",
    "MarkdownNormalizer",
    "NormalizedDocument",
    "Paragraph",
    "ParameterTableFactExtractor",
    "PatternCard",
    "ProjectContract",
    "Provenance",
    "ResolvedApiCall",
    "RetrievalTraceEntry",
    "SnapshotView",
    "StructuredFailure",
    "SemanticIssue",
    "SemanticValidator",
    "TableNode",
    "locate_snapshot",
    "render_bundle",
    "build_incident",
    "persist_incident",
    "promote_confirmed_experience",
    "reached_frontier",
]
