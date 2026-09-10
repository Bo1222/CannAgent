from __future__ import annotations

import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .knowledge_build import load_knowledge_build
from .schema import KnowledgeBundle, KnowledgeContext, RetrievalTraceEntry

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]{2,}")


def locate_knowledge_build(
    store: Path,
    version: str,
    knowledge_build_id: str | None = None,
) -> Path:
    version_root = store.expanduser().resolve() / "cann" / version
    builds_root = version_root / "builds"
    if knowledge_build_id:
        candidate = builds_root / knowledge_build_id
        load_knowledge_build(candidate)
        return candidate
    current = version_root / "current.json"
    if not current.is_file():
        raise ValueError(
            f"structured knowledge for CANN {version} is not installed under {version_root}; "
            "run python -m ascendc_multi_turn.knowledge.build first"
        )
    payload = json.loads(current.read_text(encoding="utf-8"))
    selected_id = payload.get("knowledge_build_id")
    if not isinstance(selected_id, str) or not selected_id:
        raise ValueError(f"invalid structured knowledge publication pointer: {current}")
    candidate = builds_root / selected_id
    load_knowledge_build(candidate)
    return candidate


class KnowledgeBuild:
    def __init__(self, path: Path):
        self.path = path
        self.manifest = load_knowledge_build(path)
        self.symbols: dict[str, str] = json.loads((path / "indexes/symbols.json").read_text())
        self.facts: list[dict[str, Any]] = json.loads(
            (path / "facts/atomic_facts.json").read_text()
        )
        self.api_cards: list[dict[str, Any]] = json.loads(
            (path / "cards/api_cards.json").read_text()
        )
        self.failure_cards: list[dict[str, Any]] = json.loads(
            (path / "cards/failure_cards.json").read_text()
        )
        self.pattern_cards: list[dict[str, Any]] = json.loads(
            (path / "cards/pattern_cards.json").read_text()
        )
        contracts_path = path / "cards/project_contracts.json"
        self.project_contracts: list[dict[str, Any]] = (
            json.loads(contracts_path.read_text()) if contracts_path.is_file() else []
        )
        self.cards_by_id = {item["card_id"]: item for item in self.api_cards}
        self.facts_by_id = {item["fact_id"]: item for item in self.facts}


def _tokens(value: Any) -> set[str]:
    return {item.lower() for item in _TOKEN.findall(json.dumps(value, ensure_ascii=False))}


def _cosine(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


class StructuredKnowledgeRouter:
    """Deterministic router over a validated structured knowledge build."""

    def __init__(self, knowledge: KnowledgeBuild):
        self.knowledge = knowledge

    def route(self, context: KnowledgeContext) -> KnowledgeBundle:
        trace: list[RetrievalTraceEntry] = []
        selected_cards: list[dict[str, Any]] = []
        exact = set(context.source_symbols)
        for symbol in sorted(exact):
            card_id = self.knowledge.symbols.get(symbol)
            if card_id:
                card = self.knowledge.cards_by_id[card_id]
                selected_cards.append(card)
                trace.append(RetrievalTraceEntry("exact_api", symbol, "selected", "exact source symbol"))
            else:
                trace.append(
                    RetrievalTraceEntry(
                        "exact_api",
                        symbol,
                        "rejected",
                        "no exact symbol in the selected knowledge build",
                    )
                )
        selected_names = {item["api"] for item in selected_cards}
        for candidate in self.knowledge.api_cards:
            name = candidate["api"]
            if name in selected_names:
                continue
            if any(name.startswith(symbol) or symbol.startswith(name) for symbol in exact):
                trace.append(
                    RetrievalTraceEntry(
                        "exact_api",
                        name,
                        "rejected",
                        "similar spelling is not an exact API identity",
                    )
                )

        selected_fact_ids = {
            fact_id for card in selected_cards for fact_id in card.get("fact_ids", [])
        }
        relevant_facts = []
        for fact_id in sorted(selected_fact_ids):
            fact = self.knowledge.facts_by_id.get(fact_id)
            if not fact:
                continue
            applicability = fact.get("applicability", {})
            fact_api = applicability.get("api")
            if fact_api not in selected_names:
                trace.append(RetrievalTraceEntry("context", fact_id, "rejected", "API context mismatch"))
                continue
            relevant_facts.append(fact)
            trace.append(RetrievalTraceEntry("context", fact_id, "selected", "exact API applicability"))

        failure_cards = []
        failure_tokens = _tokens(context.failure or {})
        for card in self.knowledge.failure_cards:
            score = _cosine(failure_tokens, _tokens(card.get("signals", [])))
            if score > 0:
                failure_cards.append(card)
                trace.append(RetrievalTraceEntry("metadata", card["card_id"], "selected", "failure metadata match", score))

        query_tokens = _tokens(
            {
                "operator": context.operator,
                "phase": context.phase,
                "symbols": context.source_symbols,
                "failure": context.failure,
                "active_plan": context.active_plan,
            }
        )
        examples: list[str] = []
        supplemental = []
        for card in self.knowledge.pattern_cards:
            lexical_score = len(query_tokens & _tokens(card))
            if lexical_score:
                supplemental.append((float(lexical_score), card, "fts"))
        if not supplemental:
            for card in self.knowledge.pattern_cards:
                vector_score = _cosine(query_tokens, _tokens(card))
                if vector_score > 0:
                    supplemental.append((vector_score, card, "vector"))
        for score, card, stage in sorted(supplemental, key=lambda item: (-item[0], item[1]["card_id"]))[:3]:
            examples.append(card.get("description", ""))
            trace.append(RetrievalTraceEntry(stage, card["card_id"], "selected", "supplemental pattern only", score))

        provenance = []
        seen_provenance = set()
        selected_patterns = [
            card
            for _, card, _ in sorted(
                supplemental,
                key=lambda item: (-item[0], item[1]["card_id"]),
            )[:3]
        ]
        for knowledge in [
            *relevant_facts,
            *failure_cards,
            *self.knowledge.project_contracts,
            *selected_patterns,
        ]:
            item = knowledge.get("provenance", {})
            if not item:
                continue
            key = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if key not in seen_provenance:
                seen_provenance.add(key)
                provenance.append(item)
        return KnowledgeBundle(
            context=context,
            api_semantics=selected_cards,
            relevant_facts=relevant_facts,
            examples=examples,
            failure_cards=failure_cards,
            project_contracts=self.knowledge.project_contracts,
            provenance=provenance,
            retrieval_trace=trace,
        )


def render_bundle(bundle: KnowledgeBundle, *, max_chars: int = 24000) -> str:
    payload = bundle.to_dict()
    payload["retrieval_trace"] = [asdict(item) for item in bundle.retrieval_trace]
    text = "# Verified structured AscendC knowledge\n" + json.dumps(
        payload, ensure_ascii=False, indent=2
    )
    return text[:max_chars]
