from __future__ import annotations

from collections.abc import Iterable

from .schema import AtomicFact


class FactQuery:
    def __init__(self, facts: Iterable[AtomicFact]):
        self._facts = tuple(facts)

    def find(
        self,
        *,
        subject: str | None = None,
        api: str | None = None,
        parameter: str | None = None,
        predicate: str | None = None,
    ) -> list[AtomicFact]:
        matches = []
        for fact in self._facts:
            if subject is not None and fact.subject != subject:
                continue
            if api is not None and fact.applicability.get("api") != api:
                continue
            if parameter is not None and fact.value.get("parameter") != parameter:
                continue
            if predicate is not None and fact.predicate != predicate:
                continue
            matches.append(fact)
        return matches
