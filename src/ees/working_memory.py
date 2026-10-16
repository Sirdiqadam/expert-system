"""Working memory: the fact base for a single essay (MVP plan component 4)."""

from __future__ import annotations

from typing import Any

from .schemas import Fact, Span


class WorkingMemory:
    """A typed, ordered store of facts for one essay.

    Facts keep their ``source`` and ``evidence`` so the UI can badge them and
    the explainer can attribute every score back to evidence.
    """

    def __init__(self, facts: list[Fact] | None = None) -> None:
        self._facts: dict[str, Fact] = {}
        for fact in facts or []:
            self.set(fact)

    def set(self, fact: Fact) -> None:
        """Insert or replace a fact by name (later sources win)."""

        self._facts[fact.name] = fact

    def set_value(
        self,
        name: str,
        value: Any,
        source: str = "rule",
        confidence: float | None = None,
        evidence: list[Span] | None = None,
    ) -> Fact:
        fact = Fact(
            name=name,
            value=value,
            source=source,  # type: ignore[arg-type]
            confidence=confidence,
            evidence=evidence or [],
        )
        self.set(fact)
        return fact

    def get(self, name: str) -> Fact | None:
        return self._facts.get(name)

    def value(self, name: str, default: Any = None) -> Any:
        fact = self._facts.get(name)
        return fact.value if fact is not None else default

    def has(self, name: str) -> bool:
        return name in self._facts

    def all(self) -> list[Fact]:
        """Facts sorted by name for deterministic display and logging."""

        return [self._facts[k] for k in sorted(self._facts)]

    def by_source(self, source: str) -> list[Fact]:
        return [f for f in self.all() if f.source == source]

    def __len__(self) -> int:
        return len(self._facts)
