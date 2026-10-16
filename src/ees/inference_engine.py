"""Forward-chaining inference engine (MVP plan component 5).

Applies IF-THEN rules to working memory to a fixpoint, resolving conflicts by
priority (ties broken by specificity) and recording a full firing log. The
engine is deliberately small and explicit so every step is demonstrable and
explainable. It never asks the LLM for a score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .kb_loader import KnowledgeBase
from .schemas import Fact, FiringLogEntry, Rule, RuleAction
from .working_memory import WorkingMemory

DEFAULT_MAX_CYCLES = 50


@dataclass
class EngineResult:
    """The outcome of running the engine on one fact set."""

    facts: list[Fact]
    firing_log: list[FiringLogEntry]
    scores: dict[str, float] = field(default_factory=dict)
    levels: dict[str, int] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    penalties: dict[str, float] = field(default_factory=dict)
    steps: int = 0


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class InferenceEngine:
    """A small, deterministic forward-chaining engine over :class:`Rule` objects."""

    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        # Sort by id for deterministic iteration regardless of YAML order.
        self.rules: list[Rule] = sorted(kb.rules, key=lambda r: r.id)

    # ------------------------------------------------------------------ #
    # Condition evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, cond, wm: WorkingMemory) -> bool:
        """Evaluate a condition tree against working memory."""

        if cond.fact is not None:
            return self._eval_leaf(cond.fact, cond.op, cond.value, wm)
        if cond.all is not None:
            return all(self.evaluate(c, wm) for c in cond.all)
        if cond.any is not None:
            return any(self.evaluate(c, wm) for c in cond.any)
        if cond.not_ is not None:
            return not self.evaluate(cond.not_, wm)
        return False

    @staticmethod
    def _eval_leaf(fact: str, op: str | None, value: Any, wm: WorkingMemory) -> bool:
        if not wm.has(fact):
            # An unknown fact never satisfies a comparison.
            return False
        actual = wm.value(fact)
        try:
            if op == "==":
                return actual == value
            if op == "!=":
                return actual != value
            if op == ">":
                return actual > value
            if op == ">=":
                return actual >= value
            if op == "<":
                return actual < value
            if op == "<=":
                return actual <= value
            if op == "in":
                return actual in value
            if op == "not_in":
                return actual not in value
            if op == "contains":
                return value in actual
        except TypeError:
            return False
        return False

    # ------------------------------------------------------------------ #
    # Conflict resolution helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _conflict_key(rule: Rule) -> tuple[str, str] | None:
        """The target a rule competes over, or ``None`` if it cannot conflict.

        A ``score`` action dominates; otherwise an ``assert`` does. Rules that
        only ``flag`` or ``penalty`` never conflict and always fire.
        """

        for action in rule.then:
            if action.action == "score":
                crit = action.criterion or rule.criterion
                if crit:
                    return ("score", crit)
        for action in rule.then:
            if action.action == "assert" and action.fact:
                return ("assert", action.fact)
        return None

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(
        self,
        facts: list[Fact],
        max_cycles: int = DEFAULT_MAX_CYCLES,
        clock: Callable[[], str] = _utcnow,
    ) -> EngineResult:
        """Run forward chaining to a fixpoint and return the full trace."""

        wm = WorkingMemory(facts)
        log: list[FiringLogEntry] = []
        scores: dict[str, float] = {}
        levels: dict[str, int] = {}
        flags: list[str] = []
        penalties: dict[str, float] = {}
        penalty_caps: dict[str, float] = {}
        fired: set[str] = set()
        step = 0

        for _cycle in range(max_cycles):
            matching = [
                r
                for r in self.rules
                if r.id not in fired and self.evaluate(r.if_, wm)
            ]
            if not matching:
                break

            winners, losers_by_winner = self._resolve(matching)
            # Every rule in the conflict set is now "handled": losers are
            # blocked so they cannot fire in a later cycle.
            fired.update(r.id for r in matching)
            for rule in winners:
                step += 1
                asserted: dict[str, Any] = {}
                for action in rule.then:
                    self._apply(
                        action, rule, wm, scores, levels, flags, penalties, penalty_caps, asserted
                    )
                log.append(
                    FiringLogEntry(
                        step=step,
                        rule_id=rule.id,
                        matched_facts=self._matched_facts(rule, wm),
                        asserted=asserted,
                        conflicts_resolved_against=losers_by_winner.get(rule.id, []),
                        timestamp=clock(),
                        explanation=rule.explain,
                        criterion=rule.criterion,
                    )
                )
        else:  # pragma: no cover - only on pathological rule sets
            raise RuntimeError(
                f"inference did not reach a fixpoint within {max_cycles} cycles "
                "(possible rule loop)"
            )

        # Apply cumulative penalties (bounded by the strictest cap).
        for crit, total in penalties.items():
            cap = penalty_caps.get(crit)
            applied = min(total, cap) if cap is not None else total
            scores[crit] = round(scores.get(crit, 0.0) - applied, 4)

        return EngineResult(
            facts=wm.all(),
            firing_log=log,
            scores=scores,
            levels=levels,
            flags=flags,
            penalties=penalties,
            steps=step,
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve(
        self, matching: list[Rule]
    ) -> tuple[list[Rule], dict[str, list[str]]]:
        """Pick winners from the conflict set; record losers per winner."""

        # Group conflicting rules by their conflict key.
        groups: dict[tuple[str, str], list[Rule]] = {}
        keyless: list[Rule] = []
        for rule in matching:
            key = self._conflict_key(rule)
            if key is None:
                keyless.append(rule)
            else:
                groups.setdefault(key, []).append(rule)

        winners: list[Rule] = list(keyless)
        losers_by_winner: dict[str, list[str]] = {}
        for _key, rules in groups.items():
            ordered = sorted(
                rules, key=lambda r: (r.priority, r.specificity, r.id), reverse=True
            )
            winner = ordered[0]
            winners.append(winner)
            losers = [r.id for r in ordered[1:]]
            if losers:
                losers_by_winner[winner.id] = sorted(losers)

        winners.sort(key=lambda r: r.id)
        return winners, losers_by_winner

    def _matched_facts(self, rule: Rule, wm: WorkingMemory) -> list[str]:
        return sorted(n for n in set(rule.if_.referenced_facts()) if wm.has(n))

    def _apply(
        self,
        action: RuleAction,
        rule: Rule,
        wm: WorkingMemory,
        scores: dict[str, float],
        levels: dict[str, int],
        flags: list[str],
        penalties: dict[str, float],
        penalty_caps: dict[str, float],
        asserted: dict[str, Any],
    ) -> None:
        if action.action == "assert":
            assert action.fact is not None
            wm.set_value(action.fact, action.value, source="rule")
            asserted[action.fact] = action.value
            if action.fact.endswith("_level"):
                levels[action.fact[: -len("_level")]] = int(action.value)
        elif action.action == "score":
            crit = action.criterion or rule.criterion
            assert crit is not None
            value = action.value
            if value is None and action.level is not None:
                crit_obj = self.kb.rubric.criterion(crit)
                lvl_obj = crit_obj.level(action.level) if crit_obj else None
                value = lvl_obj.representative_score if lvl_obj else 0
            scores[crit] = float(value)
            asserted[f"score:{crit}"] = float(value)
        elif action.action == "add":
            crit = action.criterion or rule.criterion
            assert crit is not None and action.amount is not None
            scores[crit] = round(scores.get(crit, 0.0) + float(action.amount), 4)
            asserted[f"add:{crit}"] = float(action.amount)
        elif action.action == "penalty":
            crit = action.criterion or rule.criterion
            assert crit is not None and action.amount is not None
            penalties[crit] = penalties.get(crit, 0.0) + float(action.amount)
            if action.cap is not None:
                penalty_caps[crit] = max(penalty_caps.get(crit, 0.0), float(action.cap))
            asserted[f"penalty:{crit}"] = float(action.amount)
        elif action.action == "flag":
            if action.reason and action.reason not in flags:
                flags.append(action.reason)
            asserted[f"flag:{action.reason}"] = True
