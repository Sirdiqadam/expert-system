"""Explanation facility (MVP plan component 6, M6).

Builds a proof tree (score <- rules <- facts <- evidence) from the firing log,
renders deterministic template explanations per fired rule, answers
"Why not level X?" and derives counterfactual advice. The optional LLM rewrite
never adds a reason that is not in the tree.
"""

from __future__ import annotations

import re

from .kb_loader import KnowledgeBase
from .schemas import Condition, Fact, FiringLogEntry, ProofNode, Rule

TEMPLATE_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")


def _facts_by_name(facts: list[Fact]) -> dict[str, Fact]:
    return {f.name: f for f in facts}


def format_template(template: str, facts_by_name: dict[str, Fact]) -> str:
    """Fill ``{fact_name}`` placeholders in a rule's explain text."""

    def repl(match: re.Match[str]) -> str:
        fact = facts_by_name.get(match.group(1))
        return str(fact.value) if fact is not None else "?"

    return TEMPLATE_RE.sub(repl, template or "")


def _rule_affects_score(rule: Rule) -> bool:
    return any(a.action in {"score", "penalty", "assert"} for a in rule.then)


# --------------------------------------------------------------------------- #
# Proof tree + template explanations
# --------------------------------------------------------------------------- #
def build_proof_tree(
    kb: KnowledgeBase,
    facts: list[Fact],
    firing_log: list[FiringLogEntry],
    scores: dict[str, float],
    levels: dict[str, int],
) -> list[ProofNode]:
    """One root node per criterion: score -> rules -> facts -> evidence."""

    by_name = _facts_by_name(facts)
    roots: list[ProofNode] = []
    for crit in kb.rubric.criteria:
        children: list[ProofNode] = []
        for entry in firing_log:
            rule = kb.rule(entry.rule_id)
            if rule is None or rule.criterion != crit.id or not _rule_affects_score(rule):
                continue
            fact_nodes: list[ProofNode] = []
            for fname in entry.matched_facts:
                fact = by_name.get(fname)
                if fact is None:
                    continue
                evidence = [
                    ProofNode(label=span.quote, kind="evidence", detail=f"[{span.start}:{span.end}]")
                    for span in fact.evidence
                ]
                fact_nodes.append(
                    ProofNode(
                        label=fname,
                        kind="fact",
                        value=fact.value,
                        detail=f"source={fact.source}",
                        children=evidence,
                    )
                )
            children.append(
                ProofNode(
                    label=entry.rule_id,
                    kind="rule",
                    detail=format_template(rule.explain, by_name),
                    value={"priority": rule.priority},
                    children=fact_nodes,
                )
            )
        roots.append(
            ProofNode(
                label=crit.name,
                kind="score",
                value=scores.get(crit.id, 0.0),
                detail=f"level {levels.get(crit.id, '?')} of {crit.max_score}",
                children=children,
            )
        )
    return roots


def template_explanations(
    kb: KnowledgeBase, facts: list[Fact], firing_log: list[FiringLogEntry]
) -> list[str]:
    """A deterministic, always-available explanation sentence per fired rule."""

    by_name = _facts_by_name(facts)
    lines: list[str] = []
    for entry in firing_log:
        rule = kb.rule(entry.rule_id)
        if rule is None:
            continue
        text = format_template(rule.explain, by_name) or rule.description
        lines.append(f"[{entry.rule_id}] {text}")
    return lines


# --------------------------------------------------------------------------- #
# Condition introspection (for counterfactuals / why-not)
# --------------------------------------------------------------------------- #
def _leaf_text(cond: Condition) -> str:
    return f"{cond.fact} {cond.op} {cond.value!r}"


def unmet_leaves(cond: Condition, values: dict[str, object]) -> list[str]:
    """Leaf conditions in *cond* that are not currently satisfied."""

    unmet: list[str] = []
    if cond.fact is not None:
        if not _leaf_true(cond, values):
            unmet.append(_leaf_text(cond))
        return unmet
    if cond.all is not None:
        for c in cond.all:
            unmet.extend(unmet_leaves(c, values))
        return unmet
    if cond.any is not None:
        # If any branch is satisfied the group is met; otherwise report all.
        if any(not unmet_leaves(c, values) for c in cond.any):
            return []
        for c in cond.any:
            unmet.extend(unmet_leaves(c, values))
        return unmet
    if cond.not_ is not None:
        if _leaf_true_of(cond.not_, values):
            unmet.append(f"NOT ({_describe(cond.not_)})")
        return unmet
    return unmet


def _describe(cond: Condition) -> str:
    if cond.fact is not None:
        return _leaf_text(cond)
    if cond.all is not None:
        return " AND ".join(_describe(c) for c in cond.all)
    if cond.any is not None:
        return " OR ".join(_describe(c) for c in cond.any)
    if cond.not_ is not None:
        return f"NOT ({_describe(cond.not_)})"
    return "?"


def _leaf_true(cond: Condition, values: dict[str, object]) -> bool:
    from .inference_engine import InferenceEngine

    class _WM:
        def has(self, name: str) -> bool:
            return name in values

        def value(self, name: str, default=None):  # noqa: ANN001
            return values.get(name, default)

    return InferenceEngine._eval_leaf(cond.fact, cond.op, cond.value, _WM())  # type: ignore[arg-type]


def _leaf_true_of(cond: Condition, values: dict[str, object]) -> bool:
    """Whether the whole sub-condition is satisfied (for NOT branches)."""

    if cond.fact is not None:
        return _leaf_true(cond, values)
    if cond.all is not None:
        return all(_leaf_true_of(c, values) for c in cond.all)
    if cond.any is not None:
        return any(_leaf_true_of(c, values) for c in cond.any)
    if cond.not_ is not None:
        return not _leaf_true_of(cond.not_, values)
    return False


# --------------------------------------------------------------------------- #
# Counterfactuals / why-not
# --------------------------------------------------------------------------- #
def why_not_level(
    kb: KnowledgeBase,
    criterion_id: str,
    target_level: int,
    facts: list[Fact],
) -> str:
    """Explain what is missing to reach *target_level* on a criterion."""

    values = {f.name: f.value for f in facts}
    crit = kb.rubric.criterion(criterion_id)
    name = crit.name if crit else criterion_id
    rules = [
        r
        for r in kb.rules
        if r.criterion == criterion_id
        and any(a.action == "score" and a.level == target_level for a in r.then)
    ]
    if not rules:
        return f"No rule defines level {target_level} for {name}."
    # Pick the rule whose conditions are closest to satisfied (fewest unmet).
    best = min(rules, key=lambda r: len(unmet_leaves(r.if_, values)))
    unmet = unmet_leaves(best.if_, values)
    if not unmet:
        return f"{name} already satisfies level {target_level} (rule {best.id})."
    missing = "; ".join(unmet)
    return (
        f"To reach level {target_level} for {name}, the essay needs: {missing} "
        f"(rule {best.id})."
    )


def counterfactuals(
    kb: KnowledgeBase,
    facts: list[Fact],
    levels: dict[str, int],
) -> list[str]:
    """For each criterion below its top level, the nearest unmet higher level."""

    tips: list[str] = []
    for crit in kb.rubric.criteria:
        current = levels.get(crit.id, 0)
        top = max(lvl.level for lvl in crit.levels)
        if current >= top:
            continue
        tips.append(why_not_level(kb, crit.id, current + 1, facts))
    return tips


# --------------------------------------------------------------------------- #
# Optional LLM rewrite (strict)
# --------------------------------------------------------------------------- #
REWRITE_SYSTEM = (
    "You rewrite an existing marking explanation into encouraging feedback for a "
    "student. Use ONLY the reasons, facts and quotes provided in the proof tree. "
    "Cite the rule id in square brackets after each claim, e.g. [R-CONT-01]. Do "
    "not mention any score that is not in the tree. Do not add advice that is not "
    "derivable from the tree. Group feedback by criterion and end with 2-3 "
    "improvement actions."
)


def build_rewrite_user(
    proof_tree: list[ProofNode],
    scores: dict[str, float],
    counterfactuals: list[str],
) -> str:
    import json

    tree = [node.model_dump() for node in proof_tree]
    return (
        "PROOF TREE (JSON):\n"
        + json.dumps(tree, indent=2)
        + "\n\nSCORES:\n"
        + json.dumps(scores)
        + "\n\nCOUNTERFACTUALS:\n- "
        + "\n- ".join(counterfactuals)
    )


def verify_feedback(feedback: str, firing_log: list[FiringLogEntry]) -> tuple[bool, list[str]]:
    """Check every ``[R-...]`` citation exists in the firing log (post-check)."""

    known = {e.rule_id for e in firing_log}
    cited = set(re.findall(r"\[(R-[A-Z0-9\-]+)\]", feedback))
    unknown = sorted(cited - known)
    return (not unknown), unknown
