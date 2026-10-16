"""Validation and human-review routing (MVP plan component 7, M7).

Runs explicit consistency checks and adds review flags. Flags never change a
score; they route the script to a human.
"""

from __future__ import annotations

from .config import AppConfig
from .grading import boundary_marks
from .kb_loader import KnowledgeBase
from .schemas import Fact, FiringLogEntry


def _referenced_facts(firing_log: list[FiringLogEntry]) -> set[str]:
    names: set[str] = set()
    for entry in firing_log:
        names.update(entry.matched_facts)
    return names


def validate(
    *,
    kb: KnowledgeBase,
    config: AppConfig,
    facts: list[Fact],
    firing_log: list[FiringLogEntry],
    scores: dict[str, float],
    levels: dict[str, int],
    total: float,
    llm_total: float | None = None,
) -> list[str]:
    """Return the list of review flags (deduplicated, ordered)."""

    flags: list[str] = []
    by_name = {f.name: f for f in facts}
    thresholds = config.thresholds

    # 1. Scores within bounds and consistent with their level range.
    for crit in kb.rubric.criteria:
        score = scores.get(crit.id, 0.0)
        if score < 0 or score > crit.max_score:
            flags.append(f"score-out-of-bounds:{crit.id}")
        level = levels.get(crit.id)
        if level is not None:
            lvl = crit.level(level)
            if lvl is not None and not (lvl.range[0] <= score <= lvl.range[1]):
                flags.append(f"score-outside-level-range:{crit.id}")

    # 2. Low-confidence LLM facts that actually influenced the mark.
    referenced = _referenced_facts(firing_log)
    for name in sorted(referenced):
        fact = by_name.get(name)
        if (
            fact is not None
            and fact.source == "llm"
            and fact.confidence is not None
            and fact.confidence < thresholds.low_confidence
        ):
            flags.append("low-confidence-evidence")

    # 3. LLM-vs-rule disagreement (LLM holistic estimate vs rule total).
    if llm_total is not None and abs(llm_total - total) > thresholds.disagreement_threshold:
        flags.append("llm-rule-disagreement")

    # 4. Borderline marks near a grade boundary.
    for boundary in boundary_marks(kb.rubric.max_total):
        if abs(total - boundary) <= thresholds.borderline_margin:
            flags.append("borderline-mark")
            break

    # 5. Very short essays.
    word_count = by_name.get("word_count")
    if word_count is not None and isinstance(word_count.value, (int, float)):
        if word_count.value < thresholds.min_words:
            flags.append("essay-too-short")

    # 6. Off-topic.
    on_topic = by_name.get("on_topic_rating")
    if on_topic is not None and isinstance(on_topic.value, (int, float)) and on_topic.value <= 2:
        flags.append("off-topic")

    # 7. Prompt injection detected in the essay text.
    injection = by_name.get("injection_detected")
    if injection is not None and injection.value is True:
        flags.append("possible-prompt-injection")

    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            ordered.append(f)
    return ordered
