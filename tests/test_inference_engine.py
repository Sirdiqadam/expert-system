"""Tests for the forward-chaining inference engine (M2 acceptance)."""

from __future__ import annotations

import pytest

from ees.config import PROJECT_ROOT
from ees.inference_engine import InferenceEngine
from ees.kb_loader import load_knowledge
from ees.schemas import Fact


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    return InferenceEngine(load_knowledge(PROJECT_ROOT / "kb"))


def make_facts(**values) -> list[Fact]:
    return [Fact(name=k, value=v, source="nlp") for k, v in values.items()]


STRONG = dict(
    thesis_present=True,
    num_supporting_arguments=3,
    counterargument_rebutted=True,
    arguments_without_evidence=0,
    paragraph_count=4,
    coherence_rating=5,
    connective_count=8,
    avg_sentence_length=15.0,
    spelling_error_rate=0.005,
    grammar_error_count=0,
    word_count=250,
    on_topic_rating=5,
    injection_detected=False,
)


def test_strong_essay_scores_full_marks(engine: InferenceEngine) -> None:
    result = engine.run(make_facts(**STRONG), clock=lambda: "T")
    assert result.levels["content"] == 4
    assert result.scores["content"] == 10
    assert result.scores["organization"] == 10
    assert result.scores["language"] == 10
    assert result.scores["mechanics"] == 10
    assert sum(result.scores.values()) == 40
    assert result.flags == []


def test_weak_essay_scores_low(engine: InferenceEngine) -> None:
    facts = make_facts(
        thesis_present=False,
        num_supporting_arguments=0,
        counterargument_rebutted=False,
        arguments_without_evidence=0,
        paragraph_count=1,
        coherence_rating=1,
        connective_count=1,
        avg_sentence_length=30.0,
        spelling_error_rate=0.2,
        grammar_error_count=0,
        word_count=150,
        on_topic_rating=3,
        injection_detected=False,
    )
    result = engine.run(facts, clock=lambda: "T")
    assert result.levels["content"] == 1
    assert result.scores["content"] == 2  # representative score for range [0,3]
    assert "single-paragraph" in result.flags
    assert "high-spelling-errors" in result.flags


def test_determinism_same_facts_identical_log(engine: InferenceEngine) -> None:
    a = engine.run(make_facts(**STRONG), clock=lambda: "T")
    b = engine.run(make_facts(**STRONG), clock=lambda: "T")
    assert a.firing_log == b.firing_log
    assert a.scores == b.scores


def test_conflict_is_resolved_and_logged(engine: InferenceEngine) -> None:
    result = engine.run(make_facts(**STRONG), clock=lambda: "T")
    cont = next(e for e in result.firing_log if e.rule_id == "R-CONT-01")
    # Lower levels also matched but lost; the winner records the losers.
    # (R-CONT-04 does not match here because a thesis is present.)
    assert set(cont.conflicts_resolved_against) == {"R-CONT-02", "R-CONT-03"}
    # Only the winner actually set the level.
    assert result.levels["content"] == 4


def test_scores_within_rubric_bounds(engine: InferenceEngine) -> None:
    result = engine.run(make_facts(**STRONG), clock=lambda: "T")
    for crit in engine.kb.rubric.criteria:
        assert 0 <= result.scores[crit.id] <= crit.max_score


def test_missing_facts_never_satisfy_positive_conditions(engine: InferenceEngine) -> None:
    # Only a thesis is provided; everything else is unknown.
    result = engine.run(make_facts(thesis_present=True), clock=lambda: "T")
    # No level-4 content rule can match without the argument facts.
    assert result.levels.get("content") in (None, 1, 2, 3)


def test_flag_rule_always_fires_even_with_conflicts(engine: InferenceEngine) -> None:
    facts = make_facts(**{**STRONG, "injection_detected": True})
    result = engine.run(facts, clock=lambda: "T")
    assert "possible-prompt-injection" in result.flags
    # The injection flag must not change the mark.
    assert sum(result.scores.values()) == 40
