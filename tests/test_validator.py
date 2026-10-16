"""Tests for validation and review routing (M7 acceptance)."""

from __future__ import annotations

import pytest

from ees.config import PROJECT_ROOT, load_config
from ees.kb_loader import load_knowledge
from ees.schemas import Fact, FiringLogEntry
from ees.validator import validate


@pytest.fixture(scope="module")
def kb():
    return load_knowledge(PROJECT_ROOT / "kb")


@pytest.fixture(scope="module")
def config():
    return load_config()


def run(kb, config, **kwargs):
    defaults = dict(
        kb=kb,
        config=config,
        facts=[],
        firing_log=[],
        scores={},
        levels={},
        total=0.0,
        llm_total=None,
    )
    defaults.update(kwargs)
    return validate(**defaults)


def test_score_out_of_bounds(kb, config) -> None:
    flags = run(kb, config, scores={"content": 99}, total=99)
    assert "score-out-of-bounds:content" in flags


def test_llm_rule_disagreement(kb, config) -> None:
    flags = run(kb, config, scores={"content": 5}, total=5, llm_total=30)
    assert "llm-rule-disagreement" in flags


def test_borderline_mark(kb, config) -> None:
    flags = run(kb, config, scores={"content": 8}, total=32)  # 32 is a grade boundary
    assert "borderline-mark" in flags


def test_short_essay_flag(kb, config) -> None:
    facts = [Fact(name="word_count", value=50, source="nlp")]
    flags = run(kb, config, facts=facts, total=10)
    assert "essay-too-short" in flags


def test_off_topic_flag(kb, config) -> None:
    facts = [Fact(name="on_topic_rating", value=1, source="llm", confidence=0.9)]
    flags = run(kb, config, facts=facts, total=10)
    assert "off-topic" in flags


def test_prompt_injection_flag(kb, config) -> None:
    facts = [Fact(name="injection_detected", value=True, source="nlp")]
    flags = run(kb, config, facts=facts, total=10)
    assert "possible-prompt-injection" in flags


def test_low_confidence_evidence_flag(kb, config) -> None:
    facts = [Fact(name="thesis_present", value=True, source="llm", confidence=0.1)]
    log = [FiringLogEntry(step=1, rule_id="R-CONT-01", matched_facts=["thesis_present"])]
    flags = run(kb, config, facts=facts, firing_log=log, scores={"content": 8}, total=8)
    assert "low-confidence-evidence" in flags


def test_clean_case_has_no_flags(kb, config) -> None:
    facts = [
        Fact(name="word_count", value=200, source="nlp"),
        Fact(name="on_topic_rating", value=4, source="llm", confidence=0.9),
        Fact(name="injection_detected", value=False, source="nlp"),
    ]
    flags = run(kb, config, facts=facts, scores={"content": 7}, total=7)
    assert flags == []
