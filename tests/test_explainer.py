"""Tests for the explanation facility (M6 acceptance)."""

from __future__ import annotations

import pytest

from ees.config import PROJECT_ROOT, load_config
from ees.data_io import load_sample_essays
from ees.explainer import (
    build_proof_tree,
    counterfactuals,
    template_explanations,
    verify_feedback,
    why_not_level,
)
from ees.kb_loader import load_knowledge
from ees.llm_gateway import LLMGateway
from ees.pipeline import mark
from ees.schemas import FiringLogEntry


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    config = load_config()
    kb = load_knowledge(PROJECT_ROOT / "kb")
    gateway = LLMGateway(provider="mock", cache_dir=tmp_path_factory.mktemp("cache"))
    essays = load_sample_essays(config)
    return config, kb, gateway, essays


def test_proof_tree_has_one_root_per_criterion(ctx) -> None:
    config, kb, gateway, essays = ctx
    e = next(x for x in essays if x.essay_id == "essay_01")
    result = mark(e.text, kb, config, gateway, mode="hybrid", prompt_id=e.prompt_id)
    roots = build_proof_tree(kb, result.facts, result.firing_log, {}, {})
    assert len(roots) == 4
    assert all(r.kind == "score" for r in roots)
    # each score root has at least one rule child
    assert all(r.children for r in roots)


def test_template_explanations_cite_rule_ids(ctx) -> None:
    config, kb, gateway, essays = ctx
    e = next(x for x in essays if x.essay_id == "essay_01")
    result = mark(e.text, kb, config, gateway, mode="hybrid", prompt_id=e.prompt_id)
    lines = template_explanations(kb, result.facts, result.firing_log)
    assert lines
    assert all(line.startswith("[R-") for line in lines)


def test_why_not_level_and_counterfactuals(ctx) -> None:
    config, kb, gateway, essays = ctx
    weak = next(x for x in essays if x.essay_id == "essay_09")
    result = mark(weak.text, kb, config, gateway, mode="hybrid", prompt_id=weak.prompt_id)
    msg = why_not_level(kb, "content", 4, result.facts)
    assert "level 4" in msg
    tips = counterfactuals(kb, result.facts, {c.id: 1 for c in kb.rubric.criteria})
    assert len(tips) == 4


def test_verify_feedback_flags_unknown_rule_ids() -> None:
    log = [FiringLogEntry(step=1, rule_id="R-CONT-01")]
    ok, unknown = verify_feedback("Good work [R-CONT-01]. Also [R-FAKE-99].", log)
    assert ok is False
    assert unknown == ["R-FAKE-99"]
    ok2, unknown2 = verify_feedback("Good work [R-CONT-01].", log)
    assert ok2 is True and unknown2 == []


def test_explanations_never_cite_a_fact_absent_from_memory(ctx) -> None:
    config, kb, gateway, essays = ctx
    e = next(x for x in essays if x.essay_id == "essay_03")
    result = mark(e.text, kb, config, gateway, mode="hybrid", prompt_id=e.prompt_id)
    present = {f.name for f in result.facts}
    for entry in result.firing_log:
        for name in entry.matched_facts:
            assert name in present
