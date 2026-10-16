"""Tests for LLM structured extraction and quote verification (M4 acceptance)."""

from __future__ import annotations

from pathlib import Path

from ees.extractor_llm import (
    drop_unverifiable,
    extract,
    extraction_to_facts,
    find_span,
    heuristic_extraction,
    quote_violations,
)
from ees.llm_gateway import LLMGateway
from ees.schemas import ArgumentInfo, LLMExtraction, ThesisInfo

ESSAY = (
    "School uniforms should be abolished because they are expensive for many "
    "families. For example, uniforms can cost over one hundred dollars per child. "
    "Furthermore, they limit students' self-expression. However, some may argue "
    "that uniforms reduce inequality. Nevertheless, the cost outweighs the "
    "benefits, so schools should not require them."
)

POINT_DESC = ["Mentions the cost to families of buying uniforms."]


def _gw(tmp_path: Path) -> LLMGateway:
    return LLMGateway(provider="mock", cache_dir=tmp_path / "cache")


def test_find_span_verbatim_only() -> None:
    assert find_span("expensive for many families", ESSAY) is not None
    assert find_span("this text is not present", ESSAY) is None


def test_heuristic_extraction_quotes_are_verbatim() -> None:
    ex = heuristic_extraction(ESSAY, "Abolish uniforms?", [])
    assert ex.thesis.present is True
    assert find_span(ex.thesis.quote, ESSAY) is not None
    assert ex.counterargument.present is True
    assert ex.counterargument.rebutted is True


def test_extract_produces_llm_facts_with_spans(tmp_path: Path) -> None:
    result = extract(ESSAY, _gw(tmp_path), "Abolish uniforms?", [])
    names = {f.name for f in result.facts}
    assert "thesis_present" in names
    assert "num_supporting_arguments" in names
    assert "counterargument_rebutted" in names
    for fact in result.facts:
        assert fact.source == "llm"
    # thesis fact carries a verified evidence span
    thesis = next(f for f in result.facts if f.name == "thesis_present")
    assert thesis.evidence and thesis.evidence[0].quote in ESSAY


def test_quote_violation_detected_and_dropped() -> None:
    bad = LLMExtraction(
        thesis=ThesisInfo(present=True, quote="This quote is fabricated entirely"),
        arguments=[ArgumentInfo(summary="s", quote="Also fabricated")],
    )
    assert set(quote_violations(bad, ESSAY)) == {"thesis.quote", "arguments[0].quote"}
    cleaned, warnings = drop_unverifiable(bad.model_copy(deep=True), ESSAY)
    assert cleaned.thesis.present is False
    assert cleaned.arguments == []
    assert warnings


def test_hybrid_never_accepts_a_score_from_the_llm() -> None:
    # The extraction contract has no score field, and fact conversion must never
    # introduce one: the inference engine alone sets scores.
    ex = heuristic_extraction(ESSAY, "Abolish uniforms?", [])
    facts = extraction_to_facts(ex, ESSAY)
    for fact in facts:
        assert "score" not in fact.name
        assert not fact.name.endswith("_level")


def test_extraction_runs_offline_and_caches(tmp_path: Path) -> None:
    gw = _gw(tmp_path)
    first = extract(ESSAY, gw, "Abolish uniforms?", [])
    assert first.provider == "mock"
    second = extract(ESSAY, gw, "Abolish uniforms?", [])
    assert second.from_cache is True
    assert {f.name: f.value for f in first.facts} == {f.name: f.value for f in second.facts}
