"""Tests for LLM structured extraction and quote verification (M4 acceptance).

Quote/schema/fact-conversion tests are deterministic and use no LLM. The live
extraction test uses the configured provider and is skipped without a key.
"""

from __future__ import annotations

from ees.extractor_llm import (
    drop_unverifiable,
    extract,
    extraction_to_facts,
    find_span,
    quote_violations,
)
from ees.schemas import ArgumentInfo, CoherenceInfo, LLMExtraction, ThesisInfo

ESSAY = (
    "School uniforms should be abolished because they are expensive for many "
    "families. For example, uniforms can cost over one hundred dollars per child. "
    "Furthermore, they limit students' self-expression. However, some may argue "
    "that uniforms reduce inequality. Nevertheless, the cost outweighs the "
    "benefits, so schools should not require them."
)

POINT_DESC = ["Mentions the cost to families of buying uniforms."]


def test_find_span_verbatim_only() -> None:
    assert find_span("expensive for many families", ESSAY) is not None
    assert find_span("this text is not present", ESSAY) is None


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


def test_extraction_to_facts_never_yields_a_score() -> None:
    # The extraction contract has no score field, and fact conversion must never
    # introduce one: the inference engine alone sets scores.
    ex = LLMExtraction(
        thesis=ThesisInfo(present=True, quote="School uniforms should be abolished"),
        arguments=[
            ArgumentInfo(
                summary="cost",
                quote="they are expensive for many families",
                has_evidence=True,
                evidence_quote="uniforms can cost over one hundred dollars per child",
            )
        ],
        coherence=CoherenceInfo(rating=4, justification_quote="Furthermore, they limit"),
    )
    facts = extraction_to_facts(ex, ESSAY)
    for fact in facts:
        assert "score" not in fact.name
        assert not fact.name.endswith("_level")


def test_extract_from_live_provider_has_verified_spans(live_gateway) -> None:
    result = extract(ESSAY, live_gateway, "Abolish uniforms?", [])
    names = {f.name for f in result.facts}
    for expected in ("thesis_present", "num_supporting_arguments", "counterargument_rebutted"):
        assert expected in names
    assert result.provider == live_gateway.provider
    for fact in result.facts:
        assert fact.source == "llm"
    # A present thesis must carry a quote verified verbatim against the essay.
    thesis = next(f for f in result.facts if f.name == "thesis_present")
    if thesis.value:
        assert thesis.evidence and thesis.evidence[0].quote in ESSAY
