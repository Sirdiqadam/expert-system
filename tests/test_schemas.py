"""Tests for the Pydantic schemas (MVP plan Section 5)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ees.schemas import Condition, Fact, Rubric, RubricLevel, Rule, Span


def test_fact_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Fact(name="x", value=1, source="llm", confidence=1.5)
    f = Fact(name="x", value=1, source="llm", confidence=0.4, evidence=[Span(start=0, end=1, quote="a")])
    assert f.confidence == 0.4
    assert f.evidence[0].quote == "a"


def test_condition_leaf_requires_op() -> None:
    with pytest.raises(ValidationError):
        Condition(fact="thesis_present", value=True)


def test_condition_group_only_one_shape() -> None:
    leaf = Condition(fact="a", op="==", value=1)
    with pytest.raises(ValidationError):
        Condition(all=[leaf], any=[leaf])
    grp = Condition(all=[leaf, Condition(fact="b", op=">=", value=2)])
    assert grp.leaf_count() == 2
    assert set(grp.referenced_facts()) == {"a", "b"}


def test_condition_not_alias() -> None:
    c = Condition.model_validate({"not": {"fact": "x", "op": "==", "value": True}})
    assert c.not_ is not None
    assert c.not_.fact == "x"


def test_rule_specificity_and_alias_if() -> None:
    rule = Rule.model_validate(
        {
            "id": "R1",
            "if": {
                "all": [
                    {"fact": "a", "op": "==", "value": True},
                    {"fact": "b", "op": ">=", "value": 3},
                ]
            },
            "then": [{"action": "assert", "fact": "c", "value": 1}],
        }
    )
    assert rule.specificity == 2
    assert rule.then[0].action == "assert"


def test_rubric_representative_score() -> None:
    lvl = RubricLevel(level=4, range=[9, 10], descriptor="x")
    assert lvl.representative_score == 10
    lvl2 = RubricLevel(level=2, range=[4, 6], descriptor="y")
    assert lvl2.representative_score == 5


def test_rubric_level_lookup() -> None:
    rubric = Rubric(
        rubric_id="r",
        max_total=10,
        criteria=[
            {
                "id": "content",
                "name": "Content",
                "max_score": 10,
                "levels": [{"level": 4, "range": [9, 10], "descriptor": "d"}],
            }
        ],
    )
    crit = rubric.criterion("content")
    assert crit is not None
    assert crit.level(4) is not None
    assert crit.level(3) is None
