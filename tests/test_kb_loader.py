"""Tests for the knowledge base loader and authoring rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from ees.config import PROJECT_ROOT
from ees.kb_loader import KnowledgeError, load_knowledge


@pytest.fixture()
def kb():
    return load_knowledge(PROJECT_ROOT / "kb")


def test_loads_real_kb(kb) -> None:
    assert kb.rubric.rubric_id == "argumentative_v1"
    assert len(kb.rubric.criteria) == 4
    assert len(kb.rules) >= 25
    assert kb.version
    assert {p.id for p in kb.prompts} == {"school_uniforms", "remote_learning"}


def test_every_criterion_can_reach_every_level(kb) -> None:
    for crit in kb.rubric.criteria:
        levels = {lvl.level for lvl in crit.levels}
        reached = set()
        for rule in kb.rules:
            for action in rule.then:
                if action.action == "score" and (action.criterion or rule.criterion) == crit.id:
                    if action.level is not None:
                        reached.add(action.level)
        assert levels.issubset(reached), f"{crit.id} missing {levels - reached}"


def _write_kb(tmp_path: Path, name: str, text: str) -> Path:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir(exist_ok=True)
    # copy the valid files, then override one
    src = PROJECT_ROOT / "kb"
    for f in src.iterdir():
        if not f.is_file():
            continue
        (kb_dir / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    (kb_dir / name).write_text(text, encoding="utf-8")
    return kb_dir


def test_invalid_yaml_reports_file_and_line(tmp_path: Path) -> None:
    kb_dir = _write_kb(tmp_path, "rules.yaml", "- id: R1\n  if: {fact: a, op: '=='\n")
    with pytest.raises(KnowledgeError) as exc:
        load_knowledge(kb_dir)
    assert "rules.yaml" in str(exc.value)


def test_unknown_criterion_is_rejected(tmp_path: Path) -> None:
    bad = (
        "- id: R-BAD\n"
        "  criterion: nonexistent\n"
        "  if: {fact: a, op: '==', value: true}\n"
        "  then: [{action: assert, fact: b, value: 1}]\n"
    )
    kb_dir = _write_kb(tmp_path, "rules.yaml", bad)
    with pytest.raises(KnowledgeError) as exc:
        load_knowledge(kb_dir)
    assert "nonexistent" in str(exc.value)


def test_missing_level_reachable_is_rejected(tmp_path: Path) -> None:
    # remove all rules -> no criterion can reach any level
    kb_dir = _write_kb(tmp_path, "rules.yaml", "[]\n")
    with pytest.raises(KnowledgeError) as exc:
        load_knowledge(kb_dir)
    assert "reaching levels" in str(exc.value)


def test_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeError) as exc:
        load_knowledge(tmp_path / "does-not-exist")
    assert "not found" in str(exc.value)
