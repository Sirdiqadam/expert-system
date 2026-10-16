"""Tests for the COM 226 exam-marking (point-based) capability.

These cover the new ``add`` action and point-based rubric criteria in the loader
and inference engine, the question splitter, the offline keyword matcher and the
best-N aggregation, plus an end-to-end regression on the committed sample script.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ees.config import PROJECT_ROOT, load_config
from ees.exam import (
    keyword_covers,
    load_exam_paper,
    mark_exam,
    segment_by_pages,
    split_answers,
)
from ees.inference_engine import InferenceEngine
from ees.kb_loader import KnowledgeError, load_knowledge
from ees.llm_gateway import LLMGateway
from ees.schemas import ExamPaper, Fact

EXAM_DIR = PROJECT_ROOT / "kb" / "exams" / "com226"
SAMPLE = PROJECT_ROOT / "data" / "exam" / "com226" / "sample_student_script.txt"

RUBRIC_SUM = """\
rubric_id: mini
max_total: 10
aggregate: sum
criteria:
  - {id: q1, name: "Q1", max_score: 10, mode: points}
"""

RUBRIC_BEST1 = """\
rubric_id: mini
max_total: 10
aggregate: best_n
best_n: 1
criteria:
  - {id: q1, name: "Q1", max_score: 10, mode: points}
  - {id: q2, name: "Q2", max_score: 10, mode: points}
"""

POLICIES = "late_penalty_enabled: false\nmin_words: 0\nmax_words: null\nborderline_margin: 2\n"
EMPTY_POINTS = "prompts: []\ncontent_points: []\n"


def _write_kb(tmp_path: Path, rubric: str, rules: str) -> Path:
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "rubric.yaml").write_text(rubric, encoding="utf-8")
    (kb_dir / "rules.yaml").write_text(rules, encoding="utf-8")
    (kb_dir / "content_points.yaml").write_text(EMPTY_POINTS, encoding="utf-8")
    (kb_dir / "policies.yaml").write_text(POLICIES, encoding="utf-8")
    return kb_dir


@pytest.fixture()
def paper() -> ExamPaper:
    return load_exam_paper(EXAM_DIR / "questions.yaml")


@pytest.fixture()
def kb():
    return load_knowledge(EXAM_DIR)


@pytest.fixture()
def config():
    return load_config()


def _gateway(tmp_path: Path) -> LLMGateway:
    return LLMGateway(provider="mock", model="mock", cache_dir=tmp_path / "cache")


# --------------------------------------------------------------------------- #
# Paper + KB structure
# --------------------------------------------------------------------------- #
def test_paper_structure(paper: ExamPaper) -> None:
    assert paper.exam_id == "com226"
    assert paper.answer_count == 5
    assert paper.total_marks == 100
    assert len(paper.questions) == 7
    for q in paper.questions:
        assert q.available_marks() == q.max_marks == 20


def test_kb_uses_point_criteria(kb) -> None:
    assert kb.rubric.aggregate == "best_n"
    assert kb.rubric.best_n == 5
    assert kb.rubric.max_total == 100
    assert all(c.mode == "points" for c in kb.rubric.criteria)
    total_points = sum(len(q.all_points()) for q in load_exam_paper(EXAM_DIR / "questions.yaml").questions)
    assert len(kb.rules) == total_points
    assert all(r.then[0].action == "add" for r in kb.rules)


def test_score_action_on_point_criterion_is_rejected(tmp_path: Path) -> None:
    bad = (
        "- id: R-BAD\n"
        "  criterion: q1\n"
        "  if: {fact: a, op: '==', value: true}\n"
        "  then: [{action: score, criterion: q1, value: 5}]\n"
    )
    kb_dir = _write_kb(tmp_path, RUBRIC_SUM, bad)
    with pytest.raises(KnowledgeError) as exc:
        load_knowledge(kb_dir)
    assert "add" in str(exc.value)


# --------------------------------------------------------------------------- #
# Splitter + keyword matcher
# --------------------------------------------------------------------------- #
def test_split_answers(paper: ExamPaper) -> None:
    script = "QUESTION ONE\n(a) alpha\n\nQ2: beta answer\n\n7\ngamma\n"
    segments, flags = split_answers(script, paper)
    assert flags == []
    assert set(segments) == {"q1", "q2", "q7"}
    assert "alpha" in segments["q1"]
    assert "beta" in segments["q2"]
    assert "gamma" in segments["q7"]


def test_split_answers_without_headings_flags(paper: ExamPaper) -> None:
    segments, flags = split_answers("just some prose with no headings", paper)
    assert "no-question-headings" in flags
    assert set(segments) == {q.id for q in paper.questions}


def _mini_paper() -> ExamPaper:
    return ExamPaper.model_validate(
        {
            "exam_id": "mini",
            "total_marks": 20,
            "answer_count": 2,
            "questions": [
                {
                    "id": "q1", "number": 1, "max_marks": 10,
                    "parts": [{"id": "q1a", "points": [
                        {"id": "p1", "marks": 2, "keywords": [["alpha"]]}]}],
                },
                {
                    "id": "q2", "number": 2, "max_marks": 10,
                    "parts": [{"id": "q2a", "points": [
                        {"id": "p2", "marks": 2, "keywords": [["beta"]]}]}],
                },
            ],
        }
    )


def test_segment_by_pages_uses_header_number() -> None:
    paper = _mini_paper()
    pages = ["CANDIDATE: 001\nQUESTION: 1\nalpha answer", "QUESTION: 2\nbeta answer"]
    segments, flags = segment_by_pages(pages, paper)
    assert set(segments) == {"q1", "q2"}
    assert "segmented-by-page" in flags
    assert "page-resolved" not in " ".join(flags)


def test_segment_by_pages_concatenates_same_question() -> None:
    paper = _mini_paper()
    pages = ["QUESTION: 1\nalpha one", "QUESTION: 1\nalpha two"]
    segments, _flags = segment_by_pages(pages, paper)
    assert segments["q1"].count("alpha") == 2


def test_segment_by_pages_trusts_unique_header() -> None:
    paper = _mini_paper()
    # A unique header number is trusted even when content keywords do not match.
    segments, flags = segment_by_pages(["QUESTION: 1\nbeta discussion"], paper)
    assert "q1" in segments
    assert "page-resolved" not in " ".join(flags)


def test_segment_by_pages_reassigns_duplicate_header_by_content() -> None:
    paper = _mini_paper()
    # Question 1 cannot start twice: the second page is really a question-2 answer.
    pages = ["QUESTION: 1\nalpha answer", "QUESTION: 1\nbeta answer"]
    segments, flags = segment_by_pages(pages, paper)
    assert "alpha" in segments["q1"]
    assert "beta" in segments["q2"]
    assert any("page-resolved" in f for f in flags)


def test_segment_by_pages_uses_resolver_for_ambiguous_pages() -> None:
    paper = _mini_paper()
    calls: list[str] = []

    def resolver(text: str) -> str:
        calls.append(text)
        return "q2"

    pages = ["QUESTION: 1\nalpha answer", "QUESTION: 1\nalpha again"]
    segments, _flags = segment_by_pages(pages, paper, resolve=resolver)
    assert "q1" in segments and "q2" in segments
    # The resolver is consulted only for the page whose header repeats.
    assert calls == ["QUESTION: 1\nalpha again"]


def test_segment_by_pages_falls_back_to_content() -> None:
    paper = _mini_paper()
    segments, _flags = segment_by_pages(["a lone beta answer"], paper)
    assert "q2" in segments


def test_segment_by_pages_reports_unassigned() -> None:
    paper = _mini_paper()
    segments, flags = segment_by_pages(["nothing relevant here"], paper)
    assert segments == {}
    assert any("unassigned-pages" in f for f in flags)


def test_keyword_covers_requires_every_group() -> None:
    from ees.schemas import ExamPoint

    point = ExamPoint(
        id="p", marks=2, keywords=[["alpha"], ["beta", "gamma"]]
    )
    assert keyword_covers("Alpha and beta here", point)[0] is True
    assert keyword_covers("alpha only", point)[0] is False


# --------------------------------------------------------------------------- #
# add action (engine)
# --------------------------------------------------------------------------- #
def test_add_action_accumulates(tmp_path: Path) -> None:
    rules = (
        "- id: R1\n"
        "  criterion: q1\n"
        "  if: {fact: a, op: '==', value: true}\n"
        "  then: [{action: add, criterion: q1, amount: 3}]\n"
        "  explain: a\n"
        "- id: R2\n"
        "  criterion: q1\n"
        "  if: {fact: b, op: '==', value: true}\n"
        "  then: [{action: add, criterion: q1, amount: 4}]\n"
        "  explain: b\n"
    )
    kb = load_knowledge(_write_kb(tmp_path, RUBRIC_SUM, rules))
    result = InferenceEngine(kb).run(
        [Fact(name="a", value=True, source="nlp"), Fact(name="b", value=True, source="nlp")]
    )
    assert result.scores["q1"] == 7
    assert {entry.rule_id for entry in result.firing_log} == {"R1", "R2"}


# --------------------------------------------------------------------------- #
# End-to-end marking (offline keyword path, deterministic)
# --------------------------------------------------------------------------- #
def test_mark_sample_offline(paper: ExamPaper, kb, config, tmp_path: Path) -> None:
    script = SAMPLE.read_text(encoding="utf-8")
    result = mark_exam(
        script, paper, kb, config, _gateway(tmp_path), mode="rule_only"
    )
    by_id = {q.question_id: q for q in result.questions}

    assert result.answer_count == 5
    assert by_id["q6"].answered is False and by_id["q7"].answered is False
    assert by_id["q1"].raw_marks == 20
    assert by_id["q3"].raw_marks == 18  # overlap function omitted
    assert by_id["q5"].raw_marks == 14  # only four operations, no diagram
    assert result.total == 89
    assert result.grade == "A"
    assert by_id["q1"].counts and by_id["q5"].counts

    # credited points carry a verifiable verbatim quote
    credited = [p for p in by_id["q1"].points if p.covered]
    assert credited
    assert all(p.evidence_ok and p.quote for p in credited)
    assert all(p.method == "keyword" for p in credited)


def test_best_n_aggregation(paper: ExamPaper, tmp_path: Path) -> None:
    rules = (
        "- id: R1\n"
        "  criterion: q1\n"
        "  if: {fact: covered_p1, op: '==', value: true}\n"
        "  then: [{action: add, criterion: q1, amount: 5}]\n"
        "  explain: p1\n"
        "- id: R2\n"
        "  criterion: q2\n"
        "  if: {fact: covered_p2, op: '==', value: true}\n"
        "  then: [{action: add, criterion: q2, amount: 9}]\n"
        "  explain: p2\n"
    )
    kb = load_knowledge(_write_kb(tmp_path, RUBRIC_BEST1, rules))
    mini = ExamPaper.model_validate(
        {
            "exam_id": "mini",
            "total_marks": 10,
            "answer_count": 1,
            "questions": [
                {
                    "id": "q1", "number": 1, "max_marks": 10,
                    "parts": [{"id": "q1a", "points": [
                        {"id": "p1", "marks": 5, "keywords": [["alpha"]]}]}],
                },
                {
                    "id": "q2", "number": 2, "max_marks": 10,
                    "parts": [{"id": "q2a", "points": [
                        {"id": "p2", "marks": 9, "keywords": [["beta"]]}]}],
                },
            ],
        }
    )
    result = mark_exam(
        "",
        mini,
        kb,
        load_config(),
        _gateway(tmp_path),
        mode="rule_only",
        segments={"q1": "alpha is discussed", "q2": "beta is discussed"},
    )
    by_id = {q.question_id: q for q in result.questions}
    assert by_id["q2"].counts is True and by_id["q1"].counts is False
    assert result.total == 9  # only the best question counts
    assert result.answer_count == 2
    assert any("answered-2-of-1" in f for f in result.flags)


def test_unknown_mode_raises(paper: ExamPaper, kb, config, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        mark_exam("x", paper, kb, config, _gateway(tmp_path), mode="nonsense")
