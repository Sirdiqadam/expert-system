"""Pydantic schemas for the essay expert system (MVP plan Section 5).

These models are the contract between every component: the knowledge base,
the fact extractors, working memory, the inference engine, the explainer and
the evaluation harness.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FactSource = Literal["nlp", "llm", "rule"]
ActionKind = Literal["assert", "score", "penalty", "flag", "add"]
Operator = Literal["==", "!=", ">", ">=", "<", "<=", "in", "not_in", "contains"]


# --------------------------------------------------------------------------- #
# Knowledge base models
# --------------------------------------------------------------------------- #
class Span(BaseModel):
    """A character span in the essay, with the exact quoted text."""

    start: int
    end: int
    quote: str


class Fact(BaseModel):
    """A single working-memory item.

    ``source`` records where the fact came from so the UI can badge it and the
    explainer can attribute it. LLM facts carry a confidence in ``[0, 1]``.
    """

    name: str
    value: bool | int | float | str
    source: FactSource
    confidence: float | None = None
    evidence: list[Span] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_confidence(self) -> "Fact":
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be within [0, 1]")
        return self


class Condition(BaseModel):
    """A rule condition: a leaf comparison or a logical group.

    Exactly one shape must be supplied:
    * a leaf: ``{fact, op, value}``
    * a group: ``{all: [...]}``, ``{any: [...]}`` or ``{not: {...}}``
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    fact: str | None = None
    op: Operator | None = None
    value: Any = None
    all: list["Condition"] | None = None
    any: list["Condition"] | None = None
    not_: Condition | None = Field(default=None, alias="not")

    @model_validator(mode="after")
    def _exactly_one(self) -> "Condition":
        group = [self.all is not None, self.any is not None, self.not_ is not None]
        leaf = self.fact is not None
        if sum(group) + int(leaf) != 1:
            raise ValueError(
                "condition must be exactly one of a leaf {fact,op,value} or a "
                "group {all|any|not}"
            )
        if leaf and self.op is None:
            raise ValueError("leaf condition requires an 'op'")
        return self

    def leaf_count(self) -> int:
        """Number of leaf comparisons (used for rule specificity)."""

        if self.fact is not None:
            return 1
        total = 0
        for group in (self.all, self.any):
            if group:
                total += sum(c.leaf_count() for c in group)
        if self.not_ is not None:
            total += self.not_.leaf_count()
        return total

    def referenced_facts(self) -> list[str]:
        """All fact names referenced anywhere in this condition tree."""

        if self.fact is not None:
            return [self.fact]
        names: list[str] = []
        for group in (self.all, self.any):
            if group:
                for c in group:
                    names.extend(c.referenced_facts())
        if self.not_ is not None:
            names.extend(self.not_.referenced_facts())
        return names


class RuleAction(BaseModel):
    """An action performed when a rule fires.

    * ``assert``  -> set ``fact`` to ``value``
    * ``score``   -> set (overwrite) the score for ``criterion`` (explicit
      ``value`` or a rubric ``level``)
    * ``add``     -> add ``amount`` to ``criterion`` (point-based/exam marking)
    * ``penalty`` -> subtract ``amount`` from ``criterion`` bounded by ``cap``
    * ``flag``    -> add a human-review flag with ``reason``
    """

    model_config = ConfigDict(extra="forbid")

    action: ActionKind
    fact: str | None = None
    value: Any = None
    criterion: str | None = None
    level: int | None = None
    amount: float | None = None
    cap: float | None = None
    reason: str = ""


class Rule(BaseModel):
    """An IF-THEN marking rule with an explanation template."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str
    criterion: str | None = None
    priority: int = 0
    description: str = ""
    if_: Condition = Field(alias="if")
    then: list[RuleAction]
    explain: str = ""

    @property
    def specificity(self) -> int:
        """Number of leaf conditions; used to break priority ties."""

        return self.if_.leaf_count()


class RubricLevel(BaseModel):
    """One level of a rubric criterion, with its score range."""

    level: int
    range: list[int]
    descriptor: str

    @model_validator(mode="after")
    def _check_range(self) -> "RubricLevel":
        if len(self.range) != 2 or self.range[0] > self.range[1]:
            raise ValueError("range must be a [lo, hi] pair with lo <= hi")
        return self

    @property
    def midpoint(self) -> float:
        lo, hi = self.range
        return (lo + hi) / 2.0

    @property
    def representative_score(self) -> int:
        """A deterministic score inside the range (rounded midpoint)."""

        return int(round(self.midpoint))


class RubricCriterion(BaseModel):
    """A rubric criterion.

    ``mode="levels"`` (default) is the analytic essay rubric: ordered levels,
    each with a score range. ``mode="points"`` is the exam scheme: no levels, the
    score is the sum of marks added by point rules (``add`` actions).
    """

    id: str
    name: str
    max_score: int
    weight: float = 1.0
    mode: Literal["levels", "points"] = "levels"
    levels: list[RubricLevel] = Field(default_factory=list)

    def level(self, level: int) -> RubricLevel | None:
        for lvl in self.levels:
            if lvl.level == level:
                return lvl
        return None


class Rubric(BaseModel):
    """The full rubric.

    ``aggregate="sum"`` (default) totals every criterion. ``aggregate="best_n"``
    (exam with "answer any N questions") counts only the best ``best_n``
    criterion scores toward ``max_total``.
    """

    rubric_id: str
    max_total: int
    criteria: list[RubricCriterion]
    aggregate: Literal["sum", "best_n"] = "sum"
    best_n: int | None = None

    def criterion(self, cid: str) -> RubricCriterion | None:
        for crit in self.criteria:
            if crit.id == cid:
                return crit
        return None


class Policies(BaseModel):
    """Marking policy knobs (``kb/policies.yaml``)."""

    late_penalty_enabled: bool = False
    late_penalty_per_day: float = 0.0
    min_words: int = 0
    max_words: int | None = None
    borderline_margin: int = 1


class ContentPoint(BaseModel):
    """A key content point expected for a task prompt."""

    id: str
    description: str
    prompt_id: str | None = None


class TaskPrompt(BaseModel):
    """A writing task prompt with its expected key content points."""

    id: str
    text: str
    key_points: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# LLM extraction contract (Section 5.4)
# --------------------------------------------------------------------------- #
class ThesisInfo(BaseModel):
    present: bool = False
    quote: str = ""


class ArgumentInfo(BaseModel):
    summary: str = ""
    quote: str = ""
    has_evidence: bool = False
    evidence_quote: str = ""


class CounterargumentInfo(BaseModel):
    present: bool = False
    rebutted: bool = False
    quote: str = ""


class CoherenceInfo(BaseModel):
    rating: int = 3
    justification_quote: str = ""


class OnTopicInfo(BaseModel):
    rating: int = 3


class GrammarError(BaseModel):
    quote: str = ""
    type: str = ""


class LLMExtraction(BaseModel):
    """The JSON schema the LLM must return (facts, never scores)."""

    thesis: ThesisInfo = Field(default_factory=ThesisInfo)
    arguments: list[ArgumentInfo] = Field(default_factory=list)
    counterargument: CounterargumentInfo = Field(default_factory=CounterargumentInfo)
    coherence: CoherenceInfo = Field(default_factory=CoherenceInfo)
    on_topic: OnTopicInfo = Field(default_factory=OnTopicInfo)
    key_points_covered: list[str] = Field(default_factory=list)
    grammar_error_examples: list[GrammarError] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Runtime / result models
# --------------------------------------------------------------------------- #
class FiringLogEntry(BaseModel):
    """One entry in the inference engine's firing log (Section 5.5)."""

    step: int
    rule_id: str
    matched_facts: list[str] = Field(default_factory=list)
    asserted: dict[str, Any] = Field(default_factory=dict)
    conflicts_resolved_against: list[str] = Field(default_factory=list)
    timestamp: str = ""
    explanation: str = ""
    criterion: str | None = None


class CriterionScore(BaseModel):
    """The final score for one rubric criterion."""

    criterion: str
    name: str = ""
    level: int | None = None
    score: float = 0.0
    max_score: float = 0.0
    fired_rules: list[str] = Field(default_factory=list)


class ProofNode(BaseModel):
    """A node in the proof tree: score <- rules <- facts <- evidence."""

    label: str
    kind: Literal["score", "rule", "fact", "evidence"]
    detail: str = ""
    value: Any = None
    children: list["ProofNode"] = Field(default_factory=list)


class StageTiming(BaseModel):
    """Wall-clock time spent in one pipeline stage."""

    stage: str
    seconds: float


class PageTranscript(BaseModel):
    """The transcription of one booklet page (scanned-input provenance)."""

    index: int
    text: str = ""
    source: str = ""  # vision | cache | embedded | mock | error
    provider: str = ""
    tokens: int = 0
    error: str = ""


class MarkingResult(BaseModel):
    """Everything produced by one marking run, for UI and evaluation."""

    mode: str
    rubric_id: str = ""
    facts: list[Fact] = Field(default_factory=list)
    firing_log: list[FiringLogEntry] = Field(default_factory=list)
    criterion_scores: list[CriterionScore] = Field(default_factory=list)
    total: float = 0.0
    max_total: float = 0.0
    grade: str = ""
    flags: list[str] = Field(default_factory=list)
    explanations: list[str] = Field(default_factory=list)
    feedback: str = ""
    proof_tree: list[ProofNode] = Field(default_factory=list)
    counterfactuals: list[str] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    extraction: dict[str, Any] | None = None
    disagreement: float | None = None
    tokens: int = 0
    # scanned-booklet provenance
    input_type: str = "typed"  # typed | scan
    ocr_text: str = ""
    vision_model: str = ""
    pages: list[PageTranscript] = Field(default_factory=list)
    error: str | None = None

    def fact(self, name: str) -> Fact | None:
        for f in self.facts:
            if f.name == name:
                return f
        return None


# --------------------------------------------------------------------------- #
# Exam scheme (point-based marking, e.g. COM 226)
# --------------------------------------------------------------------------- #
class ExamPoint(BaseModel):
    """One marking point in an exam scheme (worth ``marks`` if covered)."""

    id: str
    marks: float
    answer: str = ""
    # Offline keyword fallback: a point is covered when EVERY group has at least
    # one phrase present in the candidate's answer (a group = list of synonyms).
    keywords: list[list[str]] = Field(default_factory=list)


class ExamPart(BaseModel):
    """A sub-part of a question, e.g. ``(a)``."""

    id: str
    label: str = ""
    text: str = ""
    points: list[ExamPoint] = Field(default_factory=list)


class ExamQuestion(BaseModel):
    """One exam question with its sub-parts and marking points."""

    id: str
    number: int
    title: str = ""
    max_marks: float = 20.0
    parts: list[ExamPart] = Field(default_factory=list)

    def all_points(self) -> list[tuple["ExamPart", "ExamPoint"]]:
        return [(part, pt) for part in self.parts for pt in part.points]

    def available_marks(self) -> float:
        return sum(pt.marks for _, pt in self.all_points())


class ExamPaper(BaseModel):
    """A whole exam paper: ``answer_count`` of ``questions`` are answered."""

    exam_id: str
    title: str = ""
    total_marks: float = 100.0
    answer_count: int = 5
    instruction: str = ""
    questions: list[ExamQuestion] = Field(default_factory=list)

    def question(self, qid: str) -> ExamQuestion | None:
        for q in self.questions:
            if q.id == qid:
                return q
        return None


class PointResult(BaseModel):
    """The outcome for one marking point."""

    point_id: str
    part_id: str = ""
    marks_available: float = 0.0
    marks_awarded: float = 0.0
    covered: bool = False
    quote: str = ""
    evidence_ok: bool = False
    method: str = ""  # keyword | llm | cache


class ExamQuestionResult(BaseModel):
    """Per-question result with the point breakdown."""

    question_id: str
    number: int = 0
    title: str = ""
    max_marks: float = 0.0
    answered: bool = False
    raw_marks: float = 0.0
    counts: bool = False  # included in the best-N total
    points: list[PointResult] = Field(default_factory=list)


class ExamResult(BaseModel):
    """Everything produced by one exam-marking run."""

    exam_id: str
    mode: str = "hybrid"
    total: float = 0.0
    max_total: float = 100.0
    grade: str = ""
    answer_count: int = 0
    best_n: int = 5
    questions: list[ExamQuestionResult] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    firing_log: list[FiringLogEntry] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    timings: list[StageTiming] = Field(default_factory=list)
    tokens: int = 0
    input_type: str = "typed"
    ocr_text: str = ""
    vision_model: str = ""
    pages: list[PageTranscript] = Field(default_factory=list)
    error: str | None = None


Condition.model_rebuild()
ProofNode.model_rebuild()
