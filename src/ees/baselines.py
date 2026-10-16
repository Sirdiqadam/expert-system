"""Baseline marking modes: ``rule_only`` and ``llm_only`` (M5).

Used for the three-way ablation. ``rule_only`` simply restricts the fact layer to
deterministic NLP facts (handled by the pipeline). ``llm_only`` asks the LLM for
rubric scores directly, with an offline heuristic stand-in, and is never used to
produce the final mark in hybrid mode.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .extractor_llm import heuristic_extraction
from .features_nlp import count_connectives, extract_features, split_paragraphs, tokenize
from .kb_loader import KnowledgeBase
from .llm_gateway import LLMGateway
from .logging_setup import get_logger
from .schemas import ContentPoint

log = get_logger("ees.baselines")

LLM_ONLY_SYSTEM = (
    "You are an essay marker using the provided rubric. Return JSON only, giving "
    "a score for each criterion and a short justification. This mode is a "
    "research baseline only."
)


@dataclass
class BaselineResult:
    """Per-criterion scores from a baseline mode."""

    scores: dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    justification: str = ""
    provider: str = "mock"
    tokens: int = 0


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _heuristic_llm_only(
    essay: str, kb: KnowledgeBase, prompt_text: str, content_points: list[ContentPoint]
) -> BaselineResult:
    """A black-box estimate from surface signals (offline stand-in)."""

    feat = {f.name: f.value for f in extract_features(essay, prompt_text, [p.description for p in content_points])}
    ex = heuristic_extraction(essay, prompt_text, content_points)
    spelling = float(feat.get("spelling_error_rate", 0.0))
    conn = count_connectives(essay)
    paras = len(split_paragraphs(essay))
    words = max(1, len(tokenize(essay)))

    content_q = _clamp(
        (0.3 if ex.thesis.present else 0.0)
        + min(len(ex.arguments), 4) / 4 * 0.5
        + (0.2 if ex.counterargument.rebutted else 0.0)
    )
    organization_q = _clamp(min(paras, 4) / 4 * 0.5 + min(conn, 10) / 10 * 0.5)
    language_q = _clamp(1 - spelling * 5) * 0.7 + _clamp(conn / 10) * 0.3
    mechanics_q = _clamp(1 - spelling * 8)

    quality = {
        "content": content_q,
        "organization": organization_q,
        "language": language_q,
        "mechanics": mechanics_q,
    }
    scores: dict[str, float] = {}
    for crit in kb.rubric.criteria:
        q = quality.get(crit.id, 0.5)
        scores[crit.id] = float(round(q * crit.max_score))
    return BaselineResult(
        scores=scores,
        total=round(sum(scores.values()), 4),
        justification="Holistic heuristic estimate (offline stand-in).",
        provider="mock",
    )


def _parse_scores(text: str, kb: KnowledgeBase) -> tuple[dict[str, float], str]:
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    data = json.loads(cleaned[start : end + 1])
    scores: dict[str, float] = {}
    justification = str(data.get("justification", ""))
    for crit in kb.rubric.criteria:
        entry = data.get(crit.id) or data.get("scores", {}).get(crit.id)
        if isinstance(entry, dict):
            scores[crit.id] = float(entry.get("score", 0))
        elif isinstance(entry, (int, float)):
            scores[crit.id] = float(entry)
    return scores, justification


def llm_only(
    essay: str,
    gateway: LLMGateway,
    kb: KnowledgeBase,
    prompt_text: str = "",
    content_points: list[ContentPoint] | None = None,
) -> BaselineResult:
    """Score the essay directly with the LLM (baseline, never used in hybrid)."""

    content_points = content_points or []
    rubric_text = json.dumps(kb.rubric.model_dump(), indent=2)
    user = (
        f"RUBRIC:\n{rubric_text}\n\nTASK PROMPT:\n{prompt_text}\n\n"
        f"ESSAY:\n<<<ESSAY\n{essay}\nESSAY>>>\n\n"
        "Return JSON: {\"content\": {\"score\": n, \"justification\": s}, ... }"
    )
    mock = lambda: json.dumps(  # noqa: E731
        {
            "scores": _heuristic_llm_only(essay, kb, prompt_text, content_points).scores,
            "justification": "Holistic heuristic estimate (offline stand-in).",
        }
    )
    resp = gateway.complete(task="llm_only", system=LLM_ONLY_SYSTEM, user=user, mock=mock)
    try:
        scores, justification = _parse_scores(resp.text, kb)
    except Exception:  # pragma: no cover - defensive
        log.warning("llm_only returned unparseable JSON; using heuristic estimate")
        fallback = _heuristic_llm_only(essay, kb, prompt_text, content_points)
        return fallback
    result = BaselineResult(
        scores=scores,
        total=round(sum(scores.values()), 4),
        justification=justification,
        provider=resp.provider,
        tokens=resp.tokens,
    )
    return result
