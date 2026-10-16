"""Baseline marking modes: ``rule_only`` and ``llm_only`` (M5).

Used for the three-way ablation. ``rule_only`` simply restricts the fact layer to
deterministic NLP facts (handled by the pipeline). ``llm_only`` asks the LLM for
rubric scores directly and is never used to produce the final mark in hybrid mode.
There is no offline fallback: a missing key or invalid JSON is an error.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .kb_loader import KnowledgeBase
from .llm_gateway import LLMGateway, LLMGatewayError
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
    provider: str = ""
    tokens: int = 0


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
    resp = gateway.complete(task="llm_only", system=LLM_ONLY_SYSTEM, user=user)
    try:
        scores, justification = _parse_scores(resp.text, kb)
    except Exception as exc:  # pragma: no cover - defensive
        raise LLMGatewayError(f"llm_only returned unparseable JSON: {exc}") from exc
    return BaselineResult(
        scores=scores,
        total=round(sum(scores.values()), 4),
        justification=justification,
        provider=resp.provider,
        tokens=resp.tokens,
    )
