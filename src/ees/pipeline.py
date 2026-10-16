"""Pipeline orchestration (MVP plan component: ties it all together, M5-M7).

``mark`` runs one essay through the classic expert-system stages and returns a
full :class:`MarkingResult` for the UI, the explainer and the evaluation harness.
"""

from __future__ import annotations

import time

from . import baselines, explainer, ocr_llm
from .config import AppConfig
from .extractor_llm import extract
from .features_nlp import extract_features
from .grading import grade_for
from .inference_engine import InferenceEngine
from .kb_loader import KnowledgeBase
from .llm_gateway import LLMGateway, LLMGatewayError
from .logging_setup import get_logger
from .schemas import (
    ContentPoint,
    CriterionScore,
    Fact,
    MarkingResult,
    PageTranscript,
    StageTiming,
)
from .validator import validate

log = get_logger("ees.pipeline")

MODES = {"rule_only", "llm_only", "hybrid"}


class _Timer:
    def __init__(self) -> None:
        self.timings: list[StageTiming] = []
        self._t0 = time.perf_counter()

    def mark(self, stage: str) -> None:
        now = time.perf_counter()
        self.timings.append(StageTiming(stage=stage, seconds=round(now - self._t0, 4)))
        self._t0 = now


def _resolve_prompt(
    kb: KnowledgeBase, prompt_id: str | None, prompt_text: str
) -> tuple[str, list[ContentPoint]]:
    if prompt_id:
        prompt = kb.prompt(prompt_id)
        if prompt is None:
            raise ValueError(f"unknown prompt id '{prompt_id}'")
        prompt_text = prompt.text
        content_points = [c for c in kb.content_points if c.prompt_id == prompt_id]
        return prompt_text, content_points
    return prompt_text, []


def mark(
    essay: str,
    kb: KnowledgeBase,
    config: AppConfig,
    gateway: LLMGateway,
    mode: str | None = None,
    prompt_id: str | None = None,
    prompt_text: str = "",
    explain_with_llm: bool = False,
    compute_holistic: bool = True,
) -> MarkingResult:
    """Mark a single essay and return the complete trace."""

    mode = mode or config.mode
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}' (expected one of {sorted(MODES)})")

    timer = _Timer()
    prompt_text, content_points = _resolve_prompt(kb, prompt_id, prompt_text)
    key_point_texts = [c.description for c in content_points]

    # --- Stage 1: deterministic NLP features ------------------------------ #
    nlp_facts = extract_features(essay, prompt_text, key_point_texts)
    timer.mark("features_nlp")

    # --- Stage 2: LLM structured extraction (perception) ------------------ #
    llm_facts: list[Fact] = []
    extraction_dict: dict | None = None
    provider = gateway.provider
    tokens = 0
    if mode in {"hybrid", "llm_only"}:
        result = extract(essay, gateway, prompt_text, content_points)
        llm_facts = result.facts
        extraction_dict = result.extraction.model_dump()
        provider = result.provider
        tokens += result.tokens
        if result.warnings:
            log.info("extraction warnings: %s", result.warnings)
    timer.mark("extraction_llm")

    # --- Stage 3/4/5: working memory + inference engine ------------------- #
    facts = list(nlp_facts)
    if mode == "hybrid":
        facts = facts + llm_facts

    scores: dict[str, float] = {}
    levels: dict[str, int] = {}
    flags: list[str] = []
    firing_log = []
    penalties: dict[str, float] = {}

    if mode == "llm_only":
        baseline = baselines.llm_only(essay, gateway, kb, prompt_text, content_points)
        scores = dict(baseline.scores)
        tokens += baseline.tokens
    else:
        engine = InferenceEngine(kb)
        engine_result = engine.run(facts)
        facts = engine_result.facts
        scores = engine_result.scores
        levels = engine_result.levels
        flags = list(engine_result.flags)
        firing_log = engine_result.firing_log
        penalties = engine_result.penalties
    timer.mark("inference_engine")

    total = round(sum(scores.values()), 4)
    max_total = kb.rubric.max_total
    grade = grade_for(total, max_total)

    # --- independent holistic estimate (for the disagreement check) ------- #
    llm_total: float | None = None
    if compute_holistic and mode != "llm_only":
        try:
            holistic = baselines.llm_only(essay, gateway, kb, prompt_text, content_points)
        except LLMGatewayError as exc:
            log.warning("holistic estimate failed: %s", exc)
            flags.append("holistic-estimate-error")
        else:
            llm_total = holistic.total
            tokens += holistic.tokens
    disagreement = round(llm_total - total, 4) if llm_total is not None else None

    # --- Stage 6: validation / review routing ----------------------------- #
    review_flags = validate(
        kb=kb,
        config=config,
        facts=facts,
        firing_log=firing_log,
        scores=scores,
        levels=levels,
        total=total,
        llm_total=llm_total,
    )
    flags = _dedupe(flags + review_flags)
    timer.mark("validation")

    # --- Stage 7: explanation facility ------------------------------------ #
    criterion_scores = _criterion_scores(kb, scores, levels, firing_log)
    proof_tree = explainer.build_proof_tree(kb, facts, firing_log, scores, levels)
    templates = explainer.template_explanations(kb, facts, firing_log)
    counterfacts = explainer.counterfactuals(kb, facts, levels)
    if mode == "llm_only":
        feedback = "LLM-only baseline (no rule trace available)."
        explanations: list[str] = []
    else:
        feedback = "\n".join(templates)
        explanations = templates
        if explain_with_llm:
            try:
                feedback, flags = _llm_rewrite(
                    gateway, proof_tree, scores, counterfacts, firing_log, feedback, flags
                )
            except LLMGatewayError as exc:
                log.warning("LLM explanation rewrite failed: %s", exc)
                flags = _dedupe(flags + ["explanation-llm-error"])
    timer.mark("explanation")

    return MarkingResult(
        mode=mode,
        rubric_id=kb.rubric.rubric_id,
        facts=facts,
        firing_log=firing_log,
        criterion_scores=criterion_scores,
        total=total,
        max_total=max_total,
        grade=grade,
        flags=flags,
        explanations=explanations,
        feedback=feedback,
        proof_tree=proof_tree,
        counterfactuals=counterfacts,
        timings=timer.timings,
        extraction=extraction_dict,
        disagreement=disagreement,
        tokens=tokens,
    )


def mark_pages(
    pages,
    kb: KnowledgeBase,
    config: AppConfig,
    gateway: LLMGateway,
    *,
    vision_gateway: LLMGateway | None = None,
    mode: str | None = None,
    prompt_id: str | None = None,
    prompt_text: str = "",
    explain_with_llm: bool = False,
    compute_holistic: bool = True,
) -> MarkingResult:
    """OCR a booklet's pages with the vision model, then mark the transcript.

    The transcript is treated as the essay: NLP features and LLM fact extraction
    run on it exactly as for a typed essay, so every evidence quote is verified
    against the recognised text and the mark comes only from the rule engine.
    """

    vision = vision_gateway or ocr_llm.vision_gateway_for(config, gateway)
    transcript = ocr_llm.transcribe_pages(
        pages, vision, min_embedded_chars=config.ocr.min_embedded_chars
    )
    if not transcript.text.strip():
        log.warning("booklet produced an empty transcript")

    result = mark(
        transcript.text,
        kb,
        config,
        gateway,
        mode=mode,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        explain_with_llm=explain_with_llm,
        compute_holistic=compute_holistic,
    )
    result.input_type = "scan"
    result.ocr_text = transcript.text
    result.vision_model = vision.model
    result.pages = [
        PageTranscript(
            index=p.index,
            text=p.text,
            source=p.source,
            provider=p.provider,
            tokens=p.tokens,
            error=p.error,
        )
        for p in transcript.pages
    ]
    result.tokens += transcript.tokens
    scan_flags: list[str] = []
    if any(p.source == "error" for p in transcript.pages):
        scan_flags.append("ocr-page-error")
    if not transcript.text.strip():
        scan_flags.append("ocr-empty")
    result.flags = _dedupe(result.flags + scan_flags)
    return result


def mark_document(
    data: bytes,
    filename: str,
    kb: KnowledgeBase,
    config: AppConfig,
    gateway: LLMGateway,
    *,
    mode: str | None = None,
    prompt_id: str | None = None,
    prompt_text: str = "",
    explain_with_llm: bool = False,
    compute_holistic: bool = True,
    vision_gateway: LLMGateway | None = None,
) -> MarkingResult:
    """Convenience entry point: ingest an uploaded PDF/image, then mark it."""

    from .document_ingest import load_pages

    pages = load_pages(
        data,
        filename,
        dpi=config.ocr.dpi,
        max_pages=config.ocr.max_pages,
    )
    return mark_pages(
        pages,
        kb,
        config,
        gateway,
        vision_gateway=vision_gateway,
        mode=mode,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        explain_with_llm=explain_with_llm,
        compute_holistic=compute_holistic,
    )


def _criterion_scores(
    kb: KnowledgeBase, scores: dict[str, float], levels: dict[str, int], firing_log
) -> list[CriterionScore]:
    out: list[CriterionScore] = []
    for crit in kb.rubric.criteria:
        fired = [
            e.rule_id
            for e in firing_log
            if e.criterion == crit.id and any(
                a in {"score", "penalty", "assert"} for a in _actions_of(kb, e.rule_id)
            )
        ]
        out.append(
            CriterionScore(
                criterion=crit.id,
                name=crit.name,
                level=levels.get(crit.id),
                score=float(scores.get(crit.id, 0.0)),
                max_score=float(crit.max_score),
                fired_rules=fired,
            )
        )
    return out


def _actions_of(kb: KnowledgeBase, rule_id: str) -> set[str]:
    rule = kb.rule(rule_id)
    return {a.action for a in rule.then} if rule else set()


def _llm_rewrite(
    gateway: LLMGateway,
    proof_tree,
    scores: dict[str, float],
    counterfacts: list[str],
    firing_log,
    deterministic: str,
    flags: list[str],
) -> tuple[str, list[str]]:
    user = explainer.build_rewrite_user(proof_tree, scores, counterfacts)
    resp = gateway.complete(
        task="explanation",
        system=explainer.REWRITE_SYSTEM,
        user=user,
    )
    ok, unknown = explainer.verify_feedback(resp.text, firing_log)
    if ok:
        return resp.text, flags
    log.warning("LLM feedback cited unknown rules: %s", unknown)
    return deterministic, _dedupe(flags + ["explanation-citation-error"])


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
