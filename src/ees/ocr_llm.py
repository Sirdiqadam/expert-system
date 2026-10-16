"""Vision OCR: transcribe scanned booklet pages into text (M12).

The vision LLM *reads* each page image and returns the text exactly as written.
The transcript then flows through the normal pipeline (NLP features + LLM fact
extraction + rule-based inference), so the mark is still produced solely by the
inference engine and every quote is verified verbatim against the transcript.

A page with an embedded text layer (digital PDF) is used directly; otherwise the
vision LLM reads the page image.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import AppConfig
from .document_ingest import PageImage
from .llm_gateway import LLMGateway, LLMGatewayError
from .logging_setup import get_logger

log = get_logger("ees.ocr")

OCR_SYSTEM = (
    "You are a precise OCR and transcription engine for scanned student exam "
    "booklets. Convert the page image into plain text. Transcribe every word "
    "exactly as written, including the student's spelling and punctuation "
    "mistakes. Preserve paragraph breaks. Do not summarise, translate, correct, "
    "grade, or add any commentary. If a word is genuinely unreadable, write "
    "[illegible]. Output only the transcribed text."
)

OCR_INSTRUCTION = "Transcribe all text on this page exactly as written."

# The exam path uses a header-aware prompt so the handwritten header fields - in
# particular the question number, which drives attribution - are captured. It is
# a separate task label so the two OCR paths are logged distinctly.
OCR_EXAM_SYSTEM = (
    "You are a precise OCR engine for scanned handwritten exam booklets. "
    "Convert the page image into plain text. Transcribe every word exactly as "
    "written, including the student's spelling and punctuation mistakes. "
    "Preserve paragraph breaks. First record the two handwritten header fields, "
    "one per line, in this exact form (omit a line when that field is blank):\n"
    "CANDIDATE: <the candidate number written next to 'Candidate Number'>\n"
    "QUESTION: <the question number written next to the printed word 'Question'>\n"
    "Then transcribe the rest of the page. Do not summarise, translate, correct, "
    "grade, or add any commentary. If a word is genuinely unreadable, write "
    "[illegible]. Output only the transcription."
)
OCR_EXAM_INSTRUCTION = (
    "Transcribe this page exactly as written, and do not miss the handwritten "
    "Question number in the header."
)


@dataclass
class PageTranscript:
    """The transcription of one page plus provenance."""

    index: int
    text: str = ""
    source: str = ""  # vision | embedded | error
    provider: str = ""
    tokens: int = 0
    error: str = ""


@dataclass
class Transcript:
    """The full booklet transcript assembled from its pages."""

    text: str
    pages: list[PageTranscript]
    provider: str = ""
    tokens: int = 0
    warnings: list[str] = field(default_factory=list)


def vision_gateway_for(config: AppConfig, base: LLMGateway) -> LLMGateway:
    """Build a gateway that talks to the configured vision model.

    Reuses the base gateway's provider, key and base URL so the same credentials
    and endpoint serve both text and vision requests.
    """

    return LLMGateway(
        provider=base.provider,
        model=config.ocr.vision_model,
        temperature=base.temperature,
        max_retries=base.max_retries,
        api_key=base.api_key,
        base_url=base.base_url,
    )


def _normalise(text: str) -> str:
    """Tidy OCR output: trim line ends and cap runs of blank lines."""

    lines = [line.rstrip() for line in (text or "").replace("\r\n", "\n").split("\n")]
    out = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", out)


def transcribe_page(
    page: PageImage,
    gateway: LLMGateway,
    *,
    min_embedded_chars: int = 40,
    system: str = OCR_SYSTEM,
    instruction: str = OCR_INSTRUCTION,
    task: str = "ocr",
) -> PageTranscript:
    """Transcribe a single page (embedded text layer, else the vision model)."""

    # A digital PDF page with a decent text layer needs no OCR at all.
    if len(page.embedded_text.strip()) >= min_embedded_chars:
        return PageTranscript(page.index, _normalise(page.embedded_text), "embedded")

    try:
        resp = gateway.complete(
            task=task,
            system=system,
            user=instruction,
            images=[page.png],
            max_tokens=4096,
            timeout=600,
        )
    except LLMGatewayError as exc:
        log.warning("page %d OCR failed: %s", page.index + 1, exc)
        return PageTranscript(page.index, "", "error", error=str(exc))

    return PageTranscript(
        page.index, _normalise(resp.text), "vision", resp.provider, resp.tokens
    )


def transcribe_pages(
    pages: list[PageImage],
    gateway: LLMGateway,
    *,
    min_embedded_chars: int = 40,
    page_separator: str = "\n\n",
    system: str = OCR_SYSTEM,
    instruction: str = OCR_INSTRUCTION,
    task: str = "ocr",
) -> Transcript:
    """Transcribe every page and concatenate them into one booklet transcript."""

    transcripts: list[PageTranscript] = []
    parts: list[str] = []
    warnings: list[str] = []
    providers: set[str] = set()
    tokens = 0
    for page in pages:
        pt = transcribe_page(
            page,
            gateway,
            min_embedded_chars=min_embedded_chars,
            system=system,
            instruction=instruction,
            task=task,
        )
        transcripts.append(pt)
        if pt.text.strip():
            parts.append(pt.text)
        tokens += pt.tokens
        if pt.provider:
            providers.add(pt.provider)
        if pt.source == "error":
            warnings.append(f"page {pt.index + 1}: OCR error ({pt.error})")
        elif not pt.text.strip():
            warnings.append(f"page {pt.index + 1}: no text recognised")

    text = page_separator.join(parts).strip()
    if not text:
        warnings.append("no text recognised in the uploaded booklet")
    provider = ",".join(sorted(providers)) if providers else gateway.provider
    return Transcript(
        text=text,
        pages=transcripts,
        provider=provider,
        tokens=tokens,
        warnings=warnings,
    )
