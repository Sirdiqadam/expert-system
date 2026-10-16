"""Tests for vision OCR transcription and the scanned-booklet marking path."""

from __future__ import annotations

import io
from pathlib import Path

from ees.config import PROJECT_ROOT, load_config
from ees.document_ingest import PageImage
from ees.kb_loader import load_knowledge
from ees.llm_gateway import LLMGateway, LLMGatewayError, LLMResponse
from ees.ocr_llm import OCR_INSTRUCTION, OCR_SYSTEM, transcribe_pages
from ees.pipeline import mark_document, mark_pages

ESSAY = (
    "School uniforms should be abolished. They are expensive for families and "
    "they stop students from expressing themselves. For example, a blazer can "
    "cost more than seventy dollars. Although some argue uniforms reduce "
    "inequality, a gap between new and hand-me-down uniforms persists. In "
    "conclusion, schools should let students wear their own clothes because it "
    "is cheaper and fairer for everyone."
)


class StubVision:
    """A duck-typed stand-in for a vision gateway (no network)."""

    def __init__(self, text: str = ESSAY, provider: str = "nvidia", error: str | None = None):
        self.provider = provider
        self.model = "stub-vision"
        self._text = text
        self._error = error
        self.calls = 0

    def complete(
        self, *, task, system, user, mock=None, images=None, max_tokens=4096, timeout=180
    ) -> LLMResponse:
        self.calls += 1
        if self._error:
            raise LLMGatewayError(self._error)
        return LLMResponse(self._text, provider=self.provider, from_cache=False, tokens=7)


def _page(idx: int = 0, embedded: str = "", png: bytes = b"fake-png") -> PageImage:
    return PageImage(idx, png, 10, 10, embedded)


def _png_booklet() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (300, 400), "white").save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
def test_transcribe_uses_vision_model() -> None:
    t = transcribe_pages([_page()], StubVision())
    assert t.text == ESSAY
    assert t.pages[0].source == "vision"
    assert t.tokens == 7


def test_embedded_text_used_for_digital_pdf() -> None:
    stub = StubVision()
    t = transcribe_pages([_page(embedded="A" * 60)], stub)
    assert t.pages[0].source == "embedded"
    assert stub.calls == 0  # no vision call was needed


def test_ocr_error_is_reported_and_flagged() -> None:
    t = transcribe_pages([_page()], StubVision(error="boom"))
    assert t.pages[0].source == "error"
    assert t.text == ""
    assert t.warnings


def test_multipage_transcript_is_joined() -> None:
    t = transcribe_pages([_page(0), _page(1)], StubVision(text="page body"))
    assert t.text == "page body\n\npage body"
    assert len(t.pages) == 2


def test_cached_vision_replays_offline(tmp_path: Path) -> None:
    # A previous live transcription is stored under the vision model's key...
    live = LLMGateway(
        provider="nvidia", model="stub-vision", cache_dir=tmp_path, api_key="k"
    )
    key = live._key("ocr", OCR_SYSTEM, OCR_INSTRUCTION, [b"fake-png"])
    (tmp_path / "nvidia").mkdir(parents=True, exist_ok=True)
    (tmp_path / "nvidia" / f"{key}.json").write_text(
        '{"text": "CACHED TRANSCRIPT"}', encoding="utf-8"
    )
    # ...and the offline mock replays it without any network.
    mock = LLMGateway(provider="mock", model="stub-vision", cache_dir=tmp_path)
    t = transcribe_pages([_page()], mock)
    assert t.text == "CACHED TRANSCRIPT"
    assert t.pages[0].source == "cache"


# --------------------------------------------------------------------------- #
# Scanned-booklet marking (offline)
# --------------------------------------------------------------------------- #
def _ctx(tmp_path: Path):
    config = load_config()
    kb = load_knowledge(PROJECT_ROOT / "kb")
    gateway = LLMGateway(provider="mock", cache_dir=tmp_path)
    return config, kb, gateway


def test_mark_pages_marks_the_transcript(tmp_path: Path) -> None:
    config, kb, gateway = _ctx(tmp_path)
    result = mark_pages(
        [_page()],
        kb,
        config,
        gateway,
        vision_gateway=StubVision(),
        mode="hybrid",
        prompt_id="school_uniforms",
    )
    assert result.input_type == "scan"
    assert result.ocr_text == ESSAY
    assert result.vision_model == "stub-vision"
    assert len(result.pages) == 1
    assert result.error is None
    assert 0 < result.total <= result.max_total
    assert result.firing_log  # the rule engine still produced the mark


def test_mark_document_ingests_then_marks(tmp_path: Path) -> None:
    config, kb, gateway = _ctx(tmp_path)
    result = mark_document(
        _png_booklet(),
        "scan.png",
        kb,
        config,
        gateway,
        vision_gateway=StubVision(),
        prompt_id="school_uniforms",
    )
    assert result.input_type == "scan"
    assert result.ocr_text == ESSAY
    assert result.pages and result.pages[0].source == "vision"


def test_mark_document_flags_empty_ocr(tmp_path: Path) -> None:
    config, kb, gateway = _ctx(tmp_path)
    result = mark_document(
        _png_booklet(),
        "scan.png",
        kb,
        config,
        gateway,
        vision_gateway=StubVision(text=""),
        prompt_id="school_uniforms",
    )
    assert "ocr-empty" in result.flags
