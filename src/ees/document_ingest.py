"""Document ingestion: turn an uploaded booklet into page images.

A "booklet" is either a PDF (a scan or a digital export) or a single image
(PNG/JPEG/TIFF/BMP/WebP). Each page becomes a :class:`PageImage` holding the
rendered PNG bytes (fed to the vision model) and, for digital PDFs, any text
already embedded in the page (used to skip OCR when it is good enough).

Nothing here talks to a provider; it is pure I/O and rendering.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

from .logging_setup import get_logger

log = get_logger("ees.ingest")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = IMAGE_SUFFIXES | PDF_SUFFIXES


class IngestionError(RuntimeError):
    """Raised when an uploaded file cannot be turned into page images."""


@dataclass
class PageImage:
    """One rendered page of a booklet."""

    index: int  # 0-based page number
    png: bytes
    width: int
    height: int
    embedded_text: str = ""

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.png).hexdigest()


def _to_png(image) -> bytes:  # PIL.Image.Image -> bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _load_image(data: bytes, max_pages: int) -> list[PageImage]:
    from PIL import Image, ImageSequence

    try:
        img = Image.open(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - surface a friendly error
        raise IngestionError(f"could not read image: {exc}") from exc

    frames: list[PageImage] = []
    for i, frame in enumerate(ImageSequence.Iterator(img)):
        if i >= max_pages:
            raise IngestionError(
                f"document has more than {max_pages} pages; raise ocr.max_pages to allow it"
            )
        rgb = frame.convert("RGB")
        frames.append(PageImage(i, _to_png(rgb), rgb.width, rgb.height))
    return frames


def _load_pdf(data: bytes, dpi: int, max_pages: int) -> list[PageImage]:
    try:
        import pypdfium2 as pdfium
    except Exception as exc:  # pragma: no cover - optional dependency
        raise IngestionError(
            "PDF support needs the 'pypdfium2' package (`pip install pypdfium2`)"
        ) from exc

    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:  # noqa: BLE001
        raise IngestionError(f"could not open PDF: {exc}") from exc

    n = len(pdf)
    if n == 0:
        raise IngestionError("PDF has no pages")
    if n > max_pages:
        raise IngestionError(
            f"PDF has {n} pages; the limit is {max_pages} (raise ocr.max_pages to allow it)"
        )

    scale = max(1.0, dpi / 72.0)
    pages: list[PageImage] = []
    for i in range(n):
        page = pdf[i]
        try:
            textpage = page.get_textpage()
            try:
                embedded = (textpage.get_text_range() or "").strip()
            finally:
                textpage.close()
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil().convert("RGB")
        finally:
            page.close()
        pages.append(PageImage(i, _to_png(pil), pil.width, pil.height, embedded))
    return pages


def load_pages(
    data: bytes,
    filename: str,
    *,
    dpi: int = 150,
    max_pages: int = 30,
) -> list[PageImage]:
    """Turn *data* (bytes of *filename*) into an ordered list of page images."""

    if not data:
        raise IngestionError("uploaded file is empty")
    suffix = Path(filename or "").suffix.lower()
    if suffix in PDF_SUFFIXES:
        pages = _load_pdf(data, dpi, max_pages)
    elif suffix in IMAGE_SUFFIXES:
        pages = _load_image(data, max_pages)
    else:
        # Try to sniff: PDFs start with the magic bytes %PDF.
        if data[:4] == b"%PDF":
            pages = _load_pdf(data, dpi, max_pages)
        else:
            try:
                pages = _load_image(data, max_pages)
            except IngestionError as exc:
                raise IngestionError(
                    f"unsupported file type '{suffix or 'unknown'}'; "
                    f"expected one of {sorted(SUPPORTED_SUFFIXES)}"
                ) from exc
    log.info("ingested %s -> %d page(s) at %d dpi", filename, len(pages), dpi)
    return pages
