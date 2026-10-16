"""Tests for scanned-booklet ingestion (PDF / image -> page images)."""

from __future__ import annotations

import io

import pytest

from ees.document_ingest import IngestionError, load_pages


def _png_bytes(text: str = "Hello") -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (200, 80), "white")
    ImageDraw.Draw(img).text((10, 20), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _pdf_bytes(pages: int = 1) -> bytes:
    from PIL import Image

    imgs = [Image.new("RGB", (300, 400), "white") for _ in range(pages)]
    buf = io.BytesIO()
    imgs[0].save(buf, format="PDF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def _tiff_bytes(pages: int = 3) -> bytes:
    from PIL import Image

    imgs = [Image.new("RGB", (120, 60), "white") for _ in range(pages)]
    buf = io.BytesIO()
    imgs[0].save(buf, format="TIFF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def test_load_single_image() -> None:
    pages = load_pages(_png_bytes(), "booklet.png")
    assert len(pages) == 1
    assert pages[0].png[:4] == b"\x89PNG"
    assert pages[0].width == 200 and pages[0].height == 80


def test_load_pdf_renders_all_pages() -> None:
    pages = load_pages(_pdf_bytes(2), "booklet.pdf", dpi=72)
    assert [p.index for p in pages] == [0, 1]
    assert all(p.png[:4] == b"\x89PNG" for p in pages)


def test_load_multipage_tiff() -> None:
    pages = load_pages(_tiff_bytes(3), "booklet.tiff")
    assert len(pages) == 3


def test_pdf_page_limit_enforced() -> None:
    with pytest.raises(IngestionError):
        load_pages(_pdf_bytes(4), "booklet.pdf", max_pages=2)


def test_unsupported_type_and_empty_raise() -> None:
    with pytest.raises(IngestionError):
        load_pages(b"this is not an image", "notes.docx")
    with pytest.raises(IngestionError):
        load_pages(b"", "booklet.pdf")


def test_pdf_is_sniffed_without_extension() -> None:
    pages = load_pages(_pdf_bytes(1), "booklet.bin")
    assert len(pages) == 1


def test_page_sha_is_content_addressed() -> None:
    pages = load_pages(_png_bytes("A"), "a.png")
    pages2 = load_pages(_png_bytes("A"), "a.png")
    assert pages[0].sha256 == pages2[0].sha256
