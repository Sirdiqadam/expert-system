"""Generate sample scanned-style booklets (PDF) from the sample essays.

Used to demo the scanned-booklet path (upload -> vision OCR -> rules -> mark).
Each essay is rendered as one or more page. Run:

    python tools/make_sample_booklets.py
"""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "sample_essays"
OUT = ROOT / "data" / "sample_booklets"

PAGE_W, PAGE_H = 1000, 1400
MARGIN = 80
LINE_H = 34
FONT_SIZE = 22


def _font() -> ImageFont.ImageFont:
    for name in ("arial.ttf", "calibri.ttf", "times.ttf"):
        try:
            return ImageFont.truetype(name, FONT_SIZE)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _wrap(text: str, width: int = 78) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=width) or [""])
    return lines


def render_pages(text: str, title: str) -> list[Image.Image]:
    font = _font()
    lines = _wrap(text)
    per_page = (PAGE_H - 2 * MARGIN - LINE_H) // LINE_H
    pages: list[Image.Image] = []
    for start in range(0, max(len(lines), 1), per_page):
        chunk = lines[start : start + per_page]
        img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        draw = ImageDraw.Draw(img)
        draw.text((MARGIN, MARGIN // 2), title, fill="#333333", font=font)
        y = MARGIN
        for line in chunk:
            draw.text((MARGIN, y), line, fill="black", font=font)
            y += LINE_H
        pages.append(img)
    return pages


def build(essay_path: Path, out_path: Path, title: str) -> int:
    text = essay_path.read_text(encoding="utf-8").strip()
    pages = render_pages(text, title)
    first, rest = pages[0], pages[1:]
    buf = io.BytesIO()
    first.save(buf, format="PDF", save_all=True, append_images=rest, resolution=150)
    out_path.write_bytes(buf.getvalue())
    return len(pages)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "essay_01.txt": "essay_01_booklet.pdf",
        "essay_04.txt": "essay_04_booklet.pdf",
        "essay_09.txt": "essay_09_booklet.pdf",
    }
    for src_name, out_name in plan.items():
        src = SRC / src_name
        if not src.exists():
            print(f"skip missing {src}")
            continue
        n = build(src, OUT / out_name, f"Student booklet - {src_name}")
        print(f"{out_name}: {n} page(s) -> {OUT / out_name}")


if __name__ == "__main__":
    main()
