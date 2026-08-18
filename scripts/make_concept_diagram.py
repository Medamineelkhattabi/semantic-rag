"""Render the 'how it works' concept diagram used in the README and social posts.

Produces a 1600x900 PNG (16:9, the aspect LinkedIn renders largest in-feed)
using the same palette as the app so it sits alongside the UI screenshots.

    python scripts/make_concept_diagram.py [output.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1600, 940
BG = "#f6f8fb"
INK = "#0f172a"
MUTED = "#475569"
FAINT = "#94a3b8"

AMBER_BG, AMBER_BR, AMBER_TX = "#fffbeb", "#fcd34d", "#b45309"
EMER_BG, EMER_BR, EMER_TX = "#ecfdf5", "#6ee7b7", "#047857"
SLATE_BG, SLATE_BR = "#ffffff", "#cbd5e1"

FONTS = "C:/Windows/Fonts/"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in (name, "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(FONTS + candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_TITLE = font("segoeuib.ttf", 46)
F_SUB = font("segoeui.ttf", 25)
F_LANE = font("segoeuib.ttf", 27)
F_BOX = font("segoeuisb.ttf", 20)
F_SMALL = font("segoeui.ttf", 18)
F_QUOTE = font("segoeuii.ttf", 24)
F_TAG = font("segoeuib.ttf", 17)


def rounded(d, box, radius, fill, outline=None, width=2):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def centred(d, text, cx, cy, fnt, fill):
    left, top, right, bottom = d.textbbox((0, 0), text, font=fnt)
    d.text((cx - (right - left) / 2 - left, cy - (bottom - top) / 2 - top), text, font=fnt, fill=fill)


def arrow(d, x1, y, x2, colour, width=3):
    d.line([(x1, y), (x2 - 9, y)], fill=colour, width=width)
    d.polygon([(x2, y), (x2 - 12, y - 6), (x2 - 12, y + 6)], fill=colour)


def lane(d, y, label, label_colour, steps, bg, br, tx, note):
    """One pipeline row: a label, then boxes joined by arrows."""
    d.text((70, y - 62), label, font=F_LANE, fill=label_colour)
    d.text((70, y - 28), note, font=F_SMALL, fill=MUTED)

    x = 70
    box_h = 84
    gap = 30
    for i, (line1, line2) in enumerate(steps):
        widths = [d.textbbox((0, 0), t, font=F_BOX)[2] for t in (line1, line2) if t]
        box_w = max(max(widths) + 46, 150)
        rounded(d, (x, y + 18, x + box_w, y + 18 + box_h), 14, bg, br, 2)
        if line2:
            centred(d, line1, x + box_w / 2, y + 18 + box_h / 2 - 13, F_BOX, tx)
            centred(d, line2, x + box_w / 2, y + 18 + box_h / 2 + 13, F_SMALL, MUTED)
        else:
            centred(d, line1, x + box_w / 2, y + 18 + box_h / 2, F_BOX, tx)
        x += box_w
        if i < len(steps) - 1:
            arrow(d, x + 6, y + 18 + box_h / 2, x + gap - 4, FAINT)
            x += gap


def main(out: Path) -> None:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Header ---------------------------------------------------------
    d.text((70, 58), "Basic RAG vs Semantic RAG", font=F_TITLE, fill=INK)
    d.text(
        (70, 118),
        "Same documents. Same embedding model. Same LLM. Only the retrieval differs.",
        font=F_SUB,
        fill=MUTED,
    )

    # Shared input + headline result, side by side under the title.
    # No connector lines: they had to cross the lanes to reach both, which
    # read as noise. The label carries the meaning instead.
    rounded(d, (70, 176, 470, 250), 14, SLATE_BG, SLATE_BR, 2)
    d.text((94, 190), "SHARED INPUT", font=F_TAG, fill=MUTED)
    d.text((94, 214), "12 enterprise documents — identical for both", font=F_SMALL, fill=INK)

    rounded(d, (1058, 156, 1530, 306), 16, "#ffffff", "#93c5fd", 2)
    d.text((1086, 178), "ANSWER CORRECTNESS", font=F_TAG, fill="#1d4ed8")
    d.text((1086, 204), "15 questions, ground truth, same LLM", font=F_SMALL, fill=MUTED)
    for row_y, label, a, b in ((240, "single-hop", "1.00", "1.00"), (272, "four-hop", "0.21", "0.75")):
        d.text((1086, row_y + 4), label, font=F_SMALL, fill=MUTED)
        d.text((1290, row_y), a, font=F_BOX, fill=AMBER_TX)
        d.text((1380, row_y + 4), "vs", font=F_SMALL, fill=FAINT)
        d.text((1430, row_y), b, font=F_BOX, fill=EMER_TX)

    # Lanes ----------------------------------------------------------
    lane(
        d, 372, "BASIC RAG", AMBER_TX,
        [
            ("Chunk", "fixed windows"),
            ("Embed", "dense vectors"),
            ("Vector search", "top-K cosine"),
            ("LLM", "answer"),
        ],
        AMBER_BG, AMBER_BR, AMBER_TX,
        "retrieves text that looks like the question",
    )

    lane(
        d, 578, "SEMANTIC RAG", EMER_TX,
        [
            ("Extract", "entities + relations"),
            ("Knowledge graph", "typed, with provenance"),
            ("Traverse", "multi-hop paths"),
            ("Expand", "grounded passages"),
            ("LLM", "answer"),
        ],
        EMER_BG, EMER_BR, EMER_TX,
        "retrieves the facts that connect to the question",
    )

    # The chain, full width so nothing clips -------------------------
    rounded(d, (70, 716, 1530, 826), 16, "#ffffff", EMER_BR, 2)
    d.text((94, 734), "THE CHAIN BASIC RAG CANNOT WALK", font=F_TAG, fill=EMER_TX)
    d.text((470, 734), "no single document holds two consecutive links", font=F_SMALL, fill=MUTED)

    chain = ["Project Phoenix", "C-17", "Alpha Precision Systems", "C-2048", "R-17"]
    widths = [d.textbbox((0, 0), n, font=F_BOX)[2] + 30 for n in chain]
    total = sum(widths) + 30 * (len(chain) - 1)
    cx = (W - total) / 2
    for i, node in enumerate(chain):
        w = widths[i]
        rounded(d, (cx, 766, cx + w, 806), 10, EMER_BG, EMER_BR, 2)
        centred(d, node, cx + w / 2, 786, F_BOX, INK)
        cx += w
        if i < len(chain) - 1:
            arrow(d, cx + 5, 786, cx + 25, FAINT, 2)
            cx += 30

    # The two questions ----------------------------------------------
    y = 852
    rounded(d, (70, y, 760, y + 68), 14, AMBER_BG, AMBER_BR, 2)
    d.text((94, y + 12), "Basic RAG asks", font=F_TAG, fill=AMBER_TX)
    d.text((94, y + 34), "“Which text is most similar?”", font=F_QUOTE, fill=INK)

    rounded(d, (800, y, 1530, y + 68), 14, EMER_BG, EMER_BR, 2)
    d.text((824, y + 12), "Semantic RAG asks", font=F_TAG, fill=EMER_TX)
    d.text((824, y + 34), "“Which entities and relationships are relevant?”", font=F_QUOTE, fill=INK)

    img.save(out, "PNG")
    print(f"wrote {out} ({img.width}x{img.height})")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("images/00-how-it-works.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    main(target)
