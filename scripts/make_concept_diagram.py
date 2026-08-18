"""Render the 'how it works' hero image for the README and social posts.

Deliberately drawn in the app's own visual language — same palette, same card
treatment, same entity colours — so it reads as part of the product rather than
a generic boxes-and-arrows diagram. Soft shadows and a faint dot grid give it
the depth a flat vector export lacks.

    python scripts/make_concept_diagram.py [output.png]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1600, 1000
BG = "#f7f9fc"
GRID = "#dde5ee"
INK = "#0f172a"
MUTED = "#475569"
FAINT = "#94a3b8"
HAIRLINE = "#e2e8f0"

AMBER = {"bg": "#fffbeb", "br": "#fcd34d", "tx": "#b45309", "dot": "#f59e0b"}
EMER = {"bg": "#ecfdf5", "br": "#6ee7b7", "tx": "#047857", "dot": "#10b981"}

# Entity palette, matching ENTITY_COLORS in the frontend.
ENTITY = {
    "Project": {"bg": "#f0f9ff", "br": "#7dd3fc", "tx": "#0369a1", "dot": "#38bdf8"},
    "Component": {"bg": "#f5f3ff", "br": "#c4b5fd", "tx": "#6d28d9", "dot": "#a78bfa"},
    "Supplier": {"bg": "#fffbeb", "br": "#fcd34d", "tx": "#b45309", "dot": "#fbbf24"},
    "Contract": {"bg": "#ecfdf5", "br": "#6ee7b7", "tx": "#047857", "dot": "#34d399"},
    "Risk": {"bg": "#fff1f2", "br": "#fda4af", "tx": "#be123c", "dot": "#fb7185"},
    "Employee": {"bg": "#f0fdfa", "br": "#5eead4", "tx": "#0f766e", "dot": "#2dd4bf"},
}

FONTS = "C:/Windows/Fonts/"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    for candidate in (name, "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(FONTS + candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


F_EYEBROW = font("segoeuib.ttf", 19)
F_TITLE = font("segoeuib.ttf", 58)
F_SUB = font("segoeui.ttf", 26)
F_LANE = font("segoeuib.ttf", 24)
F_NODE = font("segoeuisb.ttf", 22)
F_KIND = font("segoeuib.ttf", 14)
F_STEP = font("segoeuisb.ttf", 20)
F_SMALL = font("segoeui.ttf", 18)
F_REL = font("consola.ttf", 16)
F_QUOTE = font("segoeuii.ttf", 25)
F_BIG = font("segoeuib.ttf", 40)
F_TAG = font("segoeuib.ttf", 16)


class Canvas:
    """Base image plus a shadow layer, composited before anything is drawn."""

    def __init__(self) -> None:
        self.img = Image.new("RGB", (W, H), BG)
        self.shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        self.sd = ImageDraw.Draw(self.shadow)

    def drop(self, box, radius: int, alpha: int = 26, dy: int = 6) -> None:
        x1, y1, x2, y2 = box
        self.sd.rounded_rectangle(
            (x1, y1 + dy, x2, y2 + dy), radius=radius, fill=(15, 23, 42, alpha)
        )

    def bake(self) -> ImageDraw.ImageDraw:
        self.shadow = self.shadow.filter(ImageFilter.GaussianBlur(14))
        self.img = Image.alpha_composite(self.img.convert("RGBA"), self.shadow).convert("RGB")
        return ImageDraw.Draw(self.img)


def dot_grid(d: ImageDraw.ImageDraw, gap: int = 26) -> None:
    for x in range(60, W - 40, gap):
        for y in range(150, H - 40, gap):
            d.point((x, y), fill=GRID)


def centred(d, text, cx, cy, fnt, fill):
    left, top, right, bottom = d.textbbox((0, 0), text, font=fnt)
    d.text((cx - (right - left) / 2 - left, cy - (bottom - top) / 2 - top), text, font=fnt, fill=fill)


def width_of(d, text, fnt) -> int:
    return d.textbbox((0, 0), text, font=fnt)[2]


def chevron(d, x, y, colour, size=9, w=3):
    d.line([(x, y - size), (x + size - 2, y), (x, y + size)], fill=colour, width=w, joint="curve")


# ----------------------------------------------------------------------
CHAIN = [
    ("Project", "Project Phoenix", "uses"),
    ("Component", "C-17", "supplied_by"),
    ("Supplier", "Alpha Precision", "governed_by"),
    ("Contract", "C-2048", "has_risk"),
    ("Risk", "R-17", None),
]

BASIC_STEPS = ["Chunk", "Embed", "Vector search", "Top-K"]
SEMANTIC_STEPS = ["Extract entities", "Build graph", "Traverse", "Expand context"]


def main(out: Path) -> None:
    c = Canvas()
    probe = ImageDraw.Draw(c.img)

    # --- plan geometry, register shadows -----------------------------
    # The dot sits at +18..28 and the kind label starts at +36, so the kind row
    # needs more padding than the centred node label.
    chain_w = [
        max(width_of(probe, label, F_NODE) + 52, width_of(probe, kind, F_KIND) + 62)
        for kind, label, _rel in CHAIN
    ]
    # Wide enough that a relation label ("supplied_by") never reaches a card.
    gap = 148
    total = sum(chain_w) + gap * (len(CHAIN) - 1)
    chain_top, chain_h = 268, 92

    x = (W - total) / 2
    chain_boxes = []
    for w in chain_w:
        chain_boxes.append((x, chain_top, x + w, chain_top + chain_h))
        c.drop((x, chain_top, x + w, chain_top + chain_h), 16)
        x += w + gap

    lane_boxes = []
    for ly in (534, 726):
        lane_boxes.append((70, ly, 1530, ly + 132))
        c.drop((70, ly, 1530, ly + 132), 20, alpha=20)

    d = c.bake()
    dot_grid(d)

    # --- header -------------------------------------------------------
    d.text((70, 62), "RAG INTELLIGENCE LAB", font=F_EYEBROW, fill=EMER["tx"])
    d.text((70, 96), "Retrieval is not one thing.", font=F_TITLE, fill=INK)
    d.text(
        (70, 180),
        "Same documents. Same embedding model. Same LLM. Only the retrieval differs.",
        font=F_SUB,
        fill=MUTED,
    )

    # --- the chain (hero) ---------------------------------------------
    for (kind, label, rel), box in zip(CHAIN, chain_boxes):
        pal = ENTITY[kind]
        d.rounded_rectangle(box, radius=16, fill=pal["bg"], outline=pal["br"], width=2)
        x1, y1, x2, y2 = box
        d.ellipse((x1 + 18, y1 + 25, x1 + 28, y1 + 35), fill=pal["dot"])
        d.text((x1 + 36, y1 + 21), kind.upper(), font=F_KIND, fill=MUTED)
        centred(d, label, (x1 + x2) / 2, y1 + 64, F_NODE, pal["tx"])

        if rel:
            mid = x2 + gap / 2
            d.line([(x2 + 12, y1 + 62), (x2 + gap - 18, y1 + 62)], fill="#cbd5e1", width=2)
            chevron(d, x2 + gap - 22, y1 + 62, "#94a3b8", 7, 2)
            rw = width_of(d, rel, F_REL)
            d.rounded_rectangle(
                (mid - rw / 2 - 10, y1 + 18, mid + rw / 2 + 10, y1 + 44),
                radius=7, fill="#ffffff", outline=HAIRLINE, width=1,
            )
            centred(d, rel, mid, y1 + 31, F_REL, MUTED)

    d.text(
        (70, 402),
        "One real question spans five documents. No single document holds two consecutive links.",
        font=F_SMALL,
        fill=MUTED,
    )
    d.line([(70, 446), (1530, 446)], fill=HAIRLINE, width=2)

    # --- the two lanes -------------------------------------------------
    lanes = (
        (
            lane_boxes[0], "BASIC RAG", "finds text that resembles the question",
            AMBER, BASIC_STEPS, "\u201cWhich text is most similar?\u201d",
        ),
        (
            lane_boxes[1], "SEMANTIC RAG", "follows the facts that connect to it",
            EMER, SEMANTIC_STEPS, "\u201cWhich entities and relationships are relevant?\u201d",
        ),
    )
    for box, name, note, pal, steps, quote in lanes:
        x1, y1, x2, y2 = box
        d.rounded_rectangle(box, radius=20, fill="#ffffff", outline=HAIRLINE, width=2)
        d.rounded_rectangle((x1, y1 + 2, x1 + 8, y2 - 2), radius=4, fill=pal["dot"])

        d.text((x1 + 36, y1 + 24), name, font=F_LANE, fill=pal["tx"])
        d.text((x1 + 36, y1 + 58), note, font=F_SMALL, fill=MUTED)
        d.text((x1 + 36, y1 + 88), quote, font=F_QUOTE, fill=INK)

        sx = x1 + 700
        for j, step in enumerate(steps):
            sw = width_of(d, step, F_STEP) + 40
            d.rounded_rectangle(
                (sx, y1 + 40, sx + sw, y1 + 90), radius=12,
                fill=pal["bg"], outline=pal["br"], width=2,
            )
            centred(d, step, sx + sw / 2, y1 + 65, F_STEP, pal["tx"])
            sx += sw
            if j < len(steps) - 1:
                chevron(d, sx + 9, y1 + 65, "#cbd5e1", 6, 2)
                sx += 28

        d.text((x1 + 700, y1 + 98), "then the same LLM, the same prompt", font=F_SMALL, fill=FAINT)

    # --- footer metrics -------------------------------------------------
    y = 902
    d.text((70, y - 4), "ANSWER CORRECTNESS", font=F_TAG, fill=MUTED)
    d.text((70, y + 22), "15 questions \u00b7 ground truth \u00b7 same LLM", font=F_SMALL, fill=FAINT)

    for lx, label, a, b in ((520, "SINGLE-HOP", "1.00", "1.00"), (930, "FOUR-HOP", "0.21", "0.75")):
        d.text((lx, y - 4), label, font=F_TAG, fill=MUTED)
        d.text((lx, y + 18), a, font=F_BIG, fill=AMBER["tx"])
        aw = width_of(d, a, F_BIG)
        d.text((lx + aw + 18, y + 34), "vs", font=F_SMALL, fill=FAINT)
        d.text((lx + aw + 62, y + 18), b, font=F_BIG, fill=EMER["tx"])

    d.text((1340, y - 4), "MIT \u00b7 OPEN SOURCE", font=F_TAG, fill=MUTED)
    d.text((1340, y + 22), "built with Semantica", font=F_SMALL, fill=FAINT)

    c.img.save(out, "PNG")
    print(f"wrote {out} ({c.img.width}x{c.img.height})")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("images/00-how-it-works.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    main(target)
