"""Table state, layout and renderer.

The renderer draws a heads-up No-Limit Hold'em table to a PIL image, the way a
poker client window would look. The vision agent never gets the state object,
only the pixels this module produces (after the noise in noise.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "..", "assets", "fonts")

W, H = 960, 600                      # reference window size
CARD_W, CARD_H = 70, 98
RANKS = "23456789TJQKA"
SUITS = "shdc"
SUIT_GLYPH = {"s": "\u2660", "h": "\u2665", "d": "\u2666", "c": "\u2663"}
RANK_LABEL = {r: r for r in RANKS}
RANK_LABEL["T"] = "10"

# Fixed layout in reference pixels (top-left corners / centres).
HERO_CARDS = [(405, 430), (485, 430)]
VILL_CARDS = [(405, 40), (485, 40)]
BOARD = [(285 + i * 80, 245) for i in range(5)]
PLATES = {                            # name -> centre, width, height
    "villain_stack": ((480, 160), 150, 30),
    "villain_bet": ((480, 205), 120, 26),
    "pot": ((480, 368), 170, 30),
    "hero_bet": ((480, 403), 120, 26),
    "hero_stack": ((480, 557), 150, 30),
}
BUTTON = {"villain": (590, 160), "hero": (590, 557)}
BUTTON_R = 14
FELT = (31, 96, 54)
FELT_EDGE = (22, 70, 40)


@dataclass
class TableState:
    hero_cards: list[str]
    board: list[str]
    pot: int                 # collected pot, excluding this street's bets
    hero_stack: int
    villain_stack: int
    hero_bet: int
    villain_bet: int
    button: str              # "hero" or "villain"

    def as_dict(self):
        return dict(hero_cards=list(self.hero_cards), board=list(self.board),
                    pot=self.pot, hero_stack=self.hero_stack,
                    villain_stack=self.villain_stack, hero_bet=self.hero_bet,
                    villain_bet=self.villain_bet, button=self.button)


@dataclass
class Theme:
    name: str
    face: tuple
    border: tuple
    colors: dict              # suit -> rgb
    font: str
    back: tuple = (150, 30, 40)


THEMES = {
    "classic": Theme("classic", (250, 250, 247), (120, 120, 120),
                     {"s": (20, 20, 20), "c": (20, 20, 20), "h": (200, 25, 30), "d": (200, 25, 30)},
                     "DejaVuSans-Bold.ttf"),
    "four_color": Theme("four_color", (250, 250, 247), (120, 120, 120),
                        {"s": (20, 20, 20), "c": (20, 130, 40), "h": (200, 25, 30), "d": (30, 70, 200)},
                        "DejaVuSans-Bold.ttf"),
    "dark": Theme("dark", (45, 47, 55), (90, 90, 100),
                  {"s": (235, 235, 235), "c": (235, 235, 235), "h": (255, 110, 120), "d": (255, 110, 120)},
                  "DejaVuSerif-Bold.ttf", back=(30, 60, 120)),
}

_font_cache: dict = {}


def font(name: str, size: int):
    key = (name, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    return _font_cache[key]


def fmt_chips(v: int) -> str:
    return f"{v:,}"


def draw_card(d: ImageDraw.ImageDraw, x: int, y: int, card: str, th: Theme):
    r, s = card[0], card[1]
    col = th.colors[s]
    d.rounded_rectangle([x, y, x + CARD_W, y + CARD_H], radius=7, fill=th.face, outline=th.border, width=2)
    d.text((x + 6, y + 3), RANK_LABEL[r], font=font(th.font, 26), fill=col)
    d.text((x + 7, y + 32), SUIT_GLYPH[s], font=font(th.font, 22), fill=col)
    d.text((x + 33, y + 50), SUIT_GLYPH[s], font=font(th.font, 34), fill=col)


def draw_back(d, x, y, th: Theme):
    d.rounded_rectangle([x, y, x + CARD_W, y + CARD_H], radius=7, fill=th.back, outline=(240, 240, 240), width=3)
    for k in range(8, CARD_W - 4, 10):
        d.line([x + k, y + 6, x + k, y + CARD_H - 6], fill=tuple(min(255, c + 40) for c in th.back), width=2)


def draw_plate(d, name: str, value: int, label: str | None = None):
    (cx, cy), w, h = PLATES[name]
    text = fmt_chips(value) if label is None else label
    is_bet = name.endswith("_bet")
    fill = (15, 40, 25) if is_bet else (12, 14, 18)
    d.rounded_rectangle([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2], radius=h // 2, fill=fill,
                        outline=(200, 170, 60) if name == "pot" else (60, 60, 60), width=2)
    f = font("DejaVuSans-Bold.ttf", 20 if not is_bet else 18)
    tw = d.textlength(text, font=f)
    d.text((cx - tw / 2, cy - 12 if not is_bet else cy - 11), text, font=f, fill=(245, 245, 245))


def render(state: TableState, theme: str = "classic", offset=(0, 0)) -> Image.Image:
    th = THEMES[theme]
    img = Image.new("RGB", (W, H), FELT_EDGE)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([20, 20, W - 20, H - 20], radius=180, fill=FELT, outline=(90, 60, 30), width=10)
    ox, oy = offset

    def sh(p):
        return p[0] + ox, p[1] + oy

    for (x, y) in VILL_CARDS:
        draw_back(d, *sh((x, y)), th)
    for (x, y), c in zip(HERO_CARDS, state.hero_cards):
        draw_card(d, *sh((x, y)), c, th)
    for (x, y), c in zip(BOARD, state.board):
        draw_card(d, *sh((x, y)), c, th)
    # plates are drawn in shifted coordinates too
    global PLATES
    saved = PLATES
    PLATES = {k: ((v[0][0] + ox, v[0][1] + oy), v[1], v[2]) for k, v in saved.items()}
    try:
        draw_plate(d, "villain_stack", state.villain_stack)
        draw_plate(d, "hero_stack", state.hero_stack)
        draw_plate(d, "pot", state.pot)
        if state.villain_bet:
            draw_plate(d, "villain_bet", state.villain_bet)
        if state.hero_bet:
            draw_plate(d, "hero_bet", state.hero_bet)
    finally:
        PLATES = saved
    bx, by = sh(BUTTON[state.button])
    d.ellipse([bx - BUTTON_R, by - BUTTON_R, bx + BUTTON_R, by + BUTTON_R], fill=(245, 245, 240), outline=(30, 30, 30), width=2)
    f = font("DejaVuSans-Bold.ttf", 16)
    d.text((bx - 6, by - 10), "D", font=f, fill=(20, 20, 20))
    return img
