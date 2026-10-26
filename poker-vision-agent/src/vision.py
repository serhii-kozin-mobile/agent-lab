"""Perception: screenshot pixels -> TableState, with no access to the engine.

Classic screen-scraping: a hand-calibrated layout (where the client draws
things), template matching for card ranks/suits, a tiny connected-component
digit reader for stacks/pot/bets, and a brightness test for the dealer button.
Templates are built ONCE from clean renders of the classic theme, the way you
would calibrate against the client you developed on.

Two modes:
  naive    - grayscale template matching as calibrated.
  hardened - polarity normalisation + Otsu binarisation before matching, so a
             skin change (dark cards) does not flip the correlation sign.
Decision-level guardrails (re-reads, validation, fallback) live in agent.py.
"""
from __future__ import annotations

import cv2
cv2.setNumThreads(1)
import numpy as np
from PIL import Image

import table as T
from table import TableState, render, CARD_W, CARD_H, RANKS, SUITS

RANK_BOX = (2, 1, 46, 33)      # x0,y0,x1,y1 relative to card top-left
SUIT_BOX = (3, 31, 30, 58)
MARGIN = 6                      # search slack for window offset


def _gray(img):
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)


def _norm(g):
    """Polarity-normalise + binarise: ink becomes 255 on 0 background."""
    g = g.astype(np.uint8)
    if np.median(g) < 128:             # mostly dark card face -> ink is bright already
        inv = g
    else:
        inv = 255 - g
    _, b = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def _ink_bbox(g, face_val):
    mask = np.abs(g.astype(int) - face_val) > 60
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


class Vision:
    def __init__(self, mode: str = "naive"):
        assert mode in ("naive", "hardened")
        self.mode = mode
        self._build_templates()

    # ---------- calibration ----------
    def _build_templates(self):
        self.rank_t, self.suit_t = {}, {}
        base = dict(board=[], pot=0, hero_stack=1, villain_stack=1, hero_bet=0, villain_bet=0, button="hero")
        for r in RANKS:
            g = _gray(render(TableState(hero_cards=[r + "s", r + "h"], **base)))
            x, y = T.HERO_CARDS[0]
            crop = g[y + RANK_BOX[1]: y + RANK_BOX[3], x + RANK_BOX[0]: x + RANK_BOX[2]]
            x0, y0, x1, y1 = _ink_bbox(crop, 250)
            self.rank_t[r] = crop[max(0, y0 - 1): y1 + 1, max(0, x0 - 1): x1 + 1]
        for s in SUITS:
            g = _gray(render(TableState(hero_cards=["A" + s, "K" + s], **base)))
            x, y = T.HERO_CARDS[0]
            crop = g[y + SUIT_BOX[1]: y + SUIT_BOX[3], x + SUIT_BOX[0]: x + SUIT_BOX[2]]
            x0, y0, x1, y1 = _ink_bbox(crop, 250)
            self.suit_t[s] = crop[max(0, y0 - 1): y1 + 1, max(0, x0 - 1): x1 + 1]
        # digits: render each digit in the stack plate and the bet plate font
        self.digit_t = {}
        for kind, name in (("big", "hero_stack"), ("small", "hero_bet")):
            ts = {}
            for dgt in "0123456789":
                st = TableState(hero_cards=["As", "Kd"], **{**base, "hero_stack": 1, "hero_bet": 1})
                img = render(st)
                # overwrite the plate with the digit (render() uses fmt_chips)
                from PIL import ImageDraw
                d = ImageDraw.Draw(img)
                T.draw_plate(d, name, 0, label=dgt)
                comps = self._components(_gray(img), name)
                ts[dgt] = self._digit_vec(comps[0][1])
            self.digit_t[kind] = ts
        self.digit_w = {"big": 13, "small": 12}

    # ---------- primitives ----------
    def _prep(self, g):
        return _norm(g) if self.mode == "hardened" else g

    def _match(self, region, templates):
        region = self._prep(region)
        best, second, label = -2.0, -2.0, None
        for k, t in templates.items():
            tt = self._prep(t)
            if tt.shape[0] > region.shape[0] or tt.shape[1] > region.shape[1]:
                continue
            sc = float(cv2.matchTemplate(region, tt, cv2.TM_CCOEFF_NORMED).max())
            if sc > best:
                best, second, label = sc, best, k
            elif sc > second:
                second = sc
        return label, best, best - second

    def read_card(self, g, x, y):
        m = MARGIN
        rr = g[y + RANK_BOX[1] - m: y + RANK_BOX[3] + m, x + RANK_BOX[0] - m: x + RANK_BOX[2] + m]
        sr = g[y + SUIT_BOX[1] - m: y + SUIT_BOX[3] + m, x + SUIT_BOX[0] - m: x + SUIT_BOX[2] + m]
        r, rs, rmarg = self._match(rr, self.rank_t)
        s, ss, smarg = self._match(sr, self.suit_t)
        return r + s, min(rs, ss)

    @staticmethod
    def card_present(rgb, x, y):
        patch = rgb[y + 25: y + CARD_H - 25, x + 15: x + CARD_W - 15].reshape(-1, 3).astype(int)
        hsv = cv2.cvtColor(patch.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        # felt is strongly saturated green; card faces (light or dark) are not
        green_felt = (np.abs(hsv[:, 0].astype(int) - 70) < 15) & (hsv[:, 1] > 90)
        return green_felt.mean() < 0.5

    def _components(self, g, name, hardened=False):
        (cx, cy), w, h = T.PLATES[name]
        x0, y0 = cx - w // 2 + 8, cy - h // 2 + 3
        crop = g[y0: cy + h // 2 - 3, x0: cx + w // 2 - 8]
        if hardened:
            _, b = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, b = cv2.threshold(crop, 150, 255, cv2.THRESH_BINARY)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
        out = []
        if hardened:
            # keep glyph-height components only, then cut every glyph to the
            # shared digit band so a comma fused by blur does not distort it
            exp_h = 13 if name.endswith("_bet") else 15
            cand = [i for i in range(1, n) if stats[i][3] >= 0.7 * exp_h and stats[i][4] >= 15
                    and stats[i][2] <= 5 * exp_h]
            if not cand:
                return out
            top = int(np.median([stats[i][1] for i in cand]))
            for i in cand:
                x, y, ww, hh, area = stats[i]
                m = (lab[top: top + exp_h, x: x + ww] == i).astype(np.uint8) * 255
                if m.sum() > 0:
                    out.append((x, m))
        else:
            for i in range(1, n):
                x, y, ww, hh, area = stats[i]
                if hh >= 9 and area >= 12:        # drop commas/specks
                    out.append((x, (lab[y: y + hh, x: x + ww] == i).astype(np.uint8) * 255))
        out.sort(key=lambda t: t[0])
        return out

    @staticmethod
    def _digit_vec(m):
        v = cv2.resize(m, (10, 16), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        v -= v.mean()
        n = np.linalg.norm(v)
        return v / n if n else v

    def read_number(self, g, name):
        kind = "small" if name.endswith("_bet") else "big"
        comps = self._components(g, name, hardened=self.mode == "hardened")
        if not comps:
            return 0, 0.0
        digits, conf = [], 1.0
        dw = self.digit_w[kind]
        for _, m in comps:
            pieces = [m]
            if m.shape[1] > 1.6 * dw:                       # merged glyphs (blur/scale)
                k = int(round(m.shape[1] / dw))
                pieces = np.array_split(m, k, axis=1)
            for p in pieces:
                v = self._digit_vec(p)
                scores = {d: float(v @ t) for d, t in self.digit_t[kind].items()}
                d = max(scores, key=scores.get)
                digits.append(d)
                conf = min(conf, scores[d])
        try:
            return int("".join(digits)), conf
        except ValueError:
            return 0, 0.0

    @staticmethod
    def plate_present(rgb, name):
        (cx, cy), w, h = T.PLATES[name]
        patch = rgb[cy - h // 2 + 4: cy + h // 2 - 4, cx - w // 2 + 6: cx + w // 2 - 6].astype(int)
        return np.median(patch[..., 1]) < 75   # dark plate vs green felt (median: long numbers are mostly white)

    @staticmethod
    def read_button(rgb):
        sc = {}
        for who, (bx, by) in T.BUTTON.items():
            p = rgb[by - 9: by + 10, bx - 9: bx + 10].astype(int)
            sc[who] = p.mean()
        who = max(sc, key=sc.get)
        return who, (sc[who] - min(sc.values())) / 255.0

    # ---------- full read ----------
    def perceive(self, img: Image.Image):
        if img.size != (T.W, T.H):
            img = img.resize((T.W, T.H), Image.BILINEAR)   # agent knows the window size
        rgb = np.asarray(img)
        g = _gray(img)
        conf = {}
        hero = []
        for i, (x, y) in enumerate(T.HERO_CARDS):
            c, s = self.read_card(g, x, y)
            hero.append(c)
            conf[f"hero{i}"] = s
        board = []
        for i, (x, y) in enumerate(T.BOARD):
            if not self.card_present(rgb, x, y):
                break
            c, s = self.read_card(g, x, y)
            board.append(c)
            conf[f"board{i}"] = s
        nums = {}
        for name in ("pot", "hero_stack", "villain_stack"):
            nums[name], conf[name] = self.read_number(g, name)
        for name in ("hero_bet", "villain_bet"):
            if self.plate_present(rgb, name):
                nums[name], conf[name] = self.read_number(g, name)
            else:
                nums[name] = 0
        button, conf["button"] = self.read_button(rgb)
        st = TableState(hero_cards=hero, board=board, button=button, **nums)
        return st, conf
