"""Hero agents. All of them call the same policy.decide(); they differ only in
what state they hand it.

  GroundTruthAgent - reads the engine state directly (the upper bound).
  VisionAgent      - screenshot -> perceive -> decide. Optional guardrails:
                     validate the read, re-capture up to 2 more frames, and
                     fall back to check/fold if nothing valid comes back.
"""
from __future__ import annotations
from seeds import stable_seed

import time

import numpy as np

from noise import CONDITIONS
from policy import decide
from table import render
from vision import Vision

FIELDS = ["hero_cards", "board", "pot", "hero_stack", "villain_stack", "hero_bet", "villain_bet", "button"]


def dec_seed(i, d):
    return i * 1000 + d


class GroundTruthAgent:
    def __init__(self):
        self.log = []

    def __call__(self, ts, i, d, ctx):
        a, amt, eq = decide(ts, dec_seed(i, d))
        return a, amt


class VisionAgent:
    def __init__(self, condition, mode="naive", guard=False, base_seed=0, thresholds=None):
        self.cond = condition
        self.theme, self.noise = CONDITIONS[condition]
        self.vision = Vision(mode)
        self.mode = mode
        self.guard = guard
        self.base_seed = base_seed
        self.th = thresholds or {"card": 0.0, "number": 0.0}
        self.log = []
        self._hand = None

    # --- guardrail checks: things the agent can verify without ground truth ---
    def _problems(self, p, conf):
        probs = []
        cards = p.hero_cards + p.board
        if len(set(cards)) != len(cards):
            probs.append("duplicate_card")
        if len(p.board) not in (0, 3, 4, 5):
            probs.append("board_count")
        if self._mem.get("board") and len(p.board) < len(self._mem["board"]):
            probs.append("board_shrank")
        if self._mem.get("hero_cards") and p.hero_cards != self._mem["hero_cards"]:
            probs.append("hole_cards_changed")
        cconf = [v for k, v in conf.items() if k.startswith(("hero", "board")) and k[-1].isdigit()]
        if cconf and min(cconf) < self.th["card"]:
            probs.append("low_card_conf")
        nconf = [conf[k] for k in ("pot", "hero_stack", "villain_stack", "hero_bet", "villain_bet") if k in conf]
        if nconf and min(nconf) < self.th["number"]:
            probs.append("low_number_conf")
        total = p.pot + p.hero_stack + p.villain_stack + p.hero_bet + p.villain_bet
        if self._mem.get("total") is not None and total != self._mem["total"]:
            probs.append("chips_not_conserved")
        return probs

    def __call__(self, ts, i, d, ctx):
        if self._hand != i:
            self._hand, self._mem = i, {}
        offset = ctx["setup"]["offset"]
        t0 = time.perf_counter()
        frames, probs_all, p, conf = 0, [], None, None
        perceive_ms = 0.0
        max_frames = 3 if self.guard else 1
        valid = False
        while frames < max_frames:
            rng = np.random.default_rng(stable_seed(self.base_seed, i, d, frames))
            img = self.noise(render(ts, self.theme, offset), rng)
            tp = time.perf_counter()
            p, conf = self.vision.perceive(img)
            perceive_ms += (time.perf_counter() - tp) * 1000
            frames += 1
            if not self.guard:
                valid = True
                break
            probs = self._problems(p, conf)
            probs_all.append(probs)
            if not probs:
                valid = True
                break
        fallback = False
        tdec = time.perf_counter()
        if valid:
            a, amt, eq_v = decide(p, dec_seed(i, d))
            if self.guard:
                self._mem.update(hero_cards=p.hero_cards, board=p.board,
                                 total=p.pot + p.hero_stack + p.villain_stack + p.hero_bet + p.villain_bet)
        else:
            fallback = True
            a, amt, eq_v = ("check", 0, float("nan"))   # check/fold button
        policy_ms = (time.perf_counter() - tdec) * 1000
        total_ms = (time.perf_counter() - t0) * 1000
        a_gt, amt_gt, eq_gt = decide(ts, dec_seed(i, d))
        true = ts.as_dict()
        seen = p.as_dict()
        row = dict(hand=i, dec=d, street=len(ts.board), condition=self.cond, mode=self.mode, guard=self.guard,
                   frames=frames, fallback=fallback,
                   problems="|".join(sorted({x for pr in probs_all for x in pr})),
                   gt_action=a_gt, gt_amount=amt_gt, vis_action=a, vis_amount=amt,
                   eq_gt=eq_gt, eq_vis=eq_v, perceive_ms=perceive_ms, policy_ms=policy_ms, total_ms=total_ms,
                   state_correct=(true == seen))
        for f in FIELDS:
            row[f"ok_{f}"] = true[f] == seen[f]
            row[f"true_{f}"] = " ".join(true[f]) if isinstance(true[f], list) else true[f]
            row[f"seen_{f}"] = " ".join(seen[f]) if isinstance(seen[f], list) else seen[f]
        self.log.append(row)
        return a, amt
