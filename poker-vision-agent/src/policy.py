"""A deliberately simple equity-based policy (the 'brain').

It is the SAME function for the ground-truth agent and the vision agent; only
its input differs. Equity is Monte-Carlo vs one random hand (treys evaluator),
seeded per decision so both agents draw identical random runouts.
"""
from __future__ import annotations

import random

from treys import Card, Evaluator

EVAL = Evaluator()
FULL_DECK = [r + s for r in "23456789TJQKA" for s in "shdc"]
_cache: dict = {}


def _c(cs):
    return [Card.new(c) for c in cs]


def equity(hole, board, seed, n=300):
    """P(win)+0.5*P(tie) of `hole` vs one uniformly random hand.
    Robust to impossible inputs (duplicate cards from a misread)."""
    key = (tuple(hole), tuple(board), seed, n)
    if key in _cache:
        return _cache[key]
    rng = random.Random(seed)
    known = set(hole) | set(board)
    deck = [c for c in FULL_DECK if c not in known]
    h = _c(hole)
    wins = 0.0
    need = 5 - len(board)
    for _ in range(n):
        draw = rng.sample(deck, 2 + need)
        opp = _c(draw[:2])
        full = _c(list(board) + draw[2:])
        try:
            a = EVAL.evaluate(full, h)
            b = EVAL.evaluate(full, opp)
        except KeyError:                       # duplicate cards -> invalid hand
            return 0.5
        wins += 1.0 if a < b else 0.5 if a == b else 0.0
    eq = wins / n
    if len(_cache) > 200000:
        _cache.clear()
    _cache[key] = eq
    return eq


def decide(st, seed, bb=100):
    """st: TableState as perceived. Returns (action, amount) where action is
    'fold' | 'check' | 'call' | 'bet' and amount is the raise-to total for bets."""
    eq = equity(st.hero_cards, st.board, seed)
    to_call = max(0, st.villain_bet - st.hero_bet)
    pot_total = st.pot + st.hero_bet + st.villain_bet
    ip_bonus = 0.03 if st.button == "hero" and st.board else 0.0   # in position postflop
    max_to = st.hero_bet + st.hero_stack
    if to_call == 0:
        if eq > 0.62 - ip_bonus:
            size = max(bb, int(round(0.66 * max(pot_total, bb) / 10.0)) * 10)
            return "bet", min(max_to, st.hero_bet + size), eq
        return "check", 0, eq
    pot_odds = to_call / (pot_total + to_call)
    pot_before = max(bb, pot_total - to_call)
    # big bets from these bots mean strength; equity vs a random hand overstates
    # our chances, so demand more the bigger the bet is relative to the pot
    strength_tax = 0.15 * min(1.0, to_call / pot_before)
    if eq > 0.72 - ip_bonus and 2 * to_call <= pot_total and st.hero_stack > to_call:
        raise_to = st.villain_bet + (pot_total + to_call)   # pot-sized raise
        return "bet", min(max_to, raise_to), eq
    if eq >= pot_odds + 0.03 + strength_tax - ip_bonus:
        return "call", 0, eq
    return "fold", 0, eq
