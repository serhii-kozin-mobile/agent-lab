"""Heads-up NLHE driven by PokerKit, with duplicate (seeded) hands.

Hand i always gets the same deck, starting stacks, button and opponent style,
so a ground-truth agent and a vision agent can be compared hand by hand.
"""
from __future__ import annotations
from seeds import stable_seed

import random

from pokerkit import Automation, Mode, NoLimitTexasHoldem

from policy import equity
from table import TableState

BB = 100
AUTOS = (Automation.ANTE_POSTING, Automation.BET_COLLECTION, Automation.BLIND_OR_STRADDLE_POSTING,
         Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING, Automation.CHIPS_PUSHING,
         Automation.CHIPS_PULLING, Automation.CARD_BURNING)
DECK = [r + s for r in "23456789TJQKA" for s in "shdc"]


def hand_setup(i, base_seed):
    rng = random.Random(base_seed * 1_000_003 + i)
    deck = DECK[:]
    rng.shuffle(deck)
    return dict(
        deck=deck,
        hero_start=rng.randrange(300, 1501) * 10,     # 30-150 bb, varied digits
        vill_start=rng.randrange(300, 1501) * 10,
        hero_button=(i % 2 == 0),
        opp_style=rng.choice(["tag", "station"]),
        offset=(rng.randint(-4, 4), rng.randint(-4, 4)),   # window position
    )


def opponent(style, cards, board, to_call, pot_total, stack, my_bet, seed):
    rng = random.Random(seed)
    eq = equity(cards, board, seed, n=150)
    if style == "tag":
        if to_call == 0:
            if eq > 0.6 or rng.random() < 0.08:
                return "bet", my_bet + max(BB, int(0.6 * pot_total) // 10 * 10)
            return "check", 0
        if eq > 0.75:
            return "bet", my_bet + to_call + pot_total + to_call
        return ("call", 0) if eq >= to_call / (pot_total + to_call) + 0.05 else ("fold", 0)
    # station
    if to_call == 0:
        if eq > 0.7:
            return "bet", my_bet + max(BB, int(0.5 * pot_total) // 10 * 10)
        return "check", 0
    return ("call", 0) if eq > 0.3 else ("fold", 0)


def apply(state, action, amount):
    """Map an intended click to the engine. Returns what actually happened.
    UI assumption: the 'check' button is a check/fold button, so clicking
    check while facing a bet folds."""
    owe = state.checking_or_calling_amount
    if action == "bet" and state.can_complete_bet_or_raise_to():
        lo = state.min_completion_betting_or_raising_to_amount
        hi = state.max_completion_betting_or_raising_to_amount
        state.complete_bet_or_raise_to(min(hi, max(lo, int(amount))))
        return "bet"
    if action == "fold" and owe > 0:
        state.fold()
        return "fold"
    if action == "check" and owe > 0:
        state.fold()
        return "fold"
    state.check_or_call()
    return "call" if owe > 0 else "check"


def play_hand(i, base_seed, hero_agent):
    """hero_agent(true_state: TableState, hand_i, dec_i, ctx) -> (action, amount)."""
    s = hand_setup(i, base_seed)
    h = 1 if s["hero_button"] else 0          # pokerkit HU: seat 1 is SB/button
    v = 1 - h
    stacks = [0, 0]
    stacks[h], stacks[v] = s["hero_start"], s["vill_start"]
    st = NoLimitTexasHoldem.create_state(AUTOS, True, 0, (50, 100), BB, tuple(stacks), 2, mode=Mode.CASH_GAME)
    deck = s["deck"]
    hole = {h: deck[0:2], v: deck[2:4]}
    st.deal_hole("".join(hole[0]))
    st.deal_hole("".join(hole[1]))
    board_src = deck[4:9]
    ctx = {"hand": i, "setup": s}
    dec_h = dec_v = 0
    while st.status:
        if st.can_deal_board():
            n = len(st.board_cards)
            k = 3 if n == 0 else 1
            st.deal_board("".join(board_src[n:n + k]))
            continue
        a = st.actor_index
        if a is None:
            break
        board = [c[0].rank + c[0].suit for c in st.board_cards]
        pot = st.total_pot_amount - sum(st.bets)
        if a == h:
            ts = TableState(hero_cards=hole[h], board=board, pot=pot, hero_stack=st.stacks[h],
                            villain_stack=st.stacks[v], hero_bet=st.bets[h], villain_bet=st.bets[v],
                            button="hero" if s["hero_button"] else "villain")
            act, amt = hero_agent(ts, i, dec_h, ctx)
            apply(st, act, amt)
            dec_h += 1
        else:
            to_call = st.bets[h] - st.bets[v]
            act, amt = opponent(s["opp_style"], hole[v], board, max(0, to_call),
                                pot + sum(st.bets), st.stacks[v], st.bets[v],
                                seed=stable_seed(base_seed, i, "opp", dec_v))
            apply(st, act, amt)
            dec_v += 1
    return (st.stacks[h] - s["hero_start"]) / BB, dec_h
