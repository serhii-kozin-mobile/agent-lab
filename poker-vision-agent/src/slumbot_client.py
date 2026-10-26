"""Phase 2: play against Slumbot's public bot API (https://slumbot.com).

Slumbot is a bot-vs-bot arena: heads-up NLHE, blinds 50/100, 200bb stacks
reset every hand, no money. We are polite: one request at a time, a fixed
delay between requests, hard cap on hands, stop on any error.

Modes:
  gt     - policy gets the state parsed from the API (ground truth).
  vision - the API state is rendered to a screenshot (noise condition), read
           back by the vision pipeline, and the policy acts on THAT. Every
           decision also logs what the ground-truth policy and the hardened
           single-read pipeline would have done on the same state/frame.

Usage: python slumbot_client.py MODE N_HANDS OUT_CSV [CONDITION] [DELAY_S]
Resumable: appends to OUT_CSV and continues counting from existing rows.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
import requests

from noise import CONDITIONS
from policy import decide
from table import TableState, render
from vision import Vision

HOST = "https://slumbot.com/slumbot/api"
STACK, SB, BB = 20000, 50, 100
UA = {"Content-Type": "application/json", "User-Agent": "serhiibuilds-lab-study/1.0 (research bot, polite rate)"}


def post(path, data, delay):
    time.sleep(delay)
    for attempt in range(3):
        try:
            r = requests.post(f"{HOST}/{path}", json=data, headers=UA, timeout=30)
            if r.status_code == 200:
                j = r.json()
                if "error_msg" in j:
                    raise RuntimeError(j["error_msg"])
                return j
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        except (requests.RequestException, ValueError) as e:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def parse(action):
    """Replay Slumbot's action string. Positions: 1 = small blind/button
    (acts first preflop), 0 = big blind. Returns per-player street bets,
    collected pot, total contributions, whose turn, last raise size."""
    street = {0: BB, 1: SB}
    total = {0: BB, 1: SB}
    pot = 0
    pos, last_size, i = 1, BB - SB, 0
    while i < len(action):
        ch = action[i]
        i += 1
        if ch == "/":
            pot += street[0] + street[1]
            street = {0: 0, 1: 0}
            pos, last_size = 0, 0
        elif ch == "k":
            pos = 1 - pos
        elif ch == "c":
            add = max(street.values()) - street[pos]
            street[pos] += add
            total[pos] += add
            pos = 1 - pos
        elif ch == "f":
            pos = -1
        elif ch == "b":
            j = i
            while i < len(action) and action[i].isdigit():
                i += 1
            to = int(action[j:i])
            last_size = to - max(street.values())
            total[pos] += to - street[pos]
            street[pos] = to
            pos = 1 - pos
    return dict(street=street, total=total, pot=pot, pos=pos, last_size=max(last_size, BB))


def to_state(r):
    me = r["client_pos"]
    p = parse(r["action"])
    return TableState(hero_cards=r["hole_cards"], board=r.get("board", []), pot=p["pot"],
                      hero_stack=STACK - p["total"][me], villain_stack=STACK - p["total"][1 - me],
                      hero_bet=p["street"][me], villain_bet=p["street"][1 - me],
                      button="hero" if me == 1 else "villain"), p


def to_incr(action, amount, ts, p):
    """Map the policy's intended click to a legal Slumbot increment. Same UI
    semantics as the local sim: 'check' is a check/fold button."""
    facing = ts.villain_bet > ts.hero_bet
    if action == "bet":
        max_to = ts.hero_bet + ts.hero_stack
        if max(ts.villain_bet, ts.hero_bet) >= max_to:     # cannot raise, just call
            return "c" if facing else "k"
        min_to = min(max_to, max(ts.villain_bet, ts.hero_bet) + p["last_size"])
        return f"b{int(min(max_to, max(min_to, amount)))}"
    if action == "call":
        return "c" if facing else "k"
    return "f" if facing else "k"          # fold or check/fold


def main(mode, n_hands, out_csv, condition="realistic_mix", delay=0.4):
    fields = ["hand", "dec", "mode", "condition", "client_pos", "street", "action_str", "incr",
              "gt_action", "vis_action", "hard_action", "state_correct", "hard_state_correct",
              "perceive_ms", "winnings", "token_changed"]
    done = 0
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            done = len({row["hand"] for row in csv.DictReader(f) if row["winnings"] != ""})
    new = not os.path.exists(out_csv)
    f = open(out_csv, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    theme, noise = CONDITIONS[condition]
    naive, hard = Vision("naive"), Vision("hardened")
    token = None
    for h in range(done, n_hands):
        r = post("new_hand", {"token": token} if token else {}, delay)
        token = r.get("token", token)
        dec = 0
        while "winnings" not in r:
            ts, p = to_state(r)
            seed = h * 1000 + dec
            a_gt, amt_gt, _ = decide(ts, seed)
            row = dict(hand=h, dec=dec, mode=mode, condition=condition if mode == "vision" else "",
                       client_pos=r["client_pos"], street=len(ts.board), action_str=r["action"],
                       gt_action=a_gt, winnings="")
            if mode == "gt":
                a, amt = a_gt, amt_gt
            else:
                img = noise(render(ts, theme, (0, 0)), np.random.default_rng(seed))
                t = time.perf_counter()
                seen, _ = naive.perceive(img)
                row["perceive_ms"] = round((time.perf_counter() - t) * 1000, 2)
                a, amt, _ = decide(seen, seed)
                seen_h, _ = hard.perceive(img)
                row["hard_action"] = decide(seen_h, seed)[0]
                row["state_correct"] = seen.as_dict() == ts.as_dict()
                row["hard_state_correct"] = seen_h.as_dict() == ts.as_dict()
            incr = to_incr(a, amt, ts, p)
            row.update(vis_action=a, incr=incr)
            old = token
            r = post("act", {"token": token, "incr": incr}, delay)
            token = r.get("token", token)
            row["token_changed"] = token != old
            if "winnings" in r:
                row["winnings"] = r["winnings"]
            w.writerow(row)
            dec += 1
        if dec == 0:      # hand ended before we acted (Slumbot folded SB)
            w.writerow(dict(hand=h, dec=-1, mode=mode, client_pos=r.get("client_pos"),
                            action_str=r.get("action"), winnings=r["winnings"]))
        f.flush()
        if (h + 1) % 50 == 0:
            print(f"{mode}: {h + 1}/{n_hands} hands", flush=True)
    f.close()


if __name__ == "__main__":
    a = sys.argv
    main(a[1], int(a[2]), a[3], a[4] if len(a) > 4 else "realistic_mix", float(a[5]) if len(a) > 5 else 0.4)
