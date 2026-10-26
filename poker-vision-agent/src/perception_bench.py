"""Perception benchmark: real game states (from ground-truth play) rendered
under every condition and read by both vision modes. Field-level accuracy,
card confusion pairs, digit errors, latency."""
from __future__ import annotations
from seeds import stable_seed

import json
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from agent import FIELDS
from game import hand_setup, play_hand
from noise import CONDITIONS
from table import render, TableState
from vision import Vision

warnings.filterwarnings("ignore")


def collect_states(n_states, seed):
    states = []

    def grab(ts, i, d, ctx):
        states.append((ts, ctx["setup"]["offset"], i, d))
        from policy import decide
        a, amt, _ = decide(ts, i * 1000 + d)
        return a, amt

    i = 0
    while len(states) < n_states:
        play_hand(i, seed, grab)
        i += 1
    return states[:n_states]


def run(n_states, seed, out_csv):
    states = collect_states(n_states, seed)
    rows = []
    for mode in ("naive", "hardened"):
        v = Vision(mode)
        for cond, (theme, fn) in CONDITIONS.items():
            for k, (ts, off, i, d) in enumerate(states):
                rng = np.random.default_rng(stable_seed(seed, "bench", k, cond))
                img = fn(render(ts, theme, off), rng)
                t = time.perf_counter()
                p, conf = v.perceive(img)
                ms = (time.perf_counter() - t) * 1000
                true, seen = ts.as_dict(), p.as_dict()
                row = dict(mode=mode, condition=cond, k=k, ms=ms, state_correct=true == seen)
                # card-level detail
                tc = true["hero_cards"] + true["board"]
                sc = seen["hero_cards"] + seen["board"]
                row["cards_total"] = len(tc)
                row["cards_correct"] = sum(1 for a, b in zip(tc, sc) if a == b)
                row["card_errors"] = ";".join(f"{a}>{b}" for a, b in zip(tc, sc) if a != b)
                row["board_len_true"], row["board_len_seen"] = len(true["board"]), len(seen["board"])
                for f in FIELDS:
                    row[f"ok_{f}"] = true[f] == seen[f]
                    if f not in ("hero_cards", "board", "button"):
                        row[f"true_{f}"], row[f"seen_{f}"] = true[f], seen[f]
                row["min_card_conf"] = min(v_ for kk, v_ in conf.items() if kk[-1].isdigit())
                row["min_num_conf"] = min(conf[kk] for kk in ("pot", "hero_stack", "villain_stack", "hero_bet", "villain_bet") if kk in conf)
                rows.append(row)
            print(f"bench {mode:8s} {cond:18s} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    run(n, seed=11, out_csv=sys.argv[2] if len(sys.argv) > 2 else "results/perception_bench.csv")
