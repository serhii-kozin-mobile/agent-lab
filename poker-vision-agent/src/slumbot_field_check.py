"""Which fields break on Slumbot-sized numbers (200bb stacks, 5-digit values)?

Slumbot's logged action strings give the exact chip numbers of every state we
saw; card identities are not logged, so cards are re-dealt at random (same
count). Renders each state under realistic_mix and reads it with both readers.
Offline: no network."""
import csv
import sys
import warnings

import numpy as np
import pandas as pd

from noise import CONDITIONS
from seeds import stable_seed
from slumbot_client import STACK, parse
from table import RANKS, SUITS, TableState, render
from vision import Vision

warnings.filterwarnings("ignore")
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["dec"] not in ("", "-1")]
theme, fn = CONDITIONS["realistic_mix"]
readers = {m: Vision(m) for m in ("naive", "hardened")}
out = []
for k, r in enumerate(rows):
    me, p = int(r["client_pos"]), parse(r["action_str"])
    rng = np.random.default_rng(stable_seed("slumbot_check", k))
    deck = [a + b for a in RANKS for b in SUITS]
    cards = list(rng.choice(deck, 2 + int(r["street"]), replace=False))
    ts = TableState(cards[:2], cards[2:], p["pot"], STACK - p["total"][me], STACK - p["total"][1 - me],
                    p["street"][me], p["street"][1 - me], "hero" if me == 1 else "villain")
    img = fn(render(ts, theme, (0, 0)), rng)
    for m, v in readers.items():
        seen, _ = v.perceive(img)
        t, s = ts.as_dict(), seen.as_dict()
        out.append(dict(mode=m, k=k, state_correct=t == s, **{f"ok_{f}": t[f] == s[f] for f in t}))
d = pd.DataFrame(out)
res = d.groupby("mode").mean(numeric_only=True).drop(columns="k").round(3)
print(res.T.to_string())
res.to_csv(sys.argv[2])
