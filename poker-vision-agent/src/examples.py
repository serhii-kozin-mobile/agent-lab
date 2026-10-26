"""Save one example screenshot per noise condition (same game state)."""
import os
import warnings

import numpy as np

from noise import CONDITIONS
from perception_bench import collect_states
from seeds import stable_seed
from table import render

warnings.filterwarnings("ignore")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "charts", "examples")
os.makedirs(out, exist_ok=True)
states = collect_states(40, seed=11)
ts, off, i, d = next(s for s in states if len(s[0].board) >= 3 and s[0].villain_bet > 0) if any(
    len(s[0].board) >= 3 and s[0].villain_bet > 0 for s in states) else states[-1]
for cond, (theme, fn) in CONDITIONS.items():
    img = fn(render(ts, theme, off), np.random.default_rng(stable_seed("example", cond)))
    img.save(os.path.join(out, f"{cond}.png"))
print("saved", len(CONDITIONS), "examples to", out)
