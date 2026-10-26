"""Duplicate matches: same N seeded hands for the ground-truth agent and for
every (condition x agent-config). Writes per-hand and per-decision CSVs."""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import pandas as pd

warnings.filterwarnings("ignore")

from agent import GroundTruthAgent, VisionAgent
from game import play_hand
from noise import CONDITIONS

BASE_SEED = 2026
# Calibrated on the perception benchmark (seed 11, disjoint from match seed):
# roughly the 5th percentile of confidence on CORRECT hardened reads.
THRESHOLDS = {"card": 0.50, "number": 0.45}
CONFIGS = {
    "naive": dict(mode="naive", guard=False),
    "hardened_guarded": dict(mode="hardened", guard=True),
}


def job(args):
    kind, cond, cfg, n = args
    warnings.filterwarnings("ignore")
    if kind == "gt":
        agent = GroundTruthAgent()
    else:
        agent = VisionAgent(cond, base_seed=BASE_SEED, thresholds=THRESHOLDS, **CONFIGS[cfg])
    t = time.time()
    hands = []
    for i in range(n):
        r, d = play_hand(i, BASE_SEED, agent)
        hands.append(dict(hand=i, result_bb=r, hero_decisions=d))
    tag = "gt" if kind == "gt" else f"{cfg}__{cond}"
    print(f"{tag}: {n} hands in {time.time() - t:.0f}s", flush=True)
    return tag, hands, agent.log


def main(n, outdir, workers):
    os.makedirs(outdir, exist_ok=True)
    jobs = [("gt", None, None, n)] + [("vis", c, cfg, n) for cfg in CONFIGS for c in CONDITIONS]
    hand_rows, dec_rows = [], []
    with Pool(workers) as pool:
        for tag, hands, log in pool.imap_unordered(job, jobs):
            for h in hands:
                h["run"] = tag
            hand_rows += hands
            for r in log:
                r["run"] = tag
            dec_rows += log
    pd.DataFrame(hand_rows).to_csv(os.path.join(outdir, "hands.csv"), index=False)
    pd.DataFrame(dec_rows).to_csv(os.path.join(outdir, "decisions.csv.gz"), index=False)
    json.dump({"n_hands": n, "base_seed": BASE_SEED, "thresholds": THRESHOLDS}, open(os.path.join(outdir, "match_config.json"), "w"))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    main(n, sys.argv[2] if len(sys.argv) > 2 else "../results", int(sys.argv[3]) if len(sys.argv) > 3 else 6)
