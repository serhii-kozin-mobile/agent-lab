"""Review check (2026-10-26): is the Elo baseline result an artefact of its settings?
Re-runs the Elo part of features.py over a grid of K and home advantage and scores RPS on 2025-26.
Writes out/elo_sensitivity.json. Seconds on a CPU."""
import json, numpy as np
src = open("features.py").read().split("# last season final table")[0]
def run(K, H):
    g = {"__file__": "features.py"}
    exec(src.replace("K, HFA = 20.0, 60.0", f"K, HFA = {K}, {H}"), g)
    P, Y = [], []
    for _, r in g["test"].iterrows():
        P.append(g["elo_probs"](r.HomeTeam, r.AwayTeam)); g["upd"](r.HomeTeam, r.AwayTeam, r.FTR)
        Y.append([r.FTR == k for k in "HDA"])
    P, Y = np.array(P), np.array(Y, float)
    return float((((np.cumsum(P, 1)[:, :2] - np.cumsum(Y, 1)[:, :2]) ** 2).sum(1) / 2).mean())
grid = {f"K={K},HFA={H}": run(float(K), float(H)) for K in (10, 15, 20, 25, 30, 40) for H in (0, 30, 60, 90, 120)}
v = [grid[f"K={K},HFA={H}"] for K in (10, 15, 20, 25, 30, 40) for H in (30, 60, 90)]
out = dict(grid=grid, used="K=20,HFA=60", rps_used=grid["K=20,HFA=60"], best=min(grid, key=grid.get),
           range_K10_40_HFA30_90=[min(v), max(v)], llm_stats_only_rps=0.2255)
json.dump(out, open("out/elo_sensitivity.json", "w"), indent=1); print(json.dumps(out, indent=1))
