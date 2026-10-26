"""Charts + sensitivity table from out/costs_per_run.csv and out/results.json."""
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

per = pd.read_csv("out/costs_per_run.csv")
calls = pd.read_csv("out/calls.csv")
res = json.load(open("out/results.json"))

# Sensitivity: reasoning models emit many more output tokens than these (non-reasoning) runs.
lines = ["| Output tokens x | Output share of bill (no cache) | Bill with cache (read 0.10, write 1.25) vs no cache |", "|---|---|---|"]
for m in (1, 3, 10):
    out = per.out_units.sum() * m
    base = per.no_cache.sum()
    c = per["cache_read0.10_write1.25"].sum()
    lines.append(f"| {m} | {out/(base+out):.1%} | {(c+out)/(base+out):.1%} |")
open("out/sensitivity.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# Chart 1: input cost per scenario, all runs
labels = {
    "no_cache": "No caching",
    "cache_read0.10_write1.25": "Cache: read 0.10x, write 1.25x",
    "cache_read0.05_write1.25": "Cache: read 0.05x, write 1.25x",
    "cache_read0.10_write2.00_1h": "Cache, 1-hour write 2x",
    "legacy_read0.50_write1.00": "2024-style: read 0.5x, no write fee",
    "miss10pct_read0.10_write1.25": "Cache, 10% of calls miss",
    "miss25pct_read0.10_write1.25": "Cache, 25% of calls miss",
    "volatile_prefix_write1.25": "Timestamp in system prompt",
}
s = res["harness"]["all"]["scenarios"]
names = list(labels)
vals = [s[n]["input_cost_vs_no_cache"] * 100 for n in names]
fig, ax = plt.subplots(figsize=(9, 4.8))
bars = ax.barh([labels[n] for n in names][::-1], vals[::-1], color="#4C8BF5")
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center")
ax.axvline(100, color="gray", ls="--", lw=1)
ax.set_xlabel("Input-token cost vs no caching (%)")
ax.set_title(f"Same {res['harness']['all']['runs']:,} real agent runs, priced under different cache rules")
plt.tight_layout(); plt.savefig("out/input_cost_by_scenario.png", dpi=150)

# Chart 2: how prompt size grows across calls (median per call index)
g = calls[calls.call <= 60].groupby("call").input_tokens.median()
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(g.index, g.values / 1000, color="#4C8BF5")
ax.set_xlabel("Model call number within a run"); ax.set_ylabel("Median prompt size (k tokens)")
ax.set_title("Every call re-sends the whole history")
plt.tight_layout(); plt.savefig("out/prompt_growth.png", dpi=150)
print("charts ok")
