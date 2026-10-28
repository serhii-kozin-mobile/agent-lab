"""Combine summary JSONs into results/summary.csv and a markdown table (results/summary_table.md)."""
import csv, json
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "results"
PIPES = {"baseline": ["bm25", "dense", "hybrid"], "sqlite": ["fts", "vec", "hybrid"],
         "memvid": ["lex-raw", "lex-or", "lex-or-score", "lex-or-score-deep", "sem", "hybrid-raw", "hybrid-or"]}
rows = []
for v in ["fixed", "ctx"]:
    for s, pipes in PIPES.items():
        d = json.load(open(R / f"{s}_{v}_summary.json"))
        for p in pipes:
            m = d[p]
            rows.append({"variant": v, "system": f"{s}:{p}", **{k: m[k] for k in ["hit@1", "hit@5", "hit@10", "mrr@10", "p50_ms", "p95_ms"]}})
with open(R / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
lines = ["| variant | system | hit@1 | hit@5 | hit@10 | MRR@10 | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|---|"]
lines += [f"| {r['variant']} | {r['system']} | {r['hit@1']} | {r['hit@5']} | {r['hit@10']} | {r['mrr@10']} | {r['p50_ms']} | {r['p95_ms']} |" for r in rows]
open(R / "summary_table.md", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
