"""hit@5 by question category (single 70, table 16, multihop 12, neardup 12) for the main systems."""
import csv, json, collections
R = str(__import__("pathlib").Path(__file__).resolve().parent.parent) + "/results"; out = {}
for v in ["fixed", "ctx"]:
    for f in ["baseline", "sqlite", "memvid"]:
        for r in csv.DictReader(open(f"{R}/{f}_{v}_per_query.csv")):
            k = (v, r["system"], r["category"]); out.setdefault(k, []).append(1 if r["first_hit_rank"] and int(r["first_hit_rank"]) <= 5 else 0)
res = {f"{v}|{s}|{c}": {"n": len(x), "hit@5": round(sum(x)/len(x), 3)} for (v, s, c), x in sorted(out.items())}
json.dump(res, open(f"{R}/by_category_hit5.json", "w"), indent=1)
for k, x in res.items():
    if any(s in k for s in ["sqlite-fts", "sqlite-vec", "memvid-sem", "memvid-lex-or-score-deep", "baseline-bm25"]): print(k, x)
