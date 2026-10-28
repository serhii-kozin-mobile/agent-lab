"""Paired bootstrap 95% CI (10k resamples, seed 0) for hit@5 differences A-B on the 110 questions."""
import csv, json, random, sys
R = str(__import__("pathlib").Path(__file__).resolve().parent.parent) + "/results"
def load(v):
    d = {}
    for f in ["baseline", "sqlite", "memvid"]:
        for r in csv.DictReader(open(f"{R}/{f}_{v}_per_query.csv")):
            rk = r["first_hit_rank"]; d.setdefault(r["system"], {})[r["qid"]] = 1 if rk and int(rk) <= 5 else 0
    return d
pairs = [("memvid-lex-or-score","sqlite-fts"),("memvid-lex-or-score-deep","sqlite-fts"),("memvid-lex-or-score","baseline-bm25"),("memvid-lex-raw","sqlite-fts"),
         ("memvid-sem","sqlite-vec"),("memvid-sem","baseline-dense"),("memvid-hybrid-raw","sqlite-hybrid"),("sqlite-fts","baseline-bm25")]
out = {}
for v in ["fixed","ctx"]:
    d = load(v); rnd = random.Random(0)
    for a,b in pairs:
        if a not in d or b not in d: continue
        q = sorted(d[a]); diffs = [d[a][x]-d[b][x] for x in q]; n=len(q)
        bs = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n))/n for _ in range(10000))
        out[f"{v}:{a}-{b}"] = {"diff": round(sum(diffs)/n,3), "ci95": [round(bs[250],3), round(bs[9749],3)]}
print(json.dumps(out, indent=1)); json.dump(out, open(f"{R}/paired_ci_hit5.json","w"), indent=1)
