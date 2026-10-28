"""Is Memvid lex top-k exact? Compare top score at k=30 vs k=300 per shard-0 query (both variants)."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bench_sqlite import lab_tokens
from memvid_sdk import use
ROOT = Path(__file__).resolve().parent.parent; out = {}
qs = json.load(open(ROOT / "data/queries.json"))
for v in ("ctx", "fixed"):
    h = use("basic", str(ROOT / f"data/memvid_{v}_shard0.mv2"), mode="open", read_only=True)
    c = {"n": 0, "higher_top_at_300": 0, "same_top_score_diff_chunk": 0, "same_top": 0}
    for g in qs:
        q = " OR ".join(lab_tokens(g["question"]))
        try:
            a = h.find(q, k=30, mode="lex", snippet_chars=40)["hits"]; b = h.find(q, k=300, mode="lex", snippet_chars=40)["hits"]
        except BaseException: continue
        if not a or not b: continue
        c["n"] += 1
        if b[0]["score"] > a[0]["score"] + 1e-6: c["higher_top_at_300"] += 1
        elif b[0]["uri"] != a[0]["uri"]: c["same_top_score_diff_chunk"] += 1
        else: c["same_top"] += 1
    out[v] = c; print(v, c, flush=True)
json.dump(out, open(ROOT / "results/topk_exactness.json", "w"), indent=1)
