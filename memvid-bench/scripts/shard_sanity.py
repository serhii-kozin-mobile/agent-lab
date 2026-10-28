"""Sanity check: is Memvid's weak lexical score caused by our 4-shard workaround?
Compare ONE Memvid shard (ctx shard0, chunks i%4==0) against SQLite FTS5 built on exactly the same
chunk subset, with the same OR-joined lab tokens, on the questions whose gold span is reachable in that shard."""
import json, re, sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bench_sqlite import lab_tokens, first_hit_rank
from memvid_sdk import use
ROOT = Path(__file__).resolve().parent.parent; LAB = ROOT / "prep"
chunks = [json.loads(l) for l in open(LAB / "data/index/ctx/chunks.jsonl")]
queries = json.load(open(ROOT / "data/queries.json"))
members = [i for i in range(len(chunks)) if i % 4 == 0]
db = sqlite3.connect(":memory:")
db.execute("CREATE VIRTUAL TABLE fts USING fts5(text)")
db.executemany("INSERT INTO fts(rowid,text) VALUES(?,?)", [(i, chunks[i]["index_text"]) for i in members])
h = use("basic", str(ROOT / "data/memvid_ctx_shard0.mv2"), mode="open", read_only=True)
def idx(hit):
    return int((hit.get("uri") or "").split("://",1)[-1].split("#",1)[0])
res = {"memvid": [], "sqlite": []}; n = 0
for g in queries:
    spans = [s for grp in g["gold_spans"] for s in grp]
    if not first_hit_rank(members, chunks, spans):  # gold not in this shard
        continue
    n += 1; q = " OR ".join(lab_tokens(g["question"]))
    try: hits = h.find(q, k=int(__import__("os").environ.get("KLEX","30")), mode="lex", snippet_chars=80)["hits"]
    except BaseException: hits = []
    seen = []
    for x in hits:
        i = idx(x)
        if i not in seen: seen.append(i)
    res["memvid"].append(first_hit_rank(seen[:10], chunks, spans))
    fq = " OR ".join('"%s"' % t for t in lab_tokens(g["question"]))
    s = [r[0] for r in db.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT 10", (fq,))]
    res["sqlite"].append(first_hit_rank(s, chunks, spans))
out = {"n_questions_reachable_in_shard0": n}
for k, v in res.items():
    out[k] = {f"hit@{c}": round(sum(1 for r in v if r and r <= c)/n, 3) for c in (1,5,10)}
import os; print(json.dumps(out, indent=1)); json.dump(out, open(ROOT / ("results/shard_sanity_ctx_k%s.json" % os.environ.get("KLEX","30")),"w"), indent=1)
