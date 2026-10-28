"""Optional single-file comparator: SQLite FTS5 (BM25) + sqlite-vec.

Run with the project venv:
  .venv/bin/python scripts/bench_sqlite.py --variant fixed

One .db file holds chunk text, FTS5 lexical index and float vectors —
the same "single portable file" proposition as .mv2, built from stdlib
SQLite. Identical chunks, identical precomputed bge-small vectors and
query vectors, identical gold-span hit rule as every other system here.
Modes: fts (FTS5 BM25, OR-joined lab tokens), vec (cosine KNN via
sqlite-vec), hybrid (RRF k=60 over both top-50 lists, as in the lab).
"""
import argparse
import csv
import json
import re
import resource
import sqlite3
import time
from pathlib import Path

import numpy as np
import sqlite_vec

ROOT = Path(__file__).resolve().parent.parent
LAB = Path(__file__).resolve().parent.parent / "prep"
K = 10
TOK = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
STOP = set("a an and are as at be by for from has have how in is it its of on or that the this to was what when where which who why will with does do can should i".split())


def lab_tokens(t):
    return [w for w in TOK.findall(t.lower()) if w not in STOP]


def chunk_hits_span(c, span):
    did, st, en = span
    if c["doc_id"] != did:
        return False
    ov = min(c["end"], en) - max(c["start"], st)
    return ov > 0 and ov >= 0.5 * min(en - st, c["end"] - c["start"])


def first_hit_rank(idxs, chunks, spans):
    for r, i in enumerate(idxs, 1):
        if any(chunk_hits_span(chunks[i], s) for s in spans):
            return r
    return None


def summarise(rows):
    ranks = [r["first_hit_rank"] for r in rows]
    hit = lambda k: float(np.mean([1.0 if (x != "" and x <= k) else 0.0 for x in ranks]))
    mrr = float(np.mean([1.0 / x if x != "" else 0.0 for x in ranks]))
    lat = [r["latency_ms"] for r in rows]
    return {"n": len(rows), "hit@1": round(hit(1), 3), "hit@5": round(hit(5), 3),
            "hit@10": round(hit(10), 3), "mrr@10": round(mrr, 3),
            "p50_ms": round(float(np.percentile(lat, 50)), 2),
            "p95_ms": round(float(np.percentile(lat, 95)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fixed", choices=["fixed", "ctx"])
    args = ap.parse_args()
    variant = args.variant

    idx_dir = LAB / "data/index" / variant
    chunks = [json.loads(l) for l in open(idx_dir / "chunks.jsonl")]
    emb = np.load(idx_dir / "emb.npy")
    queries = json.load(open(ROOT / "data/queries.json"))
    qemb = np.load(ROOT / "data/query_emb.npy")

    db_path = ROOT / "data" / f"sqlite_{variant}.db"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)

    t0 = time.perf_counter()
    db.execute("CREATE TABLE chunks(idx INTEGER PRIMARY KEY, text TEXT)")
    db.execute("CREATE VIRTUAL TABLE fts USING fts5(text, content='chunks', content_rowid='idx')")
    db.execute("CREATE VIRTUAL TABLE vec USING vec0(embedding float[384])")
    db.executemany("INSERT INTO chunks(idx, text) VALUES (?, ?)",
                   [(i, c["index_text"]) for i, c in enumerate(chunks)])
    db.execute("INSERT INTO fts(fts) VALUES ('rebuild')")
    db.executemany("INSERT INTO vec(rowid, embedding) VALUES (?, ?)",
                   [(i, emb[i].tobytes()) for i in range(len(chunks))])
    db.commit()
    build_s = time.perf_counter() - t0
    db_bytes = db_path.stat().st_size
    print(f"build_s={build_s:.2f} db_bytes={db_bytes}", flush=True)

    def search_fts(q, k):
        match = " OR ".join(f'"{t}"' for t in lab_tokens(q)) or '"emptyquery"'
        cur = db.execute("SELECT rowid FROM fts WHERE fts MATCH ? ORDER BY rank LIMIT ?",
                         (match, k))
        return [r[0] for r in cur.fetchall()]

    def search_vec(qi, k):
        cur = db.execute(
            "SELECT rowid FROM vec WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (qemb[qi].tobytes(), k))
        return [r[0] for r in cur.fetchall()]

    rows = []
    for label in ["fts", "vec", "hybrid"]:
        for qi, g in enumerate(queries):
            t0 = time.perf_counter()
            if label == "fts":
                idxs = search_fts(g["question"], K)
            elif label == "vec":
                idxs = search_vec(qi, K)
            else:
                a, b = search_fts(g["question"], 50), search_vec(qi, 50)
                sc = {}
                for lst in (a, b):
                    for r, i in enumerate(lst):
                        sc[i] = sc.get(i, 0.0) + 1.0 / (60 + r + 1)
                idxs = [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])[:K]]
            ms = (time.perf_counter() - t0) * 1000
            spans = [s for grp in g["gold_spans"] for s in grp]
            r = first_hit_rank(idxs, chunks, spans)
            rows.append({"system": f"sqlite-{label}", "variant": variant,
                         "qid": g["qid"], "category": g["category"],
                         "first_hit_rank": r or "", "latency_ms": round(ms, 3)})
        print(f"mode {label} done", flush=True)
    db.close()

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    out_csv = ROOT / "results" / f"sqlite_{variant}_per_query.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    summary = {"system": "sqlite", "variant": variant, "build_s": round(build_s, 2),
               "db_bytes": db_bytes, "peak_rss_bytes": int(peak_rss)}
    for label in ["fts", "vec", "hybrid"]:
        summary[label] = summarise([r for r in rows if r["system"] == f"sqlite-{label}"])
    with open(ROOT / "results" / f"sqlite_{variant}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
