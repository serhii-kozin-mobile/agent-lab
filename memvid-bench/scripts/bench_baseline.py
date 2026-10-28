"""Re-run the lab baseline (fixed/ctx) with the lab's own code, same metrics.

Run with the lab venv:
  prep/.venv/bin/python scripts/bench_baseline.py --variant fixed

Loads the prebuilt lab index (chunks.jsonl + emb.npy + bm25.pkl), evaluates
bm25 / dense / hybrid at k=10 on the 110 answerable golden questions using
the lab's hit rule, and records load time, BM25 rebuild time (fresh
BM25Okapi over the same chunks), per-query latency and peak RSS.
Embedding computation is excluded (prebuilt emb.npy), matching the Memvid
setup where identical precomputed vectors are supplied.
"""
import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent.parent / "prep"
ROOT = Path(__file__).resolve().parent.parent
K = 10
sys.path.insert(0, str(LAB / "lab"))
import retrieval  # noqa: E402


def chunk_hits_span(c, span):
    did, st, en = span
    if c["doc_id"] != did:
        return False
    ov = min(c["end"], en) - max(c["start"], st)
    return ov > 0 and ov >= 0.5 * min(en - st, c["end"] - c["start"])


def first_hit_rank(chs, spans):
    for r, c in enumerate(chs, 1):
        if any(chunk_hits_span(c, s) for s in spans):
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
    import os
    os.chdir(LAB)  # lab Index.load uses relative data/index paths (read-only)

    import time as _t
    t0 = _t.perf_counter()
    idx = retrieval.Index.load(variant)
    load_s = _t.perf_counter() - t0

    # fresh BM25 build over the same chunks (index-build cost excl. embeddings)
    from rank_bm25 import BM25Okapi
    t0 = _t.perf_counter()
    _bm = BM25Okapi([retrieval.tokenize(c["index_text"]) for c in idx.chunks])
    bm25_build_s = _t.perf_counter() - t0
    del _bm

    d = LAB / "data/index" / variant
    sizes = {f: (d / f).stat().st_size for f in ("chunks.jsonl", "emb.npy", "bm25.pkl")}

    queries = json.load(open(ROOT / "data/queries.json"))
    idx.search("bm25", "warm up query about python logging", k=K)
    idx.search("dense", "warm up query about python logging", k=K)

    rows = []
    for pipe in ["bm25", "dense", "hybrid"]:
        for g in queries:
            t0 = time.perf_counter()
            res = idx.search(pipe, g["question"], k=K)
            ms = (time.perf_counter() - t0) * 1000
            chs = [idx.chunks[i] for i, _ in res]
            spans = [s for grp in g["gold_spans"] for s in grp]
            r = first_hit_rank(chs, spans)
            rows.append({"system": f"baseline-{pipe}", "variant": variant,
                         "qid": g["qid"], "category": g["category"],
                         "first_hit_rank": r or "", "latency_ms": round(ms, 3)})
        print(f"pipeline {pipe} done", flush=True)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    out_csv = ROOT / "results" / f"baseline_{variant}_per_query.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = {"system": "baseline", "variant": variant,
               "load_s": round(load_s, 3), "bm25_build_s": round(bm25_build_s, 2),
               "index_bytes": sizes, "index_total_bytes": sum(sizes.values()),
               "peak_rss_bytes": int(peak_rss)}
    for pipe in ["bm25", "dense", "hybrid"]:
        summary[pipe] = summarise([r for r in rows if r["system"] == f"baseline-{pipe}"])
    with open(ROOT / "results" / f"baseline_{variant}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
