"""Benchmark Memvid (memvid-sdk) on the lab's chunks and golden set.

Run with the project venv:
  .venv/bin/python scripts/bench_memvid.py --variant fixed
  .venv/bin/python scripts/bench_memvid.py --variant ctx

Method (identical chunks / embeddings / gold matching as the lab baseline):
  * One Memvid frame per lab chunk. Frame text = chunk["index_text"], so
    lexical and vector indexes see exactly the text the baseline indexes.
  * No per-chunk metadata is stored: Memvid appends title/uri/metadata to
    the indexed text (verified in smoke tests), so metadata would leak
    doc-identifying tokens into the lexical index. Hits are mapped back to
    chunks via the frame URI "bench://<chunk_index>" (frame_id is the
    fallback). Title is the constant "chunk" for every frame.
  * Document vectors are the lab's prebuilt bge-small-en-v1.5 embeddings
    (emb.npy, L2-normalised) passed via put_many(embeddings=...); query
    vectors come from scripts/prep_queries.py (same model + query prefix).
    Build time therefore excludes embedding computation on both sides.
  * Modes: lex (Tantivy BM25), sem (HNSW vector), hybrid (core fusion;
    requires query_embedding, otherwise it silently falls back to lex).
  * Gold hit rule copied from lab/scripts/eval_retrieval.py: a retrieved
    chunk hits if it overlaps a gold span by >= half the shorter span.
"""
import argparse
import csv
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
LAB = Path(__file__).resolve().parent.parent / "prep"
K = 10
# put_many cost is superlinear in batch size (1000-doc batch did not finish
# in 280 s; 100-doc batches are steady at ~3.8 s). Keep batches small.
BATCH = 100


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
    ap.add_argument("--reuse-build", action="store_true",
                    help="reuse existing shard .mv2 files (skip ingest)")
    args = ap.parse_args()
    variant = args.variant
    proc = psutil.Process()

    idx_dir = LAB / "data/index" / variant
    chunks = [json.loads(l) for l in open(idx_dir / "chunks.jsonl")]
    emb = np.load(idx_dir / "emb.npy")
    queries = json.load(open(ROOT / "data/queries.json"))
    qemb = np.load(ROOT / "data/query_emb.npy")
    assert len(chunks) == emb.shape[0] and len(queries) == qemb.shape[0]
    print(f"variant={variant} chunks={len(chunks)} emb={emb.shape} queries={len(queries)}", flush=True)

    from memvid_sdk import create, use
    import memvid_sdk
    print("memvid_sdk version:", getattr(memvid_sdk, "__version__", "unknown"), flush=True)

    # Single-file ingest of all 8112 chunks fails on the free tier:
    # CapacityExceededError (50 MB cap) at 5000/8112 chunks, no API key
    # available. Fallback that still covers the full corpus with identical
    # chunks/vectors: NSHARDS round-robin shards (chunk i -> shard i % N),
    # queried jointly and merged (sem: by cosine score, which is comparable
    # across shards; lex/hybrid: reciprocal-rank fusion, k=60 as in the lab,
    # because BM25/fusion scores are not comparable across shards).
    NSHARDS = 4
    shard_of = [i % NSHARDS for i in range(len(chunks))]

    # ---- build (one .mv2 per shard) ----
    t_build0 = time.perf_counter()
    shard_paths, shard_stats = [], []
    for sh in range(NSHARDS):
        mv2 = ROOT / "data" / f"memvid_{variant}_shard{sh}.mv2"
        if args.reuse_build and mv2.exists():
            _h = use("basic", str(mv2), mode="open", read_only=True)
            shard_stats.append(_h.stats())
            _h.close() if hasattr(_h, "close") else None
            shard_paths.append(mv2)
            continue
        if mv2.exists():
            mv2.unlink()
        mv = create(str(mv2), enable_vec=True, enable_lex=True)
        members = [i for i in range(len(chunks)) if shard_of[i] == sh]
        for s in range(0, len(members), BATCH):
            idxs_b = members[s:s + BATCH]
            docs = [{"title": "chunk", "label": "bench",
                     "text": chunks[i]["index_text"], "uri": f"bench://{i}"}
                    for i in idxs_b]
            mv.put_many(docs, embeddings=emb[idxs_b].tolist())
        mv.seal()
        shard_stats.append(mv.stats())
        shard_paths.append(mv2)
        mv.close() if hasattr(mv, "close") else None
        print(f"  shard {sh} done: {len(members)} chunks, {mv2.stat().st_size} bytes", flush=True)
    build_s = time.perf_counter() - t_build0
    mv2_bytes = sum(p.stat().st_size for p in shard_paths)
    stats = {"frame_count": sum(s.get("frame_count", 0) for s in shard_stats),
             "lex_index_bytes": sum(s.get("lex_index_bytes", 0) for s in shard_stats),
             "vec_index_bytes": sum(s.get("vec_index_bytes", 0) for s in shard_stats)}
    rss_after_build = proc.memory_info().rss
    print(f"build_s={build_s:.2f} total_mv2_bytes={mv2_bytes} stats={stats}", flush=True)

    # ---- query (fresh read-only handles, like a cold consumer) ----
    t0 = time.perf_counter()
    handles = [use("basic", str(p), mode="open", read_only=True) for p in shard_paths]
    open_s = time.perf_counter() - t0
    for h in handles:
        h.find("warm up query about python logging", k=K, mode="lex")

    def hit_idx(h):
        # Memvid may split a frame into page sub-frames, returning URIs like
        # "7976#page-2" instead of "bench://7976". Normalise to the chunk no.
        uri = h.get("uri") or ""
        core = uri.split("://", 1)[-1].split("#", 1)[0]
        try:
            return int(core)
        except ValueError:
            return int(h["frame_id"])

    import re as _re
    def safe_find(handle, question, kw):
        # find() feeds the raw string to the Tantivy query parser: golden
        # questions containing ( ) [ ] " : etc. raise MV999 "Invalid query".
        # Retry once with parser specials stripped; caller counts fallbacks.
        # Returns (result_or_None, fell_back, error_str).
        try:
            return handle.find(question, **kw), False, ""
        except BaseException as e1:  # Rust panics surface as PanicException (BaseException)
            clean = _re.sub(r"[()\[\]{}^\"~*?:\\/+]", " ", question)
            try:
                return handle.find(clean, **kw), True, ""
            except BaseException as e2:
                return None, True, f"{type(e1).__name__}: {e1} | retry: {type(e2).__name__}: {e2}"

    # Memvid lex ANDs space-separated terms: a full natural question matches
    # nothing (measured). Baseline BM25 is OR-ranked. Report both the naive
    # default (raw question) and the OR form using the lab's exact tokenizer.
    _TOK = _re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
    _STOP = set("a an and are as at be by for from has have how in is it its of on or that the this to was what when where which who why will with does do can should i".split())
    def lab_tokens(t):
        return [w for w in _TOK.findall(t.lower()) if w not in _STOP]

    # (system label, core mode, query style)
    modes = [("lex-raw", "lex", "raw"), ("lex-or", "lex", "or"),
             ("sem", "sem", "raw"),
             ("hybrid-raw", "hybrid", "raw"), ("hybrid-or", "hybrid", "or"),
             # score-merge: pool shard hits by raw BM25 score instead of RRF.
             # RRF ties every shard's rank-1 hit, which unfairly hurts hit@1.
             ("lex-or-score", "lex", "or"),
             # deep: ask each shard for 300 lex hits (duplicates eat the list; see shard_sanity)
             ("lex-or-score-deep", "lex", "or")]
    rows = []
    for label, mode, style in modes:
        for qi, g in enumerate(queries):
            qtext = g["question"] if style == "raw" else " OR ".join(lab_tokens(g["question"]))
            # Ask lex shards for 30 and dedupe by chunk below: Memvid can
            # return the same frame several times in one hit list. Vector
            # modes stay at k=10: k=30 panics in the HNSW layer (measured).
            kw = {"k": (300 if label.endswith("-deep") else 30) if mode == "lex" else K, "snippet_chars": 240, "mode": mode}
            if mode in ("sem", "hybrid"):
                kw["query_embedding"] = qemb[qi].tolist()
            t0 = time.perf_counter()
            per_shard, fell_back, qerr = [], False, ""
            for h in handles:
                res, fb, err = safe_find(h, qtext, kw)
                if res is None:
                    qerr = err
                    res = {"hits": [], "engine": "error"}
                per_shard.append(res); fell_back = fell_back or fb
            ms = (time.perf_counter() - t0) * 1000
            if mode == "sem" or label.endswith(("-score", "-deep")):  # cosine scores are comparable across shards
                pooled = sorted(((hit_idx(h), h["score"]) for res in per_shard for h in res["hits"]),
                                key=lambda x: -x[1])
                seen, idxs = set(), []
                for i, _s in pooled:
                    if i not in seen:
                        seen.add(i); idxs.append(i)
                    if len(idxs) == K: break
            else:  # RRF merge (k=60, same constant as the lab hybrid)
                sc2 = {}
                for res in per_shard:
                    seen_r, rank = set(), 0
                    for h in res["hits"]:
                        i = hit_idx(h)
                        if i in seen_r: continue
                        seen_r.add(i); rank += 1
                        sc2[i] = sc2.get(i, 0.0) + 1.0 / (60 + rank)
                idxs = [i for i, _ in sorted(sc2.items(), key=lambda x: -x[1])[:K]]
            spans = [s for grp in g["gold_spans"] for s in grp]
            r = first_hit_rank(idxs, chunks, spans)
            rows.append({"system": f"memvid-{label}", "variant": variant,
                         "qid": g["qid"], "category": g["category"],
                         "first_hit_rank": r or "", "latency_ms": round(ms, 3),
                         "engine": per_shard[0].get("engine", ""), "n_hits": len(idxs),
                         "sanitized_retry": int(fell_back), "error": qerr})
        print(f"mode {label} done", flush=True)
    for h in handles:
        h.close() if hasattr(h, "close") else None

    peak_rss = max(rss_after_build, proc.memory_info().rss,
                   resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)

    out_csv = ROOT / "results" / f"memvid_{variant}_per_query.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary = {"system": "memvid", "variant": variant,
               "reused_build": bool(args.reuse_build),
               "build_s": round(build_s, 2),
               "open_s": round(open_s, 3), "mv2_bytes": mv2_bytes,
               "lex_index_bytes": stats.get("lex_index_bytes"),
               "vec_index_bytes": stats.get("vec_index_bytes"),
               "frame_count": stats.get("frame_count"),
               "peak_rss_bytes": int(peak_rss)}
    for label, mode, style in modes:
        summary[label] = summarise([r for r in rows if r["system"] == f"memvid-{label}"])
        summary[label]["sanitized_retries"] = int(sum(
            r["sanitized_retry"] for r in rows if r["system"] == f"memvid-{label}"))
        summary[label]["query_errors"] = int(sum(
            1 for r in rows if r["system"] == f"memvid-{label}" and r["error"]))
    with open(ROOT / "results" / f"memvid_{variant}_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
