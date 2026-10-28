"""Time bge-small query encoding alone (baseline dense/hybrid latency includes it; Memvid got precomputed vectors)."""
import sys, time, json, statistics
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "prep/lab"))
import retrieval
q = json.load(open(str(__import__("pathlib").Path(__file__).resolve().parent.parent) + "/data/queries.json"))
m = retrieval.embedder(); m.encode(["warm up"], normalize_embeddings=True)
ts = []
for g in q:
    t = time.perf_counter(); m.encode([retrieval.QUERY_PREFIX + g["question"]], normalize_embeddings=True, show_progress_bar=False); ts.append((time.perf_counter()-t)*1000)
ts.sort(); out = {"n": len(ts), "p50_ms": round(statistics.median(ts),2), "p95_ms": round(ts[int(0.95*len(ts))-1],2)}
print(out); json.dump(out, open(str(__import__("pathlib").Path(__file__).resolve().parent.parent) + "/results/query_encode_timing.json","w"), indent=2)
