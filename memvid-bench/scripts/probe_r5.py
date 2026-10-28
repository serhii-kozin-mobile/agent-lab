"""r5 probes on existing ctx shards: determinism, k-dependence of rank 1,
duplicate URIs across all 110 questions, MV999 per variant, exact KNN vs sem."""
import json, re, sys, collections
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from bench_sqlite import lab_tokens, first_hit_rank
from memvid_sdk import use
ROOT = Path(__file__).resolve().parent.parent; LAB = ROOT / "prep"
out = {}
for variant in ("ctx", "fixed"):
    chunks = [json.loads(l) for l in open(LAB / f"data/index/{variant}/chunks.jsonl")]
    emb = np.load(LAB / f"data/index/{variant}/emb.npy")
    qs = json.load(open(ROOT / "data/queries.json")); qemb = np.load(ROOT / "data/query_emb.npy")
    hs = [use("basic", str(ROOT / f"data/memvid_{variant}_shard{s}.mv2"), mode="open", read_only=True) for s in range(4)]
    idx = lambda x: int((x.get("uri") or "").split("://",1)[-1].split("#",1)[0])
    V = {"dup": {}, "errors": {}, "determinism_mismatch": 0, "rank1_changes_k": 0, "top1_examples": []}
    for k in (30, 100, 300):
        tot = uniq = 0; pagey = 0; errs = []
        for g in qs:
            q = " OR ".join(lab_tokens(g["question"]))
            for s, h in enumerate(hs):
                try: hits = h.find(q, k=k, mode="lex", snippet_chars=80)["hits"]
                except BaseException as e: errs.append((g["qid"], s, str(e)[:120])); continue
                tot += len(hits); uniq += len({idx(x) for x in hits})
                pagey += sum(1 for x in hits if "#" in (x.get("uri") or ""))
        V["dup"][k] = {"total_hits": tot, "unique_chunks": uniq, "uris_with_#page": pagey}
        V["errors"][k] = errs
    # raw-question errors (MV999) per variant
    raw_err = []
    for g in qs:
        for s, h in enumerate(hs):
            try: h.find(g["question"], k=10, mode="lex", snippet_chars=80)
            except BaseException as e: raw_err.append((g["qid"], s, str(e)[:160]))
    V["raw_errors"] = raw_err
    # determinism + rank-1 vs k on shard0
    for g in qs:
        q = " OR ".join(lab_tokens(g["question"]))
        try:
            a = [(idx(x), round(x["score"],4)) for x in hs[0].find(q, k=30, mode="lex", snippet_chars=80)["hits"]]
            b = [(idx(x), round(x["score"],4)) for x in hs[0].find(q, k=30, mode="lex", snippet_chars=80)["hits"]]
            c = [(idx(x), round(x["score"],4)) for x in hs[0].find(q, k=300, mode="lex", snippet_chars=80)["hits"]]
        except BaseException: continue
        if a != b: V["determinism_mismatch"] += 1
        if a and c and a[0] != c[0]:
            V["rank1_changes_k"] += 1
            if len(V["top1_examples"]) < 3: V["top1_examples"].append({"qid": g["qid"], "k30_top3": a[:3], "k300_top3": c[:3]})
    # exact KNN vs sem on shard0
    mem = np.array([i for i in range(len(chunks)) if i % 4 == 0]); E = emb[mem]
    agree1 = overlap10 = n = 0; maxdiff = 0.0
    for qi in range(len(qs)):
        sims = E @ qemb[qi]; ex = list(mem[np.argsort(-sims)[:10]])
        try: hits = hs[0].find(qs[qi]["question"], k=10, mode="sem", query_embedding=qemb[qi].tolist(), snippet_chars=80)["hits"]
        except BaseException: continue
        got = []
        for x in hits:
            if idx(x) not in got: got.append(idx(x))
        n += 1; agree1 += int(bool(got) and got[0] == ex[0]); overlap10 += len(set(got[:10]) & set(ex))
    V["knn_vs_sem_shard0"] = {"n": n, "top1_agree": agree1, "mean_overlap@10": round(overlap10 / max(n,1), 2)}
    out[variant] = V
    print(variant, json.dumps({k: v for k, v in V.items() if k not in ("errors",)}, indent=1)[:3000], flush=True)
json.dump(out, open(ROOT / "results/probe_r5.json", "w"), indent=1, default=str)
