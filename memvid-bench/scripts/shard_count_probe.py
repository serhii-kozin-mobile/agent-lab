"""Does one shard fit the free-tier cap with N=2 or 3 round-robin shards? Builds shard 0 only (ctx) in /tmp."""
import json, sys, time, os
import numpy as np
from memvid_sdk import create
VAR = os.environ.get("VARIANT", "ctx"); ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); LAB = os.path.join(ROOT, "prep", "data", "index", VAR)
chunks = [json.loads(l) for l in open(f"{LAB}/chunks.jsonl")]; emb = np.load(f"{LAB}/emb.npy")
res = {}
for N in map(int, sys.argv[1:]):
    p = f"/tmp/shardprobe_{N}.mv2"
    if os.path.exists(p): os.unlink(p)
    members = [i for i in range(len(chunks)) if i % N == 0]
    mv = create(p, enable_vec=True, enable_lex=True); done = 0; err = None; t = time.time()
    try:
        for s in range(0, len(members), 100):
            b = members[s:s+100]
            mv.put_many([{"title": "chunk", "label": "bench", "text": chunks[i]["index_text"], "uri": f"bench://{i}"} for i in b], embeddings=emb[b].tolist())
            done += len(b)
        mv.seal()
    except BaseException as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"
    res[N] = {"shard_chunks": len(members), "ingested_before_stop": done, "error": err,
              "bytes": os.path.getsize(p) if os.path.exists(p) else None, "seconds": round(time.time()-t, 1)}
    print(N, res[N], flush=True)
    if os.path.exists(p): os.unlink(p)
json.dump(res, open(os.path.join(ROOT, "results", f"shard_count_probe_{VAR}.json"), "w"), indent=1)
