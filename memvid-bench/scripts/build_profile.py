"""Where does Memvid build time go? 1,000 ctx chunks with supplied vectors: time put_many vs seal()."""
import json, time, tempfile, os, numpy as np, psutil
from memvid_sdk import create
LAB = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "prep/data/index/ctx")
ch = [json.loads(l) for _, l in zip(range(1000), open(f"{LAB}/chunks.jsonl"))]
emb = np.load(f"{LAB}/emb.npy")[:1000]
p = os.path.join(tempfile.mkdtemp(), "prof.mv2"); proc = psutil.Process()
t = time.perf_counter(); mv = create(p, enable_vec=True, enable_lex=True); t_create = time.perf_counter() - t
t = time.perf_counter()
for s in range(0, 1000, 100):
    mv.put_many([{"title": "chunk", "label": "bench", "text": c["index_text"], "uri": f"bench://{s+i}"} for i, c in enumerate(ch[s:s+100])], embeddings=emb[s:s+100].tolist())
t_put = time.perf_counter() - t
t = time.perf_counter(); mv.seal(); t_seal = time.perf_counter() - t
mods = [m.path for m in proc.memory_maps() if "onnx" in m.path.lower()]
out = {"chunks": 1000, "create_s": round(t_create, 3), "put_many_s": round(t_put, 2), "seal_s": round(t_seal, 2), "rss_mb": round(proc.memory_info().rss/1e6), "onnx_libs_mapped": len(mods)}
print(out); json.dump(out, open(str(__import__("pathlib").Path(__file__).resolve().parent.parent) + "/results/build_profile_ctx1000.json", "w"), indent=1)
