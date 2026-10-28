import json, sys, time
sys.path.insert(0, "lab")
import chunking, retrieval
import torch
torch.manual_seed(0)
docs = chunking.load_docs()
all_stats = []
for v in ["fixed", "struct", "ctx"]:
    t0 = time.time()
    chunks = chunking.build(v, docs)
    _, st = retrieval.Index.build(v, chunks)
    st["chunking_s"] = round(time.time() - t0 - st["bm25_build_s"] - st["dense_build_s"], 2)
    st["total_build_s"] = round(time.time() - t0, 2)
    print(st, flush=True)
    all_stats.append(st)
json.dump({"cpu_threads": torch.get_num_threads(), "embedding_model": retrieval.EMB_MODEL, "variants": all_stats},
          open("results/index_stats.json", "w"), indent=2)
