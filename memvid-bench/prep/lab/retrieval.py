"""Retrieval pipelines: BM25, dense, hybrid (RRF), hybrid + cross-encoder rerank."""
import json, os, pickle, re, time
from pathlib import Path
import numpy as np

EMB_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODELS = {"rerank": "cross-encoder/ms-marco-MiniLM-L-6-v2"}
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
INDEX_DIR = Path("data/index")
RRF_K = 60
CANDIDATES = 50      # per retriever, fed to RRF
RERANK_TOP = 30      # hybrid candidates passed to the cross-encoder

_TOK = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
STOP = set("a an and are as at be by for from has have how in is it its of on or that the this to was what when where which who why will with does do can should i".split())


def tokenize(t):
    return [w for w in _TOK.findall(t.lower()) if w not in STOP]


_models = {}
def embedder():
    if "emb" not in _models:
        from sentence_transformers import SentenceTransformer
        _models["emb"] = SentenceTransformer(EMB_MODEL, device="cpu")
    return _models["emb"]


def reranker(name="rerank"):
    if name not in _models:
        from sentence_transformers import CrossEncoder
        _models[name] = CrossEncoder(RERANK_MODELS[name], device="cpu", max_length=512)
    return _models[name]


class Index:
    def __init__(self, variant, chunks, bm25, emb):
        self.variant, self.chunks, self.bm25, self.emb = variant, chunks, bm25, emb

    @classmethod
    def build(cls, variant, chunks):
        from rank_bm25 import BM25Okapi
        d = INDEX_DIR / variant
        d.mkdir(parents=True, exist_ok=True)
        stats = {"variant": variant, "chunks": len(chunks)}
        t0 = time.time()
        bm25 = BM25Okapi([tokenize(c["index_text"]) for c in chunks])
        stats["bm25_build_s"] = round(time.time() - t0, 2)
        t0 = time.time()
        emb = embedder().encode([c["index_text"] for c in chunks], batch_size=64, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        stats["dense_build_s"] = round(time.time() - t0, 2)
        np.save(d / "emb.npy", emb)
        with open(d / "bm25.pkl", "wb") as fh:
            pickle.dump(bm25, fh)
        with open(d / "chunks.jsonl", "w") as fh:
            for c in chunks:
                fh.write(json.dumps(c) + "\n")
        stats["emb_bytes"] = os.path.getsize(d / "emb.npy")
        stats["bm25_bytes"] = os.path.getsize(d / "bm25.pkl")
        stats["chunks_bytes"] = os.path.getsize(d / "chunks.jsonl")
        return cls(variant, chunks, bm25, emb), stats

    @classmethod
    def load(cls, variant):
        d = INDEX_DIR / variant
        chunks = [json.loads(l) for l in open(d / "chunks.jsonl")]
        with open(d / "bm25.pkl", "rb") as fh:
            bm25 = pickle.load(fh)
        return cls(variant, chunks, bm25, np.load(d / "emb.npy"))

    # each search returns a list of (chunk_idx, score), best first
    def search_bm25(self, q, k=CANDIDATES, mask=None):
        s = self.bm25.get_scores(tokenize(q))
        if mask is not None: s = np.where(mask, s, -1e9)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]

    def search_dense(self, q, k=CANDIDATES, mask=None):
        qv = embedder().encode([QUERY_PREFIX + q], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)[0]
        s = self.emb @ qv
        if mask is not None: s = np.where(mask, s, -1e9)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]

    def search_hybrid(self, q, k=CANDIDATES, mask=None):
        a, b = self.search_bm25(q, CANDIDATES, mask), self.search_dense(q, CANDIDATES, mask)
        sc = {}
        for lst in (a, b):
            for r, (i, _) in enumerate(lst):
                sc[i] = sc.get(i, 0.0) + 1.0 / (RRF_K + r + 1)
        return sorted(sc.items(), key=lambda x: -x[1])[:k]

    def search_rerank(self, q, k=10, mask=None, model="rerank", base="hybrid"):
        cand = (self.search_bm25 if base == "bm25" else self.search_hybrid)(q, RERANK_TOP, mask)
        pairs = [(q, self.chunks[i]["index_text"]) for i, _ in cand]
        s = reranker(model).predict(pairs, batch_size=32, show_progress_bar=False)
        order = np.argsort(-np.asarray(s))
        return [(cand[j][0], float(s[j])) for j in order[:k]]

    def search(self, pipeline, q, k=10, mask=None):
        if pipeline == "rerank_bm25":  # cross-encoder over the BM25 top 30
            return self.search_rerank(q, k=k, mask=mask, base="bm25")
        if pipeline.startswith("rerank"):
            return self.search_rerank(q, k=k, mask=mask, model=pipeline)
        return {"bm25": self.search_bm25, "dense": self.search_dense,
                "hybrid": self.search_hybrid}[pipeline](q, k=k, mask=mask)
