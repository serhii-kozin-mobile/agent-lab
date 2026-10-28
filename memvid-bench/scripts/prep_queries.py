"""Export query embeddings for the 110 answerable golden questions.

Run with the lab venv (has sentence-transformers):
  prep/.venv/bin/python scripts/prep_queries.py

Uses the lab's own embedding model and query prefix so Memvid semantic
search gets byte-identical query vectors to the baseline dense pipeline.
Writes data/queries.json (qid, question, category, gold_spans) and
data/query_emb.npy (110 x 384 float32, L2-normalised).
"""
import json
import sys
from pathlib import Path

import numpy as np

LAB = Path(__file__).resolve().parent.parent / "prep"
OUT = Path(__file__).resolve().parent.parent / "data"
sys.path.insert(0, str(LAB / "lab"))
import retrieval  # noqa: E402  (lab code, read-only use)

golden = json.load(open(LAB / "data/golden.json"))
answerable = [g for g in golden if g["category"] != "noanswer"]
print(f"answerable: {len(answerable)}")

emb = retrieval.embedder().encode(
    [retrieval.QUERY_PREFIX + g["question"] for g in answerable],
    normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False,
).astype(np.float32)
print("query emb:", emb.shape, emb.dtype)

OUT.mkdir(parents=True, exist_ok=True)
np.save(OUT / "query_emb.npy", emb)
with open(OUT / "queries.json", "w") as fh:
    json.dump([{ "qid": g["qid"], "question": g["question"],
                 "category": g["category"], "gold_spans": g["gold_spans"]}
               for g in answerable], fh)
print("wrote", OUT / "query_emb.npy", OUT / "queries.json")
