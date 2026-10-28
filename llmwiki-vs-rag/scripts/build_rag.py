"""Build the plain-RAG baseline index: BM25 over fixed+header chunks (lab's best
retriever per rag_experiment.md: fixed windows + header, BM25), restricted to the
same 100-doc subset and the same raw text the wiki ingested.

Chunks: 200-word windows, 40-word overlap (lab `fixed`), index_text prefixed with
'Title > heading' (lab `fixedh`). Tokenizer/BM25: lab's (rank_bm25 BM25Okapi).
Also builds a BM25 index over wiki pages for system (c). Run after ingest.
"""
import json, pickle, re, sys
from pathlib import Path

sys.path.insert(0, __import__("os").environ.get("LAB_DIR", "lab"))
import retrieval  # noqa: E402  (lab tokenizer)
from rank_bm25 import BM25Okapi  # noqa: E402

ROOT = Path(".")
MAX_WORDS, OVERLAP = 200, 40


def canonical(doc):
    parts, offs, pos = [], [], 0
    for s in doc["sections"]:
        offs.append((pos, pos + len(s["text"]), s["heading"]))
        parts.append(s["text"])
        pos += len(s["text"]) + 2
    return "\n\n".join(parts), offs


def fixed_chunks(doc):
    text, offs = canonical(doc)
    ws = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out, i, n = [], 0, 0
    while i < len(ws):
        j = min(i + MAX_WORDS, len(ws))
        st, en = ws[i][0], ws[j - 1][1]
        sec = next((o for o in offs if o[0] <= st < o[1] + 2), None)
        heading = sec[2] if sec else ""
        head = f"{doc['title']} > {heading}" if heading and heading != doc["title"] else doc["title"]
        out.append({"chunk_id": f"{doc['doc_id']}#f{n}", "doc_id": doc["doc_id"],
                    "text": text[st:en], "index_text": head + "\n" + text[st:en]})
        n += 1
        if j == len(ws):
            break
        i = j - OVERLAP
    return out


def main():
    docs = [json.loads(l) for l in open(ROOT / "data/subset_docs.jsonl")]
    chunks = []
    for d in docs:
        chunks += fixed_chunks(d)
    bm25 = BM25Okapi([retrieval.tokenize(c["index_text"]) for c in chunks])
    out = ROOT / "data/rag_index.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"chunks": chunks, "bm25": bm25}, fh)
    print(f"rag chunks: {len(chunks)} from {len(docs)} docs -> {out}")

    pages = []
    for p in sorted((ROOT / "wiki").rglob("*.md")):
        if p.name in ("index.md", "log.md"):
            continue
        pages.append({"page": str(p.relative_to(ROOT / "wiki")), "text": p.read_text()})
    if not pages:
        print("wiki pages indexed: 0 (run after ingest)")
        return
    wbm25 = BM25Okapi([retrieval.tokenize(p["text"]) for p in pages])
    with open(ROOT / "data/wiki_index.pkl", "wb") as fh:
        pickle.dump({"pages": pages, "bm25": wbm25}, fh)
    print(f"wiki pages indexed: {len(pages)}")


if __name__ == "__main__":
    main()
