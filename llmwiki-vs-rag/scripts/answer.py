"""Answer questions with three systems (same answerer model for all):
  wiki     : two-step — pick up to 5 pages from wiki/index.md, then answer from those pages only.
  rag      : BM25 (fixed+header, subset) top-5 chunks, answer from chunks only.
  wiki_bm25: BM25 over wiki pages top-5 pages, answer from those pages only.
All systems: answer only from provided context, else reply exactly NOT_FOUND.
Writes results/answers_<system>[_<label>].csv + .jsonl raw log. Resumable per qid.
Usage: answer.py --system wiki|rag|wiki_bm25 --questions data/questions.json+data/noanswer.json [--wiki-dir wiki] [--label main] [--limit N]
"""
import argparse, csv, json, pickle, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, __import__("os").environ.get("LAB_DIR", "lab"))
import retrieval  # noqa: E402
from common import ROOT, call_answerer  # noqa: E402

ANSWER_RULE = ("Answer the question using ONLY the context below. Give a concise answer with exact values. "
               "Cite sources inline. If the context does not contain the answer, reply exactly NOT_FOUND.")


def topk(index, query, k):
    scores = index["bm25"].get_scores(retrieval.tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return order, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["wiki", "rag", "wiki_bm25", "wiki_full", "wiki_links"])
    ap.add_argument("--questions", nargs="+", default=["data/questions.json", "data/noanswer.json"])
    ap.add_argument("--wiki-dir", default="wiki")
    ap.add_argument("--label", default="main")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--ctx-cap", type=int, default=26000)  # r4: wiki_full at 36000 to match wiki_links
    args = ap.parse_args()

    questions = []
    for f in args.questions:
        questions += json.load(open(ROOT / f))
    if args.limit:
        questions = questions[: args.limit]
    wiki_dir = ROOT / args.wiki_dir
    schema = (ROOT / "CLAUDE.md").read_text()
    out_csv = ROOT / f"results/answers_{args.system}_{args.label}.csv"
    raw_log = ROOT / f"results/answers_{args.system}_{args.label}.jsonl"
    done = set()
    if raw_log.exists():
        done = {json.loads(l)["qid"] for l in open(raw_log)}
    todo = [q for q in questions if q["qid"] not in done]
    print(f"{args.system}/{args.label}: {len(done)} done, {len(todo)} todo", flush=True)

    rag_idx = pickle.load(open(ROOT / "data/rag_index.pkl", "rb")) if args.system == "rag" else None
    wiki_idx = pickle.load(open(ROOT / "data/wiki_index.pkl", "rb")) if args.system == "wiki_bm25" else None
    index_text = (wiki_dir / "index.md").read_text() if args.system.startswith("wiki") and args.system != "wiki_bm25" else ""

    IDX_CAP = 16000 if args.system == "wiki" else len(index_text) + 1  # r3: full index for wiki_full/wiki_links
    LINK_CAP_PAGES, LINK_CAP_CHARS, CTX_CAP = 5, 4000, 36000  # r3 link-hop caps

    def read_pages(paths, cap):
        pages, used = [], []
        for p in paths:
            fp = wiki_dir / p
            if fp.exists() and fp.name not in ("index.md", "log.md") and p not in used:
                pages.append(f"--- PAGE {p} ---\n{fp.read_text()[:cap]}")
                used.append(p)
        return pages, used

    def answer_wiki_links(q):
        pick_prompt = (f"{schema}\n\nWIKI INDEX:\n{index_text}\n\nQUESTION: {q['question']}\n\n"
                       "List up to 5 wiki page paths from the index that can answer this question, "
                       "one per line, paths only (e.g. sources/foo.md). If none, output NOT_FOUND.")
        r1 = call_answerer(pick_prompt, tag=f"wikilinks-pick:{q['qid']}")
        paths = [ln.strip("- []() ") for ln in r1["text"].splitlines() if ".md" in ln][:5]
        pages, used = read_pages(paths, 7000)
        links = []
        for p in used:
            for l in re.findall(r"\[\[([^\]|#]+)", (wiki_dir / p).read_text()):
                l = l.strip(); l = l if l.endswith(".md") else l + ".md"
                if (wiki_dir / l).exists() and l not in used and l not in links:
                    links.append(l)
        r2 = None; hop = []
        if links:
            hop_prompt = (f"QUESTION: {q['question']}\n\nPAGES ALREADY OPENED:\n" + "\n\n".join(pages)[:26000] +
                          "\n\nLINKED PAGES YOU MAY ALSO OPEN (one hop):\n" + "\n".join(links) +
                          f"\n\nIf the opened pages lack the exact fact, list up to {LINK_CAP_PAGES} linked page paths to open, "
                          "one per line, paths only. Otherwise output NONE.")
            r2 = call_answerer(hop_prompt, tag=f"wikilinks-hop:{q['qid']}")
            want = [ln.strip("- []() ") for ln in r2["text"].splitlines() if ".md" in ln]
            want = [w for w in want if w in links][:LINK_CAP_PAGES]
            extra, hop = read_pages(want, LINK_CAP_CHARS)
            pages += extra
        ctx = "\n\n".join(pages)[:CTX_CAP] or "(no pages selected)"
        r3 = call_answerer(f"{ANSWER_RULE}\n\nCONTEXT (wiki pages):\n{ctx}\n\nQUESTION: {q['question']}",
                           tag=f"wikilinks-answer:{q['qid']}")
        calls = [c for c in (r1, r2, r3) if c]
        return {"answer": r3["text"], "context_ids": used + ["HOP:" + h for h in hop], "context_chars": len(ctx),
                "links_offered": len(links), "hop_pages": hop, "calls": calls,
                "latency_s": sum(c["latency_s"] for c in calls), "prompt_chars": sum(c["prompt_chars"] for c in calls),
                "output_chars": sum(c["output_chars"] for c in calls)}

    def answer_wiki(q):
        pick_prompt = (f"{schema}\n\nWIKI INDEX:\n{index_text[:IDX_CAP]}\n\nQUESTION: {q['question']}\n\n"
                       "List up to 5 wiki page paths from the index that can answer this question, "
                       "one per line, paths only (e.g. sources/foo.md). If none, output NOT_FOUND.")
        r1 = call_answerer(pick_prompt, tag=f"wiki-pick:{q['qid']}")
        paths = [ln.strip("- []() ") for ln in r1["text"].splitlines() if ".md" in ln]
        pages, used = [], []
        for p in paths[:5]:
            fp = wiki_dir / p
            if fp.exists() and fp.name not in ("index.md", "log.md"):
                pages.append(f"--- PAGE {p} ---\n{fp.read_text()[:7000]}")
                used.append(p)
        ctx = "\n\n".join(pages)[:args.ctx_cap] or "(no pages selected)"
        r2 = call_answerer(f"{ANSWER_RULE}\n\nCONTEXT (wiki pages):\n{ctx}\n\nQUESTION: {q['question']}",
                           tag=f"wiki-answer:{q['qid']}")
        return {"answer": r2["text"], "context_ids": used, "context_chars": len(ctx),
                "calls": [r1, r2], "latency_s": r1["latency_s"] + r2["latency_s"],
                "prompt_chars": r1["prompt_chars"] + r2["prompt_chars"],
                "output_chars": r1["output_chars"] + r2["output_chars"]}

    def answer_rag(q):
        order, _ = topk(rag_idx, q["question"], 5)
        chs = [rag_idx["chunks"][i] for i in order]
        ctx = "\n\n".join(f"--- CHUNK {c['chunk_id']} ---\n{c['index_text'][:3500]}" for c in chs)[:26000]
        r = call_answerer(f"{ANSWER_RULE}\n\nCONTEXT (retrieved chunks):\n{ctx}\n\nQUESTION: {q['question']}",
                          tag=f"rag-answer:{q['qid']}")
        return {"answer": r["text"], "context_ids": [c["chunk_id"] for c in chs], "context_chars": len(ctx),
                "calls": [r], "latency_s": r["latency_s"], "prompt_chars": r["prompt_chars"], "output_chars": r["output_chars"]}

    def answer_wiki_bm25(q):
        order, _ = topk(wiki_idx, q["question"], 5)
        pgs = [wiki_idx["pages"][i] for i in order]
        ctx = "\n\n".join(f"--- PAGE {p['page']} ---\n{p['text'][:7000]}" for p in pgs)[:26000]
        r = call_answerer(f"{ANSWER_RULE}\n\nCONTEXT (wiki pages):\n{ctx}\n\nQUESTION: {q['question']}",
                          tag=f"wikibm25-answer:{q['qid']}")
        return {"answer": r["text"], "context_ids": [p["page"] for p in pgs], "context_chars": len(ctx),
                "calls": [r], "latency_s": r["latency_s"], "prompt_chars": r["prompt_chars"], "output_chars": r["output_chars"]}

    fn = {"wiki": answer_wiki, "wiki_full": answer_wiki, "wiki_links": answer_wiki_links, "rag": answer_rag, "wiki_bm25": answer_wiki_bm25}[args.system]

    def one(q):
        t0 = time.time()
        try:
            res = fn(q)
            res["ok"] = all(c["ok"] for c in res.pop("calls"))
        except Exception as e:  # noqa: BLE001
            res = {"answer": "", "context_ids": [], "context_chars": 0, "latency_s": time.time() - t0,
                   "prompt_chars": 0, "output_chars": 0, "ok": False, "error": str(e)[:300]}
        return {**q, **res, "system": args.system, "label": args.label}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for rec in ex.map(one, todo):
            with open(raw_log, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(f"{rec['qid']} ok={rec.get('ok')} {rec['latency_s']:.0f}s ans={rec['answer'][:60]!r}", flush=True)

    rows = [json.loads(l) for l in open(raw_log)]
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "label", "qid", "category", "answer", "latency_s", "prompt_chars", "output_chars", "context_ids"])
        for r in sorted(rows, key=lambda x: x["qid"]):
            w.writerow([r["system"], r["label"], r["qid"], r["category"], r["answer"], round(r["latency_s"], 2),
                        r["prompt_chars"], r["output_chars"], ";".join(r["context_ids"])])
    print(f"wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
