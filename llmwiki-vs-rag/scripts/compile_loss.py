"""r3: compile-loss check + RAG home advantage + wiki_bm25 error breakdown.
For every answerable miss (verdict != correct) of wiki_main / wiki_bm25_main:
 1) normalized string match of each gold evidence quote anywhere in wiki pages (and ref answer);
 2) else judge (claude-llm) on the top-5 wiki pages by BM25(evidence) -> does the wiki contain the fact?
Also: does the fact sit in pages the arm actually retrieved? -> retrieval vs compile loss.
Home advantage: gold quote fully inside one RAG chunk.
Writes results/compile_loss.json"""
import json, pickle, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, __import__("os").environ.get("LAB_DIR", "lab"))
import retrieval
from common import ROOT, call_judge
norm = lambda t: re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
Q = {x["qid"]: x for x in json.load(open(ROOT / "data/questions.json"))}
W = pickle.load(open(ROOT / "data/wiki_index.pkl", "rb")); R = pickle.load(open(ROOT / "data/rag_index.pkl", "rb"))
pages = {p["page"]: norm(p["text"]) for p in W["pages"]}
chunks = [norm(c["text"]) for c in R["chunks"]]
out = {"home_advantage": {}, "arms": {}}
inside = sum(1 for q in Q.values() if q["evidence"] and all(any(norm(e) in c for c in chunks) for e in q["evidence"]))
out["home_advantage"] = {"n": len(Q), "all_quotes_inside_one_chunk_each": inside,
    "per_quote_inside_single_chunk": sum(any(norm(e) in c for c in chunks) for q in Q.values() for e in q["evidence"]),
    "n_quotes": sum(len(q["evidence"]) for q in Q.values())}
cache = {}
def wiki_has(qid):
    if qid in cache: return cache[qid]
    q = Q[qid]; hits = [p for p, t in pages.items() if any(norm(e) in t for e in q["evidence"])]
    if hits: cache[qid] = {"method": "string", "pages": hits, "has": True}; return cache[qid]
    sc = W["bm25"].get_scores(retrieval.tokenize(" ".join(q["evidence"]) + " " + q["question"]))
    top = sorted(range(len(sc)), key=lambda i: -sc[i])[:5]
    ctx = "\n\n".join(f"--- PAGE {W['pages'][i]['page']} ---\n{W['pages'][i]['text'][:6000]}" for i in top)
    r = call_judge(f"QUESTION: {q['question']}\nREFERENCE ANSWER: {q['answer']}\nGOLD EVIDENCE: {' | '.join(q['evidence'])}\n\n"
        f"WIKI PAGES:\n{ctx}\n\nDo these wiki pages, taken together, contain enough of the specific fact to give the reference answer "
        "(exact values included)? Output ONLY JSON: {\"has\": true|false, \"page\": \"path or null\", \"why\": \"short\"}", tag="compile-loss-judge")
    m = re.search(r"\{.*\}", r["text"], re.S)
    try: v = json.loads(m.group(0))
    except Exception: v = {"has": None, "page": None, "why": "parse_error"}
    cache[qid] = {"method": "judge", "pages": [v.get("page")] if v.get("has") else [], "has": v.get("has"), "why": v.get("why")}
    return cache[qid]
for arm in ("wiki_main", "wiki_bm25_main"):
    J = json.load(open(ROOT / f"results/judged_answers_{arm}.json")); rows = []
    for r in J:
        if r["category"] == "noanswer" or r["verdict"] == "correct": continue
        h = wiki_has(r["qid"]); ret = set(r.get("context_ids", []))
        cls = "compile_loss" if h["has"] is False else ("retrieval_miss" if not (set(h["pages"]) & ret) else "in_context_but_missed")
        rows.append({"qid": r["qid"], "category": r["category"], "verdict": r["verdict"], "not_found": "NOT_FOUND" in r["answer"], **h, "class": cls})
        print(arm, r["qid"], cls, h["method"], flush=True)
    from collections import Counter
    out["arms"][arm] = {"misses": len(rows), "classes": Counter(x["class"] for x in rows), "rows": rows}
(ROOT / "results/compile_loss.json").write_text(json.dumps(out, indent=1, default=list))
print(json.dumps({k: (v if k == "home_advantage" else {a: {"misses": x["misses"], "classes": x["classes"]} for a, x in v.items()}) for k, v in out.items()}, default=dict))
