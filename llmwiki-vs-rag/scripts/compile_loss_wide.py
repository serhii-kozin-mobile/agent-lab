"""r4: wider recheck of the 16 compile-loss questions. Judge reads every wiki page whose text cites a gold doc
(source tag / slug) plus top-15 wiki pages by BM25(gold quote + question + reference answer), in batches of 8 pages.
Writes results/compile_loss_wide.json"""
import json, pickle, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, __import__("os").environ.get("LAB_DIR", "lab"))
import retrieval
from common import ROOT, call_judge
Q = {x["qid"]: x for x in json.load(open(ROOT / "data/questions.json"))}
W = pickle.load(open(ROOT / "data/wiki_index.pkl", "rb"))
cl = json.load(open(ROOT / "results/compile_loss.json"))
loss = sorted({r["qid"] for a in cl["arms"].values() for r in a["rows"] if r["class"] == "compile_loss"})
res = []
for qid in loss:
    q = Q[qid]
    cand = [i for i, p in enumerate(W["pages"]) if any(g in p["text"] for g in q["gold_docs"])]
    sc = W["bm25"].get_scores(retrieval.tokenize(" ".join(q["evidence"]) + " " + q["question"] + " " + q["answer"]))
    for i in sorted(range(len(sc)), key=lambda i: -sc[i])[:15]:
        if i not in cand: cand.append(i)
    found = None
    for b in range(0, len(cand), 8):
        ctx = "\n\n".join(f"--- PAGE {W['pages'][i]['page']} ---\n{W['pages'][i]['text'][:8000]}" for i in cand[b:b+8])
        r = call_judge(f"QUESTION: {q['question']}\nREFERENCE ANSWER: {q['answer']}\nGOLD EVIDENCE: {' | '.join(q['evidence'])}\n\nWIKI PAGES:\n{ctx}\n\n"
            "Does any page (or pages together) state the specific fact needed for the reference answer, exact values included? "
            'Output ONLY JSON: {"has": true|false, "page": "path or null"}', tag="compile-loss-wide")
        m = re.search(r"\{.*\}", r["text"], re.S)
        try: v = json.loads(m.group(0))
        except Exception: v = {"has": None}
        if v.get("has"): found = v.get("page"); break
    res.append({"qid": qid, "category": q["category"], "pages_checked": len(cand), "found_page": found})
    print(res[-1], flush=True)
out = {"n": len(res), "still_absent": sum(r["found_page"] is None for r in res), "rows": res}
(ROOT / "results/compile_loss_wide.json").write_text(json.dumps(out, indent=1)); print(out["n"], out["still_absent"])
