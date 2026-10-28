"""Generate 8 cross-source synthesis questions with ONE judge-model (claude-llm) call.

Input to the generator: gold question/evidence pairs from the subset whose docs are
topically pairable (MCP sampling/elicitation across revisions, Python logging docs,
asyncio, etc. — chosen by shared title tokens, computed here). The generator must
combine TWO different source docs per question and quote both evidences verbatim.
This script then verifies both quotes occur verbatim (whitespace-normalized) in the
subset raw text; unverified questions are dropped. Output: data/synthesis.json.
"""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, call_judge  # noqa: E402


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    questions = json.load(open(ROOT / "data/questions.json"))
    docs = {json.loads(l)["doc_id"]: json.loads(l) for l in open(ROOT / "data/subset_docs.jsonl")}
    raw_norm = {did: norm("\n".join(s["text"] for s in d["sections"])) for did, d in docs.items()}

    pairs = []
    singles = [q for q in questions if q["evidence"]]
    for i, a in enumerate(singles):
        for b in singles[i + 1:]:
            da, db = a["seed_doc"], b["seed_doc"]
            if not da or not db or da == db:
                continue
            ta = set(re.findall(r"[a-z]+", docs[da]["title"].lower())) if da in docs else set()
            tb = set(re.findall(r"[a-z]+", docs[db]["title"].lower())) if db in docs else set()
            stem_a = re.sub(r"[^a-z]", "", da.split("/")[-1].split(".")[0].lower())
            stem_b = re.sub(r"[^a-z]", "", db.split("/")[-1].split(".")[0].lower())
            if stem_a == stem_b or (ta & tb) or ("mcp" in da and "mcp" in db) or ("logging" in da and "logging" in db) or ("asyncio" in da and "asyncio" in db):
                pairs.append((a, b))
    pairs = pairs[:14]
    print(f"candidate pairs: {len(pairs)}")

    blocks = []
    for i, (a, b) in enumerate(pairs):
        blocks.append(f"PAIR {i}: doc A={a['seed_doc']} evidence A: {a['evidence'][0][:400]}\n"
                      f"doc B={b['seed_doc']} evidence B: {b['evidence'][0][:400]}")
    prompt = ("Create exactly 8 cross-source synthesis questions. Each question MUST require combining "
              "information from BOTH docs of one pair below (comparison, version difference, or combined "
              "procedure) — it must NOT be answerable from one doc alone. For each, give: pair number, "
              "question, reference answer (using only the two evidences), quote_a (copied VERBATIM from "
              "evidence A, 40-200 chars), quote_b (copied VERBATIM from evidence B, 40-200 chars).\n\n"
              + "\n\n".join(blocks) +
              "\n\nOutput ONLY a JSON array of objects with keys: pair, question, answer, quote_a, quote_b.")

    r = call_judge(prompt, tag="synthesis-gen")
    (ROOT / "results/synthesis_raw.txt").write_text(r["text"])
    m = re.search(r"\[.*\]", r["text"], re.S)
    items = json.loads(m.group(0)) if m else []
    out = []
    for it in items:
        a, b = pairs[it["pair"]] if 0 <= it.get("pair", -1) < len(pairs) else (None, None)
        if not a:
            continue
        qa, qb = norm(it.get("quote_a", "")), norm(it.get("quote_b", ""))
        ok_a = qa[:80] in raw_norm.get(a["seed_doc"], "") if qa else False
        ok_b = qb[:80] in raw_norm.get(b["seed_doc"], "") if qb else False
        out.append({"qid": f"s{len(out):02d}", "category": "synthesis", "question": it["question"],
                    "answer": it["answer"], "evidence": [it.get("quote_a", ""), it.get("quote_b", "")],
                    "gold_docs": [a["seed_doc"], b["seed_doc"]], "seed_doc": a["seed_doc"],
                    "verified_a": ok_a, "verified_b": ok_b})
        print(out[-1]["qid"], "verified:", ok_a, ok_b, "|", it["question"][:90])
    out = [o for o in out if o["verified_a"] and o["verified_b"]][:8]
    (ROOT / "data/synthesis.json").write_text(json.dumps(out, indent=1))
    print(f"kept {len(out)} verified synthesis questions -> data/synthesis.json")


if __name__ == "__main__":
    main()
