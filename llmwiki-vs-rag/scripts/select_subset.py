"""Select the benchmark subset (seeded, stratified) and build raw/ + subset docs.

Rules (documented in REPORT.md):
- Answerable: single 24, table 8, multihop 6 (two-passage), neardup 8 (version-specific) = 46.
  Eligible if the seed doc's canonical text <= 72000 chars; single must include the only
  eligible NIST questions (NIST.CSWP.29, q058/q059) for source diversity.
- No-answer: 10 of 12, seeded.
- Gold docs = every doc containing a gold span for a sampled question (all verbatim locations).
- Distractors: random non-gold docs, canonical <= 36000 chars, fill to 100 docs total.
- Raw text per doc capped at 36000 chars, section-aware: sections containing a gold span
  of a sampled question are always kept; then earliest sections until the cap. Both wiki
  and RAG see exactly these raw/ files (fair, identical input).
- Ingest order: subset docs shuffled with seed 123 (fixed).
"""
import json, random, re
from pathlib import Path

ROOT = Path(".")
LAB = Path(__import__("os").environ.get("LAB_DIR", "lab"))
SEED_Q, SEED_DISTRACT, SEED_ORDER = 42, 43, 123
CAP = 36000
TARGETS = {"single": 24, "table": 8, "multihop": 6, "neardup": 8}
N_NOANSWER = 10
TOTAL_DOCS = 100


def canonical(doc):
    parts, offs, pos = [], [], 0
    for s in doc["sections"]:
        offs.append((pos, pos + len(s["text"]), s["heading"]))
        parts.append(s["text"])
        pos += len(s["text"]) + 2
    return "\n\n".join(parts), offs


def slug(doc_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", doc_id).strip("-")[:90]


def main():
    docs = [json.loads(l) for l in open(LAB / "data/corpus/docs.jsonl")]
    docmap = {d["doc_id"]: d for d in docs}
    canon = {d["doc_id"]: canonical(d) for d in docs}
    golden = json.load(open(LAB / "data/golden.json"))

    def seed_doc(q):
        return q["seed_chunk"].split("#")[0] if q.get("seed_chunk") else None

    def gold_docs(q):
        out = []
        for grp in q.get("gold_spans", []):
            for span in grp:
                if span[0] not in out:
                    out.append(span[0])
        return out

    rng = random.Random(SEED_Q)
    sampled = []
    for cat, n in TARGETS.items():
        pool = [q for q in golden if q["category"] == cat
                and seed_doc(q) in docmap and len(canon[seed_doc(q)][0]) <= 72000]
        if cat == "single":
            nist = [q for q in pool if docmap[seed_doc(q)]["source"] == "nist"]
            rest = [q for q in pool if docmap[seed_doc(q)]["source"] != "nist"]
            rng.shuffle(nist); rng.shuffle(rest)
            picked = nist[:2] + rest[: n - 2]
        else:
            rng.shuffle(pool)
            picked = pool[:n]
        sampled += picked
        print(f"{cat}: pool {len(pool)} picked {[q['qid'] for q in picked]}")
    noans_pool = [q for q in golden if q["category"] == "noanswer"]
    rng.shuffle(noans_pool)
    noans = noans_pool[:N_NOANSWER]
    print(f"noanswer picked {[q['qid'] for q in noans]}")

    gold_doc_ids = []
    for q in sampled:
        for d in gold_docs(q):
            if d not in gold_doc_ids:
                gold_doc_ids.append(d)
    print(f"gold docs: {len(gold_doc_ids)}")

    rng_d = random.Random(SEED_DISTRACT)
    distract_pool = [d for d in docs if d["doc_id"] not in gold_doc_ids and len(canon[d["doc_id"]][0]) <= CAP]
    rng_d.shuffle(distract_pool)
    distract = distract_pool[: TOTAL_DOCS - len(gold_doc_ids)]
    subset_ids = gold_doc_ids + [d["doc_id"] for d in distract]
    print(f"distractors: {len(distract)} total docs: {len(subset_ids)}")

    # gold sections per doc (canonical offsets -> section indexes)
    gold_secs = {d: set() for d in subset_ids}
    for q in sampled:
        for grp in q.get("gold_spans", []):
            for did, st, en in grp:
                if did in gold_secs:
                    _, offs = canon[did]
                    for i, (s0, s1, _) in enumerate(offs):
                        if s0 < en and st < s1 + 2:
                            gold_secs[did].add(i)

    raw_dir = ROOT / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    subset_docs, manifest = [], []
    for did in subset_ids:
        d = docmap[did]
        text, offs = canon[did]
        keep, kept_chars = [], 0
        order = sorted(gold_secs[did]) + [i for i in range(len(d["sections"])) if i not in gold_secs[did]]
        chosen = set()
        for i in order:
            sec_len = len(d["sections"][i]["text"])
            if chosen and kept_chars + sec_len > CAP and i not in gold_secs[did]:
                continue
            if not chosen or i in gold_secs[did] or kept_chars + sec_len <= CAP:
                chosen.add(i); kept_chars += sec_len
        secs = [d["sections"][i] for i in sorted(chosen)]
        raw_text = "\n\n".join(f"## {s['heading']}\n\n{s['text']}" if s.get("heading") else s["text"] for s in secs)
        fname = slug(did) + ".md"
        (raw_dir / fname).write_text(f"# {d['title']}\n\nSource: {did} ({d['source']})\n\n" + raw_text + "\n")
        subset_docs.append({"doc_id": did, "source": d["source"], "title": d["title"],
                            "raw_file": fname, "canonical_chars": len(text), "raw_chars": len(raw_text),
                            "truncated": len(raw_text) < len(text) - 200, "is_gold": did in gold_doc_ids,
                            "sections": [{"heading": s["heading"], "text": s["text"]} for s in secs]})
        manifest.append({k: subset_docs[-1][k] for k in ("doc_id", "source", "title", "raw_file", "canonical_chars", "raw_chars", "truncated", "is_gold")})

    rng_o = random.Random(SEED_ORDER)
    ingest_order = subset_ids[:]
    rng_o.shuffle(ingest_order)

    questions = [{"qid": q["qid"], "category": q["category"], "question": q["question"],
                  "answer": q["answer"], "evidence": q["evidence"], "gold_docs": gold_docs(q),
                  "seed_doc": seed_doc(q)} for q in sampled]
    noanswer = [{"qid": q["qid"], "category": "noanswer", "question": q["question"], "answer": "", "evidence": [], "gold_docs": []} for q in noans]

    (ROOT / "data/subset_docs.jsonl").write_text("\n".join(json.dumps(x) for x in subset_docs) + "\n")
    (ROOT / "data/questions.json").write_text(json.dumps(questions, indent=1))
    (ROOT / "data/noanswer.json").write_text(json.dumps(noanswer, indent=1))
    (ROOT / "data/ingest_order.json").write_text(json.dumps(ingest_order, indent=1))
    (ROOT / "results/subset_manifest.json").write_text(json.dumps(
        {"seed_questions": SEED_Q, "seed_distractors": SEED_DISTRACT, "seed_order": SEED_ORDER,
         "cap_chars": CAP, "n_answerable": len(questions), "n_noanswer": len(noanswer),
         "n_docs": len(subset_ids), "n_gold_docs": len(gold_doc_ids),
         "raw_chars_total": sum(m["raw_chars"] for m in manifest),
         "n_truncated": sum(1 for m in manifest if m["truncated"]), "docs": manifest}, indent=1))
    print(f"raw chars total {sum(m['raw_chars'] for m in manifest)} truncated {sum(1 for m in manifest if m['truncated'])}")
    print("wrote data/subset_docs.jsonl data/questions.json data/noanswer.json data/ingest_order.json")


if __name__ == "__main__":
    main()
