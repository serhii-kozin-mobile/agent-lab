"""Judge answers with claude-llm (Opus, different model from the answerer), batched 8 per call.

Verdicts per answer: verdict in {correct, partial, wrong, correct_abstention, wrong_abstention},
unsupported_claims = number of factual claims in the answer not supported by the reference
answer/evidence (hallucination proxy, judged against gold evidence — see REPORT limitations).
Usage: judge.py answers_<sys>_<label>.jsonl [...]  -> results/judged_<sys>_<label>.csv
"""
import csv, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, call_judge  # noqa: E402

BATCH = 8


def judge_batch(items):
    blocks = []
    for i, it in enumerate(items):
        ref = it.get("answer_ref") or "(no-answer question: the correct response is NOT_FOUND / abstention)"
        ev = " | ".join(e[:300] for e in it.get("evidence", [])) or "(none)"
        blocks.append(f"ITEM {i}\nQUESTION: {it['question']}\nREFERENCE: {ref}\nEVIDENCE: {ev}\nCANDIDATE ANSWER: {it['answer'][:1500]}")
    prompt = ("You are a strict judge. For each item, compare the CANDIDATE ANSWER with the REFERENCE and EVIDENCE.\n"
              "verdict: 'correct' (matches reference, no invented facts), 'partial' (partly right / missing key part), "
              "'wrong' (wrong or invented), 'correct_abstention' (candidate says NOT_FOUND / not in sources AND reference "
              "is empty/no-answer), 'wrong_abstention' (candidate abstains but reference has an answer).\n"
              "unsupported_claims: count of factual claims in the candidate not supported by REFERENCE+EVIDENCE (0 if abstention).\n\n"
              + "\n\n".join(blocks) +
              "\n\nOutput ONLY a JSON array, one object per item in order: "
              '[{"verdict": "...", "unsupported_claims": 0}]')
    r = call_judge(prompt, tag="judge-batch")
    m = re.search(r"\[.*\]", r["text"], re.S)
    try:
        verdicts = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        verdicts = []
    while len(verdicts) < len(items):
        verdicts.append({"verdict": "judge_parse_error", "unsupported_claims": -1})
    return verdicts[: len(items)], r


def main():
    refs = {}
    for f in ("data/questions.json", "data/noanswer.json", "data/synthesis.json"):
        p = ROOT / f
        if p.exists():
            for q in json.load(open(p)):
                refs[q["qid"]] = q
    for name in sys.argv[1:]:
        src = ROOT / "results" / name
        rows = [json.loads(l) for l in open(src)]
        items = []
        for r in rows:
            ref = refs.get(r["qid"], {})
            items.append({**r, "answer_ref": ref.get("answer", ""), "evidence": ref.get("evidence", []),
                          "question": ref.get("question", r.get("question", ""))})
        judged = []
        for i in range(0, len(items), BATCH):
            batch = items[i: i + BATCH]
            verdicts, _ = judge_batch(batch)
            for it, v in zip(batch, verdicts):
                judged.append({**it, "verdict": v.get("verdict", "judge_parse_error"),
                               "unsupported_claims": v.get("unsupported_claims", -1)})
                print(f"{it['qid']} -> {judged[-1]['verdict']} (unsupported={judged[-1]['unsupported_claims']})", flush=True)
        out = ROOT / f"results/judged_{src.stem}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["system", "label", "qid", "category", "verdict", "unsupported_claims", "answer"])
            for r in judged:
                w.writerow([r["system"], r["label"], r["qid"], r["category"], r["verdict"], r["unsupported_claims"], r["answer"]])
        (ROOT / f"results/judged_{src.stem}.json").write_text(json.dumps(judged, indent=1))
        print(f"wrote {out} ({len(judged)} rows)")


if __name__ == "__main__":
    main()
