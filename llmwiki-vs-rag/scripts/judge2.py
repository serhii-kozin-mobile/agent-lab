"""r3: second judge pass on a 20-answer sample (seed 7, stratified across rag_main/wiki_main/wiki_bm25_main,
answerable+noanswer). Judge 2 = Muse (muse exec, empty temp dir), different prompt wording, one call per item.
Agreement on 3 classes: correct / partial / wrong(incl. wrong_abstention); correct_abstention kept separate.
Writes results/judge2_sample.json"""
import json, random, re, subprocess, tempfile, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, _log_usage, est_tokens
refs = {q["qid"]: q for f in ("data/questions.json", "data/noanswer.json") for q in json.load(open(ROOT / f))}
rng = random.Random(7); sample = []
for arm, k in (("rag_main", 7), ("wiki_main", 6), ("wiki_bm25_main", 7)):
    J = json.load(open(ROOT / f"results/judged_answers_{arm}.json")); sample += [(arm, r) for r in rng.sample(J, k)]
cls = lambda v: {"wrong_abstention": "wrong"}.get(v, v)
out = []
for arm, r in sample:
    q = refs[r["qid"]]
    p = ("Grade one answer. Read the reference carefully.\n"
         f"Question: {q['question']}\nReference answer: {q.get('answer') or 'NONE - question is not answerable from the corpus; abstaining is right'}\n"
         f"Supporting quote(s): {' | '.join(q.get('evidence', [])) or 'none'}\nAnswer to grade: {r['answer'][:1500]}\n\n"
         "Labels: correct = gives the reference fact without errors; partial = some of it; wrong = incorrect, invented, "
         "or says not found when a reference exists; correct_abstention = says not found and reference is NONE.\n"
         'Reply with ONLY one JSON object: {"label": "..."}')
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        pr = subprocess.run(["muse", "exec", "--reasoning-effort", "medium", "--contributor-mode", "off", p], cwd=td, capture_output=True, text=True, timeout=600)
    txt = pr.stdout
    _log_usage({"tag": "judge2-muse", "model": "muse", "latency_s": round(time.time() - t0, 2), "prompt_chars": len(p), "output_chars": len(txt),
                "prompt_tokens_est": est_tokens(len(p)), "output_tokens_est": est_tokens(len(txt)), "ok": pr.returncode == 0})
    m = re.findall(r'\{\s*"label"\s*:\s*"([a-z_]+)"\s*\}', txt)
    lab = m[-1] if m else "parse_error"
    out.append({"arm": arm, "qid": r["qid"], "judge1": r["verdict"], "judge2": lab, "agree": cls(r["verdict"]) == cls(lab)})
    print(out[-1], flush=True)
agree = sum(o["agree"] for o in out)
(ROOT / "results/judge2_sample.json").write_text(json.dumps({"n": len(out), "agree": agree, "items": out}, indent=1))
print("agreement", agree, "/", len(out))
