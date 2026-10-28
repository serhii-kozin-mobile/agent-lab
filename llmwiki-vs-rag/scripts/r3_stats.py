"""r3: per-arm accuracy + paired bootstrap vs RAG (10,000 resamples, seed 7), tokens/latency, hop stats, RAG repeat variance."""
import json, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from metrics import load, acc, boot_diff, boot_mean
ROOT = Path("."); out = {}
arms = ["rag_main", "rag_rep2", "wiki_bm25_rep2", "wiki_full_cap36", "wiki_main", "wiki_full_main", "wiki_links_main", "wiki_bm25_main"]
J = {a: load(f"judged_answers_{a}.csv") for a in arms}
raw = {a: [json.loads(l) for l in open(ROOT / f"results/answers_{a}.jsonl")] for a in arms}
for a in arms:
    ans = [r for r in J[a] if r["category"] != "noanswer"]; na = [r for r in J[a] if r["category"] == "noanswer"]
    out[a] = {"answerable": acc(ans), "noanswer_ok": f"{sum(r['verdict']=='correct_abstention' for r in na)}/{len(na)}",
              "not_found_on_answerable": sum("NOT_FOUND" in r["answer"] for r in ans),
              "tokens_est": boot_mean([(r["prompt_chars"] + r["output_chars"]) / 4 for r in raw[a]]),
              "latency_s": boot_mean([r["latency_s"] for r in raw[a]])}
    if a != "rag_main": out[a]["vs_rag_main"] = boot_diff(J[a], J["rag_main"])
out["wiki_bm25_main"]["vs_wiki_full_main"] = boot_diff(J["wiki_bm25_main"], J["wiki_full_main"])
out["wiki_links_main"]["vs_wiki_full_main"] = boot_diff(J["wiki_links_main"], J["wiki_full_main"])
L = raw["wiki_links_main"]; out["wiki_links_main"]["hop"] = {"queries_with_hop": sum(1 for r in L if r.get("hop_pages")),
    "hop_pages_total": sum(len(r.get("hop_pages", [])) for r in L), "mean_links_offered": round(sum(r.get("links_offered", 0) for r in L) / len(L), 1)}
a1 = {r["qid"]: r["verdict"] for r in J["rag_main"]}; a2 = {r["qid"]: r["verdict"] for r in J["rag_rep2"]}
b1 = {r["qid"]: r["verdict"] for r in J["wiki_bm25_main"]}; b2 = {r["qid"]: r["verdict"] for r in J["wiki_bm25_rep2"]}
out["wiki_bm25_repeat_same_verdict"] = f"{sum(b1[q]==b2[q] for q in b1)}/{len(b1)}"
out["wiki_links_main"]["vs_wiki_full_cap36"] = boot_diff(J["wiki_links_main"], J["wiki_full_cap36"]); out["wiki_full_cap36"]["vs_wiki_full_main"] = boot_diff(J["wiki_full_cap36"], J["wiki_full_main"])
c1 = {r["qid"]: r["verdict"] for r in J["wiki_full_main"]}; c2 = {r["qid"]: r["verdict"] for r in J["wiki_full_cap36"]}; out["wiki_full_cap26_vs_cap36_same_verdict"] = f"{sum(c1[q]==c2[q] for q in c1)}/{len(c1)}"
out["rag_repeat_same_verdict"] = f"{sum(a1[q]==a2[q] for q in a1)}/{len(a1)}"
(ROOT / "results/r3_stats.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
