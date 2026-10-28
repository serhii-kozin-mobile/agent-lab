"""Compute all report metrics from judged CSVs + usage log + wiki files.
Bootstrap 95% CIs: paired resampling over questions (10,000 draws, seed 7).
Lint: broken [[links]], orphans, pages missing from index (mechanical counts).
Writes results/metrics.json and prints a markdown table.
"""
import csv, json, random, re
from collections import defaultdict
from pathlib import Path

ROOT = Path(".")


def load(name):
    p = ROOT / f"results/{name}"
    return list(csv.DictReader(open(p))) if p.exists() else []


def acc(rows, cats=None):
    sel = [r for r in rows if cats is None or r["category"] in cats]
    if not sel:
        return None
    score = sum({"correct": 1.0, "partial": 0.5}.get(r["verdict"], 0.0) for r in sel)
    strict = sum(1 for r in sel if r["verdict"] == "correct")
    return {"n": len(sel), "accuracy_half": round(score / len(sel), 3), "accuracy_strict": round(strict / len(sel), 3),
            "correct": sum(1 for r in sel if r["verdict"] == "correct"),
            "partial": sum(1 for r in sel if r["verdict"] == "partial"),
            "wrong": sum(1 for r in sel if r["verdict"] in ("wrong", "wrong_abstention")),
            "abstain_wrong": sum(1 for r in sel if r["verdict"] == "wrong_abstention")}


def boot_diff(a_rows, b_rows, seed=7, draws=10000):
    """Paired bootstrap of accuracy_half(a) - accuracy_half(b) on shared answerable qids."""
    a = {r["qid"]: r for r in a_rows if r["category"] != "noanswer"}
    b = {r["qid"]: r for r in b_rows if r["category"] != "noanswer"}
    qids = sorted(set(a) & set(b))
    if len(qids) < 10:
        return {"n": len(qids), "diff": None, "ci": None, "note": "n too small for CI"}
    s = lambda r: {"correct": 1.0, "partial": 0.5}.get(r["verdict"], 0.0)  # noqa: E731
    diffs_obs = [s(a[q]) - s(b[q]) for q in qids]
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = [diffs_obs[rng.randrange(len(diffs_obs))] for _ in diffs_obs]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {"n": len(qids), "diff": round(sum(diffs_obs) / len(diffs_obs), 3),
            "ci": [round(means[int(0.025 * draws)], 3), round(means[int(0.975 * draws)], 3)]}


def boot_mean(vals, seed=7, draws=10000):
    if len(vals) < 10:
        return {"mean": round(sum(vals) / len(vals), 3) if vals else None, "ci": None, "note": "n too small"}
    rng = random.Random(seed)
    means = sorted(sum(vals[rng.randrange(len(vals))] for _ in vals) / len(vals) for _ in range(draws))
    return {"mean": round(sum(vals) / len(vals), 3), "ci": [round(means[250], 3), round(means[9750], 3)]}


def lint(wiki: Path):
    pages = {str(p.relative_to(wiki)): p for p in wiki.rglob("*.md")}
    broken, inbound = [], defaultdict(int)
    for rel, p in pages.items():
        if p.name in ("index.md", "log.md"):
            continue
        for target in re.findall(r"\[\[([^\]]+)\]\]", p.read_text()):
            t = target.strip()
            cand = t if t.endswith(".md") else t + ".md"
            if cand not in pages and f"{cand}" not in pages:
                broken.append((rel, target))
            else:
                inbound[cand] += 1
        for target in re.findall(r"\]\(([^)]+\.md)\)", p.read_text()):
            if target not in pages:
                broken.append((rel, target))
    orphans = [r for r in pages if r not in ("index.md", "log.md") and inbound[r] == 0]
    index_text = (wiki / "index.md").read_text() if (wiki / "index.md").exists() else ""
    not_indexed = [r for r in pages if r not in ("index.md", "log.md") and r not in index_text]
    total_chars = sum(len(p.read_text()) for p in pages.values())
    return {"pages": len([r for r in pages if r not in ("index.md", "log.md")]), "chars": total_chars,
            "broken_links": len(broken), "broken_sample": broken[:10], "orphans": len(orphans),
            "not_in_index": len(not_indexed)}


def main():
    out = {}
    systems = {}
    for name in ("judged_answers_wiki_main.csv", "judged_answers_rag_main.csv", "judged_answers_wiki_bm25_main.csv",
                 "judged_answers_wiki_synth.csv", "judged_answers_rag_synth.csv",
                 "judged_answers_wiki_snap25.csv", "judged_answers_wiki_snap50.csv"):
        rows = load(name)
        if rows:
            systems[name] = rows
    for name, rows in systems.items():
        ans = [r for r in rows if r["category"] != "noanswer"]
        noa = [r for r in rows if r["category"] == "noanswer"]
        per_cat = {}
        for cat in sorted(set(r["category"] for r in ans)):
            per_cat[cat] = acc(rows, {cat})
        answered = [r for r in rows if r["verdict"] not in ("correct_abstention", "wrong_abstention")]
        hall = [r for r in answered if int(r["unsupported_claims"] or 0) > 0]
        out[name] = {"overall": acc(rows), "answerable": acc(ans), "per_category": per_cat,
                     "noanswer_correct_abstention": f"{sum(1 for r in noa if r['verdict'] == 'correct_abstention')}/{len(noa)}",
                     "hallucination_rate_answered": round(len(hall) / len(answered), 3) if answered else None,
                     "answers_with_unsupported_claims": len(hall), "n_answered": len(answered)}
        raw_name = name.replace("judged_answers_", "answers_").replace(".csv", ".jsonl")
        rawp = ROOT / f"results/{raw_name}"
        if rawp.exists():
            raw = [json.loads(l) for l in open(rawp)]
            out[name]["latency_mean_s"] = boot_mean([r["latency_s"] for r in raw])
            out[name]["tokens_per_query_est"] = boot_mean([(r["prompt_chars"] + r["output_chars"]) / 4 for r in raw])

    if "judged_answers_wiki_main.csv" in systems and "judged_answers_rag_main.csv" in systems:
        out["paired_wiki_minus_rag"] = boot_diff(systems["judged_answers_wiki_main.csv"], systems["judged_answers_rag_main.csv"])
    if "judged_answers_wiki_synth.csv" in systems and "judged_answers_rag_synth.csv" in systems:
        out["paired_synth_wiki_minus_rag"] = boot_diff(systems["judged_answers_wiki_synth.csv"], systems["judged_answers_rag_synth.csv"])

    usage = [json.loads(l) for l in open(ROOT / "results/llm_usage.jsonl")] if (ROOT / "results/llm_usage.jsonl").exists() else []
    by_tag = defaultdict(lambda: {"calls": 0, "tokens_est": 0, "latency_s": 0.0})
    for u in usage:
        k = u["tag"].split(":")[0]
        by_tag[k]["calls"] += 1
        by_tag[k]["tokens_est"] += u["prompt_tokens_est"] + u["output_tokens_est"]
        by_tag[k]["latency_s"] += u["latency_s"]
    out["usage_by_tag"] = {k: {"calls": v["calls"], "tokens_est": v["tokens_est"], "latency_s": round(v["latency_s"], 1)} for k, v in sorted(by_tag.items())}
    ingest = [json.loads(l) for l in open(ROOT / "results/ingest_stats.jsonl")] if (ROOT / "results/ingest_stats.jsonl").exists() else []
    out["ingest"] = {"docs": len(ingest), "ok": sum(1 for r in ingest if r["ok"]),
                     "tokens_est_total": sum((r["prompt_chars"] + r["output_chars"]) // 4 for r in ingest),
                     "latency_sum_s": round(sum(r["latency_s"] for r in ingest), 1)}
    out["lint_final"] = lint(ROOT / "wiki")
    for snap in ("wiki_25", "wiki_50"):
        p = ROOT / f"results/snapshots/{snap}"
        if p.exists():
            out[f"lint_{snap}"] = lint(p)
    (ROOT / "results/metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
