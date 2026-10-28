"""Questions answerable at each ingest snapshot: all gold docs within the first N
ingested docs (fixed ingest order). Caps at 15 per snapshot (seeded, stratified by
category order qid) to bound query cost. Writes data/snap{N}_questions.json.
"""
import json
from pathlib import Path

ROOT = Path(".")

order = json.load(open(ROOT / "data/ingest_order.json"))
questions = json.load(open(ROOT / "data/questions.json"))
for n in (25, 50):
    have = set(order[:n])
    eligible = [q for q in questions if q["gold_docs"] and set(q["gold_docs"]) <= have]
    eligible_sorted = sorted(eligible, key=lambda q: q["qid"])[:15]
    (ROOT / f"data/snap{n}_questions.json").write_text(json.dumps(eligible_sorted, indent=1))
    print(f"snap{n}: {len(eligible)} eligible, kept {len(eligible_sorted)}: {[q['qid'] for q in eligible_sorted]}")
