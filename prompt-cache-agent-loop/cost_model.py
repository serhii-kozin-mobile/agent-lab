"""Price every model call under different prompt-caching rules.

All costs are in "base input units": 1.0 = one input token at the model's normal price.
Multipliers come from provider docs (checked 2026-10-25):
- Anthropic: 5-min write 1.25x, 1-hour write 2x, read 0.1x (0.05x Opus 5.5 / Sonnet 5.5).
  https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- OpenAI GPT-5.6+: 30-min write 1.25x, read 0.1x (0.05x GPT-6.1 Sol), min prefix 1,024 tokens.
  https://developers.openai.com/api/docs/guides/prompt-caching
- OpenAI 2024 launch (GPT-4o era): no write premium, cached input 50% off.
  https://openai.com/index/api-prompt-caching/
Output tokens are priced at OUT_RATIO x base input (5x matches the Anthropic list prices
above, e.g. $3/$15, $4/$20).

Cache mechanics modelled (automatic prefix caching, one breakpoint at the end of each request):
call k reads the previous request's prompt (I[k-1]) from cache and writes the new tail
(I[k] - I[k-1]). Prompts under MIN_PREFIX tokens are billed at the normal input price.
A "miss" (expired TTL, changed prefix, model switch) means the whole prompt is written again.
"""
import json
import pandas as pd

MIN_PREFIX = 1024
OUT_RATIO = 5.0

def run_cost(inputs, outputs, read, write, miss_prob=0.0, volatile=False, switch_at_middle=False):
    """Expected input cost of one run (base-input units). miss_prob applies to each call k>1."""
    cost, prev = 0.0, 0
    mid = len(inputs) // 2
    for k, inp in enumerate(inputs):
        if inp < MIN_PREFIX:
            cost += inp
            prev = 0
            continue
        full_write = inp * write
        if k == 0 or prev == 0 or volatile:
            c = full_write
        else:
            hit = prev * read + (inp - prev) * write
            p = 1.0 if (switch_at_middle and k == mid) else miss_prob
            c = (1 - p) * hit + p * full_write
        cost += c
        prev = inp
    return cost

SCENARIOS = {
    "no_cache":                    dict(read=1.0,  write=1.0),
    "cache_read0.10_write1.25":    dict(read=0.10, write=1.25),
    "cache_read0.05_write1.25":    dict(read=0.05, write=1.25),
    "cache_read0.10_write2.00_1h": dict(read=0.10, write=2.00),
    "legacy_read0.50_write1.00":   dict(read=0.50, write=1.00),
    "miss5pct_read0.10_write1.25": dict(read=0.10, write=1.25, miss_prob=0.05),
    "miss10pct_read0.10_write1.25":dict(read=0.10, write=1.25, miss_prob=0.10),
    "miss25pct_read0.10_write1.25":dict(read=0.10, write=1.25, miss_prob=0.25),
    "model_switch_mid_read0.10":   dict(read=0.10, write=1.25, switch_at_middle=True),
    "volatile_prefix_write1.25":   dict(read=0.10, write=1.25, volatile=True),
}

if __name__ == "__main__":
    calls = pd.read_csv("out/calls.csv").sort_values(["harness", "run_id", "call"])
    runs = pd.read_csv("out/runs.csv")
    rows = []
    for (h, rid), g in calls.groupby(["harness", "run_id"], sort=False):
        inp, out = g.input_tokens.tolist(), g.output_tokens.tolist()
        r = dict(harness=h, run_id=rid, out_units=sum(out) * OUT_RATIO)
        for name, kw in SCENARIOS.items():
            r[name] = run_cost(inp, out, **kw)
        rows.append(r)
    per = pd.DataFrame(rows)
    per.to_csv("out/costs_per_run.csv", index=False)

    result = {"min_prefix": MIN_PREFIX, "out_ratio": OUT_RATIO, "harness": {}}
    for h, g in list(per.groupby("harness")) + [("all", per)]:
        rr = runs if h == "all" else runs[runs.harness == h]
        base = g.no_cache.sum()
        total_base = base + g.out_units.sum()
        s = {
            "runs": int(len(g)),
            "calls_median": float(rr.calls.median()),
            "calls_p90": float(rr.calls.quantile(0.9)),
            "input_tokens_per_run_median": float(rr.total_input.median()),
            "output_tokens_per_run_median": float(rr.total_output.median()),
            "input_to_output_ratio_aggregate": float(rr.total_input.sum() / rr.total_output.sum()),
            "resent_share_of_input_aggregate": float(1 - rr.unique_tokens.sum() / rr.total_input.sum()),
            "input_share_of_bill_no_cache": float(base / total_base),
            "scenarios": {},
        }
        for name in SCENARIOS:
            c = g[name].sum()
            s["scenarios"][name] = {
                "input_cost_vs_no_cache": round(c / base, 4),
                "total_bill_vs_no_cache": round((c + g.out_units.sum()) / total_base, 4),
            }
        result["harness"][h] = s
    json.dump(result, open("out/results.json", "w"), indent=2)
    print(json.dumps(result, indent=2))
