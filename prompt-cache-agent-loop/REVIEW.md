# Independent review: prompt caching on a real agent loop

Reviewer: independent agent review, 2026-10-26 (before the public release). Scope: this study folder.
No paid API, no spend. The only network access was reading public provider pricing pages to check the multipliers.

## Verdict: **ready** (no number changed)

The study is a cost model on recorded traces, and it says so. The code is short and deterministic (seeded sample, fixed tokenizer), and it reproduces exactly.

## Reproduction
| Check | Result |
|---|---|
| `measure_tokens.py` → `cost_model.py` → `report.py` on the same two dataset files, clean copy of the folder | `out/results.json`, `out/runs.csv`, `out/sensitivity.md` and both charts **byte-identical** (about 16 s on CPU) |
| `out/runs.csv` recount | 2,490 runs (1,999 SWE-agent + 491 OpenHands), 61,945 model calls, aggregate input:output 107:1, median 16 calls per run: matches the README |
| Every number in the README tables vs `out/results.json` (`all`, `swe-agent`, `openhands`) | all match after rounding |

## Checks on the method
| # | Severity | Finding | Status |
|---|---|---|---|
| 1 | Info | Cache multipliers re-checked against the provider pricing pages on 2026-10-26: Anthropic 5-min write 1.25x, 1-hour write 2x, read 0.1x (0.05x on Opus 5.5 / Sonnet 5.5); OpenAI GPT-5.6 Sol cached input = 0.1x and 30-minute write = 1.25x of input, GPT-6.1 Sol cached input = 0.05x. All match the README. Anthropic now also lists 0.025x reads on its newest top tier, which is below the cheapest scenario modelled here; the 0.05x row is therefore not the floor. | Disclosed here |
| 2 | Low (method) | Misses are modelled as independent per-call events; real misses cluster (after long tool calls, idle gaps). The README already lists this under "does not prove". | No change |
| 3 | Low (method) | Output is priced at a fixed 5x input. Reasoning models emit far more output; the sensitivity table (1x/3x/10x output) covers this. | No change |
| 4 | Low (proxy) | Token counts use `o200k_base`, not each provider's tokenizer. Ratios are what the study reports, and they are robust to a constant-factor tokenizer difference. | No change |
| 5 | Open | Not validated against a billed run (reading `cached_tokens` / `cache_read_input_tokens` from real API usage). Listed as the next check in the README. | Open |

**Data licences:** `nebius/SWE-agent-trajectories` (CC-BY-4.0) and `SWE-Gym/OpenHands-SFT-Trajectories` (MIT) are credited in the README. The raw parquet files are not committed; `fetch_data.py` downloads them. `out/runs.csv` holds only per-run token counts and run IDs derived from them.

**Second opinion:** none for this study (no second-model pass); the check here is full reproduction plus a line-by-line number audit.
