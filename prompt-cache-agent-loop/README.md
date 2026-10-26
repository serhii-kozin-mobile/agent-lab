# Study: what prompt caching actually saves on a real agent loop

Status: run 2026-10-25 (data + scripts + results below). Free: no API keys, no model calls, no spend. Independent review: `REVIEW.md`.

## Question
An agent loop re-sends its whole history on every model call. On real agent runs, how much of the input is re-sent,
and what share of the input bill does prompt caching remove under current provider rules, including the realistic
ways it fails (cache misses, a model switch, a volatile prefix)?

## Data (public, licensed for reuse)
- `nebius/SWE-agent-trajectories` (CC-BY-4.0): shard 1 of 12 (6,670 runs), seeded random sample of 2,000 (seed 20261025). SWE-agent framework, open Llama models (8B/70B/405B) solving GitHub issues. 1 run had no assistant turn and was dropped -> 1,999.
- `SWE-Gym/OpenHands-SFT-Trajectories` (MIT): all 491 successful OSS trajectories (OpenHands harness).
- Total: 2,490 runs, 61,945 model calls.

## Method
1. `measure_tokens.py`: tokenize every message with tiktoken `o200k_base` (a proxy; provider tokenizers differ). Model call k's prompt = every message before the k-th assistant message; its output = that assistant message.
2. `cost_model.py`: price each call in "base input units" (1.0 = one input token at normal price). Automatic prefix caching with one breakpoint at the end of each request: call k reads the previous prompt from cache and writes the new tail. Prompts under 1,024 tokens are not cacheable. A miss = the whole prompt is written again. Output priced at 5x base input (Anthropic list-price ratio).
3. `report.py`: charts + sensitivity to output volume.
Multipliers (provider docs, checked 2026-10-25):
- Anthropic: 5-min write 1.25x, 1-hour write 2x, read 0.1x (0.05x on Opus 5.5 / Sonnet 5.5). https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- OpenAI GPT-5.6+: 30-min write 1.25x, read 0.1x (0.05x GPT-6.1 Sol), min 1,024 tokens. https://developers.openai.com/api/docs/guides/prompt-caching
- OpenAI 2024 launch (GPT-4o): cached input 50% off, no write fee. https://openai.com/index/api-prompt-caching/

Reproduce: `./run_all.sh` (Python 3, ~100 MB download, < 1 min compute). Versions in `requirements.txt`.

## Results (real output of the scripts; full numbers in out/results.json)
| | SWE-agent | OpenHands | All |
|---|---|---|---|
| Runs | 1,999 | 491 | 2,490 |
| Model calls per run, median (p90) | 16 (51) | 16 (33) | 16 (48) |
| Input tokens per run, median | 89,390 | 163,186 | 108,646 |
| Output tokens per run, median | 1,583 | 2,095 | 1,719 |
| Input : output tokens (aggregate) | 108 : 1 | 103 : 1 | 107 : 1 |
| Share of input that was re-sent history | 95.0% | 93.4% | 94.7% |
| Input share of bill, no caching (output 5x) | 95.6% | 95.4% | 95.5% |

Input cost vs no caching (all runs):
| Scenario | Input cost | Whole bill |
|---|---|---|
| No caching | 100% | 100% |
| Cache, read 0.10x, write 1.25x (Anthropic 5-min, OpenAI GPT-5.6+ default) | 16.0% | 19.8% |
| Cache, read 0.05x, write 1.25x (Opus 5.5, Sonnet 5.5, GPT-6.1 Sol) | 11.3% | 15.2% |
| Cache, 1-hour write 2x, read 0.10x | 19.9% | 23.5% |
| 2024-style: read 0.5x, no write fee | 52.6% | 54.7% |
| Cache, 5% of calls miss | 21.5% | 25.0% |
| Cache, 10% of calls miss | 26.9% | 30.2% |
| Cache, 25% of calls miss | 43.3% | 45.8% |
| Cache, one model switch mid-run | 19.2% | 22.8% |
| Volatile prefix (timestamp/ID in system prompt: every call is a full write) | 125.0% | 123.9% |

Sensitivity (these traces come from non-reasoning models; reasoning models emit far more output):
| Output tokens x | Output share of bill (no cache) | Bill with cache (0.10 / 1.25) vs no cache |
|---|---|---|
| 1 | 4.5% | 19.8% |
| 3 | 12.3% | 26.3% |
| 10 | 31.9% | 42.8% |

Charts: out/input_cost_by_scenario.png, out/prompt_growth.png (median prompt size by call number; the flattening after ~40 calls is survivorship: short runs drop out and long ones hit context limits, not a real slowdown).

## Takeaways
- On these runs about 95 of every 100 input tokens were history the model had already seen. Input, not output, is the bill.
- Working caching cuts the input bill to about a sixth; a cheaper read rate (0.05x) to about a ninth.
- The failure modes matter more than the read discount: 10% of calls missing nearly doubles the cached cost (16% -> 27%); a timestamp in the system prompt makes caching 25% MORE expensive than not caching, because every call pays the write premium and never reads.
- The 1-hour TTL is not free: on bursty runs that never idle it costs more (20% vs 16%); it only wins when gaps exceed 5 minutes.

## What this study does not prove
- It is a cost model on recorded traces, not a billed run. Real cache hit rates depend on provider routing and eviction we can't see.
- Tokenizer is a proxy (o200k_base), so absolute token counts are approximate; ratios are robust to that.
- Traces are from 2024-25 open-model agents (SWE-agent, OpenHands); current harnesses compact or truncate history, which lowers the re-sent share.
- Miss rate is modelled as an independent per-call probability; real misses cluster (e.g. after a long tool call).
- No latency measured; caching also cuts time-to-first-token, which isn't counted here.
- Not yet validated against a billed run: a small real run that reads `cache_read_input_tokens` / `cached_tokens` from the API usage fields is the planned check.

## Files
fetch_data.py · measure_tokens.py · cost_model.py · report.py · run_all.sh · requirements.txt · out/ (runs.csv, results.json, sensitivity.md, charts). `out/calls.csv` (3 MB) and `out/costs_per_run.csv` are regenerated by `run_all.sh` and not committed; `data/` (~100 MB, downloaded by `fetch_data.py`) and `.venv/` are not committed either.
