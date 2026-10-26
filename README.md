# agent-lab

Small, reproducible experiments on how AI agents behave in practice: what they cost, where they break, and how they compare to simple baselines.

Write-ups: https://serhii-kozin-mobile.github.io/lab/

## How the lab works

- **One question per study.** Each folder answers one narrow question and says what it does *not* prove.
- **Public data.** Every dataset is public and credited. Third-party data that can't be redistributed is downloaded by a script, not copied here.
- **Free or local models.** No paid API was used for any result in this repo. Local models run on a CPU (llama.cpp, open weights).
- **Reproducible.** Each study has a `run_all.sh`, pinned versions, fixed seeds, and the raw results it reports.
- **Independent review.** Each study has a `REVIEW.md`: a separate review pass that reran the code, checked every number against the result files, and listed what was fixed.

## Studies

| Study | Question | Result (from the result files) |
|---|---|---|
| [`prompt-cache-agent-loop`](prompt-cache-agent-loop/) | How much of an agent loop's input bill does prompt caching remove on real runs, and what do cache misses do to it? | On 2,490 real agent runs (61,945 model calls), 94.7% of input tokens were re-sent history. Caching (read 0.10x, write 1.25x) cuts input cost to 16.0% of uncached; 10% misses raise it to 26.9%; a timestamp in the prefix makes it 125%. (`out/results.json`) |
| [`poker-vision-agent`](poker-vision-agent/) | Where does a screenshot-only agent break, when the same policy also plays the true state on the same seeded hands? | Cards were easy, numbers were not: full-state accuracy on a realistic noisy screen was 65.1% (naive reader) / 74.5% (hardened). 35.3% of naive decisions ran on a wrong state, but only 5.8% changed the click; gap vs ground truth −73.5 ± 49.2 bb/100 over 3,000 duplicate hands. (`results/summary.json`, `results/match_summary.csv`) |
| [`llm-vs-bookmaker-epl`](llm-vs-bookmaker-epl/) | How close does a free local 3B LLM get to bookmaker closing odds on all 380 Premier League 2025-26 matches? (backtest only, no betting) | RPS (lower is better): closing odds 0.2045, pre-closing odds 0.2053, Elo 0.2095, LLM given the odds 0.2190, LLM alone 0.2255, base rates 0.2278. Given the market odds, the model made them worse (+0.0137 RPS, CI +0.0035..+0.0240). (`out/results.json`) |

## Run a study

Each study is self-contained. Tested with Python 3.13 on Linux, CPU only.

```bash
cd prompt-cache-agent-loop && ./run_all.sh   # ~100 MB public dataset download, < 1 min compute
cd poker-vision-agent      && ./run_all.sh   # local simulator, ~40 min on 8 cores
cd llm-vs-bookmaker-epl    && ./run_all.sh   # downloads data + a 2 GB open model, ~1.5 h on CPU
```

Every study's README lists its data sources, method, results, limitations and the exact commands.

## Layout

```
<study>/
  README.md      question, data, method, results, limitations
  REVIEW.md      independent review: reproduction checks, issues found, what changed
  run_all.sh     reproduces everything end to end
  requirements.txt
  *.py or src/   code
  out/ or results/   result files (JSON/CSV) the README numbers come from
  charts/ or out/*.png
```

## Licence

Code and text: MIT, see `LICENSE`. Third-party data keeps its own licence (see each study's README). The DejaVu fonts in `poker-vision-agent/assets/fonts/` are under their own free licence (`LICENSE_DEJAVU.txt`).
