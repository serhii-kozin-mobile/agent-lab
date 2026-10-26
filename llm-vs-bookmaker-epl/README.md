# Can a free local LLM beat the bookies? Premier League 2025-26, backtest only

**Question.** Given only what a fan knows before kickoff (league table, last-5 form, last season's finish), how close does a small local LLM get to bookmaker closing odds? Does handing it the bookmaker's own odds help? And does it give the same answer twice?

This is a **forecasting/calibration study on historical data**. No bets were placed, no betting account exists, and nothing here is a tip. The "backtest" is paper arithmetic.

## Setup
- **Matches:** all 380 Premier League 2025-26 matches. Results and odds come from football-data.co.uk (`mmz4281/2526/E0.csv`, free); baselines use 2021-22 to 2024-25.
- **Model:** Qwen2.5-3B-Instruct, Q4_K_M GGUF, run on CPU with llama-cpp-python 0.3.36 at 3 threads, with JSON-schema constrained output `{"home","draw","away"}` in %. It took a median 5.8 s per match. API cost: $0.
- **No look-ahead:** `features.py` builds every match's prompt from results *before* that match, and the table updates only after the prompt is recorded.
- **Leakage check** (`out/leakage_check.json`): Qwen2.5 was released in September 2024, before the 2025-26 season started (15 Aug 2025), so no result can be in its weights. As a sanity check it says the 2025-26 season "hasn't been decided yet". It also gets 2024-25 wrong (it names Manchester City as champions). So it is forecasting, not remembering.
- **Conditions:**
  1. `stats`: table + form + last season, temperature 0.
  2. `stats_odds`: the same plus the average pre-closing bookmaker odds, temperature 0. Per football-data's notes, pre-closing odds are collected on Friday afternoons for weekend games and Tuesday afternoons for midweek games, so they are always pre-kickoff.
  3. Rerun check: `stats` at temperature 0.7 with seeds 1 and 2, on the first 120 matches.
- **Baselines:** average closing odds (reference), average pre-closing odds, Pinnacle closing (only the first 210 matches; the CSV has no Pinnacle closing odds after that), a simple Elo (K=20, home advantage 60, carried over from 2021-25; a club never seen in 2021-25, i.e. Sunderland, starts at 1420), and historical base rates (H 44.5% / D 23.0% / A 32.4%). Odds become probabilities by proportional margin removal.
- **Metrics:** ranked probability score (RPS, the standard for ordered H/D/A), Brier, log loss, accuracy, and paired per-match RPS gap vs closing odds with a 4,000-sample bootstrap 95% CI.

## Results (380 matches; actual H 42.6% / D 27.4% / A 30.0%)
| Forecaster | RPS (lower=better) | Brier | Log loss | Accuracy | RPS gap vs closing (95% CI) |
|---|---|---|---|---|---|
| Closing odds (market avg) | 0.2045 | 0.6077 | 1.0118 | 49.5% | reference |
| Pre-closing odds (market avg) | 0.2053 | 0.6100 | 1.0153 | 49.5% | +0.0008 (-0.0012..+0.0028) |
| Elo baseline | 0.2095 | 0.6240 | 1.0425 | 48.7% | +0.0050 (-0.0013..+0.0110) |
| **LLM stats + odds** | 0.2190 | 0.6427 | 1.0612 | 38.9% | +0.0145 (+0.0044..+0.0248) |
| **LLM stats only** | 0.2255 | 0.6582 | 1.0894 | 39.5% | +0.0210 (+0.0091..+0.0332) |
| Base rates (2021-25) | 0.2278 | 0.6562 | 1.0845 | 42.6% (= always "home") | +0.0233 (+0.0112..+0.0345) |

What breaks:
1. **The LLM is no better than "predict the historical average".** RPS is 0.2255 vs 0.2278 for base rates (paired gap -0.0022, 95% CI -0.0140..+0.0101), and on Brier and log loss it is slightly worse than base rates. A short Elo script (0.2095) is far closer to the closing odds (0.2045) and beats the stats-only LLM by 0.0160 RPS (CI +0.0032..+0.0288). Against the LLM with odds the Elo edge (0.0094) is not significant (CI -0.0010..+0.0198). All paired gaps: `out/results.json` → `paired_head_to_head`.
2. **Given the odds, it made them worse.** The pre-closing odds alone score 0.2053. After the model "adjusted" them, the score is 0.2190: paired gap +0.0137 (CI +0.0035..+0.0240), also worse on Brier and log loss. Its home-win probability sits a mean 10.9 points away from the odds it was handed (correlation 0.73).
3. **It doesn't believe in home advantage, and it loves draws.** Average forecast: home 30.5%, draw 31.9%, away 37.7%. Reality: 42.6 / 27.4 / 30.0. It makes the draw its outright (strict) favourite in 22.1% of matches (43.7% when given odds); counting ties broken towards the draw by argmax it is 28.4% (51.6%). The closing odds never did, in any of the 380 matches.
4. **It answers from a template.** There are only 47 distinct probability triples across 380 matches, and `35/30/35` alone appears 89 times (23.4%). Its accuracy (39.5%, or 37.1% if tied top picks are split) is below the dumb rule "always pick home" (42.6%). 30% of its answers have a tie at the top (e.g. 35/30/35); `argmax` scores those as a home pick.
5. **Reruns:** at temperature 0.7, two seeds gave identical outputs on 70% of the first 120 matches. The argmax favourite flipped in 10% of them (e.g. `22/35/43` vs `20/60/20`); 7.5% are outright disagreements, the rest are ties broken differently. The mean of the largest per-match difference was 1.9 points.

Paper backtest, **not advice**: 1 unit on any outcome where the forecaster's probability beats the closing implied probability by 5 or more points, settled at the average closing odds. Results: LLM stats only -11.7% over 313 hypothetical bets (CI -30.3%..+8.1%); LLM stats + odds +3.4% over 338 (CI -15.9%..+23.3%); Elo -8.2% over 259 (CI -26.0%..+10.2%). Every CI crosses zero, so this is noise, not edge. Average closing odds are also not a price anyone actually gets.

Charts: `charts/rps_gap_vs_closing.png`, `charts/avg_forecast_vs_actual.png`, `charts/calibration.png`.

## Limits (read before quoting)
- One small 3B model at Q4 on CPU. Frontier models would do better; this measures the free-local tier, not "AI".
- One season, one league, 380 matches: the CIs are wide, so use the paired gaps, not the raw RPS bars.
- A single prompt format. Prompt wording almost certainly moves results; that is untested here.
- The Elo draw model is a simple heuristic, documented in `features.py`. K=20 / home advantage 60 happen to be close to the in-sample best for 2025-26 (best: K=15, HFA=60, RPS 0.2092); over K 10-40 and HFA 30-90 the Elo RPS ranges 0.209-0.215, always far ahead of the LLM (`elo_sensitivity.py`, `out/elo_sensitivity.json`).
- About a dozen paired comparisons are reported with unadjusted 95% CIs; treat ones that barely clear zero with caution.
- Brier here is the multi-class sum over H/D/A (range 0-2), not the binary Brier.
- Reruns cover only 120 matches (CPU time).
- No injuries, lineups, xG or news. The market has all of that, which is the point of the comparison.

## Reproduce
```
./run_all.sh          # downloads the data + the 2 GB model, ~1.5 h on 8 shared CPU cores

# rescoring only (seconds): the raw model answers are in out/llm_*.jsonl
python fetch_data.py && python features.py && python report.py
```
The football-data.co.uk CSVs and `out/matches_2526.json` (built from them) are not redistributed; see `DATA.md`.
On a shared CPU box, keep llama.cpp threads below the free core count. With 8 threads on a busy 8-core box, calls were several times slower than with 3 (an observation while setting up, not logged or benchmarked).

Files: `fetch_data.py` / `download_data.sh` (data download, see `DATA.md`), `features.py`, `leakage_check.py`, `run_llm.py`, `report.py`, `elo_sensitivity.py`, `REVIEW.md`, `out/results.json`, `out/results_table.md`, `out/llm_*.jsonl` (every raw model answer).
