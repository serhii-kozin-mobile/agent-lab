# Independent review: LLM vs bookmakers, EPL 2025-26 (backtest)

Reviewer: independent agent review, 2026-10-26. Scope: this study folder, the companion social posts, and the article draft.
No external API was called and no money was spent. The only network access was reading public web pages to check facts.

## Verdict: **ready after fixes** (fixes applied; two placeholders remain)

The core claims hold up, and the data, the code and the leakage story are sound. The draft overstated a few things, and one baseline had a small bug. Both are fixed now. The remaining pre-publish items (repo link, data redistribution) are resolved in this public release.

## Reproduction
| Check | Result |
|---|---|
| `features.py` rebuilt from the CSVs in `data/` | byte-identical `out/matches_2526.json` (before the Elo fix) |
| `report.py` rescoring | byte-identical `out/results.json` (before the fixes) |
| LLM re-run, Qwen2.5-3B Q4_K_M, temp 0, first 25 matches × both conditions (CPU) | 50/50 answers identical to `out/llm_*.jsonl` |
| Full 380-match LLM rerun | not done (~1.5 h); the deterministic subset matched exactly |

## Issues found
| # | Severity | Issue | Status |
|---|---|---|---|
| 1 | High (claim) | **"The LLM barely beat 'predict the average'."** The paired RPS gap is -0.0022 (95% CI -0.0140..+0.0101). On Brier and log loss the LLM is slightly *worse* than base rates. It's a tie, not a win. | Fixed: "no better than" (subtitle, README) |
| 2 | High (number) | **Draw-favourite share was inflated by argmax tie-breaking.** 30% of stats-only answers tie at the top (e.g. 35/30/35, 20/40/40). The "single most likely outcome" share is 22.1% (43.7% with odds), not 28.4% (51.6%). | Fixed in the draft, README and the social post; tie-aware stats added to `results.json` (`ties_and_favourites`) |
| 3 | Medium (claim) | **"Elo beat the LLM by a wide margin."** That holds against stats-only (+0.0160, CI +0.0032..+0.0288). Against the LLM with odds it's +0.0094, CI -0.0010..+0.0198, which is not significant. | Fixed: scoped to stats-only, with the CI |
| 4 | Medium (missing evidence) | **The headline "it made the odds worse" was only shown vs closing odds**, not paired against the pre-closing odds the model was given. Paired: +0.0137 RPS (CI +0.0035..+0.0240), and also worse on Brier and log loss. The headline holds. | Added to the draft and README; `paired_head_to_head` in `results.json` |
| 5 | Medium (code bug) | **Elo prior inconsistency.** Unseen clubs used 1420 to forecast but 1450 to update (`features.py`). Only Sunderland is affected in 2025-26, and in history every first appearance used 1450. | Fixed (one `NEW_TEAM = 1420` constant), reran features + report + charts. Elo RPS 0.2094 → **0.2095**, gap +0.0049 → **+0.0050** (CI -0.0013..+0.0110), accuracy 48.9% → 48.7%, Elo paper ROI -5.1%/261 bets → **-8.2%/259** (CI -26.0%..+10.2%). The shared bootstrap RNG also shifted the LLM ROI CIs slightly: stats-only now CI -30.3%..+8.1%, with odds -15.9%..+23.3%. No conclusion changed. |
| 6 | Medium (overfitting risk) | **Elo K=20 / HFA=60 sit close to the in-sample optimum** for 2025-26 (best K=15, HFA=60, 0.2092). They may have been chosen as textbook values, but nothing documents that. | Sensitivity added (`elo_sensitivity.py`, `out/elo_sensitivity.json`): over K 10-40 × HFA 30-90 the Elo RPS stays in 0.209-0.215, always far ahead of the LLM (0.2255). Disclosed in Limits |
| 7 | Medium (number) | README said "every CI **except the first** crosses zero" for the paper backtest. All three cross zero. | Fixed |
| 8 | Medium (no source) | "8 threads made the call **15x** slower" / "15-22 s instead of 1.3 s": no log supports it, and 1.3 s contradicts the measured 5.8 s median at 3 threads. | Softened to "several times slower", labelled an unlogged observation |
| 9 | Low | Rerun "favourite flipped in 10%" is argmax-based. Outright disagreements are 7.5%; the rest are ties that broke differently. | Disclosed in the draft and README; `favourite_disjoint_share` in `results.json` |
| 10 | Low | Accuracy 39.5% scores tied picks as home. With ties split it's 37.1%, so the claim "below always-home (42.6%)" still holds. | Disclosed |
| 11 | Low | Leakage: the self-report check is weak evidence on its own. The decisive argument is the release date (Qwen2.5, Sept 2024) vs the season start (15 Aug 2025). | Draft and README now lead with the release date |
| 12 | Low | "Earlier odds" timing wasn't explained. football-data's notes: pre-closing odds are collected Friday afternoon (weekend games) and Tuesday afternoon (midweek), so they're always pre-kickoff. Verified at football-data.co.uk/notes.txt. | Added to the draft and README |
| 13 | Low | Multiple comparisons: about a dozen unadjusted 95% CIs. | Caveat added to Limits |
| 14 | Low | Brier here is the multi-class sum (range 0-2). | Noted in README |
| 15 | Low | "Last season's finish" ranks 2024-25 by points only (no goal-difference tiebreak). It could misorder tied clubs in the prompt. | Not fixed (cosmetic; it's prompt text only) |
| 16 | Editorial | Generic opening hook ("'Can AI beat the bookies?' comes up a lot"). The title said "Free Local LLM", but the evidence covers one 3B model. Units were mixed (RPS vs "10.9 points"). | Rewritten: the hook leads with the result, the title says "Small Local LLM", units are spelled out as percentage points |
| 17 | Editorial | A social post was 313 characters (over 280). | Trimmed to 265 |

**Checked and correct:** RPS formula (cumulative, /(r-1)=2), proportional margin removal, paired bootstrap, no look-ahead in `features.py` (prompt recorded before the table update), all 380 matches have both odds sets, the Pinnacle closing column is empty from match 210 (17 Jan 2026) on, as stated, and every number in the draft now traces to `out/results.json` or `out/elo_sensitivity.json`.

**Content rules:** no personal data, no employer or email address. Betting is framed as a paper backtest, nothing reads as a tip, and no bookmaker brands appear. Pass.

**Second opinion:** I ran Qwen2.5-3B (local, CPU) as a second reviewer over the original draft. It also flagged the 28.4% draw claim and the "wide margin" wording, and it tripped on the mixed RPS/percentage-point units, which is now clarified. Most of its other output was noise.

## Headline numbers that changed
- Elo row: 0.2094 → 0.2095; gap +0.0049 (-0.0016..+0.0109) → +0.0050 (-0.0013..+0.0110)
- Draw as outright favourite: 28.4% / 51.6% → 22% / 44%
- Elo paper ROI: -5.1% over 261 → -8.2% over 259; LLM ROI CIs moved by ≤0.5 pt
- "barely beat the average" → "no better than the average"

## Pre-publish items
- Data: the football-data.co.uk CSVs are not redistributed in the public repo. `fetch_data.py` downloads them; see `DATA.md`. (Resolved at release.)
- Headline suggestion (optional): "I Gave a Small LLM the Bookmaker's Odds. Its Forecasts Got Worse." The current title is accurate and fine.

## Changelog (files)
- `features.py`: one `NEW_TEAM` prior for unseen clubs (bug fix).
- `report.py`: adds `paired_head_to_head` and `ties_and_favourites` plus tie-aware rerun stats, using a separate RNG so the existing numbers aren't disturbed.
- `elo_sensitivity.py` (new), `out/elo_sensitivity.json` (new); `run_all.sh` calls it.
- Regenerated: `out/matches_2526.json`, `out/results.json`, `out/results_table.md`, `charts/*.png`. Copied `charts/rps_gap_vs_closing.png` to the drafts folder.
- `README.md`, the social posts and the article draft are edited as listed above.
