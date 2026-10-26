# Independent review: poker vision agent

Reviewer: independent agent review, 2026-10-26. Scope: this study folder, the companion social posts, and the article draft.
Slumbot was not contacted again. No paid API, no spend. The only network access was reading public web pages to check facts.

## Verdict: **ready after fixes** (fixes applied; placeholders remain)

The local study is solid. It's deterministic, the paired design is sound, and the numbers reproduce exactly. The review found one real design bug in the chip-conservation guardrail, which the draft had explained the wrong way. It also found a cost-estimate error and several misquoted or unsourced numbers. All of these are fixed in the files. The bug doesn't change any headline table: those describe the guardrail as it was run, and the bug is now disclosed with an experiment. What remains: re-checking the leaderboards and prices on publish day.

## Reproduction
| Check | Result |
|---|---|
| `perception_bench.py` on the first 200 of 2,000 states, all 8 conditions × 2 readers | identical to `results/perception_bench.csv` (every field except timing) |
| `run_matches.py` on the first 200 hands × all 17 runs | per-hand results, actions, state_correct and fallback identical to `hands.csv` / `decisions.csv.gz` |
| `analyze.py` rerun | every CSV and chart byte-identical, except `cost_estimate.json`/`summary.json` (cost fix below) |
| VLM probe | not rerun (~30 min CPU). "28/50 and 33/50 frames contain the hero stack" and "~18 s median" recomputed from the saved CSV: confirmed |
| Slumbot | not rerun (by instruction). Summary numbers recomputed from the saved CSVs: confirmed. Difference −382.4 ± 304.4 |

## Issues found
| # | Severity | Issue | Status |
|---|---|---|---|
| 1 | **High (code/design bug)** | **The chip-conservation guardrail anchors on an unchecked read.** `agent.py` compares each read with the total stored at the previous *acted* decision. The first acted read of a hand has nothing to compare against, so when it's wrong (29.3% of hands on realistic_mix) every later *correct* frame fails the check and the agent falls back to check/fold. On realistic_mix, 1,307 of the 1,819 "fallback on a correct frame" cases come from this, and 81% of the 9.2% acted-on-wrong-state decisions are those unchecked first reads. The draft attributed the bluntness to overlapping confidence scores. | The draft and README now say what actually happens. Analysis is in `src/review_checks.py` → `results/review_checks.json`. Experiment: `src/review_anchor_fix.py` (anchor accepted only when 2 frames agree), 3,000 hands, same seeds. It **did not help**: fallback 32.3% vs 30.5%, gap vs station −53.2 ± 57.7 vs −51.2 ± 57.7. The folds moved to the first decision (34.9% vs 24.2%). Disclosed in "What this doesn't prove". The headline tables are unchanged (they describe the guardrail as run). |
| 2 | Medium (number) | **The cost estimate's guarded column was overstated by about 26%.** It used 3.77 decisions per hand × 1.70 frames, but the guarded agent makes fewer decisions per hand (fallbacks end hands). Measured: 5.09 frames per hand. | `analyze.py` fixed, rerun. Guarded column: $10.93 → **$8.62**, $21.86 → **$17.24**, $5.33 → **$4.21**. The one-frame column is unchanged. |
| 3 | Medium (claim) | "The worst 1% of hands cost more than the whole net loss. **The rest roughly broke even.**" The other 2,970 hands were net **+33 bb/100**; the best 30 hands won back 79% of the loss. | Rewritten: misreads changed 518 hands, and both tails are big. |
| 4 | Medium (overreach) | "The eyes can cost more than the gap between the best and the worst model on those boards." That compares bb/100 across different opponents, units (BluffBench uses chips, multi-player) and harnesses. | Rewritten as "for a sense of scale only", with the caveats. |
| 5 | Medium (rule risk) | The defender section listed the exact fingerprints a screen-reading bot leaves (pipeline next to the client, machine-speed decisions, consistent strategy) and "time to act". Read in reverse, that's an evasion checklist. | Trimmed: the "irony" paragraph is removed and the interview bullet is summarised at a high level. Policy facts are kept, all re-verified. |
| 6 | Low (number) | Hardened villain-bet accuracy on Slumbot was "75%" in one place and 76% in another (file: 0.760). | Fixed → 76% |
| 7 | Low (rounding) | Guarded fallback "30.6%" (file: 0.30546 → 30.5%). | Fixed in the draft and README |
| 8 | Low (number) | "±25 to ±70 bb/100 on the gap". The files say ±22.9 to ±126.1. | Fixed → ±23 to ±126 |
| 9 | Low (claim) | "Slumbot bets 200, 250, 300 all the time". 250 doesn't appear among the common sizes; 200 dominates. | Fixed |
| 10 | Low (claim) | "Most common **expensive** flip: 100 → 106". It's the most common misread behind a changed click (249 + 69 of 628), and cost wasn't measured. | Reworded |
| 11 | Low (claim) | Dark theme: "matched **almost every** card to the '10' template". It's 81% of misread cards. | Fixed → "four out of five" |
| 12 | Low (clarity) | Latency sentence mixed naive p50 (25 ms) with guarded p95 (87 ms). | Split: naive p50 25 ms; guarded p50 29 / p95 87 ms |
| 13 | Low (sources) | GTO Wizard "±" is the paper's standard deviation, not a 95% CI. Kaggle BB/100 is relative to the other LLMs in the pool. BluffBench was last updated 2026-05-20 and is multi-player, in chips. "Promo price" isn't Google's wording (it's a scheduled price change). | Clarified in the draft and README |
| 14 | Low (wording) | "Numbers broke everywhere". They didn't break on clean, JPEG or four-colour. | Fixed |
| 15 | Low | The guardrail description said "chips add up to what they were at the start of the hand". The code compares with the last acted read. | Fixed |
| 16 | Info | The perception threshold calibration (seed 11) is disjoint from the match seeds (2026). The hardened reader and the guardrails were designed after looking at the failures (disclosed). | No change |
| 17 | Info | A prepared, never-connected client for a third-party bot-vs-bot API was in the working folder. It is not used by any result. | Resolved: left out of the public release; nothing imports it. |
| 18 | Info | `assets/fonts` (DejaVu) can be redistributed, but add the DejaVu license file before publishing. | Resolved: `assets/fonts/LICENSE_DEJAVU.txt` |

**Verified external facts (fetched 2026-10-26):** GTO Wizard Benchmark arXiv 2603.23660: all quoted rows match Table 2, and the 2% suited/offsuit note matches §4.3. Kaggle Heads-Up Poker: updated Oct 8, 2026, GPT-5.6 Sol 30 (−5/+5), 10,000 duplicated decks, text harness. BluffBench: v0.3 beta, GPT-5.5 xHigh +1,568. Gemini pricing (page updated 2026-10-22): 3.8 Flash $0.75/$3.75 through Dec 31, 2026, then $1.50/$7.50; 3.5 Flash-Lite $0.30/$2.50. Media resolution: 1,120 tokens/image default for Gemini 3/4. PokerStars prohibited-tools page, the RTA article (>95% proactive detection, two-agent review, confiscated balances used to compensate players), the integrity interview (four data types, ML accuracy platform), and GGPoker SEP v20260313 (bots, RTA, remote access, VMs/emulators, VPNs, data mining): all match. Slumbot as the 2018 ACPC winner is confirmed by the GTO Wizard paper.

**Content rules:** no personal data, employer or email address. Nothing is exploit or anti-detection material after fix #5. Pass.

**Second opinion:** I ran Qwen2.5-3B (local, CPU) as a second reviewer on the original draft. It independently flagged the "75%" line, the "worst 1% / rest broke even" claim and the "vision layer in front of any of these" sentence. Most of its other output was noise.

## Headline/number changes in the draft
- Cost table guarded column: $10.93 / $21.86 / $5.33 → **$8.62 / $17.24 / $4.21** (header: "5.1 frames per hand")
- Fallback 30.6% → 30.5%; Slumbot hardened villain bet 75% → 76%; gap CI range ±25–70 → ±23–126
- New: the guardrail anchoring bug and the two-frame experiment (−53 vs −51 bb/100 vs station; 32.3% vs 30.5% fallback)
- No change to the perception table, the mismatch table, the bb/100 gaps or the Slumbot table

## Editorial notes (not applied, optional)
- Title is OK but long. Alternatives: "My Screen-Reading Agent Read Every Card and Misread the Numbers" or "Cards Were Easy, Numbers Were Not: A Poker Agent That Only Sees Screenshots".
- The draft is about 3,000 words. The VLM-cost and leaderboard sections could move to the README if Medium read-time matters.

## Changelog (files)
- `src/analyze.py`: the guarded cost uses measured frames per hand (`frames_per_hand_guarded_realistic_mix` added to `cost_estimate.json`).
- `src/review_checks.py` (new) → `results/review_checks.json`.
- `src/review_anchor_fix.py` (new) → `results/review_anchor_fix/` (hands, decisions, summary).
- `run_all.sh`: runs both review scripts. `README.md`, the social posts (numbers verified, all ≤280 chars) and the draft are edited as above.
- Regenerated: `results/cost_estimate.json`, `results/summary.json`, `results/analyze.log`. Every other result and chart is byte-identical.
