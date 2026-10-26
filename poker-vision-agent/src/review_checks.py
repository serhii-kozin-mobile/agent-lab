"""Independent review checks (added 2026-10-26). Reads existing results only, no new runs.
Writes results/review_checks.json. Run from src/: python review_checks.py"""
from __future__ import annotations
import json, os, re
from collections import Counter
import numpy as np, pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
dec = pd.read_csv(os.path.join(R, "decisions.csv.gz"), low_memory=False)
hands = pd.read_csv(os.path.join(R, "hands.csv"))
out = {}

# 1. Guardrail anchoring: the chip-conservation check compares each read with the total stored at the
#    previous ACTED decision of the hand. The first acted read of a hand is never checked against
#    anything, so a misread there becomes the reference and later correct frames fail the check.
anch = {}
for run in ("hardened_guarded__realistic_mix", "hardened_guarded__scale_0.5", "hardened_guarded__chip_overlap",
            "hardened_guarded__blur_1.5", "hardened_guarded__clean"):
    g = dec[dec.run == run].sort_values(["hand", "dec"])
    last, a = {}, []
    for h, fb, sc in zip(g.hand, g.fallback.astype(bool), g.state_correct.astype(bool)):
        a.append(last.get(h))
        if not fb:
            last[h] = sc
    g = g.assign(anchor_ok=a, sc=g.state_correct.astype(bool), fb=g.fallback.astype(bool))
    fb_right = g[g.fb & g.sc]
    first_acted = g[~g.fb].groupby("hand").head(1)
    anch[run] = dict(decisions=int(len(g)), fallbacks=int(g.fb.sum()),
                     fallbacks_on_correct_frame=int(len(fb_right)),
                     of_which_anchor_was_a_misread=int((fb_right.anchor_ok == False).sum()),
                     fallbacks_after_misread_anchor_any=int((g.fb & (g.anchor_ok == False)).sum()),
                     first_acted_decision_wrong_share=float((~first_acted.sc).mean()),
                     frames_per_hand=float(g.frames.sum() / g.hand.nunique()))
out["guardrail_anchoring"] = anch

# 2. Where the naive realistic_mix loss sits (per-hand paired gap).
gt = hands[hands.run == "gt"].set_index("hand").result_bb
v = hands[hands.run == "naive__realistic_mix"].set_index("hand").result_bb
d = (v - gt).sort_values()
out["naive_realistic_tail"] = dict(total_gap_bb=float(d.sum()), worst30_bb=float(d.iloc[:30].sum()),
                                   best30_bb=float(d.iloc[-30:].sum()), other_2970_bb=float(d.iloc[30:].sum()),
                                   other_2970_bb100=float(d.iloc[30:].mean() * 100),
                                   hands_with_nonzero_gap=int((d != 0).sum()))

# 3. Misreads behind changed clicks (naive realistic_mix).
g = dec[dec.run == "naive__realistic_mix"]; mm = g[g.gt_action != g.vis_action]
c = Counter()
for f in ("pot", "hero_stack", "villain_stack", "hero_bet", "villain_bet"):
    b = mm[~mm[f"ok_{f}"].astype(bool)]
    for _, r in b.iterrows():
        c[f"{f}:{r[f'true_{f}']}->{r[f'seen_{f}']}:street{r.street}"] += 1
out["naive_realistic_changed_click_misreads_top"] = c.most_common(5)
out["naive_realistic_changed_clicks"] = int(len(mm))

# 4. Slumbot paired difference CI (independent samples) and villain bet sizes containing a '2'.
s = json.load(open(os.path.join(R, "slumbot_summary.json")))
out["slumbot_diff"] = dict(diff=s["vision"]["bb100"] - s["gt"]["bb100"],
                           ci95=float(np.hypot(s["vision"]["bb100_ci95"], s["gt"]["bb100_ci95"])))
sv = pd.read_csv(os.path.join(R, "slumbot", "slumbot_vision.csv"))
bets = Counter(int(x) for a in sv.action_str.dropna() for x in re.findall(r"b(\d+)", a))
out["slumbot_bet_sizes_top"] = bets.most_common(8)
out["slumbot_bets_with_digit_2_share"] = float(sum(n for b_, n in bets.items() if "2" in str(b_)) / max(1, sum(bets.values())))
json.dump(out, open(os.path.join(R, "review_checks.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
