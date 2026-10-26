"""Turn raw run outputs into summary CSV/JSON + charts. No new runs here."""
from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
C = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "charts")
os.makedirs(C, exist_ok=True)
ORDER = ["clean", "jpeg_q20", "theme_four_color", "theme_dark", "chip_overlap", "scale_0.5", "blur_1.5", "realistic_mix"]
NUMF = ["pot", "hero_stack", "villain_stack", "hero_bet", "villain_bet"]


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def perception():
    d = pd.read_csv(os.path.join(R, "perception_bench.csv"))
    rows = []
    for (mode, cond), g in d.groupby(["mode", "condition"]):
        r = dict(mode=mode, condition=cond, frames=len(g),
                 card_acc=g.cards_correct.sum() / g.cards_total.sum(),
                 full_state_acc=g.state_correct.mean(),
                 ms_mean=g.ms.mean(), ms_p95=g.ms.quantile(.95))
        for f in ["hero_cards", "board"] + NUMF + ["button"]:
            r[f"acc_{f}"] = g[f"ok_{f}"].mean()
        r["numbers_all_ok"] = g[[f"ok_{f}" for f in NUMF]].all(axis=1).mean()
        rows.append(r)
    acc = pd.DataFrame(rows)
    acc["condition"] = pd.Categorical(acc.condition, ORDER)
    acc = acc.sort_values(["mode", "condition"])
    acc.to_csv(os.path.join(R, "perception_accuracy.csv"), index=False, float_format="%.4f")

    # card confusions
    conf = Counter()
    kinds = Counter()
    for (mode, cond), g in d.groupby(["mode", "condition"]):
        for s in g.card_errors.dropna():
            for e in s.split(";"):
                a, b = e.split(">")
                conf[(mode, cond, a, b)] += 1
                k = "rank+suit" if a[0] != b[0] and a[1] != b[1] else ("rank" if a[0] != b[0] else "suit")
                kinds[(mode, cond, k)] += 1
    cc = pd.DataFrame([dict(mode=m, condition=c, true=a, read=b, n=n) for (m, c, a, b), n in conf.items()])
    if len(cc):
        cc = cc.sort_values("n", ascending=False)
    cc.to_csv(os.path.join(R, "card_confusions.csv"), index=False)
    kk = pd.DataFrame([dict(mode=m, condition=c, error_kind=k, n=n) for (m, c, k), n in kinds.items()])
    kk.to_csv(os.path.join(R, "card_error_kinds.csv"), index=False)

    # digit errors: examples + a crude taxonomy
    ex = []
    for f in NUMF:
        bad = d[~d[f"ok_{f}"]]
        for _, r in bad.iterrows():
            t, s = int(r[f"true_{f}"]), int(r[f"seen_{f}"])
            ts, ss = str(t), str(s)
            if s == 0 and t > 0:
                kind = "missed (read 0)"
            elif len(ss) > len(ts):
                kind = "extra digits"
            elif len(ss) < len(ts):
                kind = "dropped digits"
            else:
                kind = "wrong digit(s)"
            ex.append(dict(mode=r["mode"], condition=r.condition, field=f, true=t, read=s, kind=kind,
                           ratio=(s / t if t else np.nan)))
    ex = pd.DataFrame(ex)
    ex.to_csv(os.path.join(R, "number_errors.csv"), index=False)
    return acc, cc, kk, ex


def matches():
    h = pd.read_csv(os.path.join(R, "hands.csv"))
    dec = pd.read_csv(os.path.join(R, "decisions.csv.gz"), low_memory=False)
    gt = h[h.run == "gt"].set_index("hand").result_bb
    rows = []
    for run, g in h.groupby("run"):
        g = g.set_index("hand").result_bb
        m, e = ci(g)
        r = dict(run=run, hands=len(g), bb100=m * 100, bb100_ci95=e * 100)
        if run != "gt":
            diff = (g - gt.loc[g.index])
            dm, de = ci(diff)
            r.update(gap_bb100=dm * 100, gap_ci95=de * 100, hands_changed=(diff != 0).mean())
            dd = dec[dec.run == run]
            r.update(decisions=len(dd), wrong_state_rate=1 - dd.state_correct.mean(),
                     action_mismatch_rate=(dd.gt_action != dd.vis_action).mean(),
                     mismatch_given_wrong_state=(dd[~dd.state_correct].gt_action != dd[~dd.state_correct].vis_action).mean()
                     if (~dd.state_correct).any() else 0.0,
                     fallback_rate=dd.fallback.mean(), frames_mean=dd.frames.mean(),
                     ms_mean=dd.total_ms.mean(), ms_p50=dd.total_ms.median(), ms_p95=dd.total_ms.quantile(.95),
                     perceive_ms_mean=dd.perceive_ms.mean(), policy_ms_mean=dd.policy_ms.mean())
            cfg, cond = run.split("__")
            r.update(config=cfg, condition=cond)
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(R, "match_summary.csv"), index=False, float_format="%.4f")

    # split by opponent style: folding more "helps" against the bot our policy loses to
    from game import hand_setup
    style = pd.Series({i: hand_setup(i, json.load(open(os.path.join(R, "match_config.json")))["base_seed"])["opp_style"]
                       for i in gt.index})
    srows = []
    for run, g in h.groupby("run"):
        g = g.set_index("hand").result_bb
        for st in ("tag", "station"):
            idx = style[style == st].index
            m, e = ci(g.loc[idx])
            r = dict(run=run, opponent=st, hands=len(idx), bb100=m * 100, bb100_ci95=e * 100)
            if run != "gt":
                dm, de = ci(g.loc[idx] - gt.loc[idx])
                r.update(gap_bb100=dm * 100, gap_ci95=de * 100)
            srows.append(r)
    pd.DataFrame(srows).to_csv(os.path.join(R, "match_by_opponent.csv"), index=False, float_format="%.3f")

    # guardrail diagnostics: what fired, and were the fallbacks needed?
    grows = []
    for run, dd in dec[dec.guard == True].groupby("run"):
        acted = dd[~dd.fallback]
        probs = dd.problems.fillna("").astype(str).str.split("|").explode()
        probs = probs[probs != ""].value_counts()
        grows.append(dict(run=run, decisions=len(dd), fallback_rate=dd.fallback.mean(),
                          acted_on_wrong_state=(~acted.state_correct).sum() / len(dd),
                          wrong_given_acted=(~acted.state_correct).mean(),
                          fallback_but_last_frame_was_right=dd[dd.fallback].state_correct.mean() if dd.fallback.any() else np.nan,
                          **{f"flag_{k}": int(v) for k, v in probs.items()}))
    pd.DataFrame(grows).to_csv(os.path.join(R, "guardrail_diagnostics.csv"), index=False, float_format="%.4f")
    vis = dec.copy()
    mt = vis[vis.gt_action != vis.vis_action].groupby(["run", "gt_action", "vis_action"]).size().reset_index(name="n")
    mt.to_csv(os.path.join(R, "action_mismatches.csv"), index=False)
    return s, mt


def slumbot():
    out = {}
    done = {}
    for mode in ("gt", "vision"):
        p = os.path.join(R, "slumbot", f"slumbot_{mode}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            done[mode] = d[d.winnings.notna()].hand.nunique()
    # same, even number of completed hands for both modes (a killed run may leave a partial hand)
    cap = (min(done.values()) // 2) * 2 if done else 0
    out["hands_used_per_mode"] = int(cap)
    for mode in ("gt", "vision"):
        p = os.path.join(R, "slumbot", f"slumbot_{mode}.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d.hand < cap]
        w = d[d.winnings.notna()].groupby("hand").winnings.last()
        # bb/100 = mean chips per hand / 100 chips per bb * 100 hands
        m, e = ci(w.values)
        r = dict(hands=int(len(w)), bb100=float(m), bb100_ci95=float(e),
                 decisions=int((d.dec >= 0).sum()))
        if mode == "vision":
            v = d[d.dec >= 0]
            r.update(wrong_state_rate=float(1 - v.state_correct.astype(bool).mean()),
                     action_mismatch_vs_gt=float((v.gt_action != v.vis_action).mean()),
                     hardened_wrong_state_rate=float(1 - v.hard_state_correct.astype(bool).mean()),
                     hardened_action_mismatch_vs_gt=float((v.gt_action != v.hard_action).mean()),
                     perceive_ms_mean=float(v.perceive_ms.mean()))
        out[mode] = r
    json.dump(out, open(os.path.join(R, "slumbot_summary.json"), "w"), indent=2)
    return out


def cost_estimate(s):
    """ESTIMATE, not a run: what the perception step would cost with a hosted
    VLM reading one screenshot per decision. Prices read from Google's public
    pricing page on 2026-10-26 (see README for URLs)."""
    gt_dec = None
    dec = pd.read_csv(os.path.join(R, "decisions.csv.gz"), usecols=["run", "hand"])
    per_hand = dec[dec.run == "naive__clean"].groupby("hand").size()
    h = pd.read_csv(os.path.join(R, "hands.csv"))
    n_hands = h[h.run == "naive__clean"].shape[0]
    dph = len(dec[dec.run == "naive__clean"]) / n_hands
    guarded = s[s.run == "hardened_guarded__realistic_mix"].frames_mean.iloc[0] if "frames_mean" in s else 1.0
    # Review fix (2026-10-26): the guarded agent makes fewer decisions per hand (fallback folds end hands
    # early), so price the frames it actually captured per hand instead of dph * frames_per_decision.
    gd = pd.read_csv(os.path.join(R, "decisions.csv.gz"), usecols=["run", "hand", "frames"])
    gd = gd[gd.run == "hardened_guarded__realistic_mix"]
    frames_per_hand_guarded = gd.frames.sum() / h[h.run == "hardened_guarded__realistic_mix"].shape[0]
    assumptions = dict(image_tokens=1120, prompt_tokens=400, output_tokens=150,
                       image_tokens_source="Gemini API docs, media_resolution: default/high = 1120 tokens per image for Gemini 3/4 models",
                       note="no thinking tokens assumed; thinking would add output-priced tokens")
    models = {
        "gemini-3.8-flash (standard, through 2026-12-31)": (0.75, 3.75),
        "gemini-3.8-flash (standard, from 2027-01-01)": (1.50, 7.50),
        "gemini-3.5-flash-lite (standard)": (0.30, 2.50),
    }
    est = {}
    for name, (pin, pout) in models.items():
        per_call = (assumptions["image_tokens"] + assumptions["prompt_tokens"]) * pin / 1e6 + assumptions["output_tokens"] * pout / 1e6
        est[name] = dict(usd_per_call=per_call, usd_per_1000_hands_one_frame=per_call * dph * 1000,
                         usd_per_1000_hands_guarded=per_call * frames_per_hand_guarded * 1000)
    out = dict(kind="ESTIMATE (not a run)", decisions_per_hand_measured=dph,
               frames_per_decision_guarded_realistic_mix=float(guarded),
               frames_per_hand_guarded_realistic_mix=float(frames_per_hand_guarded),
               assumptions=assumptions, estimates=est,
               pricing_source="https://ai.google.dev/gemini-api/docs/pricing (read 2026-10-26)")
    json.dump(out, open(os.path.join(R, "cost_estimate.json"), "w"), indent=2)
    return out


def charts(acc, s, kk):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    # 1. full-state accuracy
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(ORDER))
    for k, (mode, col) in enumerate([("naive", "#c0392b"), ("hardened", "#2471a3")]):
        a = acc[acc["mode"] == mode].set_index("condition").reindex(ORDER)
        b = ax.bar(x + (k - .5) * .38, a.full_state_acc * 100, .38, label=f"{mode} reader", color=col)
        for xi, v in zip(x + (k - .5) * .38, a.full_state_acc * 100):
            ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x, [o.replace("_", "\n") for o in ORDER])
    ax.set_ylabel("% screenshots read 100% correctly")
    ax.set_ylim(0, 118)
    ax.set_title("Same table, same code: how often the agent reads EVERYTHING right")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(C, "1_full_state_accuracy.png"), dpi=150)
    plt.close(fig)

    # 2. decision gap vs ground truth (paired), split by opponent
    bo = pd.read_csv(os.path.join(R, "match_by_opponent.csv"))
    bo = bo[bo.run != "gt"].copy()
    bo["config"] = bo.run.str.split("__").str[0]
    bo["condition"] = bo.run.str.split("__").str[1]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, opp, title in [(axes[0], "station", "vs calling station (our policy wins: +59 bb/100 on true state)"),
                           (axes[1], "tag", "vs tight-aggressive bot (our policy loses: -215 bb/100)")]:
        for k, (cfg, col, lab) in enumerate([("naive", "#c0392b", "naive reader"),
                                             ("hardened_guarded", "#2471a3", "hardened reader + guardrails")]):
            a = bo[(bo.config == cfg) & (bo.opponent == opp)].set_index("condition").reindex(ORDER)
            ax.bar(x + (k - .5) * .38, a.gap_bb100.clip(lower=-400), .38, yerr=a.gap_ci95, capsize=3, label=lab, color=col)
            for xi, v in zip(x + (k - .5) * .38, a.gap_bb100):
                if v < -400:
                    ax.text(xi, -395, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="white", rotation=90)
        ax.axhline(0, color="black", lw=.8)
        ax.set_xticks(x, [o.replace("_", "\n") for o in ORDER], fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-400, 320)
    axes[0].set_ylabel("bb/100: vision agent minus true-state agent\n(same hands, 95% CI)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("What misreads cost in money terms. Falling back to check/fold looks like a 'win' only against the bot our policy already loses to.", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(C, "2_decision_gap_bb100.png"), dpi=150)
    plt.close(fig)

    # 3. which fields break (naive), heatmap
    fields = ["hero_cards", "board"] + NUMF + ["button"]
    a = acc[acc["mode"] == "naive"].set_index("condition").reindex(ORDER)
    m = np.array([[100 * (1 - a.loc[c, f"acc_{f}"]) for f in fields] for c in ORDER])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(m, cmap="Reds", vmin=0, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if m[i, j] > 55 else "black")
    ax.set_xticks(range(len(fields)), [f.replace("_", "\n") for f in fields])
    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.set_title("Naive reader: % of screenshots where this field is wrong")
    fig.colorbar(im, ax=ax, fraction=.03)
    fig.tight_layout()
    fig.savefig(os.path.join(C, "3_field_error_heatmap.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    acc, cc, kk, ex = perception()
    s, mt = matches()
    sl = slumbot()
    ce = cost_estimate(s)
    charts(acc, s, kk)
    summary = dict(perception=acc.to_dict(orient="records"), matches=s.to_dict(orient="records"),
                   slumbot=sl, cost_estimate=ce)
    json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=2, default=float)
    print(acc[["mode", "condition", "card_acc", "full_state_acc", "numbers_all_ok", "ms_mean"]].to_string(index=False))
    print(s.drop(columns=[c for c in ["config", "condition"] if c in s]).to_string(index=False))
    print(json.dumps(sl, indent=1))
    print(json.dumps(ce["estimates"], indent=1))
