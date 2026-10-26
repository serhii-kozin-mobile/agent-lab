"""Score every forecaster on the 2025-26 EPL season and draw the charts. Pure numpy/matplotlib."""
import json, pathlib, numpy as np, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
rng = np.random.default_rng(7)
J = json.load(open("out/matches_2526.json")); M = J["matches"]; N = len(M)
Y = np.array([[r["result"] == k for k in "HDA"] for r in M], float)
def load_llm(name):
    p = pathlib.Path(f"out/llm_{name}.jsonl")
    if not p.exists(): return None, 0
    d = {}; bad = 0
    for l in p.open():
        r = json.loads(l); d[r["i"]] = r
    P = np.full((N, 3), np.nan); secs = []
    for i, r in d.items():
        if r["p"]: P[i] = r["p"]
        else: bad += 1
        secs.append(r["secs"])
    return P, dict(n=len(d), invalid=bad, median_secs=float(np.median(secs)) if secs else None)
F = {"Closing odds (market avg)": np.array([r["p_close"] for r in M]),
         "Pre-closing odds (market avg)": np.array([r["p_open"] for r in M]),
     "Elo baseline": np.array([r["p_elo"] for r in M]),
     "Base rates (2021-25)": np.tile(J["meta"]["base_rates"], (N, 1))}
runs = {"LLM stats only": "stats_t0.0_s0", "LLM stats + odds": "stats_odds_t0.0_s0",
        "LLM stats only (t=0.7, seed 1)": "stats_t0.7_s1", "LLM stats only (t=0.7, seed 2)": "stats_t0.7_s2"}
info = {}; R = {}
for k, v in runs.items():
    P, inf = load_llm(v)
    if P is not None:
        (R if "t=0.7" in k else F)[k] = P; info[k] = inf
# eval only on matches where every forecaster has a valid prediction
mask = np.all([~np.isnan(P).any(1) for P in F.values()], axis=0)
def brier(P, Y): return ((P - Y) ** 2).sum(1)
def rps(P, Y): c = np.cumsum(P, 1)[:, :2]; o = np.cumsum(Y, 1)[:, :2]; return ((c - o) ** 2).sum(1) / 2
def logl(P, Y): return -np.log(np.clip((P * Y).sum(1), 1e-6, 1))
def boot(x, B=4000):
    idx = rng.integers(0, len(x), (B, len(x))); m = x[idx].mean(1); return np.percentile(m, [2.5, 97.5])
res = {}
ref = "Closing odds (market avg)"
for k, P in F.items():
    P = P[mask]; y = Y[mask]
    b, r, l = brier(P, y), rps(P, y), logl(P, y)
    d = r - rps(F[ref][mask], y)
    res[k] = dict(brier=b.mean(), rps=r.mean(), rps_ci=boot(r).tolist(), logloss=l.mean(),
                  acc=(P.argmax(1) == y.argmax(1)).mean(), mean_p=P.mean(0).tolist(),
                  rps_minus_close=d.mean(), rps_minus_close_ci=boot(d).tolist(),
                  share_draw_pick=(P.argmax(1) == 1).mean())
# paper backtest (NOT betting advice): flat 1 unit on the outcome where forecaster's prob exceeds
# the closing implied prob by >= 5 points, settled at the closing average decimal odds.
odds_c = np.array([r["odds_close"] for r in M])
def backtest(P, thr=0.05):
    pnl = []
    for i in np.where(mask)[0]:
        e = P[i] - F[ref][i]; j = int(e.argmax())
        if e[j] >= thr: pnl.append(odds_c[i, j] - 1 if Y[i, j] else -1.0)
    pnl = np.array(pnl); return dict(bets=len(pnl), roi=pnl.mean() if len(pnl) else None,
                                     roi_ci=boot(pnl).tolist() if len(pnl) > 5 else None)
bt = {k: backtest(F[k]) for k in F if k.startswith("LLM") or k == "Elo baseline"}
# behaviour: output diversity / collapse
beh = {}
for k, v in runs.items():
    p = pathlib.Path(f"out/llm_{v}.jsonl")
    if not p.exists(): continue
    raws = [tuple(json.loads(l)["raw"] or ()) for l in p.open()]
    c = collections.Counter(raws)
    beh[k] = dict(distinct=len(c), top=[(list(t), n) for t, n in c.most_common(5)],
                  share_top1=c.most_common(1)[0][1] / len(raws))
# rerun consistency
cons = None
if "LLM stats only (t=0.7, seed 1)" in R and "LLM stats only (t=0.7, seed 2)" in R:
    A, B = R["LLM stats only (t=0.7, seed 1)"], R["LLM stats only (t=0.7, seed 2)"]
    ok = ~np.isnan(A).any(1) & ~np.isnan(B).any(1)
    cons = dict(n=int(ok.sum()), mean_abs_diff_home_pts=float(np.abs(A[ok, 0] - B[ok, 0]).mean() * 100),
                max_abs_diff_home_pts=float(np.abs(A[ok, 0] - B[ok, 0]).max() * 100),
                argmax_flip_share=float((A[ok].argmax(1) != B[ok].argmax(1)).mean()))
    S = F["LLM stats only"]; ok2 = ok & ~np.isnan(S).any(1)
    cons["argmax_flip_vs_t0_seed1"] = float((A[ok2].argmax(1) != S[ok2].argmax(1)).mean())
    cons["identical_output_share"] = float((np.abs(A[ok] - B[ok]).max(1) < 1e-9).mean())
    cons["mean_max_abs_diff_any_outcome_pts"] = float(np.abs(A[ok] - B[ok]).max(1).mean() * 100)
    cons["flip_examples"] = [[(A[i]*100).round().tolist(), (B[i]*100).round().tolist()] for i in np.where(ok & (A.argmax(1) != B.argmax(1)))[0][:5]]
    cons["mean_abs_diff_home_vs_t0_pts"] = float(np.abs(A[ok2, 0] - S[ok2, 0]).mean() * 100)
# how much the LLM "listens" to the odds it was given
listen = None
if "LLM stats + odds" in F:
    P = F["LLM stats + odds"]; ok = ~np.isnan(P).any(1); O = F["Pre-closing odds (market avg)"]
    listen = dict(corr_home_with_given_odds=float(np.corrcoef(P[ok, 0], O[ok, 0])[0, 1]),
                  mean_abs_gap_home_pts=float(np.abs(P[ok, 0] - O[ok, 0]).mean() * 100))
    S = F["LLM stats only"]; ok3 = ok & ~np.isnan(S).any(1)
    listen["corr_home_stats_only_with_odds"] = float(np.corrcoef(S[ok3, 0], O[ok3, 0])[0, 1])
PIN = np.array([r["p_pin_close"] for r in M]); okp = mask & ~np.isnan(PIN).any(1)
pin_note = dict(n=int(okp.sum()), note="Pinnacle closing odds missing in the CSV from match index 210 on",
                rps_pin=float(rps(PIN[okp], Y[okp]).mean()), rps_avg_close_same=float(rps(F[ref][okp], Y[okp]).mean()))
# ---- review additions (2026-10-26): paired head-to-head gaps and tie-aware behaviour stats.
# Separate RNG so the numbers above are unchanged by these extra bootstraps.
rng2 = np.random.default_rng(11)
def boot2(x, B=4000):
    idx = rng2.integers(0, len(x), (B, len(x))); return np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist()
pairs = {}
for a, b in (("LLM stats + odds", "Pre-closing odds (market avg)"), ("LLM stats only", "Base rates (2021-25)"),
             ("LLM stats only", "Elo baseline"), ("LLM stats + odds", "Elo baseline"), ("LLM stats + odds", "LLM stats only"),
             ("Elo baseline", "Base rates (2021-25)")):
    if a not in F or b not in F: continue
    y = Y[mask]; pa, pb = F[a][mask], F[b][mask]
    pairs[f"{a} minus {b}"] = {nm: dict(mean=float((f(pa, y) - f(pb, y)).mean()), ci=boot2(f(pa, y) - f(pb, y)))
                               for nm, f in (("rps", rps), ("brier", brier), ("logloss", logl))}
ties = {}
for k in ("LLM stats only", "LLM stats + odds", "Closing odds (market avg)"):
    if k not in F: continue
    P = F[k][mask]; y = Y[mask]; mx = P.max(1, keepdims=True); top = (np.abs(P - mx) < 1e-12)
    w = top / top.sum(1, keepdims=True)
    ties[k] = dict(tied_top_share=float((top.sum(1) > 1).mean()),
                   strict_draw_favourite_share=float(((P[:, 1] > P[:, 0]) & (P[:, 1] > P[:, 2])).mean()),
                   strict_home_favourite_share=float(((P[:, 0] > P[:, 1]) & (P[:, 0] > P[:, 2])).mean()),
                   acc_ties_split=float((w * y).sum(1).mean()))
if cons is not None:
    def topset(P): return [frozenset(np.where(np.abs(r - r.max()) < 1e-12)[0]) for r in P]
    A, B = R["LLM stats only (t=0.7, seed 1)"], R["LLM stats only (t=0.7, seed 2)"]; S0 = F["LLM stats only"][:len(A)]
    okA = ~np.isnan(A).any(1) & ~np.isnan(B).any(1); okS = okA & ~np.isnan(S0).any(1)
    ta, tb = topset(A[okA]), topset(B[okA])
    cons["favourite_set_differs_share"] = float(np.mean([x != y for x, y in zip(ta, tb)]))
    cons["favourite_disjoint_share"] = float(np.mean([not (x & y) for x, y in zip(ta, tb)]))
    cons["identical_t07s1_vs_t0"] = float((np.abs(A[okS] - S0[okS]).max(1) < 1e-9).mean())
    cons["identical_t07s2_vs_t0"] = float((np.abs(B[okS] - S0[okS]).max(1) < 1e-9).mean())
out = dict(pinnacle=pin_note, n_eval=int(mask.sum()), actual=Y[mask].mean(0).tolist(), scores=res, backtest=bt,
           behaviour=beh, rerun=cons, listens_to_odds=listen, run_info=info,
           paired_head_to_head=pairs, ties_and_favourites=ties)
json.dump(out, open("out/results.json", "w"), indent=1, default=float)
# ---- table
lines = ["| Forecaster | RPS (lower=better) | 95% CI | Brier | Log loss | Accuracy | RPS vs closing odds |", "|---|---|---|---|---|---|---|"]
for k, v in sorted(res.items(), key=lambda x: x[1]["rps"]):
    lines.append(f"| {k} | {v['rps']:.4f} | {v['rps_ci'][0]:.4f}-{v['rps_ci'][1]:.4f} | {v['brier']:.4f} | {v['logloss']:.4f} | {v['acc']*100:.1f}% | {v['rps_minus_close']:+.4f} ({v['rps_minus_close_ci'][0]:+.4f}..{v['rps_minus_close_ci'][1]:+.4f}) |")
pathlib.Path("out/results_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines)); print(json.dumps({k: out[k] for k in ("n_eval", "actual", "backtest", "behaviour", "rerun", "listens_to_odds", "run_info")}, indent=1, default=float))
# ---- chart 1: reliability diagram (all three outcomes pooled)
plt.figure(figsize=(6.4, 6.4))
bins = np.linspace(0, 1, 11)
for k, st in (("Closing odds (market avg)", "o-"), ("Elo baseline", "s-"), ("LLM stats only", "^-"), ("LLM stats + odds", "v-")):
    if k not in F: continue
    p = F[k][mask].ravel(); y = Y[mask].ravel(); xs, ys, ns = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        s = (p >= a) & (p < b if b < 1 else p <= b)
        if s.sum() >= 8: xs.append(p[s].mean()); ys.append(y[s].mean()); ns.append(s.sum())
    plt.plot(xs, ys, st, label=k, ms=5)
plt.plot([0, 1], [0, 1], "k:", lw=1, label="perfect calibration")
plt.xlabel("Forecast probability"); plt.ylabel("How often it actually happened")
plt.title(f"Premier League 2025-26, {int(mask.sum())} matches\nCalibration: bins with 8+ forecasts")
plt.legend(fontsize=8, loc="upper left"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("charts/calibration.png", dpi=160)
# ---- chart 2 (replaces bars): paired RPS gap vs closing odds, per match, 95% bootstrap CI
plt.figure(figsize=(9.5, 4.4))
ks = [k for k in sorted(res, key=lambda k: res[k]["rps_minus_close"]) if k != ref]
v = [res[k]["rps_minus_close"] for k in ks]
lo = [v[i] - res[k]["rps_minus_close_ci"][0] for i, k in enumerate(ks)]; hi = [res[k]["rps_minus_close_ci"][1] - v[i] for i, k in enumerate(ks)]
col = ["#d62728" if k.startswith("LLM") else "#1f77b4" for k in ks]
plt.barh(ks, v, xerr=[lo, hi], color=col, capsize=3); plt.gca().invert_yaxis(); plt.axvline(0, color="k", lw=1)
plt.xlabel("Extra error vs bookmaker closing odds (RPS, paired per match, 95% CI)\n0 = as good as the market, right = worse")
plt.title(f"EPL 2025-26, {int(mask.sum())} matches: nobody beats the closing odds"); plt.tight_layout()
plt.savefig("charts/rps_gap_vs_closing.png", dpi=160)
# ---- chart 3: what each forecaster expects on average vs what happened
plt.figure(figsize=(7.5, 4.2)); labs = ["Home win", "Draw", "Away win"]; x = np.arange(3); w = 0.2
series = [("What happened", Y[mask].mean(0), "#333333"), ("Closing odds", res[ref]["mean_p"], "#1f77b4"),
          ("LLM stats only", res.get("LLM stats only", {}).get("mean_p"), "#d62728"),
          ("LLM stats + odds", res.get("LLM stats + odds", {}).get("mean_p"), "#ff9896")]
for j, (n, m, c) in enumerate([s for s in series if s[1] is not None]):
    plt.bar(x + (j - 1.5) * w, np.array(m) * 100, w, label=n, color=c)
    for xi, mi in zip(x + (j - 1.5) * w, m): plt.text(xi, mi * 100 + 0.8, f"{mi*100:.0f}", ha="center", fontsize=8)
plt.xticks(x, labs); plt.ylabel("Average probability / actual share, %"); plt.legend(fontsize=8)
plt.title("The 3B model doesn't believe in home advantage"); plt.tight_layout(); plt.savefig("charts/avg_forecast_vs_actual.png", dpi=160)
