"""Build pre-match features for the 2025-26 EPL season using ONLY information known before kickoff.
Also fits a simple Elo baseline on 2021-22..2024-25 and keeps updating it through 2025-26 (no look-ahead)."""
import pandas as pd, numpy as np, json, pathlib
D = pathlib.Path(__file__).parent / "data"
def load(s):
    df = pd.read_csv(D / f"E0_{s}.csv", encoding="latin-1")
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.copy(); df["season"] = s
    return df.dropna(subset=["FTR"])
hist = pd.concat([load(s) for s in ["2122", "2223", "2324", "2425"]]).sort_values("Date")
test = load("2526").sort_values(["Date", "Time"]).reset_index(drop=True)

def implied(h, d, a):
    p = np.array([1/h, 1/d, 1/a]); return p / p.sum()   # proportional margin removal

# --- Elo with home advantage, K=20, carried over with regression to mean between seasons
K, HFA = 20.0, 60.0
NEW_TEAM = 1420.0   # prior for a club never seen in 2021-25 (only Sunderland); same value used for forecasting and updating
elo = {}
def exp_home(rh, ra): return 1 / (1 + 10 ** (-(rh + HFA - ra) / 400))
def upd(h, a, res):
    rh, ra = elo.get(h, NEW_TEAM), elo.get(a, NEW_TEAM)
    e = exp_home(rh, ra); s = {"H": 1, "D": 0.5, "A": 0}[res]
    elo[h] = rh + K * (s - e); elo[a] = ra - K * (s - e)
prev = None
for _, r in hist.iterrows():
    if r.season != prev and prev is not None:
        for t in elo: elo[t] = 0.8 * elo[t] + 0.2 * 1500
    prev = r.season; upd(r.HomeTeam, r.AwayTeam, r.FTR)
for t in elo: elo[t] = 0.8 * elo[t] + 0.2 * 1500
# draw rate from history, used to turn Elo expectation into H/D/A
draw_rate = (hist.FTR == "D").mean()
def elo_probs(h, a):
    e = exp_home(elo.get(h, NEW_TEAM), elo.get(a, NEW_TEAM))
    # draw mass peaks for even matches; simple, documented heuristic
    pd_ = draw_rate * (1 - abs(2 * e - 1)) * 1.25
    pd_ = min(max(pd_, 0.05), 0.40)
    ph = max(e - pd_ / 2, 0.01); pa = max(1 - e - pd_ / 2, 0.01)
    p = np.array([ph, pd_, pa]); return p / p.sum()

# last season final table
last = load("2425"); pts = {}
for _, r in last.iterrows():
    pts.setdefault(r.HomeTeam, 0); pts.setdefault(r.AwayTeam, 0)
    pts[r.HomeTeam] += {"H": 3, "D": 1, "A": 0}[r.FTR]; pts[r.AwayTeam] += {"H": 0, "D": 1, "A": 3}[r.FTR]
last_rank = {t: i + 1 for i, (t, _) in enumerate(sorted(pts.items(), key=lambda x: -x[1]))}

rows, table, form = [], {}, {}
def tstat(t): return table.setdefault(t, {"P": 0, "W": 0, "D": 0, "L": 0, "GF": 0, "GA": 0, "Pts": 0})
for _, r in test.iterrows():
    h, a = r.HomeTeam, r.AwayTeam
    order = sorted(table.items(), key=lambda x: (-x[1]["Pts"], -(x[1]["GF"] - x[1]["GA"]), -x[1]["GF"]))
    pos = {t: i + 1 for i, (t, _) in enumerate(order)}
    def desc(t):
        s = tstat(t)
        lr = f"{last_rank[t]}th in the Premier League last season ({pts[t]} pts)" if t in last_rank else "promoted from the Championship (not in the Premier League last season)"
        cur = (f"this season: {s['P']} played, {s['W']}W {s['D']}D {s['L']}L, goals {s['GF']}-{s['GA']}, {s['Pts']} pts, position {pos.get(t,'-')}"
               if s["P"] else "this season: no matches played yet")
        f5 = "".join(form.get(t, [])[-5:]) or "none"
        return f"{lr}; {cur}; last 5 results (oldest to newest): {f5}"
    oh, od, oa = r.AvgH, r.AvgD, r.AvgA            # pre-closing market average (as published by football-data)
    ch, cd, ca = r.AvgCH, r.AvgCD, r.AvgCA         # closing market average
    rows.append(dict(i=len(rows), date=str(r.Date.date()), home=h, away=a, result=r.FTR,
                     home_desc=desc(h), away_desc=desc(a),
                     odds_open=[oh, od, oa], odds_close=[ch, cd, ca],
                     odds_pin_close=[r.PSCH, r.PSCD, r.PSCA],
                     p_open=implied(oh, od, oa).tolist(), p_close=implied(ch, cd, ca).tolist(),
                     p_pin_close=implied(r.PSCH, r.PSCD, r.PSCA).tolist(), p_elo=elo_probs(h, a).tolist()))
    # update AFTER recording features (no look-ahead)
    hs, as_ = tstat(h), tstat(a)
    for s, gf, ga in ((hs, r.FTHG, r.FTAG), (as_, r.FTAG, r.FTHG)):
        s["P"] += 1; s["GF"] += int(gf); s["GA"] += int(ga)
    if r.FTR == "H": hs["W"] += 1; as_["L"] += 1; hs["Pts"] += 3; fh, fa = "W", "L"
    elif r.FTR == "A": as_["W"] += 1; hs["L"] += 1; as_["Pts"] += 3; fh, fa = "L", "W"
    else: hs["D"] += 1; as_["D"] += 1; hs["Pts"] += 1; as_["Pts"] += 1; fh = fa = "D"
    form.setdefault(h, []).append(fh); form.setdefault(a, []).append(fa)
    upd(h, a, r.FTR)
base = hist.FTR.value_counts(normalize=True)
meta = dict(n=len(rows), base_rates=[base["H"], base["D"], base["A"]], hist_draw_rate=draw_rate)
pathlib.Path("out").mkdir(exist_ok=True)
json.dump(dict(meta=meta, matches=rows), open("out/matches_2526.json", "w"), indent=0)
print(meta, rows[100]["home_desc"])
