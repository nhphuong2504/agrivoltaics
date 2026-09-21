"""One authoritative dump of every number quoted in the paper and the two write-ups.

The 2026-09 data refresh moved essentially every headline figure, and the values were
spread across a dozen step scripts. Rather than re-derive each one by hand and risk
transcribing a stale number into the paper, this prints them all from the frozen harness
in one pass. If a number in `paper/full_part.tex`, `METHODS_RESULTS.md` or
`SATURATION_REVIEW.md` is not produced here, it should not be quoted.

Run: ./venv/Scripts/python.exe sat_work/research/step32_paper_numbers.py
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bench as B  # noqa: E402
import canon_metrics as CM  # noqa: E402
from satsim import CONTROL_PAIRS, INVERTERS, INV_OF, PAIRS, RELIABLE  # noqa: E402

np.set_printoptions(suppress=True)

df = B.raw_df()
fe = B.raw_fe()
ref = fe["ref"]
N = len(df)
EU = [f"eu_{i}" for i in range(1, 25)]


def head(t):
    print()
    print("=" * 92)
    print(t)
    print("=" * 92)


# --------------------------------------------------------------------------- #
head("A. THE RECORD")
steps = df.index.to_series().diff().dt.total_seconds().dropna()
modal = int((steps == 300).sum())
cal = (df.index.max().normalize() - df.index.min().normalize()).days + 1
print(f"  rows {N:,}   dates {df.index.normalize().nunique()}   calendar days {cal}")
print(f"  span {df.index.min()} -> {df.index.max()}")
print(f"  modal 5-min step on {modal:,} of {len(steps):,} ({100 * modal / len(steps):.1f} %)")
print(f"  reference units |R| = {len(RELIABLE)}")

# --------------------------------------------------------------------------- #
head("B. DATA-QUALITY SHARES  (per unit-timestamp unless noted)")
cells = 24 * N
print(f"  unit-timestamp cells {cells:,}")
for name, arr in (("off", None), ("stale", None), ("neg", None)):
    pass
off = sum(int(fe["unit_off"][u].sum()) for u in EU)
stale = sum(int(fe["stale"][u].sum()) for u in EU)
neg = sum(int(fe["negative"][u].sum()) for u in EU)
inact = sum(int((~fe["active"][u]).sum()) for u in EU)
outage = int(fe["site_outage"].sum())
print(f"  off    {off:>9,}  {100 * off / cells:6.2f} %")
print(f"  stale  {stale:>9,}  {100 * stale / cells:6.2f} %")
print(f"  neg    {neg:>9,}  {100 * neg / cells:6.2f} %")
print(f"  outage {outage:>9,}  {100 * outage / N:6.2f} %   (site-wide, share of timestamps)")
print(f"  mask   {inact:>9,}  {100 * inact / cells:6.2f} %")

# --------------------------------------------------------------------------- #
head("C. DECISION DOMAIN")
can = CM.load_canonical()
for a, b in PAIRS:
    m = CM.domain_mask(fe, a, b)
    print(f"  {a}+{b:8s} {int(m.sum()):>7,} rows  {100 * m.sum() / N:5.1f} % of record")

# --------------------------------------------------------------------------- #
head("D. FLAGGED EXPOSURE  (canonical sat flag)")
pairs_for_days = []
for a, b in PAIRS:
    col = f"{a}_{b}_sat"
    f = can[col].to_numpy().astype(bool)
    days = pd.Series(df.index[f]).dt.normalize().nunique()
    pairs_for_days.append(set(pd.Series(df.index[f]).dt.normalize()))
    print(f"  {a}+{b:8s} {int(f.sum()):>7,} rows  {f.sum() / 12:7.1f} h  {days:>4} days")
tot = sum(int(can[f'{a}_{b}_sat'].sum()) for a, b in PAIRS)
union = set().union(*pairs_for_days)
print(f"  {'TOTAL':17s} {tot:>7,} rows  {tot / 12:7.1f} h  {len(union):>4} days (union)")

print("\n  affected days by calendar month:")
mo = pd.Series(sorted(union)).dt.month.value_counts().sort_index()
for m, c in mo.items():
    print(f"     month {m:2d}: {c:5d} days")
mjj = sum(c for m, c in mo.items() if m in (5, 6, 7))
print(f"     May+Jun+Jul = {mjj} of {len(union)}")

# --------------------------------------------------------------------------- #
head("E. PEER RATIO, UNBINNED OVER r >= 0.70  (the paper's tab:mppt_peer)")
PEAK = df[EU].quantile(0.999)
X = df[EU].div(PEAK)


def peer_ratio(P, Q, extra_mask=None):
    p = X[P[0]] + X[P[1]]
    q = (X[Q[0]] + X[Q[1]]).replace(0, np.nan)
    r = p / q
    calib = (ref >= 0.35) & (ref <= 0.60)
    r = r / r[calib].median()
    hi = (ref >= 0.70) & r.notna() & fe["active"][P[0]] & fe["active"][P[1]]
    if extra_mask is not None:
        hi = hi & extra_mask
    return float(r[hi].median()), int(hi.sum())


rows = [("EU8 + EU16", ("eu_8", "eu_16"), ("eu_1", "eu_3"), "shared"),
        ("EU10 + EU18", ("eu_10", "eu_18"), ("eu_1", "eu_3"), "shared"),
        ("EU13 + EU21", ("eu_13", "eu_21"), ("eu_1", "eu_3"), "shared"),
        ("EU1 + EU3", ("eu_1", "eu_3"), ("eu_4", "eu_12"), "independent"),
        ("EU4 + EU12", ("eu_4", "eu_12"), ("eu_1", "eu_3"), "independent"),
        ("EU15 + EU23", ("eu_15", "eu_23"), ("eu_1", "eu_3"), "independent")]
print(f"  {'pair':13s} {'channel':12s} {'ratio':>7s} {'vs peers':>9s} {'n':>9s}")
for nm, P, Q, ch in rows:
    ratio, n = peer_ratio(P, Q)
    print(f"  {nm:13s} {ch:12s} {ratio:7.3f} {100 * (ratio - 1):8.1f}% {n:9,}")

# Robustness: the same six rows on ONE sample set, so they are strictly comparable.
# The pair-wise mask above gives each pair every sample it can contribute; a reviewer
# may prefer a single common frame. Both must tell the same story.
inv_units = sorted({u for _, P, Q, _ in rows for u in P + Q})
common = np.ones(N, dtype=bool)
for u in inv_units:
    common &= fe["active"][u].to_numpy()
    common &= ~fe["unit_off"][u].to_numpy()
    common &= ~fe["stale"][u].to_numpy()
    common &= ~fe["negative"][u].to_numpy()
common &= ~fe["site_outage"].to_numpy()
print(f"\n  robustness - a single common mask over all {len(inv_units)} units involved "
      f"(n = {int((common & (ref >= 0.70)).sum()):,}):")
for nm, P, Q, ch in rows:
    ratio, n = peer_ratio(P, Q, extra_mask=common)
    print(f"     {nm:13s} {ch:12s} {ratio:7.3f} {100 * (ratio - 1):8.1f}% {n:9,}")

# and the spread across three different peer references, for the shared pairs
print("\n  robustness - shared pairs against each of the three disjoint peers:")
for nm, P in (("EU8+EU16", ("eu_8", "eu_16")), ("EU10+EU18", ("eu_10", "eu_18")),
              ("EU13+EU21", ("eu_13", "eu_21"))):
    vals = [peer_ratio(P, Q)[0] for Q in
            (("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23"))]
    print(f"     {nm:11s} {min(vals):.3f} - {max(vals):.3f}")

# --------------------------------------------------------------------------- #
head("F. ROLL-OFF SHAPE  (bin medians, rescaled to 1 in the mid band)")


def did_bins(P, Q, bins=np.linspace(0.3, 1.05, 26)):
    p = X[P[0]] + X[P[1]]
    q = (X[Q[0]] + X[Q[1]]).replace(0, np.nan)
    r = p / q
    calib = (ref >= 0.35) & (ref <= 0.60)
    r = r / r[calib].median()
    idx = np.digitize(ref, bins).clip(1, len(bins) - 1)
    s = (ref > 0.3) & r.notna()
    med = r[s].groupby(idx[s]).median().reindex(range(1, len(bins)))
    ctr = pd.Series(0.5 * (bins[1:] + bins[:-1]), index=range(1, len(bins)))
    return med, ctr


for nm, P in (("EU8+EU16", ("eu_8", "eu_16")), ("EU10+EU18", ("eu_10", "eu_18")),
              ("EU13+EU21", ("eu_13", "eu_21"))):
    med, ctr = did_bins(P, ("eu_1", "eu_3"))
    top = med.iloc[-1]
    hipart = med[ctr >= 0.75].dropna()
    monotone_from = None
    for i in range(len(hipart) - 1):
        if (hipart.iloc[i:].diff().dropna() <= 0).all():
            monotone_from = f"{float(ctr.loc[hipart.index[i]]):.3f}"
            break
    print(f"  {nm:11s} top bin {top:.3f}   monotone from r~{monotone_from}")
for nm, P in (("EU4+EU12", ("eu_4", "eu_12")), ("EU15+EU23", ("eu_15", "eu_23"))):
    med, _ = did_bins(P, ("eu_1", "eu_3"))
    print(f"  {nm:11s} control range {med.min():.3f} - {med.max():.3f}")

# --------------------------------------------------------------------------- #
head("G. PER-UNIT HIGH-IRRADIANCE SCORES  (r >= 0.75, each unit scored alone)")


def unit_deficit(u):
    s = df[u]
    act = fe["active"][u]
    th = B.theta_roll(s, ref, act, fe["dayidx"])
    d = (1 - s / (th * ref)).replace([np.inf, -np.inf], np.nan)
    clean = (act & ~fe["unit_off"][u] & ~fe["stale"][u]
             & ~fe["negative"][u] & ~fe["site_outage"])
    hi = clean & (ref >= 0.75)
    return d, hi


shared_units = [u for p in PAIRS for u in p]
vals = {}
for u in EU:
    d, hi = unit_deficit(u)
    vals[u] = float(d[hi].median())
sh = sorted((vals[u], u) for u in shared_units)
oth = sorted((vals[u], u) for u in EU if u not in shared_units and u != "eu_7")
print(f"  six shared-channel units : {sh[0][0]:+.3f} .. {sh[-1][0]:+.3f}")
print(f"  other 17 units           : {oth[0][0]:+.3f} .. {oth[-1][0]:+.3f}")
print(f"  excluded eu_7            : {vals['eu_7']:+.3f}")
ranked = sorted(vals.items(), key=lambda kv: -kv[1])[:8]
print("  ranked top 8:", ", ".join(f"{u} {v:+.3f}" for u, v in ranked))

# --------------------------------------------------------------------------- #
head("H. CALIBRATION AGAINST THE MODEL-FREE MEASUREMENT")

model = {}
for a, b in PAIRS + list(CONTROL_PAIRS):
    d = B.pair_deficit(df, fe, a, b)
    m = CM.domain_mask(fe, a, b)
    model[(a, b)] = float(pd.Series(d["deficit"], index=df.index)[m].median())
truth = {}
for a, b in PAIRS + list(CONTROL_PAIRS):
    # the model-free measurement is a RATIO; its implied deficit is 1 - ratio
    truth[(a, b)] = 1.0 - peer_ratio((a, b), ("eu_1", "eu_3"))[0]
print("  (medians over the decision domain; step26 prints the gate-AVERAGED pair of")
print("   numbers that the paper's calibration paragraph quotes)")
for lbl, grp in (("shared", PAIRS), ("control", CONTROL_PAIRS)):
    mm = float(np.median([model[p] for p in grp]))
    tt = float(np.median([truth[p] for p in grp]))
    print(f"  {lbl:8s} model {mm:+.3f}   truth {tt:+.3f}   residual {mm - tt:+.4f}")

print("\n  control-pair median deficit, model, by season (r >= 0.5):")
mon = pd.Series(df.index.month, index=df.index)
for m, lab in ((12, "December"), (6, "June")):
    row = []
    for a, b in CONTROL_PAIRS:
        d = B.pair_deficit(df, fe, a, b)
        sel = (ref >= 0.5) & (mon == m) & d["act"]
        row.append(float(pd.Series(d["deficit"], index=df.index)[sel].median()))
    print(f"     {lab:9s} {row[0]:+.3f} {row[1]:+.3f} {row[2]:+.3f}")

# --------------------------------------------------------------------------- #
head("I. INVERTER-TOPOLOGY NATURAL EXPERIMENT")
for p in PAIRS:
    inv = INV_OF[p[0]]
    mates = [u for u in INVERTERS[inv] if u not in p and u != "eu_7"]
    d, hi = unit_deficit(p[0])
    d2, hi2 = unit_deficit(p[1])
    paird = float(np.median([d[hi].median(), d2[hi2].median()]))
    md = float(np.median([unit_deficit(u)[0][unit_deficit(u)[1]].median() for u in mates]))
    print(f"  inv {inv}: {p[0]}+{p[1]} d {paird:+.3f}   mates {mates} d {md:+.3f}   "
          f"gap {paird - md:+.3f}")

# --------------------------------------------------------------------------- #
head("J. NAMEPLATE CROSS-CHECK")
th = []
for u in EU:
    if u == "eu_7":
        continue
    act = fe["active"][u]
    # the paper's definition: theta over the mid band, where clipping cannot bind
    mid = act & (ref >= 0.35) & (ref <= 0.60)
    th.append(float(B.theta_roll(df[u], ref, act, fe["dayidx"])[mid].median()))
th = np.array(th)
print(f"  median theta {np.median(th):,.0f} W   ratio {np.median(th) / 9000:.3f}   "
      f"sd {np.std(th / 9000):.3f}   max dev {100 * np.abs(th / 9000 - 1).max():.1f} %")

# --------------------------------------------------------------------------- #
head("K. ENERGY AND THE BIAS FLOOR")
tot_s = tot_c = 0.0
for a, b in PAIRS + list(CONTROL_PAIRS):
    d = B.pair_deficit(df, fe, a, b)
    gate = ((ref >= B.REF_ON) & (d["s"] > B.NEAR_CAP * d["expected"])
            & d["act"] & ~B.dq_mask(fe, a, b)).to_numpy()
    loss = float(np.maximum(d["expected"] - d["s"], 0).to_numpy()[gate].sum()) * 5 / 60 / 1000
    kind = "shared" if (a, b) in PAIRS else "ctrl"
    if kind == "shared":
        tot_s += loss
    else:
        tot_c += loss
    print(f"  {kind:6s} {a}+{b:8s} {loss:9,.0f} kWh")
print(f"  shared total {tot_s:,.0f} kWh   control total {tot_c:,.0f} kWh   "
      f"bias floor {100 * tot_c / tot_s:.0f} %")

# --------------------------------------------------------------------------- #
head("L. FLEET-WIDE NULL  (cross-inverter independent pairs)")
indep = [u for u in EU if u not in shared_units and u != "eu_7"]
cross = [(a, b) for i, a in enumerate(indep) for b in indep[i + 1:]
         if INV_OF[a] != INV_OF[b]]
losses = []
for a, b in cross:
    d = B.pair_deficit(df, fe, a, b)
    gate = ((ref >= B.REF_ON) & (d["s"] > B.NEAR_CAP * d["expected"])
            & d["act"] & ~B.dq_mask(fe, a, b)).to_numpy()
    losses.append(float(np.maximum(d["expected"] - d["s"], 0).to_numpy()[gate].sum())
                  * 5 / 60 / 1000)
print(f"  {len(cross)} cross-inverter independent pairs")
print(f"  median phantom loss {np.median(losses):,.0f} kWh/pair")

# --------------------------------------------------------------------------- #
head("M. CONTROL-PAIR SELF-REFERENCE  (how much do control units move r?)")
in_ref = [u for u in RELIABLE]
ctrl_units = sorted({u for a, b in CONTROL_PAIRS for u in (a, b)})
overlap = [u for u in ctrl_units if u in in_ref]
alt_ref = B.preprocess(df, [u for u in in_ref if u not in ctrl_units])["ref"]
both = ref.notna() & alt_ref.notna()
print(f"  |R| = {len(in_ref)}; control-pair units inside R: {overlap} "
      f"({len(overlap)} of 11)")
print(f"  corr(r, r without the control units) = {np.corrcoef(ref[both], alt_ref[both])[0, 1]:.4f}")
mx = 0.0
for a, b in CONTROL_PAIRS:
    d1 = B.pair_deficit(df, fe, a, b)
    g1 = CM.domain_mask(fe, a, b)
    m1 = float(pd.Series(d1["deficit"], index=df.index)[g1].median())
    mx = max(mx, abs(m1))
    print(f"     {a}+{b:8s} in-domain median deficit {m1:+.3f}")
print(f"  -> worst control-pair deficit {mx:+.3f} against a shared-pair signal of ~0.18")

# --------------------------------------------------------------------------- #
head("N. THE UNEXPLAINED CONTROL PAIR")
_, fl_ctl = B.det_rolling(df, fe, "eu_15", "eu_23")
f = np.asarray(fl_ctl, dtype=bool)
dd = pd.Series(df.index[f]).dt.normalize()
dpair = B.pair_deficit(df, fe, "eu_15", "eu_23")
dv = pd.Series(dpair["deficit"], index=df.index)
print(f"  EU15+EU23: {int(f.sum())} flagged rows on {dd.nunique()} distinct days")
print(f"     flagged deficit: median {dv[f].median():+.3f}, p90 {dv[f].quantile(.9):+.3f}, "
      f"max {dv[CM.domain_mask(fe, 'eu_15', 'eu_23')].max():.3f}")
print(f"     in-domain median {dv[CM.domain_mask(fe, 'eu_15', 'eu_23')].median():+.3f}")

# --------------------------------------------------------------------------- #
head("O. CEILING-PINNED (D4) BY REGIME  (the negative result)")
for lbl, kw in (("episodic  eu_1+eu_3", dict(duty=0.35, q=0.75, pattern="episodic")),
                ("episodic  eu_15+eu_23", dict(duty=0.35, q=0.75, pattern="episodic")),
                ("chronic   35% eu_1+eu_3", dict(duty=0.35, q=0.75, pattern="chronic")),
                ("chronic  100% eu_1+eu_3", dict(duty=1.00, q=0.75, pattern="chronic")),
                ("chronic   35% eu_4+eu_12", dict(duty=0.35, q=0.75, pattern="chronic"))):
    pair = ("eu_15", "eu_23") if "eu_15" in lbl else (("eu_4", "eu_12") if "eu_4" in lbl else ("eu_1", "eu_3"))
    sc = B.scenario(pair, **kw)
    s4 = B.score_scenario(sc, B.det_ceiling_pinned)
    s1 = B.score_scenario(sc, B.det_rolling)
    print(f"  {lbl:24s} D4 auc {s4['auc00']:.3f} fp {s4['fp_rows']:>5}   "
          f"| D1 auc {s1['auc00']:.3f}")

# --------------------------------------------------------------------------- #
head("P. SEVERITY RECOVERY FROM THE deficit COLUMN")
for a, b in PAIRS:
    dom = CM.domain_mask(fe, a, b)
    dv = pd.Series(B.pair_deficit(df, fe, a, b)["deficit"], index=df.index)
    _, flr = B.det_rolling(df, fe, a, b)
    fl = pd.Series(np.asarray(flr, dtype=bool), index=df.index)
    n30 = int((dv[dom] > 0.30).sum())
    notfl = int(((dv[dom] > 0.30) & ~fl[dom]).sum())
    sh = 100.0 * float((dv[fl] > 0.30).sum()) / max(int(fl.sum()), 1)
    print(f"  {a}+{b}: in-domain d>0.30 {n30:>5}; of those NOT flagged {notfl:>4}; "
          f"flagged rows {int(fl.sum()):>5}; of flagged, d>0.30 {sh:4.1f}%; "
          f"max flagged d {dv[fl].max():.3f}; max in-domain d {dv[dom].max():.3f}")
