"""STEP 1 - reproduce the published detector and verify every headline number."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *

df = load()
fe = preprocess(df)
ref = fe["ref"]

print("=" * 78)
print("A. DATASET")
print("=" * 78)
span = (df.index.max() - df.index.min()).total_seconds() / 300 + 1
gap = df.index.to_series().diff().dt.total_seconds().div(60)
print(f"rows={len(df):,}  span {df.index.min()} -> {df.index.max()}")
print(f"days={df.index.normalize().nunique()}  full grid={int(span):,} rows "
      f"({100*len(df)/span:.1f}% present)  gaps>1h={(gap>60).sum()}")
print(f"negative rows total={int(fe['negative'].sum().sum())}  "
      f"{fe['negative'].sum()[lambda s: s>0].to_dict()}")
print(f"site_outage={int(fe['site_outage'].sum()):,}  unit_off={int(fe['unit_off'].sum().sum()):,}  "
      f"stale={int(fe['stale'].sum().sum())}")

print()
print("=" * 78)
print("B. SHARED-PAIR DETECTOR (published settings)")
print("=" * 78)
res = {}
for a, b in PAIRS:
    r = detect(df, fe, a, b)
    res[(a, b)] = r
    fl = r["moderate"]
    print(f"{a}+{b:7s} g0={r['g0']:.3f}  cap med={np.nanmedian(r['cap_t']):,.0f} W "
          f"(range {np.nanmin(r['cap_t']):,.0f}-{np.nanmax(r['cap_t']):,.0f})")
    print(f"            flagged rows={int(fl.sum()):,}  hours={fl.sum()/12:,.1f}  "
          f"days={fl[fl].index.normalize().nunique()}  severe={int(r['severe'].sum())}")

print()
print("=" * 78)
print("C. CONTROL PAIRS (independent pseudo-pairs that cannot share an MPPT)")
print("=" * 78)
ctrl = {}
for a, b in CONTROL_PAIRS:
    r = detect(df, fe, a, b)
    ctrl[(a, b)] = r
    print(f"{a}+{b:7s} g0={r['g0']:.3f}  flagged rows={int(r['moderate'].sum())}")
c_tot = sum(int(r["moderate"].sum()) for r in ctrl.values())
s_tot = sum(int(r["moderate"].sum()) for r in res.values())
print(f"\ncontrols total={c_tot}   shared total={s_tot}   ratio={s_tot/max(c_tot,1):.0f}x")

print()
print("=" * 78)
print("D. SENSITIVITY TO DEFICIT THRESHOLD (hours)")
print("=" * 78)
rows = []
for thr in [0.10, 0.15, 0.20, 0.30, 0.40]:
    row = {"deficit>": thr}
    for a, b in PAIRS:
        r = res[(a, b)]
        raw = r["base"] & (r["deficit"] > thr)
        row[f"{a}_{b}"] = round((raw & (run_lengths(raw) >= 3)).sum() / 12, 1)
    rows.append(row)
print(pd.DataFrame(rows).set_index("deficit>").to_string())

# ------------------------------------------------------------------ #
print()
print("=" * 78)
print("E. DIAGNOSTIC: how mechanical is `deficit`?  (is it just a restatement of ref?)")
print("=" * 78)
for label, pairs in [("shared", PAIRS), ("control", CONTROL_PAIRS)]:
    for a, b in pairs:
        r = res.get((a, b)) or ctrl[(a, b)]
        sel = r["act"] & (ref > 0.3)
        d = r["deficit"][sel].replace([np.inf, -np.inf], np.nan)
        rr = ref[sel]
        ok = d.notna()
        print(f"{label:8s} {a}+{b:7s} corr(ref,deficit)={rr[ok].corr(d[ok]):+.3f}  "
              f"resid sd after binned mean = {d[ok].groupby(pd.cut(rr[ok],20)).transform('mean').sub(d[ok]).std():.3f}"
              f"  (raw sd={d[ok].std():.3f})")

print()
print("=" * 78)
print("F. DIAGNOSTIC: rolling-ceiling window semantics")
print("=" * 78)
dayidx = fe["dayidx"]
for a, b in PAIRS[:1]:
    s = df[a] + df[b]
    act = fe["active"][a] & fe["active"][b]
    ok = act & (ref > 0.3)
    per_day = s[ok].groupby(dayidx[ok]).quantile(0.99)
    alldays = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
    print(f"{a}+{b}: per_day has {len(per_day)} of {len(alldays)} calendar days "
          f"({100*len(per_day)/len(alldays):.0f}%)")
    cal_span = per_day.index.to_series().diff().dt.total_seconds().div(86400).rolling(31).sum().median()
    print(f"   rolling(31 obs) actually spans median {cal_span:.1f} calendar days "
          f"-> the '31-day' ceiling is really a {cal_span:.0f}-day median")
    print(f"   daily p99 == daily max?  "
          f"{np.isclose(per_day, s[ok].groupby(dayidx[ok]).max()).mean()*100:.0f}% of days")
    # does the window straddle the big acquisition gaps?
    d = per_day.index.to_series().diff().dt.total_seconds().div(86400)
    print(f"   largest day-step inside per_day index = {d.max():.1f} days "
          f"(a 31-obs window centred there averages epochs >2 months apart)")
    # how much do cap days move if we reindex onto a full calendar first?
    per_day_full = per_day.reindex(alldays)
    cap_obs = per_day.rolling(31, center=True, min_periods=5).median()
    cap_cal = per_day_full.rolling(31, center=True, min_periods=5).median().reindex(per_day.index)
    diff = (cap_obs - cap_cal).abs()
    print(f"   |cap(obs-window) - cap(calendar-window)| : median {diff.median():.0f} W, "
          f"p95 {diff.quantile(.95):.0f} W, max {diff.max():.0f} W  "
          f"(= {100*diff.median()/np.nanmedian(cap_obs):.1f}% of the ceiling)")
