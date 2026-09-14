"""
STEP 15 - does the published `deficit` COLUMN mislead, not just the flags?

step11 showed the published model's error is seasonally signed: expected runs 12-15%
too high in summer and ~40% too low in December. The flags only ever fire at
ref>=0.70, and midwinter never reaches that gate, so winter never produces a flag.

But `saturation_flags.csv` publishes `eu_*_deficit` as a CONTINUOUS column over every
sample. If that column runs to -0.45 on control pairs in December, then anyone reading
it as a severity measure - or aggregating it into an energy figure - inherits nonsense
that no flag-level check would ever catch.

Run:  ./venv/Scripts/python.exe sat_work/research/step15_deficitcol.py
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *

pd.set_option("display.width", 200)
df = load(); FE = preprocess(df); ref = FE["ref"]
FULLCAL = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
BAND = (0.35, 0.60)
X = df[EU].div(df[EU].quantile(0.999))
mo = df.index.to_period("M").astype(str)
mono = sorted(set(mo))


def broadcast(per_day, dayidx, labels, window="31D", minp=5):
    roll = per_day.reindex(FULLCAL).rolling(window, center=True, min_periods=minp).median()
    return pd.Series(roll.reindex(dayidx).to_numpy(), index=labels)


def theta(s, act):
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= BAND[0]) & (ref <= BAND[1]) & u.notna()
    return broadcast(u[sel].groupby(FE["dayidx"][sel]).median(), FE["dayidx"], s.index)


def truth(a, b, mask):
    """Peer-ratio shortfall on the peer CLEAN and pair CLEAN rows inside `mask`."""
    sa = X[a] + X[b]
    act = FE["active"][a] & FE["active"][b]
    vals = []
    for p1, p2 in CONTROL_PAIRS:
        if {p1, p2} & {a, b}:
            continue
        pq = X[p1] + X[p2]
        pact = FE["active"][p1] & FE["active"][p2]
        r = sa / pq.replace(0, np.nan)
        cal = act & pact & (ref >= BAND[0]) & (ref <= BAND[1])
        r = r / r[cal].median()
        m = mask & act & pact & ~dq_mask(FE, a, b) & ~dq_mask(FE, p1, p2)
        if m.sum() > 20:
            vals.append(1 - float(r[m].median()))
    return float(np.median(vals)) if vals else np.nan


print("=" * 104)
print("THE PUBLISHED `deficit` COLUMN, MONTH BY MONTH  (all daylight active rows, no ref gate)")
print("=" * 104)
print("This is exactly what saturation_flags.csv exposes to downstream users.\n")
for a, b in PAIRS + CONTROL_PAIRS:
    s = df[a] + df[b]
    act = FE["active"][a] & FE["active"][b]
    clean = act & ~dq_mask(FE, a, b)
    d_pub = pd.Series(detect(df, FE, a, b)["deficit"], index=df.index)
    d_vat = (1 - s / (theta(s, act) * ref)).replace([np.inf, -np.inf], np.nan)
    tag = "SHARED " if (a, b) in PAIRS else "control"
    rows = []
    for m in mono:
        km = (mo == m) & clean
        if km.sum() < 50 or ref[km].max() < 0.3:
            continue
        rows.append(dict(month=m, ref_max=round(float(ref[km].max()), 2),
                         d_pub=round(float(d_pub[km].median()), 3),
                         d_vat=round(float(d_vat[km].median()), 3),
                         truth=round(t, 3) if not np.isnan(t := truth(a, b, km)) else np.nan))
    T = pd.DataFrame(rows)
    T["pub_err"] = (T.d_pub - T.truth).round(3)
    T["vat_err"] = (T.d_vat - T.truth).round(3)
    print("-" * 104)
    print(f"{tag} {a}+{b}")
    print(T.to_string(index=False))
    print(f"    median |error|:  published {T.pub_err.abs().median():.3f}   vat-v1 {T.vat_err.abs().median():.3f}"
          f"      worst published month: {T.loc[T.pub_err.abs().idxmax(), 'month']}"
          f" ({T.pub_err.abs().max():.3f})")

print()
print("=" * 104)
print("AGGREGATE: if a downstream user averaged the deficit column over the whole record")
print("=" * 104)
rows = []
for a, b in PAIRS + CONTROL_PAIRS:
    s = df[a] + df[b]
    act = FE["active"][a] & FE["active"][b]
    clean = act & ~dq_mask(FE, a, b)
    d_pub = pd.Series(detect(df, FE, a, b)["deficit"], index=df.index)
    d_vat = (1 - s / (theta(s, act) * ref)).replace([np.inf, -np.inf], np.nan)
    tag = "SHARED " if (a, b) in PAIRS else "control"
    rows.append(dict(pair=f"{tag}{a}+{b}",
                     pub_mean_all=round(float(d_pub[clean].mean()), 3),
                     vat_mean_all=round(float(d_vat[clean].mean()), 3),
                     pub_mean_summer=round(float(d_pub[clean & (ref >= 0.7)].mean()), 3),
                     vat_mean_summer=round(float(d_vat[clean & (ref >= 0.7)].mean()), 3)))
A = pd.DataFrame(rows)
print(A.to_string(index=False))
print()
print("   Note the all-rows means on the CONTROL pairs: whatever they show is pure error.")
print(f"   control published mean = {A[A.pair.str.startswith('control')].pub_mean_all.mean():+.3f}   "
      f"vs vat-v1 {A[A.pair.str.startswith('control')].vat_mean_all.mean():+.3f}")
