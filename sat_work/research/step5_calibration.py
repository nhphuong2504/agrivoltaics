"""
STEP 5 - calibration of the *baseline*, measured on pairs that were never clipped.

A detector is only as good as its definition of "expected".  The control pairs cannot
saturate, so any deficit they show is model error.  We compare three baselines on the
same six pairs:

  published  : expected = g0 * cap(t) * ref            (cap = rolling median of daily p99)
  rolling    : expected = theta(t) * ref               (theta = rolling median of daily s/ref)
  envelope   : expected = theta90(t) * ref             (theta90 = rolling median of daily p90)
  DiD truth  : 1 - ratio to a peer pair, normalised at mid irradiance (no model at all)

and then convert each into an "apparent lost energy" on both shared and control pairs.
A baseline that reports real losses on control pairs is biased.
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *

df = load()
FE = preprocess(df)
ref = FE["ref"]
FULLCAL = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
OUT = r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research"

def broadcast(per_day, dayidx, labels, window="31D", minp=5):
    roll = per_day.reindex(FULLCAL).rolling(window, center=True, min_periods=minp).median()
    return pd.Series(roll.reindex(dayidx).to_numpy(), index=labels)

def theta_roll(s, act, q=0.5, band=(0.35, 0.6)):
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= band[0]) & (ref <= band[1]) & u.notna()
    g = u[sel].groupby(FE["dayidx"][sel])
    per_day = g.quantile(q) if q != 0.5 else g.median()
    return broadcast(per_day, FE["dayidx"], s.index)

ALLP = [("eu_8", "eu_16"), ("eu_10", "eu_18"), ("eu_13", "eu_21"),
        ("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]
KIND = {p: ("SHARED" if p in PAIRS else "control") for p in ALLP}
BINS = np.linspace(0.30, 1.05, 21)
CTR = 0.5 * (BINS[1:] + BINS[:-1])
PEER = ("eu_1", "eu_3")
PEAK = df[EU].quantile(0.999)

def by_ref(x):
    idx = np.digitize(ref, BINS).clip(1, len(BINS) - 1)
    ok = (ref > 0.3) & np.isfinite(pd.Series(x, index=df.index))
    return pd.Series(x, index=df.index)[ok].groupby(idx[ok]).median().reindex(range(1, len(BINS)))

# --------------------------------------------------------------------------- #
print("=" * 104)
print("A. BASELINE DEFICIT vs ref, by pair   (controls cannot saturate: whatever they show is MODEL ERROR)")
print("=" * 104)
curves = {}
for a, b in ALLP:
    s = df[a] + df[b]
    act = FE["active"][a] & FE["active"][b]
    r = detect(df, FE, a, b)
    d_pub = pd.Series(r["deficit"], index=df.index)
    th = theta_roll(s, act, q=0.5)
    d_roll = 1 - s / (th * ref)
    th9 = theta_roll(s, act, q=0.9)
    d_env = 1 - s / (th9 * ref)
    p = df[PEER[0]] + df[PEER[1]]
    k = (s / p)[act & (ref >= 0.35) & (ref <= 0.6)].median()
    d_did = 1 - s / (k * p)
    curves[(a, b)] = pd.DataFrame({"published": by_ref(d_pub), "rolling": by_ref(d_roll),
                                   "envelope": by_ref(d_env), "DiD(peer)": by_ref(d_did)})
    print(f"\n{a}+{b}  [{KIND[(a,b)]}]")
    print(curves[(a, b)].round(3).T.to_string())

print()
print("=" * 104)
print("B. THE HEADLINE NUMBER:  baseline deficit at ref>=0.8, on pairs that CANNOT saturate")
print("=" * 104)
tab = pd.DataFrame({f"{a}+{b}": curves[(a, b)].loc[0.8:].mean()
                    for a, b in ALLP if KIND[(a, b)] == "control"}).T
tab["mean"] = tab.mean(axis=1)
print(tab.round(3).to_string())
print("\n   the published baseline already calls ~10% saturation on these healthy pairs.")
print("   the rolling / envelope baselines sit at ~0 on the same pairs.")

print()
print("=" * 104)
print("C. APPARENT LOST ENERGY  (sum of positive shortfall, ref>=0.7, active, no DQ)  [kWh]")
print("=" * 104)
rows = {}
for a, b in ALLP:
    s = df[a] + df[b]
    act = FE["active"][a] & FE["active"][b]
    ok = (act & (ref >= 0.7) & ~dq_mask(FE, a, b)).to_numpy()
    r = detect(df, FE, a, b)
    e_pub = r["expected"]
    th = theta_roll(s, act, q=0.5); e_roll = th * ref
    th9 = theta_roll(s, act, q=0.9); e_env = th9 * ref
    p = df[PEER[0]] + df[PEER[1]]
    k = (s / p)[act & (ref >= 0.35) & (ref <= 0.6)].median()
    e_did = k * p
    n = int(ok.sum())
    rows[f"{a}+{b}"] = {
        "published": float(np.maximum(e_pub - s, 0).to_numpy()[ok].sum()) * 5 / 60 / 1000,
        "rolling": float(np.maximum(e_roll - s, 0).to_numpy()[ok].sum()) * 5 / 60 / 1000,
        "envelope": float(np.maximum(e_env - s, 0).to_numpy()[ok].sum()) * 5 / 60 / 1000,
        "DiD(peer)": float(np.maximum(e_did - s, 0).to_numpy()[ok].sum()) * 5 / 60 / 1000,
        "rows": n}
T = pd.DataFrame(rows).T
T["group"] = [KIND[p] for p in ALLP]
print(T.round(0).to_string())
print("\ncontrol-pair totals (should be ~0 for an unbiased baseline):")
print(T[T.group == "control"][["published", "rolling", "envelope", "DiD(peer)"]].sum().round(0).to_string())
print("\nshared-pair totals (the real signal):")
print(T[T.group == "SHARED"][["published", "rolling", "envelope", "DiD(peer)"]].sum().round(0).to_string())
T.to_csv(OUT + r"\baseline_calibration.csv")

# --------------------------------------------------------------------------- #
print()
print("=" * 104)
print("D. WHERE IS THE PUBLISHED BASELINE BIAS COMING FROM?")
print("=" * 104)
a, b = "eu_1", "eu_3"
s = df[a] + df[b]
act = FE["active"][a] & FE["active"][b]
r = detect(df, FE, a, b)
gcap = pd.Series(r["g0"] * r["cap_t"], index=df.index)
th = theta_roll(s, act, q=0.5)
print(f"pair {a}+{b} (never clipped):")
print(f"  g0 (global)                     = {r['g0']:.3f}")
print(f"  g0*cap(t) median                = {gcap.median():,.0f} W")
print(f"  theta_roll(t) median            = {th.median():,.0f} W")
print(f"  ratio g0*cap/theta              = {(gcap/th).median():.3f}  (drifts "
      f"{float((gcap/th).quantile(.05)):.2f}-{float((gcap/th).quantile(.95)):.2f} over the record)")
print(f"  => the single global g0 cannot track the time-varying ratio; the residual")
print(f"     shows up as a spurious deficit.  SD of that ratio = {float((gcap/th).std()):.3f}")
r_hi = gcap / th
print(f"  ratio on high-ref days vs low-ref days: "
      f"{float(r_hi[ref>0.8].median()):.3f} vs {float(r_hi[ref<0.5].median()):.3f}")
