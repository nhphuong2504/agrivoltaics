"""
STEP 2 - the identifiability question.

The published claim rests on `gain = s / (cap*ref)` rolling off at high `ref`.
But `ref` and `cap` are both derived from this same fleet, so a roll-off could in
principle be a normalisation artefact (e.g. `ref` saturating, or `cap` being biased).

Test that removes the artefact: a difference-in-differences on *normalised* power.
    x_c(t)  = u_c(t) / p99.9(u_c)        (unit -> [0,1], "how hard is this unit working")
    X_P(t)  = x_a + x_b                  (a pair's normalised output, in units of "unit-peaks")
    ratio   = X_P / X_Q   for a peer pair Q
Rescale ratio so its median over ref in [0.35,0.6] is 1. Nothing but a ratio of raw
measurements, no cap, no g0, no ref inside the ceiling. If shared pairs really clip,
ratio must fall at high ref while control-vs-control stays flat.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *

df = load()
fe = preprocess(df)
ref = fe["ref"]
PEAK = df[EU].quantile(0.999)
X = df[EU].div(PEAK)                      # normalised unit output, 0..1

def pairX(a, b):
    return X[a] + X[b]

def did(P, Q, bins=np.linspace(0.3, 1.05, 26)):
    """Median of (X_P/X_Q) normalised to 1 in the calibration band, per ref bin."""
    p = pairX(*P); q = pairX(*Q)
    r = p / q.replace(0, np.nan)
    cal = (ref >= 0.35) & (ref <= 0.6)
    r = r / r[cal].median()
    idx = np.digitize(ref, bins).clip(1, len(bins) - 1)
    s = (ref > 0.3) & r.notna()
    med = r[s].groupby(idx[s]).median().reindex(range(1, len(bins)))
    q25 = r[s].groupby(idx[s]).quantile(.25).reindex(range(1, len(bins)))
    q75 = r[s].groupby(idx[s]).quantile(.75).reindex(range(1, len(bins)))
    ctr = 0.5 * (bins[1:] + bins[:-1])
    return pd.Series(med.values, index=ctr), pd.Series(q25.values, index=ctr), pd.Series(q75.values, index=ctr)

print("=" * 78)
print("A. DIFFERENCE-IN-DIFFERENCES vs a peer pair  (1.0 at mid irradiance == healthy)")
print("=" * 78)
print("rows = median(normalised ratio) in each ref bin; a declining row = the pair")
print("delivers progressively less than an independent pair of the same fleet.\n")

peer = ("eu_1", "eu_3")     # arbitrary independent peer pair
tab = {}
for name, P in [("S eu_8+eu_16", ("eu_8", "eu_16")),
                ("S eu_10+eu_18", ("eu_10", "eu_18")),
                ("S eu_13+eu_21", ("eu_13", "eu_21")),
                ("C eu_4+eu_12", ("eu_4", "eu_12")),
                ("C eu_15+eu_23", ("eu_15", "eu_23")),
                ("C eu_1+eu_3", ("eu_1", "eu_3"))]:
    if P == peer and name.startswith("C"):
        tab[name] = np.nan
        continue
    med, _, _ = did(P, peer)
    tab[name] = med
print(pd.DataFrame(tab).round(3).to_string())

print()
print("... same thing against a *different* peer pair, to show it is not peer-specific:")
for other in [("eu_4", "eu_12"), ("eu_15", "eu_23")]:
    rows = {}
    for name, P in [("S eu_8+eu_16", ("eu_8", "eu_16")),
                    ("S eu_13+eu_21", ("eu_13", "eu_21")),
                    ("C eu_1+eu_3", ("eu_1", "eu_3"))]:
        med, _, _ = did(P, other)
        rows[name] = med
    t = pd.DataFrame(rows).round(3)
    print(f"\npeer = {other[0]}+{other[1]}")
    print(t.to_string())

print()
print("=" * 78)
print("B. HOW MUCH ENERGY?  implied clipping loss on shared pairs")
print("=" * 78)
# Use the DiD curve to estimate, for each shared pair, the loss factor at high ref
for name, P in [("eu_8+eu_16", ("eu_8", "eu_16")),
                ("eu_10+eu_18", ("eu_10", "eu_18")),
                ("eu_13+eu_21", ("eu_13", "eu_21")),
                ("CONTROL eu_4+eu_12", ("eu_4", "eu_12")),
                ("CONTROL eu_15+eu_23", ("eu_15", "eu_23"))]:
    med, _, _ = did(P, peer)
    hi = med.loc[0.7:].median()
    print(f"{name:20s} median ratio at ref>=0.7 : {hi:.3f}   -> {100*(1-hi):4.1f}% below peers")

print()
print("=" * 78)
print("C. DOES THE CEILING ABSORB THE CLIP?  (self-reference sanity test)")
print("=" * 78)
print("If `cap` were pinned at the clipping level, then g0*cap would be clipped too and")
print("the deficit would vanish. It does not, because g0 rescales. Check numerically:")
for a, b in PAIRS:
    r = detect(df, fe, a, b)
    cap = pd.Series(r["cap_t"], index=df.index)
    print(f"{a}+{b}: cap(t) p50={cap.median():,.0f} W ; g0*cap p50={r['g0']*cap.median():,.0f} W "
          f"=> implied unsaturated peak / observed peak = {r['g0']:.2f}")
