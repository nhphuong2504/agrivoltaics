"""
STEP 12 - what is actually special about July 2026?

step10 localised the disagreement but did not explain it: July 2026 has ~95 h of
published-only flags against 25-39 h in every other summer month, so it is not just
"summer". This script compares, month by month and at MATCHED irradiance:

    truth_hi   = the normalisation-free peer-ratio shortfall at ref >= 0.75
    d_pub_hi   = the published deficit on the same rows
    d_vat_hi   = the vat-v1 deficit on the same rows
    anchor     = the pair's mid-band efficiency relative to a peer pair
                 (a fall here means theta is being dragged down)

If `anchor` collapses in July 2026 while `truth_hi` does not, the clear-sky anchor is
contaminated and vat-v1 is under-flagging. If `anchor` is stable and d_pub is merely
high, the published model is over-claiming.

Run:  ./venv/Scripts/python.exe sat_work/research/step12_julycause.py
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *

pd.set_option("display.width", 220)
df = load(); FE = preprocess(df); ref = FE["ref"]
FULLCAL = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
BAND = (0.35, 0.60)
CONTROLS = [("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]
PEERS = [("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]
X = df[EU].div(df[EU].quantile(0.999))
mo = df.index.to_period("M").astype(str)


def broadcast(per_day, dayidx, labels, window="31D", minp=5):
    roll = per_day.reindex(FULLCAL).rolling(window, center=True, min_periods=minp).median()
    return pd.Series(roll.reindex(dayidx).to_numpy(), index=labels)


def theta(s, act):
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= BAND[0]) & (ref <= BAND[1]) & u.notna()
    return broadcast(u[sel].groupby(FE["dayidx"][sel]).median(), FE["dayidx"], s.index)


def vat_def(a, b):
    s = df[a] + df[b]
    act = FE["active"][a] & FE["active"][b]
    exp_ = theta(s, act) * ref
    return s, act, exp_, (1 - s / exp_).replace([np.inf, -np.inf], np.nan)


def peer_r(a, b):
    """Median pair/peer normalised ratio on a mask, averaged over the three peer pairs."""
    sa = X[a] + X[b]
    act = FE["active"][a] & FE["active"][b]
    out = {}
    for p1, p2 in PEERS:
        if {p1, p2} & {a, b}:
            continue
        pq = X[p1] + X[p2]
        pact = FE["active"][p1] & FE["active"][p2]
        r = sa / pq.replace(0, np.nan)
        cal = act & pact & (ref >= BAND[0]) & (ref <= BAND[1])
        out[f"{p1}+{p2}"] = (r / r[cal].median(), act & pact & ~dq_mask(FE, p1, p2))
    return out


print("=" * 130)
print("MONTH x MATCHED-IRRADIANCE DIAGNOSIS  (shared pairs; truth = peer ratio, no cap, no g0, no ref)")
print("=" * 130)
print("truth_hi  : 1 - peer-ratio shortfall at ref>=0.75  (0 = pair matches independent peers)")
print("d_pub_hi  : published deficit on the same rows")
print("d_vat_hi  : vat-v1 deficit on the same rows")
print("anchor    : pair mid-band efficiency / peer mid-band efficiency at ref in [0.35,0.60]")
print("            (tracked month by month; a collapse means theta is contaminated)\n")

for a, b in PAIRS:
    s, act, exp_vat, d_vat = vat_def(a, b)
    d_pub = pd.Series(detect(df, FE, a, b)["deficit"], index=df.index)
    pr = peer_r(a, b)
    clean = act & ~dq_mask(FE, a, b)
    rows = []
    for m in sorted(set(mo)):
        km = (mo == m)
        hi = km & clean & (ref >= 0.75)
        mid = km & clean & (ref >= BAND[0]) & (ref <= BAND[1])
        # peer-ratio on the high rows
        rs = [r[hi & ok].median() for r, ok in pr.values() if (hi & ok).sum() > 30]
        truth = 1 - float(np.median(rs)) if rs else np.nan
        # anchor: pair mid efficiency vs peer mid efficiency
        an = []
        for p1, p2 in PEERS:
            if {p1, p2} & {a, b}:
                continue
            pact = FE["active"][p1] & FE["active"][p2]
            mm = mid & pact & ~dq_mask(FE, p1, p2)
            if mm.sum() > 30:
                an.append((s[mm] / ref[mm]).median() / ((df[p1] + df[p2])[mm] / ref[mm]).median())
        rows.append(dict(month=m, n_hi=int(hi.sum()),
                         truth_hi=round(truth, 3) if rs else np.nan,
                         d_pub_hi=round(float(d_pub[hi].median()), 3) if hi.sum() else np.nan,
                         d_vat_hi=round(float(d_vat[hi].median()), 3) if hi.sum() else np.nan,
                         anchor=round(float(np.median(an)), 3) if an else np.nan))
    T = pd.DataFrame(rows)
    print("-" * 130)
    print(f"{a}+{b}")
    print(T.to_string(index=False))
    print()

print("=" * 130)
print("IS THE MID-BAND ANCHOR ITSELF CLIPPED?  pair/peer ratio, mid band vs high ref, by summer month")
print("=" * 130)
print("If a clip is deep enough to bite inside ref in [0.35,0.60] the mid-band ratio falls")
print("too, which drags theta down and makes vat-v1 under-flag.\n")
for a, b in PAIRS:
    pr = peer_r(a, b)
    clean = FE["active"][a] & FE["active"][b] & ~dq_mask(FE, a, b)
    out = []
    for m in ["2025-06", "2025-07", "2026-05", "2026-06", "2026-07"]:
        km = (mo == m) & clean
        mm = km & (ref >= BAND[0]) & (ref <= BAND[1])
        hh = km & (ref >= 0.75)
        vals = []
        for r, ok in pr.values():
            if (mm & ok).sum() > 20 and (hh & ok).sum() > 20:
                vals.append((r[mm & ok].median(), r[hh & ok].median()))
        if vals:
            out.append(dict(month=m, mid_ratio=round(float(np.median([v[0] for v in vals])), 3),
                            hi_ratio=round(float(np.median([v[1] for v in vals])), 3)))
    T = pd.DataFrame(out)
    if len(T):
        T["hi_over_mid"] = (T.hi_ratio / T.mid_ratio).round(3)
        print(f"{a}+{b}")
        print(T.to_string(index=False))
        print()
