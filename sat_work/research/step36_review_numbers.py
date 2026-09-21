"""Numbers the review's live claims need after the record refresh.

Covers the published detector's own counts, the retired severe tier, the per-pair
strict-vs-bridged comparison, and the monthly truth-vs-model table of §3.8.
"""
import sys
sys.path.insert(0, "sat_work/research")
import numpy as np
import pandas as pd
import bench as B

df = B.raw_df()
fe = B.preprocess(df, B.RELIABLE)
PAIRS = B.PAIRS


def counts(flag):
    f = pd.Series(np.asarray(flag, dtype=bool), index=df.index)
    return int(f.sum()), f.sum() / 12.0, int(f.groupby(f.index.normalize()).any().sum())


print("A. PUBLISHED (D0) vs CANONICAL (D1) vs RETIRED SEVERE")
tr = th = td = cr = ch = 0
for a, b in PAIRS:
    _, f0 = B.det_published(df, fe, a, b)
    _, f1 = B.det_rolling(df, fe, a, b)
    _, fs = B.det_severe(df, fe, a, b)
    r0, h0, d0 = counts(f0)
    r1, h1, d1 = counts(f1)
    rs, hs, ds = counts(fs)
    tr += r0; th += h0; td += d0
    print("  {}+{}: published {:>6,} rows / {:7.1f} h / {:>3} d   | "
          "vat-v1 {:>6,} / {:7.1f} / {:>3}   | severe {:>6,} / {:6.1f} / {:>3}".format(
              a, b, r0, h0, d0, r1, h1, d1, rs, hs, ds))
print("  TOTALS published {:,} rows / {:.1f} h / {} d".format(tr, th, td))

for a, b in B.CONTROL_PAIRS:
    _, f0 = B.det_published(df, fe, a, b)
    _, f1 = B.det_rolling(df, fe, a, b)
    print("  control {}+{}: published {} rows | vat-v1 {} rows".format(
        a, b, counts(f0)[0], counts(f1)[0]))

print()
print("B. STRICT vs BRIDGED PERSISTENCE  (real data, per pair)")
for a, b in PAIRS:
    for lbl, kw in (("strict", dict(gap=0)), ("bridged", {})):
        _, f = B.det_rolling(df, fe, a, b, **kw)
        r, h, d = counts(f)
        print("  {}+{} {:8s}: {:>6,} rows / {:7.1f} h / {:>3} d".format(a, b, lbl, r, h, d))
ts = tb = 0.0
for a, b in PAIRS:
    ts += counts(B.det_rolling(df, fe, a, b, gap=0)[1])[1]
    tb += counts(B.det_rolling(df, fe, a, b)[1])[1]
print("  total hours strict {:.1f} -> bridged {:.1f}  ({:+.1f} %)".format(
    ts, tb, 100 * (tb - ts) / ts))

print()
print("C. PEER RATIO AT ref >= 0.70  (% below peers)")
X = df[[c for c in df.columns if c.startswith("eu_")]]
peerof = {("eu_8", "eu_16"): ("eu_1", "eu_3"), ("eu_10", "eu_18"): ("eu_4", "eu_12"),
          ("eu_13", "eu_21"): ("eu_15", "eu_23"), ("eu_1", "eu_3"): ("eu_4", "eu_12"),
          ("eu_4", "eu_12"): ("eu_15", "eu_23"), ("eu_15", "eu_23"): ("eu_1", "eu_3")}
ref = fe["ref"]
for (a, b), (c, d) in peerof.items():
    p = X[a] + X[b]
    q = (X[c] + X[d]).replace(0, np.nan)
    r = p / q
    calib = (ref >= 0.35) & (ref <= 0.60)
    r = r / r[calib].median()
    hi = (ref >= 0.70) & r.notna() & fe["active"][a] & fe["active"][b]
    print("  {}+{} vs {}+{}: median {:.3f}  ({:+.1f} % vs peers)   n={:,}".format(
        a, b, c, d, r[hi].median(), 100 * (r[hi].median() - 1), int(hi.sum())))

print()
print("D. MONTHLY TRUTH vs MODEL, eu_8+eu_16  (ref >= 0.75)")
a, b = "eu_8", "eu_16"
p = X[a] + X[b]
q = (X["eu_1"] + X["eu_3"]).replace(0, np.nan)
r = p / q
r = r / r[(ref >= 0.35) & (ref <= 0.60)].median()
_, f0 = B.det_published(df, fe, a, b)
d0 = pd.Series(B.pair_deficit(df, fe, a, b)["deficit"], index=df.index)
d0 = d0  # vat-v1 deficit
hi = (ref >= 0.75)
mon = pd.Series(df.index, index=df.index).dt.to_period("M")
for m, g in pd.Series(np.arange(len(df)), index=df.index).groupby(mon):
    idx = g.to_numpy()
    sel = hi.to_numpy()[idx] & r.to_numpy()[idx] == r.to_numpy()[idx]
    if sel.sum() < 50:
        continue
    print("  {}  truth {:.3f}    vat-v1 {:.3f}    gap {:+.3f}".format(
        m, 1 - r.to_numpy()[idx][sel].mean() * 0 + (1 - np.median(r.to_numpy()[idx][sel])),
        np.median(d0.to_numpy()[idx][sel]),
        np.median(d0.to_numpy()[idx][sel]) - (1 - np.median(r.to_numpy()[idx][sel]))))
