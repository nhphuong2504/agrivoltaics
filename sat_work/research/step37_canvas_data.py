"""Emit the canvas's data arrays from the completed record.

The canvas hard-codes its chart series and table cells, so every one of them is a second
copy of a number that lives in the analysis. This regenerates them so the canvas cannot
drift.
"""
import sys
sys.path.insert(0, "sat_work/research")
import numpy as np
import pandas as pd
import bench as B
from satsim import EU, PAIRS, CONTROL_PAIRS

df = B.raw_df()
fe = B.raw_fe()
ref = fe["ref"]
BINS = np.linspace(0.30, 1.05, 21)
X = df[EU].div(df[EU].quantile(0.999))


def binmed(x, lo=0.3):
    idx = np.digitize(ref, BINS).clip(1, len(BINS) - 1)
    s = pd.Series(np.asarray(x, dtype=float), index=df.index)
    ok = (ref > lo) & np.isfinite(s)
    return s[ok].groupby(idx[ok]).median().reindex(range(1, len(BINS))).to_numpy()


def js(name, vals, per_line=6):
    out = []
    for i in range(0, len(vals), per_line):
        out.append(", ".join("{:.4f}".format(v) for v in vals[i:i + per_line]))
    body = (",\n      ".join(out))
    print("const {} = [\n      {}\n];".format(name, body))


print("// ---- DID series (peer-normalised pair sum, per ref bin) ----")
did = {}
for lbl, P in (("shared_eu_8_eu_16", ("eu_8", "eu_16")),
               ("shared_eu_10_eu_18", ("eu_10", "eu_18")),
               ("shared_eu_13_eu_21", ("eu_13", "eu_21")),
               ("control_eu_4_eu_12", ("eu_4", "eu_12")),
               ("control_eu_15_eu_23", ("eu_15", "eu_23"))):
    peer = ("eu_4", "eu_12") if P == ("eu_1", "eu_3") else ("eu_1", "eu_3")
    r = (X[P[0]] + X[P[1]]) / (X[peer[0]] + X[peer[1]]).replace(0, np.nan)
    r = r / r[(ref >= 0.35) & (ref <= 0.60)].median()
    did[lbl] = binmed(r)
    js("DID_" + lbl.upper(), did[lbl])
print()

print("// ---- calibration series (published vs vat-v1 vs peer truth), SHARED pairs ----")
pub, ours, tru = [], [], []
for P in PAIRS:
    d0 = pd.Series(B.pair_deficit(df, fe, P[0], P[1])["deficit"], index=df.index)
    d0 = d0.clip(lower=None)
    dd = B.build_export
    # published model
    _, _ = B.det_published(df, fe, P[0], P[1])
    cap = B.rolling_cap(df, fe, P[0], P[1]) if hasattr(B, "rolling_cap") else None
    peer = ("eu_4", "eu_12") if P == ("eu_1", "eu_3") else ("eu_1", "eu_3")
    ratio = (X[P[0]] + X[P[1]]) / (X[peer[0]] + X[peer[1]]).replace(0, np.nan)
    k = ratio[(ref >= 0.35) & (ref <= 0.60)].median()
    ours.append(binmed(d0.to_numpy()))
    tru.append(binmed((1 - ratio / k).to_numpy()))
print("// printed below by the caller")

print()
print("// ---- detector AUCs ----")
CAT = ["D1", "D2", "D5", "D0", "D4"]
FN = [B.det_rolling, B.det_envelope, B.det_control_calibrated, B.det_published, B.det_ceiling_pinned]
SC = [
    ("episodic", [("eu_1", "eu_3", 0.35, "episodic"), ("eu_15", "eu_23", 0.35, "episodic")]),
    ("anyclip", [("eu_1", "eu_3", 0.35, "chronic"), ("eu_1", "eu_3", 0.35, "episodic"),
                 ("eu_1", "eu_3", 1.00, "chronic"), ("eu_4", "eu_12", 0.35, "chronic"),
                 ("eu_15", "eu_23", 0.35, "episodic")]),
]
for lbl, scen in SC:
    vals = []
    for fn in FN:
        a = []
        for p, b2, duty, pat in scen:
            sc = B.scenario((p, b2), duty=duty, q=0.75, pattern=pat)
            a.append(B.score_scenario(sc, fn)["auc00"])
        vals.append(float(np.mean(a)))
    print("//", lbl, ["{:.3f}".format(v) for v in vals])
