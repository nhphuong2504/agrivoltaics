"""Emit chart series as JSON for the canvas."""
import sys, io, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *

df = load(); FE = preprocess(df); ref = FE["ref"]
FULLCAL = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
BINS = np.linspace(0.30, 1.05, 21)
CTR = [round(float(x), 3) for x in 0.5 * (BINS[1:] + BINS[:-1])]
X = df[EU].div(df[EU].quantile(0.999))

def by_ref(x):
    idx = np.digitize(ref, BINS).clip(1, len(BINS) - 1)
    ser = pd.Series(np.asarray(x, float), index=df.index)
    ok = (ref > 0.3) & np.isfinite(ser)
    return [round(float(v), 4) for v in
            ser[ok].groupby(idx[ok]).median().reindex(range(1, len(BINS))).to_numpy()]

def broadcast(per_day, dayidx, labels, window="31D", minp=5):
    roll = per_day.reindex(FULLCAL).rolling(window, center=True, min_periods=minp).median()
    return pd.Series(roll.reindex(dayidx).to_numpy(), index=labels)

def theta_roll(s, act, q=0.5, band=(0.35, 0.6)):
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= band[0]) & (ref <= band[1]) & u.notna()
    g = u[sel].groupby(FE["dayidx"][sel])
    per_day = g.quantile(q) if q != 0.5 else g.median()
    return broadcast(per_day, FE["dayidx"], s.index)

did = {}
for a, b, lab in [("eu_8", "eu_16", "shared eu_8+eu_16"), ("eu_10", "eu_18", "shared eu_10+eu_18"),
                  ("eu_13", "eu_21", "shared eu_13+eu_21"), ("eu_4", "eu_12", "control eu_4+eu_12"),
                  ("eu_15", "eu_23", "control eu_15+eu_23")]:
    rr = (X[a] + X[b]) / (X["eu_1"] + X["eu_3"]).replace(0, np.nan)
    cal = (ref >= 0.35) & (ref <= 0.6)
    did[lab] = by_ref(rr / rr[cal].median())

base = {}
for a, b, lab in [("eu_1", "eu_3", "control eu_1+eu_3"), ("eu_8", "eu_16", "shared eu_8+eu_16")]:
    s = df[a] + df[b]; act = FE["active"][a] & FE["active"][b]
    pub = detect(df, FE, a, b)["deficit"]
    th = theta_roll(s, act, q=0.5); roll = 1 - s / (th * ref)
    p = df["eu_1"] + df["eu_3"]; k = (s / p)[act & (ref >= 0.35) & (ref <= 0.6)].median()
    truth = 1 - s / (k * p)
    base[lab] = {"published": by_ref(pub), "rolling": by_ref(roll), "peer": by_ref(truth)}

print(json.dumps({"ref": CTR, "did": did, "base": base}, indent=1))
