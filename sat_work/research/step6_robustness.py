"""
STEP 6 - robustness of the ranking to the *shape* of the constraint.

The benchmark so far injects a hard, constant clip.  A real converter limit drifts with
temperature and softens near the knee.  We re-run the comparison with
  (i)  a drifting limit      C(t) = C0 * (1 + 0.08*sin(2*pi*t/365) + slow random walk)
  (ii) a soft knee           s -> s - (1/beta)*softplus(beta*(s-C)),  1/beta = 500 W
  (iii) both
and confirm the ordering of the detectors does not depend on that choice.
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *
from sklearn.metrics import roc_auc_score

df_raw = load()
FULLCAL = pd.date_range(df_raw.index.min().normalize(), df_raw.index.max().normalize(), freq="D")
OUT = r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research"

def broadcast(per_day, dayidx, labels, window="31D", minp=5):
    roll = per_day.reindex(FULLCAL).rolling(window, center=True, min_periods=minp).median()
    return pd.Series(roll.reindex(dayidx).to_numpy(), index=labels)

def theta_roll(s, ref, act, dayidx, q=0.5, band=(0.35, 0.6)):
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= band[0]) & (ref <= band[1]) & u.notna()
    g = u[sel].groupby(dayidx[sel])
    per_day = g.quantile(q) if q != 0.5 else g.median()
    return broadcast(per_day, dayidx, s.index)

def persist(raw, k):
    return raw & (run_lengths(raw) >= k)

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)

def make_clip(s, C, beta=1 / 500.0):
    """hard or soft min(s, C).  beta -> 0 gives the hard min."""
    if beta is None:
        return np.minimum(s, C)
    return s - softplus(beta * (s - C)) / beta

def run_case(a, b, q, frac, drift, soft, seed=11):
    rel = [c for c in RELIABLE if c not in (a, b)]
    base = df_raw.copy()
    fe0 = preprocess(base, rel)
    Ptyp = np.nanpercentile((base[a] + base[b])[fe0["ref"] >= 0.85], 95)
    C0 = q * Ptyp
    dava = base[a].groupby(base.index.normalize()).count().pipe(lambda x: x[x > 0]).index
    rng = np.random.default_rng(seed)
    on = pd.Series(rng.random(len(dava)) < frac, index=dava) \
        .reindex(base.index.normalize()).fillna(False).to_numpy()
    if drift:
        seas = 1 + 0.08 * np.sin(2 * np.pi * (base.index.dayofyear.to_numpy() - 60) / 365.0)
        walk = pd.Series(rng.standard_normal(len(dava)) * 0.004, index=dava).cumsum()
        walk = walk.clip(-0.12, 0.12).reindex(base.index.normalize()).fillna(0).to_numpy()
        Cvec = C0 * (1 + (seas - 1) + walk)
    else:
        Cvec = np.full(len(base), C0)
    Cvec = np.where(on, Cvec, np.inf)
    d2 = base.copy()
    s0 = (base[a] + base[b]).to_numpy(float)
    scl = make_clip(s0, Cvec, (1 / 500.0) if soft else None)
    f = np.where(s0 > 0, scl / np.where(s0 > 0, s0, 1.0), 1.0)
    d2[a] = d2[a] * f; d2[b] = d2[b] * f
    gt = pd.Series(scl < s0 - 1e-9, index=base.index)      # rows the constraint actually held back
    fe = preprocess(d2, rel)
    act0 = fe0["active"][a] & fe0["active"][b]
    th_true = theta_roll(base[a] + base[b], fe0["ref"], act0, fe0["dayidx"], q=0.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        sev = pd.Series(np.clip(1 - Cvec / (th_true * fe0["ref"]), 0, 1), index=base.index).fillna(0)
    # detectors
    r0 = detect(d2, fe, a, b)
    out = {}
    for nm, qq in [("D1 rolling-slope", 0.5), ("D2 envelope-q90", 0.9)]:
        s = d2[a] + d2[b]
        act = fe["active"][a] & fe["active"][b]
        th = theta_roll(s, fe["ref"], act, fe["dayidx"], q=qq)
        exp_ = th * fe["ref"]
        dv = (1 - s / exp_).replace([np.inf, -np.inf], np.nan)
        raw = (fe["ref"] >= 0.7) & (s > 0.6 * exp_) & act & ~dq_mask(fe, a, b) & (dv > 0.15)
        out[nm] = (dv, persist(raw, 3))
    out["D0 published"] = (r0["deficit"], r0["moderate"])

    res = {}
    act = fe["active"][a] & fe["active"][b]
    test = (act & (fe["ref"] >= 0.7) & ~dq_mask(fe, a, b)).to_numpy()
    for nm, (sc, fl) in out.items():
        scv = pd.Series(np.asarray(sc, float), index=fe["ref"].index).to_numpy()
        ok = test & np.isfinite(scv)
        y = (gt.to_numpy() & ok).astype(int)
        flv = np.asarray(fl, bool) & ok
        neg = ok & (y == 0)
        e = {"fp_rows": int((flv & neg).sum()),
             "auc00": roc_auc_score(y[ok], scv[ok]) if 0 < y[ok].sum() < ok.sum() else np.nan}
        pos10 = ok & (y == 1) & (sev.to_numpy() >= 0.10)
        e["auc10"] = roc_auc_score(pos10.astype(int)[ok], scv[ok]) if 0 < pos10.sum() < ok.sum() else np.nan
        tp = int((flv & (y == 1)).sum()); fp = int((flv & (y == 0)).sum()); fn = int((~flv & (y == 1)).sum())
        e["rec"] = tp / max(tp + fn, 1); e["prec"] = tp / max(tp + fp, 1)
        res[nm] = e
    return res, int(gt.sum())

CASES = [("hard, constant", False, False), ("hard, drifting", True, False),
         ("soft, constant", False, True), ("soft, drifting", True, True)]
print("=" * 100)
print("ROBUSTNESS TO CONSTRAINT SHAPE  (q=0.75, 40% of days, 3 pseudo-pairs each)")
print("=" * 100)
rows = []
for label, drift, soft in CASES:
    agg = {}
    for (a, b) in [("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]:
        res, ngt = run_case(a, b, 0.75, 0.40, drift, soft)
        for nm, e in res.items():
            agg.setdefault(nm, []).append(e)
    for nm, lst in agg.items():
        m = pd.DataFrame(lst).mean()
        m["case"] = label
        m["det"] = nm
        rows.append(m)
    line = "  ".join(f"{nm.split()[0]}: auc10={pd.DataFrame(lst).auc10.mean():.3f} "
                     f"fp={pd.DataFrame(lst).fp_rows.mean():6.0f}" for nm, lst in agg.items())
    print(f"{label:16s} {line}")

R = pd.DataFrame(rows)
R.to_csv(OUT + r"\bench_robustness.csv", index=False)
print()
print(R.pivot_table(index="case", columns="det", values=["auc00", "auc10", "fp_rows"]).round(3).to_string())
