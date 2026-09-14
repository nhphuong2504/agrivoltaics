"""
STEP 4 - why the published detector over-flags, and a synthesised replacement.

Hypothesis
----------
`cap(t)` is the rolling median of the pair's *own* daily 99th percentile.  If the pair
is clipped on a sizeable fraction of days, that ceiling is dragged down toward the clip
level.  `g0` is a single global scalar, so it cannot undo a *time-varying* drag, and the
model then predicts too little power on healthy days -> systematic false positives.

We measure the drag directly, then build the synthesised detector R:
    theta(t)  = 31-day centred rolling median of the daily 90th pct of s/ref
                (a clear-sky envelope inside a low-irradiance anchor band)
    d(t)      = 1 - s / (theta(t) * ref(t))
    null bin  = 99.5th pct of the *control pairs'* d in the same ref bin
    flag      = d > null AND s > 0.6*theta*ref AND ref>=0.7 AND active AND ~DQ
                AND persistence >= k
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df_raw = load()
FULLCAL = pd.date_range(df_raw.index.min().normalize(), df_raw.index.max().normalize(), freq="D")
OUT = r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research"
BINS = np.linspace(0.30, 1.05, 21)

# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def inject(df, a, b, C, day_on=None):
    d = df.copy()
    s = (d[a] + d[b]).to_numpy(dtype=float)
    Cv = np.asarray(C) if np.ndim(C) else np.full(len(d), float(C))
    if day_on is not None:
        Cv = np.where(day_on, Cv, np.inf)
    f = np.where(s > 0, np.minimum(s, Cv) / np.where(s > 0, s, 1.0), 1.0)
    d[a] = d[a] * f; d[b] = d[b] * f
    return d, pd.Series(s > Cv, index=d.index)


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


def null_curve(fe, pairs, q=0.9, alpha=0.995, band=(0.30, 0.60)):
    """Per-ref-bin quantile of the deficit shown by *independent* pairs."""
    pool = []
    for a, b in pairs:
        s = fe["_df"][a] + fe["_df"][b]
        act = fe["active"][a] & fe["active"][b]
        th = theta_roll(s, fe["ref"], act, fe["dayidx"], q=q, band=band)
        d = (1 - s / (th * fe["ref"])).replace([np.inf, -np.inf], np.nan)
        ok = act & (fe["ref"] >= 0.3) & ~dq_mask(fe, a, b)
        pool.append(pd.DataFrame({"ref": fe["ref"][ok], "def": d[ok]}))
    pool = pd.concat(pool).dropna()
    pool["bin"] = np.digitize(pool["ref"], BINS).clip(1, len(BINS) - 1)
    return pool.groupby("bin")["def"].quantile(alpha)


def d_baseline(df, fe, a, b, q=0.9, band=(0.30, 0.60)):
    s = df[a] + df[b]
    act = fe["active"][a] & fe["active"][b]
    th = theta_roll(s, fe["ref"], act, fe["dayidx"], q=q, band=band)
    exp_ = th * fe["ref"]
    return s, act, th, exp_, (1 - s / exp_).replace([np.inf, -np.inf], np.nan)


def R_detector(df, fe, a, b, controls=CONTROL_PAIRS, exclude=(), q=0.9,
               alpha=0.995, ref_on=0.70, k=3, band=(0.30, 0.60)):
    s, act, th, exp_, d = d_baseline(df, fe, a, b, q=q, band=band)
    pool = [c for c in controls if not (set(c) & set(exclude))]
    fe2 = dict(fe); fe2["_df"] = df
    null = null_curve(fe2, pool, q=q, alpha=alpha, band=band)
    binidx = np.digitize(fe["ref"], BINS).clip(1, len(BINS) - 1)
    qn = pd.Series(binidx, index=df.index).map(null).astype(float)
    excess = d - qn
    base = (fe["ref"] >= ref_on) & (s > 0.6 * exp_) & act & ~dq_mask(fe, a, b)
    raw = base & (excess > 0)
    return excess, persist(raw, k)


# --------------------------------------------------------------------------- #
# A. the drag diagnostic
# --------------------------------------------------------------------------- #
print("=" * 108)
print("A. WHY THE PUBLISHED CEILING OVER-FLAGS:  g0*cap versus the true unsaturated slope")
print("=" * 108)
print("scenario                      theta_true   g0*cap    ratio   -> bias in `expected`")
rowsA = []
for q, frac, label in [(None, 0.0, "no clip"), (0.75, 0.35, "clip 35% of days"),
                       (0.75, 0.65, "clip 65% of days"), (0.75, 1.0, "clip every day")]:
    a, b = "eu_1", "eu_3"
    rel = [c for c in RELIABLE if c not in (a, b)]
    base = df_raw.copy()
    fe0 = preprocess(base, rel)
    Ptyp = np.nanpercentile((base[a] + base[b])[fe0["ref"] >= 0.85], 95)
    dava = base[a].groupby(base.index.normalize()).count().pipe(lambda x: x[x > 0]).index
    day_on = pd.Series(np.random.default_rng(3).random(len(dava)) < frac, index=dava) \
        .reindex(base.index.normalize()).fillna(False).to_numpy()
    d2, gt = inject(base, a, b, (np.inf if q is None else q * Ptyp), day_on)
    fe = preprocess(d2, rel)
    act = fe["active"][a] & fe["active"][b]
    th_true = theta_roll(d2[a] + d2[b], fe["ref"], act, fe["dayidx"], q=0.5)
    r = detect(d2, fe, a, b)
    gcap = pd.Series(r["g0"] * r["cap_t"], index=d2.index)
    ratio = (gcap / th_true).median()
    rowsA.append(dict(scenario=label, theta_true=th_true.median(), g0_cap=gcap.median(),
                      ratio=ratio, deficit_on_healthy_days=float((r["deficit"][act & (fe["ref"] >= 0.9)]).median())))
    print(f"  {label:22s} {th_true.median():8.0f} {gcap.median():9.0f} {ratio:8.3f}"
          f"   median deficit at ref>=0.9: {rowsA[-1]['deficit_on_healthy_days']:+.3f}")

print("\n   ratio << 1 means `expected` is systematically too small -> saturation is claimed")
print("   even on the days that were never clipped.")

# --------------------------------------------------------------------------- #
# B. head-to-head, including the synthesised detector, on a realistic duty cycle
# --------------------------------------------------------------------------- #
def evaluate(score, flag, fe, a, b, gt, sev, ref_on=0.70):
    act = fe["active"][a] & fe["active"][b]
    test = (act & (fe["ref"] >= ref_on) & ~dq_mask(fe, a, b)).to_numpy()
    sc = pd.Series(np.asarray(score, dtype=float), index=fe["ref"].index).to_numpy()
    ok = test & np.isfinite(sc)
    gt, sev = gt.to_numpy(), np.asarray(sev, dtype=float)
    fl = np.asarray(flag, dtype=bool) & ok
    neg = ok & ~gt.astype(bool)
    out = dict(n_neg=int(neg.sum()))
    for dlt in (0.0, 0.05, 0.10, 0.20):
        pos = ok & gt.astype(bool) & (sev >= dlt)
        out[f"auc{int(dlt*100):02d}"] = (roc_auc_score(pos.astype(int)[ok], sc[ok])
                                         if 0 < pos.sum() < ok.sum() else np.nan)
    out["fp_rows"] = int((fl & neg).sum())
    tp = int((fl & ok & gt.astype(bool)).sum()); fn = int((~fl & ok & gt.astype(bool)).sum())
    out["rec"] = tp / max(tp + fn, 1)
    out["prec"] = tp / max(tp + int((fl & neg).sum()), 1)
    return out


SCEN = [(f, pat) for f in (0.35, 0.65, 1.0) for pat in ("chronic",)]
print()
print("=" * 108)
print("B. PUBLISHED vs DE-HACKIFIED vs SYNTHESISED   (duty cycle = fraction of days clipped, q=0.75)")
print("=" * 108)
rows = []
for frac, pat in SCEN:
    for (a, b) in [("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]:
        rel = [c for c in RELIABLE if c not in (a, b)]
        base = df_raw.copy()
        fe0 = preprocess(base, rel)
        Ptyp = np.nanpercentile((base[a] + base[b])[fe0["ref"] >= 0.85], 95)
        dava = base[a].groupby(base.index.normalize()).count().pipe(lambda x: x[x > 0]).index
        day_on = pd.Series(np.random.default_rng(7).random(len(dava)) < frac, index=dava) \
            .reindex(base.index.normalize()).fillna(False).to_numpy()
        d2, gt = inject(base, a, b, 0.75 * Ptyp, day_on)
        fe = preprocess(d2, rel)
        act0 = fe0["active"][a] & fe0["active"][b]
        th_true = theta_roll(base[a] + base[b], fe0["ref"], act0, fe0["dayidx"], q=0.5)
        sev = pd.Series(np.clip(1 - 0.75 * Ptyp / (th_true * fe0["ref"]), 0, 1), index=base.index).fillna(0)

        r0 = detect(d2, fe, a, b)
        s, act, th, exp_, d = d_baseline(d2, fe, a, b, q=0.5)
        baseg = (fe["ref"] >= 0.7) & (s > 0.6 * exp_) & act & ~dq_mask(fe, a, b)
        f1_ = persist(baseg & (d > 0.15), 3)
        s, act, th, exp_, d = d_baseline(d2, fe, a, b, q=0.9)
        baseg = (fe["ref"] >= 0.7) & (s > 0.6 * exp_) & act & ~dq_mask(fe, a, b)
        f2_ = persist(baseg & (d > 0.15), 3)
        ex, fR = R_detector(d2, fe, a, b, exclude=(a, b))
        cands = {"D0 published": (r0["deficit"], r0["moderate"]),
                 "D1 rolling-slope": (d_baseline(d2, fe, a, b, q=0.5)[4], f1_),
                 "D2 envelope-q90": (d_baseline(d2, fe, a, b, q=0.9)[4], f2_),
                 "R synthesised": (ex, fR)}
        for nm, (sc, fl) in cands.items():
            e = evaluate(sc, fl, fe, a, b, gt, sev)
            e.update(duty=frac, pair=f"{a}+{b}", det=nm)
            rows.append(e)
        print(f"  duty={frac:.2f} {a}+{b}: done")

B = pd.DataFrame(rows)
B.to_csv(OUT + r"\bench_synthesis.csv", index=False)
print()
print("mean over the 9 injected pairs:")
print(B.groupby("det")[["auc00", "auc05", "auc10", "auc20", "fp_rows", "rec", "prec"]]
      .mean().round(3).to_string())
print()
print("by duty cycle (false-positive rows at ref>=0.7):")
print(B.pivot_table(index="duty", columns="det", values="fp_rows").round(0).astype("Int64").to_string())
print("\nby duty cycle (AUC on all clipped rows):")
print(B.pivot_table(index="duty", columns="det", values="auc00").round(3).to_string())
print("\nby duty cycle (AUC, clips leaving >10% deficit):")
print(B.pivot_table(index="duty", columns="det", values="auc10").round(3).to_string())

# --------------------------------------------------------------------------- #
# C. real data: how do the published and synthesised outputs compare?
# --------------------------------------------------------------------------- #
print()
print("=" * 108)
print("C. REAL SHARED PAIRS: flagged hours, published vs synthesised")
print("=" * 108)
feR = preprocess(df_raw)
tab = {}
for a, b in PAIRS:
    r0 = detect(df_raw, feR, a, b)
    ex, fR = R_detector(df_raw, feR, a, b, exclude=(a, b))
    ex2, fR2 = R_detector(df_raw, feR, a, b, exclude=(a, b), k=6, alpha=0.999)
    tab[f"{a}+{b}"] = {"D0 moderate hrs": float(r0["moderate"].sum()) / 12,
                       "D0 severe hrs": float(r0["severe"].sum()) / 12,
                       "R hrs (k=3)": float(fR.sum()) / 12,
                       "R hrs (k=6, a=.999)": float(fR2.sum()) / 12,
                       "R days": int(fR[fR].index.normalize().nunique())}
print(pd.DataFrame(tab).round(1).to_string())
ctab = {}
for a, b in CONTROL_PAIRS:
    r0 = detect(df_raw, feR, a, b)
    ex, fR = R_detector(df_raw, feR, a, b, exclude=(a, b))
    ctab[f"{a}+{b}"] = {"D0 rows": int(r0["moderate"].sum()), "R rows": int(fR.sum())}
print("\ncontrol pairs (independent; every flag is a false alarm):")
print(pd.DataFrame(ctab).to_string())
