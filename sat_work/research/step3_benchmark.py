"""
STEP 3 - ground-truth benchmark (v2).

We cannot know the truth for the real shared pairs, so we manufacture it: take two
*independent* units, replace their aggregate with min(u+v, C) -- exactly what a shared
MPPT with limit C does to an observable pair sum -- and record which samples were
actually constrained.  Positives therefore sit on top of the real cloud/gap/temperature
confounds, and unmodified pseudo-pairs provide labelled negatives.

Severity is defined per sample, not per scenario:
    sev(t) = 1 - C / (theta_true(t) * ref(t))     (0 = the constraint barely binds)
and AUC is stratified by sev so that "detect any binding constraint" is not confused
with "detect meaningful saturation".

Detectors
---------
D0  published                 : deficit vs g0 * rolling-cap * ref, ref>=0.7, persist>=3
D1  rolling slope (q=0.5)     : deficit vs theta(t)*ref, theta = 30-day rolling median of s/ref
D2  clear-sky envelope (q=0.9): same, theta = 30-day rolling 90th pct of s/ref
D3  ref-gated 2-state HMM     : EM on the utilisation ratio, state 1 mean = min(1, knee/ref)
D4  ceiling-pinned            : s sitting on its own plateau while theta*ref exceeds it
D5  control-calibrated        : deficit minus the 99.5th pct of the *control pairs'* deficit
                                in the same ref bin  (empirical null, no magic constant)
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
BINS = np.linspace(0.30, 1.05, 21)
DETS = ["D0 published", "D1 rolling-slope", "D2 envelope-q90", "D3 HMM-2state",
        "D4 ceiling-pinned", "D5 control-calibrated"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def inject(df, a, b, C, day_on=None):
    d = df.copy()
    s = (d[a] + d[b]).to_numpy(dtype=float)
    Cv = np.asarray(C) if np.ndim(C) else np.full(len(d), float(C))
    if day_on is not None:
        Cv = np.where(day_on, Cv, np.inf)
    f = np.where(s > 0, np.minimum(s, Cv) / np.where(s > 0, s, 1.0), 1.0)
    d[a] = d[a] * f
    d[b] = d[b] * f
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


def pair_deficit(df, fe, a, b, q=0.5):
    s = df[a] + df[b]
    act = fe["active"][a] & fe["active"][b]
    th = theta_roll(s, fe["ref"], act, fe["dayidx"], q=q)
    exp_ = th * fe["ref"]
    deficit = (1 - s / exp_).replace([np.inf, -np.inf], np.nan)
    return dict(s=s, act=act, theta=th, expected=exp_, deficit=deficit)


# --------------------------------------------------------------------------- #
# detectors
# --------------------------------------------------------------------------- #
def d0_published(df, fe, a, b):
    r = detect(df, fe, a, b)
    return r["deficit"], r["moderate"]


def d1_or_d2(df, fe, a, b, q=0.5, thr=0.15, ref_on=0.70, pers=3):
    d = pair_deficit(df, fe, a, b, q=q)
    base = (fe["ref"] >= ref_on) & (d["s"] > 0.6 * d["expected"]) & d["act"] & ~dq_mask(fe, a, b)
    raw = base & (d["deficit"] > thr)
    return d["deficit"], persist(raw, pers)


def d4_ceiling_pinned(df, fe, a, b, ref_on=0.70, pers=3, nsigma=2.0):
    s = df[a] + df[b]
    act = fe["active"][a] & fe["active"][b]
    hi = act & (fe["ref"] >= 0.7)
    C = broadcast(s[hi].groupby(fe["dayidx"][hi]).max(), fe["dayidx"], df.index)
    th = theta_roll(s, fe["ref"], act, fe["dayidx"], q=0.9)
    potential = th * fe["ref"]
    onp = (hi & (s > 0.9 * C) & C.gt(0)).to_numpy()
    sc = np.abs(s.to_numpy() - C.to_numpy())[onp]
    sigma = max(float(np.nanmedian(sc)) if len(sc) else 1.0, 1.0)
    score = np.exp(-0.5 * ((s - C) / (nsigma * sigma)) ** 2) * (potential > C).astype(float)
    raw = (score > 0.5) & (fe["ref"] >= ref_on) & act & ~dq_mask(fe, a, b)
    return score, persist(raw, pers)


def d5_control_calibrated(df, fe, a, b, controls=CONTROL_PAIRS, q=0.5, alpha=0.995,
                          ref_on=0.70, pers=3, exclude=()):
    d = pair_deficit(df, fe, a, b, q=q)
    pool = []
    for c1, c2 in controls:
        if c1 in exclude or c2 in exclude or (c1, c2) == (a, b):
            continue
        dc = pair_deficit(df, fe, c1, c2, q=q)
        ok = dc["act"] & (fe["ref"] >= 0.3) & ~dq_mask(fe, c1, c2)
        pool.append(pd.DataFrame({"ref": fe["ref"][ok], "def": dc["deficit"][ok]}))
    pool = pd.concat(pool).dropna()
    pool["bin"] = np.digitize(pool["ref"], BINS).clip(1, len(BINS) - 1)
    null = pool.groupby("bin")["def"].quantile(alpha)
    binidx = np.digitize(fe["ref"], BINS).clip(1, len(BINS) - 1)
    q_null = pd.Series(binidx, index=df.index).map(null).astype(float)
    excess = d["deficit"] - q_null
    base = (fe["ref"] >= ref_on) & (d["s"] > 0.6 * d["expected"]) & d["act"] & ~dq_mask(fe, a, b)
    raw = base & (excess > 0)
    return excess, persist(raw, pers)


def d3_hmm(df, fe, a, b, q=0.5, ref_on=0.70, pers=3, iters=40):
    """Ref-gated 2-state Gaussian HMM on the utilisation ratio z = s/(theta*ref).

    state 0 (healthy)     : z ~ N(1, s0^2)
    state 1 (constrained) : z ~ N(min(1, knee/ref), s1^2)
    Fitted by EM; the knee is re-estimated each M-step on a grid inside (0,1) so that
    state 1 can never collapse onto state 0.
    """
    d = pair_deficit(df, fe, a, b, q=q)
    z_all = (d["s"] / d["expected"]).to_numpy(dtype=float)
    rf_all = fe["ref"].to_numpy(dtype=float)
    fitm = (rf_all >= 0.45) & np.isfinite(z_all) & d["act"].to_numpy()
    z, rf = z_all[fitm], rf_all[fitm]
    if len(z) < 500:
        return pd.Series(np.nan, index=df.index), pd.Series(False, index=df.index)
    knees = np.linspace(0.30, 0.99, 70)
    knee, s0, s1 = 0.80, 0.05, 0.08
    A = np.array([[0.97, 0.03], [0.03, 0.97]]); pi = np.array([0.6, 0.4])
    m1 = lambda g: np.minimum(1.0, g / np.maximum(rf, 1e-6))
    for _ in range(iters):
        e0 = np.exp(-0.5 * ((z - 1) / s0) ** 2) / s0
        e1 = np.exp(-0.5 * ((z - m1(knee)) / s1) ** 2) / s1
        B = np.column_stack([np.maximum(e0, 1e-300), np.maximum(e1, 1e-300)])
        B /= B.sum(axis=1, keepdims=True)
        a_f = np.zeros((len(z), 2)); a_f[0] = pi * B[0]
        for t in range(1, len(z)):
            a_f[t] = (a_f[t - 1] @ A) * B[t]
            ssum = a_f[t].sum()
            a_f[t] = a_f[t] / ssum if ssum > 0 else 1e-300
        b_b = np.ones((len(z), 2))
        for t in range(len(z) - 2, -1, -1):
            b_b[t] = A @ (B[t + 1] * b_b[t + 1])
        gam = a_f * b_b
        gam /= np.maximum(gam.sum(axis=1, keepdims=True), 1e-300)
        xi = np.zeros((2, 2))
        for t in range(len(z) - 1):
            m = a_f[t][:, None] * A * (B[t + 1] * b_b[t + 1])[None, :]
            xi += m / max(m.sum(), 1e-300)
        pi = gam[0] / max(gam[0].sum(), 1e-300)
        Ax = xi / np.maximum(xi.sum(axis=1, keepdims=True), 1e-300)
        A = 0.9 * A + 0.1 * Ax; A /= A.sum(axis=1, keepdims=True)
        w = gam[:, 1]
        s0 = np.sqrt(max(np.sum(gam[:, 0] * (z - 1) ** 2) / max(gam[:, 0].sum(), 1e-9), 1e-6))
        ll = [np.sum(w * (-0.5 * ((z - m1(gi)) / max(s1, 1e-3)) ** 2)) for gi in knees]
        knee = float(knees[int(np.argmax(ll))])
        s1 = np.sqrt(max(np.sum(w * (z - m1(knee)) ** 2) / max(w.sum(), 1e-9), 1e-6))
    p = pd.Series(np.nan, index=df.index)
    p.iloc[np.flatnonzero(fitm)] = gam[:, 1]
    score = p.bfill().ffill().fillna(0)
    raw = (score > 0.5) & (fe["ref"] >= ref_on) & d["act"] & ~dq_mask(fe, a, b)
    return score, persist(raw, pers)


# --------------------------------------------------------------------------- #
# evaluation
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
    for dlt in (0.0, 0.05, 0.10, 0.20, 0.30):
        pos = ok & gt.astype(bool) & (sev >= dlt)
        out[f"n_pos{int(dlt*100):02d}"] = int(pos.sum())
        out[f"auc{int(dlt*100):02d}"] = (roc_auc_score(pos.astype(int)[ok], sc[ok])
                                         if 0 < pos.sum() < ok.sum() else np.nan)
    out["fp_rows"] = int((fl & neg).sum())
    tp = int((fl & ok & gt.astype(bool)).sum()); fn = int((~fl & ok & gt.astype(bool)).sum())
    out["rec"] = tp / max(tp + fn, 1)
    out["prec"] = tp / max(tp + out["fp_rows"], 1)
    out["f1"] = 2 * out["prec"] * out["rec"] / max(out["prec"] + out["rec"], 1e-9)
    return out


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
SCEN = [(("eu_1", "eu_3"), None, "chronic"),
        (("eu_1", "eu_3"), 0.95, "chronic"), (("eu_1", "eu_3"), 0.85, "chronic"),
        (("eu_1", "eu_3"), 0.75, "chronic"), (("eu_1", "eu_3"), 0.65, "chronic"),
        (("eu_1", "eu_3"), 0.95, "episodic"), (("eu_1", "eu_3"), 0.85, "episodic"),
        (("eu_1", "eu_3"), 0.75, "episodic"), (("eu_1", "eu_3"), 0.65, "episodic"),
        (("eu_4", "eu_12"), 0.75, "chronic"), (("eu_4", "eu_12"), 0.75, "episodic"),
        (("eu_15", "eu_23"), 0.75, "chronic"), (("eu_15", "eu_23"), 0.75, "episodic")]

print("=" * 112)
print("CLIP-INJECTION BENCHMARK  (positives = samples where the injected constraint bit, evaluated at ref>=0.70)")
print("=" * 112)

rows = []
for (a, b), q, pattern in SCEN:
    rel = [c for c in RELIABLE if c not in (a, b)]
    base = df_raw.copy()
    fe0 = preprocess(base, rel)
    ref0 = fe0["ref"]
    Ptyp = np.nanpercentile((base[a] + base[b])[ref0 >= 0.85], 95)
    C = (q * Ptyp) if q is not None else np.inf
    # true unsaturated slope, for the per-sample severity
    act0 = fe0["active"][a] & fe0["active"][b]
    th_true = theta_roll(base[a] + base[b], ref0, act0, fe0["dayidx"], q=0.5)
    dava = base[a].groupby(base.index.normalize()).count().pipe(lambda x: x[x > 0]).index
    day_on = pd.Series(np.random.default_rng(7).random(len(dava)) < 0.35, index=dava) \
        .reindex(base.index.normalize()).fillna(False).to_numpy()

    d2, gt = inject(base, a, b, C, day_on if pattern == "episodic" else None)
    fe = preprocess(d2, rel)
    with np.errstate(invalid="ignore", divide="ignore"):
        sev = np.clip(1 - C / (th_true * ref0), 0, 1)
    sev = pd.Series(sev, index=base.index).fillna(0)

    res = {"D0 published": d0_published(d2, fe, a, b),
           "D1 rolling-slope": d1_or_d2(d2, fe, a, b, q=0.5)[:2],
           "D2 envelope-q90": d1_or_d2(d2, fe, a, b, q=0.9)[:2],
           "D3 HMM-2state": d3_hmm(d2, fe, a, b),
           "D4 ceiling-pinned": d4_ceiling_pinned(d2, fe, a, b)[:2],
           "D5 control-calibrated": d5_control_calibrated(d2, fe, a, b, exclude=(a, b))[:2]}
    for name in DETS:
        sc, fl = res[name]
        e = evaluate(sc, fl, fe, a, b, gt, sev)
        e.update(pair=f"{a}+{b}", q=(1.0 if q is None else q), pattern=pattern, det=name)
        rows.append(e)
    act = fe["active"][a] & fe["active"][b]
    cl = int((gt & act).sum()); cl_hi = int((gt & act & (fe["ref"] >= 0.7)).sum())
    print(f"  {a}+{b} q={('-' if q is None else f'{q:.2f}'):>4s} {pattern:8s}: constraint bit on "
          f"{cl:6,} rows, {cl_hi:6,} at ref>=0.7 ({100*cl_hi/max(cl,1):3.0f}%)")

R = pd.DataFrame(rows)
R.to_csv(OUT + r"\bench_results.csv", index=False)
AUC = [f"auc{int(d*100):02d}" for d in (0.0, 0.05, 0.10, 0.20, 0.30)]

print()
print("=" * 112)
print("HEADLINE  (mean over 13 scenarios; auc00 = all clipped rows, auc20 = only clips leaving >20% deficit)")
print("=" * 112)
print(R.groupby("det")[AUC + ["fp_rows", "rec", "prec", "f1"]].mean()
      .sort_values("auc10", ascending=False).round(3).to_string())

print()
print("=" * 112)
print("AUC vs per-sample CLIP SEVERITY   (severity = 1 - C/(theta_true*ref); 0 = the clip barely bites)")
print("=" * 112)
print(R.groupby("det")[AUC].mean().round(3).to_string())
print("\nsame, EPISODIC scenarios only (the realistic case: clips affect only some days):")
print(R[R.pattern == "episodic"].groupby("det")[AUC].mean().round(3).to_string())

print()
print("=" * 112)
print("SPECIFICITY: false-positive rows at ref>=0.7, EPISODIC scenarios (unclipped days are true negatives)")
print("=" * 112)
sub = R[R.pattern == "episodic"]
print(sub.pivot_table(index=["pair", "q"], columns="det", values="fp_rows").round(0).astype("Int64").to_string())
print("\nNO-CLIP control scenario (q='-', chronic): flag counts")
print(R[R["q"] == 1.0].set_index("det")[["fp_rows", "rec"]].astype("Int64").to_string())

# --------------------------------------------------------------------------- #
# apply the detectors to the REAL shared pairs
# --------------------------------------------------------------------------- #
print()
print("=" * 112)
print("APPLIED TO THE REAL SHARED PAIRS  (flagged HOURS at ref>=0.7)")
print("=" * 112)
feR = preprocess(df_raw)
real, ctrl = {}, {}
for a, b in PAIRS:
    r = {"D0 published": d0_published(df_raw, feR, a, b),
         "D1 rolling-slope": d1_or_d2(df_raw, feR, a, b, q=0.5),
         "D2 envelope-q90": d1_or_d2(df_raw, feR, a, b, q=0.9),
         "D3 HMM-2state": d3_hmm(df_raw, feR, a, b),
         "D4 ceiling-pinned": d4_ceiling_pinned(df_raw, feR, a, b),
         "D5 control-calibrated": d5_control_calibrated(df_raw, feR, a, b, exclude=(a, b))}
    real[f"{a}+{b}"] = {d: float(v[1].sum()) / 12 for d, v in r.items()}
for a, b in CONTROL_PAIRS:
    r = {"D0 published": d0_published(df_raw, feR, a, b),
         "D1 rolling-slope": d1_or_d2(df_raw, feR, a, b, q=0.5),
         "D2 envelope-q90": d1_or_d2(df_raw, feR, a, b, q=0.9),
         "D3 HMM-2state": d3_hmm(df_raw, feR, a, b),
         "D4 ceiling-pinned": d4_ceiling_pinned(df_raw, feR, a, b),
         "D5 control-calibrated": d5_control_calibrated(df_raw, feR, a, b, exclude=(a, b))}
    ctrl[f"{a}+{b}"] = {d: int(v[1].sum()) for d, v in r.items()}
print(pd.DataFrame(real).round(1).to_string())
print("\ncontrol-pair (independent) flagged ROWS with the same detectors:")
print(pd.DataFrame(ctrl).to_string())
print("\nseparation factor (shared rows / control rows):")
sd = pd.DataFrame(real).sum(axis=1) * 12
cd = pd.DataFrame(ctrl).sum(axis=1)
print((sd / cd.replace(0, np.nan)).round(0).to_string())
print("\nwrote", OUT + r"\bench_results.csv")
