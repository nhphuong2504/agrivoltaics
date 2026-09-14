"""
FROZEN REGRESSION HARNESS for the MPPT saturation detectors.

Extracted from the research phase (step3 / step4 / step6 / recommended.py) and frozen
so that future detector changes can be scored against ground truth before publication.

Why this exists
---------------
The published validation compares shared pairs against independent control pairs. That
test cannot catch the failure mode that actually matters: the control pairs are never
clipped, so a detector whose baseline is mis-calibrated on *intermittently clipped* pairs
passes cleanly. This harness manufactures ground truth instead -- it replaces an
independent pair's aggregate with min(u+v, C), which is exactly what a shared MPPT with
limit C does to an observable pair sum, and records which samples were actually
constrained. Unmodified samples give labelled negatives.

    severity(t) = 1 - C / (theta_true(t) * ref(t))     0 = the limit barely binds

so "detect any binding constraint" is not confused with "detect meaningful saturation".

Nothing here is a research conclusion; it is the fixture those conclusions were measured
on, kept stable so regressions are detectable. Research scripts in the parent directory
may drift, this file should not.

Import as a library (`from bench import ...`) or run directly to print a scoreboard:
    ./venv/Scripts/python.exe sat_work/research/bench.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from satsim import (CONTROL_PAIRS, EU, PAIRS, RELIABLE, detect, dq_mask,  # noqa: E402
                    load, preprocess, run_lengths)

BINS = np.linspace(0.30, 1.05, 21)
BAND = (0.35, 0.60)
REF_ON = 0.70
NEAR_CAP = 0.60
THR_MOD = 0.15
THR_SEV = 0.30
PERS_MOD = 3
PERS_SEV = 6
# ADOPTED 2026-09 (SATURATION_REVIEW section 3.13): bridge 1-sample gaps before the
# run-length rule for the 15-minute (k=3) tiers. A strict rule discards genuine signal --
# +6.0 pp recall on injected ground truth for +83 FP rows, and on real data the whole
# marginal cost lands on eu_15+eu_23, an already-contaminated same-inverter null, while
# the clean cross-inverter control pays nothing. This is a real precision/recall
# trade-off, not a free gain: guarding the bridged samples was tested and does not
# dominate (step18).
#
# NOT applied to the SEVERE tier (k=6): the benchmark's injected severity caps at about
# 0.25, so it produces ZERO true positives for a >0.30 tier and cannot adjudicate the
# rule there. Bridging would inflate the severe tier ~3x (7.1 -> 21.0 h on eu_8+eu_16)
# on the strength of an untested analogy, in the tier most exposed to baseline bias.
# Severe therefore stays strict until ground truth exists for it.
PERS_GAP = 1
PERS_GAP_SEV = 0

CONTROLS = list(CONTROL_PAIRS)

# --------------------------------------------------------------------------- #
# lazy singletons
# --------------------------------------------------------------------------- #
_STATE: dict = {}


def raw_df() -> pd.DataFrame:
    if "df" not in _STATE:
        df = load()
        _STATE["df"] = df
        _STATE["fullcal"] = pd.date_range(df.index.min().normalize(),
                                          df.index.max().normalize(), freq="D")
        _STATE["fe"] = preprocess(df)
    return _STATE["df"]


def fullcal() -> pd.DatetimeIndex:
    raw_df()
    return _STATE["fullcal"]


def raw_fe() -> dict:
    raw_df()
    return _STATE["fe"]


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def inject(df: pd.DataFrame, a: str, b: str, C, day_on=None):
    """Clamp the (a+b) pair sum at C. Returns the modified frame and the ground truth."""
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
    """Per-day series -> full calendar -> time-based centred rolling median -> per row.

    Reindexing to a complete daily calendar first matters: the acquisition record has
    gaps, so a `.rolling(31)` on the raw per-day index spans far more than 31 days.
    """
    roll = per_day.reindex(fullcal()).rolling(window, center=True, min_periods=minp).median()
    return pd.Series(roll.reindex(dayidx).to_numpy(), index=labels)


def theta_roll(s, ref, act, dayidx, q=0.5, band=BAND):
    """Clear-sky anchor slope: rolling median of the daily (median | q-th pct) of s/ref,
    estimated only inside `band` so it is never contaminated by the unsampled high tail."""
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= band[0]) & (ref <= band[1]) & u.notna()
    g = u[sel].groupby(dayidx[sel])
    per_day = g.median() if q == 0.5 else g.quantile(q)
    return broadcast(per_day, dayidx, s.index)


def persist(raw: pd.Series, k: int, index=None, max_gap_min: float = 6.0) -> pd.Series:
    """Keep only samples inside a run of at least `k` consecutive True values.

    With `index` supplied, a run is also broken at any acquisition gap longer than
    `max_gap_min` minutes. Without that, a run can straddle a multi-month hole in the
    record and manufacture an episode that never happened -- the export test
    `test_flags_are_not_set_across_calendar_gaps` exists because it did (9 such runs on
    the real data before this was fixed).
    """
    if index is None:
        return raw & (run_lengths(raw) >= k)
    step = pd.Series(index, index=raw.index).diff().dt.total_seconds().div(60)
    seg = ((step > max_gap_min) | step.isna()).cumsum().to_numpy()
    grp = raw.ne(raw.shift()).cumsum().to_numpy()
    key = pd.Series(seg * (int(grp.max()) + 2) + grp, index=raw.index)
    return raw & (raw.groupby(key).transform("sum") >= k)


def bridge(raw: pd.Series, g: int, within: pd.Series | None = None) -> pd.Series:
    """Fill False gaps of length <= g that sit between two True samples.

    Lets a run survive a single dropout. Measured on this harness (step14): +19 % recall
    for +138 false-positive rows, precision 0.987 -> 0.983. Opt-in, not default.

    `within` restricts the fill to samples that satisfy the detector's gate (ref, activity,
    DQ). Without it, a fill can land on a sample BELOW the ref gate -- the sample dropped
    out of the run merely because a cloud edge pushed ref under 0.70 -- which breaks the
    export contract that no flag is ever set below the gate. Pass `within=gate`.

    step18 tested whether *guarding* the fills (requiring the filled sample's own deficit
    to clear a floor) could buy the recall without the false positives. It cannot: every
    guard setting lies on the Pareto frontier, trading recall for FP roughly smoothly, so
    there is no dominating rule. This is a judgement about which error costs more -- see
    SATURATION_REVIEW.md section 3.13.
    """
    out = raw.copy()
    for k in range(1, g + 1):
        fill = raw.shift(k, fill_value=False) & raw.shift(-k, fill_value=False)
        if within is not None:
            fill = fill & within
        out = out | fill
    return out


def pair_deficit(df, fe, a, b, q=0.5):
    s = df[a] + df[b]
    act = fe["active"][a] & fe["active"][b]
    th = theta_roll(s, fe["ref"], act, fe["dayidx"], q=q)
    exp_ = th * fe["ref"]
    deficit = (1 - s / exp_).replace([np.inf, -np.inf], np.nan)
    return dict(s=s, act=act, theta=th, expected=exp_, deficit=deficit)


def null_curve(controls=CONTROLS, q=0.5, alpha=0.995, exclude=(), fe=None, df=None):
    """Per-ref-bin `alpha` quantile of the deficit shown by the INDEPENDENT pairs.

    An empirical, assumption-light null: it inherits whatever bias the baseline has, so
    it trades sensitivity for specificity rather than fixing the baseline.
    """
    df = df if df is not None else raw_df()
    fe = fe if fe is not None else raw_fe()
    ref = fe["ref"]
    pool = []
    for c1, c2 in controls:
        if c1 in exclude or c2 in exclude:
            continue
        dc = pair_deficit(df, fe, c1, c2, q=q)
        ok = dc["act"] & (ref >= 0.3) & ~dq_mask(fe, c1, c2)
        pool.append(pd.DataFrame({"ref": ref[ok], "def": dc["deficit"][ok]}))
    pool = pd.concat(pool).dropna()
    pool["bin"] = np.digitize(pool["ref"], BINS).clip(1, len(BINS) - 1)
    return pool.groupby("bin")["def"].quantile(alpha)


# --------------------------------------------------------------------------- #
# detector registry
# --------------------------------------------------------------------------- #
# Each detector returns (score, flag). `score` is a continuous severity used for the
# threshold-free AUC; `flag` is the published-style 0/1 decision at the standard rules.

def det_published(df, fe, a, b, **_):
    """D0 - the method as published: expected = g0 * cap(t) * ref.

    Reproduces saturation_detection.ipynb. Note the score legitimately contains -inf
    where ref == 0, which is a property of the published formulation (see test_exports).
    """
    r = detect(df, fe, a, b)
    return r["deficit"], r["moderate"]


def det_rolling(df, fe, a, b, q=0.5, k=PERS_MOD, gap=PERS_GAP, **_):
    """D1 - expected = theta(t) * ref, theta the rolling median of the daily s/ref.

    This IS the recommended detector's primary tier (vat-v1 `moderate`); the published
    rules are otherwise unchanged, which is what makes the fix one function.
    """
    d = pair_deficit(df, fe, a, b, q=q)
    base = ((fe["ref"] >= REF_ON) & (d["s"] > NEAR_CAP * d["expected"])
            & d["act"] & ~dq_mask(fe, a, b))
    raw = base & (d["deficit"] > THR_MOD)
    if gap:
        raw = bridge(raw, gap, within=base)
    return d["deficit"], persist(raw, k, index=df.index)


def det_severe(df, fe, a, b, q=0.5, k=PERS_SEV, gap=PERS_GAP_SEV, **_):
    """The severe tier of the recommended detector: d > 0.30 for 30 minutes."""
    d = pair_deficit(df, fe, a, b, q=q)
    base = ((fe["ref"] >= REF_ON) & (d["s"] > NEAR_CAP * d["expected"])
            & d["act"] & ~dq_mask(fe, a, b))
    raw = base & (d["deficit"] > THR_SEV)
    if gap:
        raw = bridge(raw, gap, within=base)
    return d["deficit"], persist(raw, k, index=df.index)


def det_envelope(df, fe, a, b, q=0.9, k=PERS_MOD, gap=PERS_GAP, **_):
    """D2 - as D1 but theta is the daily 90th pct, i.e. an estimate of potential."""
    return det_rolling(df, fe, a, b, q=q, k=k, gap=gap)


def det_ceiling_pinned(df, fe, a, b, ref_on=REF_ON, k=PERS_MOD, nsigma=2.0, **_):
    """D4 - is s sitting on its own plateau while theta*ref exceeds it?

    Near-perfect for a STABLE limit, unusable for an intermittent one: a plateau cannot
    be established from days that are not clipped.
    """
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
    return score, persist(raw, k, index=df.index)


def det_control_calibrated(df, fe, a, b, q=0.5, alpha=0.995, k=PERS_MOD, **_):
    """D5 - deficit minus the empirical control-pair null in the same ref bin."""
    d = pair_deficit(df, fe, a, b, q=q)
    null = null_curve(q=q, alpha=alpha, exclude=(a, b), fe=fe, df=df)
    binidx = np.digitize(fe["ref"], BINS).clip(1, len(BINS) - 1)
    qn = pd.Series(binidx, index=df.index).map(null).astype(float)
    excess = d["deficit"] - qn
    base = ((fe["ref"] >= REF_ON) & (d["s"] > NEAR_CAP * d["expected"])
            & d["act"] & ~dq_mask(fe, a, b))
    return excess, persist(base & (excess > 0), k, index=df.index)


DETECTORS = {
    "D0 published": det_published,
    "D1 rolling-slope": det_rolling,
    "D2 envelope-q90": det_envelope,
    "D4 ceiling-pinned": det_ceiling_pinned,
    "D5 control-calibrated": det_control_calibrated,
    # D3 (ref-gated 2-state HMM) is deliberately absent: it scored AUC ~0.48 on this
    # harness in step3 -- a cloud dip and a clipped sample are both "low utilisation",
    # so a 2-state Gaussian emission cannot separate them. Kept in step3_benchmark.py.
}


# --------------------------------------------------------------------------- #
# scenarios
# --------------------------------------------------------------------------- #
def scenario(pair, duty=0.35, q=0.75, pattern="chronic", seed=7):
    """Build one labelled scenario. Cached -- the preprocess step is not cheap.

    `duty`  fraction of days the constraint is active (1.0 = every day)
    `q`     clamp level as a fraction of the pair's typical clear-sky peak (None = no clip)
    """
    key = ("scenario", tuple(pair), duty, q, pattern, seed)
    if key in _STATE:
        return _STATE[key]

    a, b = pair
    base = raw_df()
    # the injected pair must be held OUT of the irradiance proxy, or it would help
    # define the very reference used to judge it
    rel = [c for c in RELIABLE if c not in (a, b)]
    fe0 = preprocess(base, rel)
    ref0 = fe0["ref"]
    act0 = fe0["active"][a] & fe0["active"][b]
    Ptyp = np.nanpercentile((base[a] + base[b])[ref0 >= 0.85], 95)
    C = (q * Ptyp) if q is not None else np.inf

    th_true = theta_roll(base[a] + base[b], ref0, act0, fe0["dayidx"], q=0.5)
    days = base[a].groupby(base.index.normalize()).count().pipe(lambda x: x[x > 0]).index
    day_on = (pd.Series(np.random.default_rng(seed).random(len(days)) < duty, index=days)
              .reindex(base.index.normalize()).fillna(False).to_numpy())

    d2, gt = inject(base, a, b, C, day_on if pattern == "episodic" else None)
    fe = preprocess(d2, rel)
    with np.errstate(invalid="ignore", divide="ignore"):
        sev = np.clip(1 - C / (th_true * ref0), 0, 1)
    sev = pd.Series(sev, index=base.index).fillna(0)

    out = dict(pair=pair, a=a, b=b, df=d2, fe=fe, gt=gt, sev=sev, C=C,
               duty=duty, q=q, pattern=pattern, fe0=fe0, ref0=ref0)
    _STATE[key] = out
    return out


def score_scenario(sc: dict, detector) -> dict:
    score, flag = detector(sc["df"], sc["fe"], sc["a"], sc["b"])
    return evaluate(score, flag, sc["fe"], sc["a"], sc["b"], sc["gt"], sc["sev"])


def evaluate(score, flag, fe, a, b, gt, sev, ref_on=REF_ON) -> dict:
    """Threshold-free AUC (stratified by clip severity) plus flag-level precision/recall.

    `ok` is the *evaluation domain*: both units active, ref >= gate, no DQ flag. AUC is
    computed on that domain only, and -inf / NaN scores are excluded, so a detector is
    never penalised or rewarded for rows it cannot see.
    """
    act = fe["active"][a] & fe["active"][b]
    test = (act & (fe["ref"] >= ref_on) & ~dq_mask(fe, a, b)).to_numpy()
    sc = pd.Series(np.asarray(score, dtype=float), index=fe["ref"].index).to_numpy()
    ok = test & np.isfinite(sc)
    gt = gt.to_numpy()
    sev = np.asarray(sev, dtype=float)
    fl = np.asarray(flag, dtype=bool) & ok

    out = dict(n_eval=int(ok.sum()), n_neg=int((ok & ~gt.astype(bool)).sum()))
    for dlt in (0.0, 0.10, 0.20):
        pos = ok & gt.astype(bool) & (sev >= dlt)
        out[f"n_pos{int(dlt * 100):02d}"] = int(pos.sum())
        out[f"auc{int(dlt * 100):02d}"] = (
            roc_auc_score(pos.astype(int)[ok], sc[ok]) if 0 < pos.sum() < ok.sum() else np.nan)
    out["fp_rows"] = int((fl & ok & ~gt.astype(bool)).sum())
    tp = int((fl & ok & gt.astype(bool)).sum())
    fn = int((~fl & ok & gt.astype(bool)).sum())
    out["tp_rows"] = tp
    out["rec"] = tp / max(tp + fn, 1)
    out["prec"] = tp / max(tp + out["fp_rows"], 1)
    return out


# --------------------------------------------------------------------------- #
# the recommended export (vat-v1), as a pure function
# --------------------------------------------------------------------------- #
def build_export(pairs=PAIRS, controls=CONTROLS, mask_below_gate=True):
    """vat-v1 flags for every pair, in the published schema (plus extra columns).

    Contract guaranteed here and asserted in test_exports.py:
      * `time` plus every column of the published saturation_flags.csv is present, so
        existing consumers keep working (the export is a superset)
      * `deficit` never contains +/-inf
      * `null_d` and `excess_vs_controls` are NaN outside the gate domain, because the
        empirical null is only meaningful where the detector would actually decide
      * `*_sat_*` flags are 0/1 integers and are never 1 where `deficit` is NaN

    Note on naming: `*_sat_conservative` is the null-calibrated tier. It is NOT more
    restrictive than `*_sat_moderate` in practice (it flags more) because the control-pair
    null often sits below 0.15. The name is a misnomer kept for continuity; treat it as an
    alternative calibration, not a stricter one.

    Persistence: all three tiers use the adopted gap-tolerant rule (PERS_GAP), i.e.
    1-sample acquisition gaps are bridged before the run-length test. See
    SATURATION_REVIEW.md section 3.13 for the measured trade-off.
    """
    df = raw_df()
    fe = raw_fe()
    ref = fe["ref"]
    null = null_curve(controls=controls)
    binidx = np.digitize(ref, BINS).clip(1, len(BINS) - 1)

    out = pd.DataFrame({"time": df.index,
                        "ref": ref.round(4),
                        "dq_site_outage": fe["site_outage"].astype(int)})
    for a, b in pairs:
        k = f"{a}_{b}"
        d = pair_deficit(df, fe, a, b)
        gate = ((ref >= REF_ON) & (d["s"] > NEAR_CAP * d["expected"])
                & d["act"] & ~dq_mask(fe, a, b))
        qn = pd.Series(binidx, index=df.index).map(null).astype(float)
        excess = d["deficit"] - qn
        domain = d["act"] & (ref >= REF_ON)
        if mask_below_gate:
            qn = qn.where(domain)
            excess = excess.where(domain)

        deficit = d["deficit"].replace([np.inf, -np.inf], np.nan)
        moderate = persist(bridge(gate & (d["deficit"] > THR_MOD), PERS_GAP, within=gate),
                           PERS_MOD, index=df.index)
        severe = persist(bridge(gate & (d["deficit"] > THR_SEV), PERS_GAP_SEV, within=gate),
                         PERS_SEV, index=df.index)
        conserv = persist(bridge(gate & (excess > 0), PERS_GAP, within=gate),
                          PERS_MOD, index=df.index)

        out[f"{k}_deficit"] = deficit.round(4)
        out[f"{k}_excess_vs_controls"] = excess.round(4)
        out[f"{k}_null_d"] = qn.round(4)
        out[f"{k}_sat_moderate"] = (moderate & deficit.notna()).astype(int)
        out[f"{k}_sat_severe"] = (severe & deficit.notna()).astype(int)
        out[f"{k}_sat_conservative"] = (conserv & deficit.notna()).astype(int)
    # match the published file's layout: a plain RangeIndex with `time` as a column
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# direct run: print the scoreboard
# --------------------------------------------------------------------------- #
DEFAULT_SCENARIOS = [
    (("eu_1", "eu_3"), 0.35, 0.75, "chronic"),
    (("eu_1", "eu_3"), 0.35, 0.75, "episodic"),
    (("eu_1", "eu_3"), 1.00, 0.75, "chronic"),
    (("eu_1", "eu_3"), 1.00, None, "chronic"),
    (("eu_4", "eu_12"), 0.35, 0.75, "chronic"),
    (("eu_15", "eu_23"), 0.35, 0.75, "episodic"),
]


def run(scenarios=None, detectors=None) -> pd.DataFrame:
    scenarios = scenarios or DEFAULT_SCENARIOS
    detectors = detectors or DETECTORS
    rows = []
    for pair, duty, q, pat in scenarios:
        sc = scenario(pair, duty, q, pat)
        gt_hi = int((sc["gt"] & sc["fe"]["active"][sc["a"]] & sc["fe"]["active"][sc["b"]]
                     & (sc["fe"]["ref"] >= REF_ON)).sum())
        for name, fn in detectors.items():
            e = score_scenario(sc, fn)
            e.update(pair=f"{pair[0]}+{pair[1]}", duty=duty,
                     q=(1.0 if q is None else q), pattern=pat, det=name)
            rows.append(e)
        print(f"  {pair[0]}+{pair[1]} duty={duty:.2f} q={'-' if q is None else f'{q:.2f}'}"
              f" {pat:8s}: {gt_hi:,} positive rows in the evaluation domain")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=" * 108)
    print("FROZEN BENCHMARK - detector scoreboard")
    print("=" * 108)
    R = run()
    cols = ["auc00", "auc10", "auc20", "fp_rows", "tp_rows", "rec", "prec"]
    print()
    print(R.groupby("det")[cols].mean().sort_values("auc00", ascending=False)
          .round(3).to_string())
    print()
    print("episodic scenarios only (the realistic case -- the limit binds on some days only):")
    print(R[R.pattern == "episodic"].groupby("det")[["auc00", "auc10", "fp_rows"]]
          .mean().round(3).to_string())
    print()
    print("false-positive rows by scenario:")
    print(R.pivot_table(index=["pair", "duty", "q", "pattern"], columns="det",
                        values="fp_rows").astype("Int64").to_string())
