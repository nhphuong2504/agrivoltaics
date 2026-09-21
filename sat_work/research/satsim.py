"""
Shared data loading + faithful re-implementation of the current saturation detector
(Nguyen, saturation_detection.ipynb, 2026-08-27) so that variants can be benchmarked
against exactly the same preprocessing.

Nothing here is imported from the notebook; every rule is re-coded from the notebook
source so that we can perturb individual pieces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ROOT = r"c:\Users\nhphuong\Desktop\Solar\all_data"
# DATA REFRESH 2026-09-16: superseded combined_may2025_jul2026_area1.csv.
# The Jul-2026 export was missing four entire months (2025-08, 2025-09, 2026-03,
# 2026-04) plus part of 2026-07 and all of 2026-08: 328 of 451 calendar dates
# present. This export covers 2026-08-31 with 487 of 488 dates present, and is a
# strict superset -- every one of the 51,005 old timestamps appears with identical
# values apart from a single rounding artefact on eu_10. See step27_data_refresh_audit.py.
CSV = ROOT + r"\combined_may2025_aug2026_area1.csv"

EU = [f"eu_{i}" for i in range(1, 25)]

# --------------------------------------------------------------------------- #
# FLEET TOPOLOGY - supplied by the site engineer 2026-09, no longer inferred.
#   MPPT channels : eu_8+eu_16, eu_10+eu_18, eu_13+eu_21
#   inverters     : 1: eu_1,eu_9,eu_17        2: eu_2,eu_3,eu_10,eu_11,eu_18,eu_19
#                   3: eu_4,eu_12,eu_20        4: eu_5,eu_6,eu_13,eu_14,eu_21,eu_22
#                   5: eu_7,eu_15,eu_23        6: eu_8,eu_16,eu_24
# The three MPPT pairs match the ones previously inferred from the data (below),
# which validates the shared/independent labelling used throughout the analysis.
# See step17_groundtruth.py for the natural experiment this enables.
# --------------------------------------------------------------------------- #
INVERTERS = {
    1: ["eu_1", "eu_9", "eu_17"],
    2: ["eu_2", "eu_3", "eu_10", "eu_11", "eu_18", "eu_19"],
    3: ["eu_4", "eu_12", "eu_20"],
    4: ["eu_5", "eu_6", "eu_13", "eu_14", "eu_21", "eu_22"],
    5: ["eu_7", "eu_15", "eu_23"],
    6: ["eu_8", "eu_16", "eu_24"],
}
INV_OF = {u: k for k, us in INVERTERS.items() for u in us}
# units on the same MPPT channel share a tracker and CAN saturate together
MPPT_CHANNELS = [("eu_8", "eu_16"), ("eu_10", "eu_18"), ("eu_13", "eu_21")]

PAIRS = [("eu_8", "eu_16"), ("eu_10", "eu_18"), ("eu_13", "eu_21")]
SHARED = [c for p in PAIRS for c in p]

# 11 independent units that are essentially never offline at midday
RELIABLE = ["eu_1", "eu_3", "eu_4", "eu_9", "eu_11", "eu_12",
            "eu_15", "eu_17", "eu_19", "eu_20", "eu_23"]

# independent pairs used as "controls" in the notebook.
# NOTE (step17): eu_4+eu_12 share inverter 3 and eu_15+eu_23 share inverter 5; only
# eu_1+eu_3 crosses inverters. That does NOT invalidate them as magnitude nulls
# (same-inverter independent pairs are flat on median), but same-inverter pairs are
# roughly twice as likely to trip the threshold via a co-movement common mode, so use
# cross-inverter pairs for flag-LEVEL nulls. See step17_groundtruth.py sections C/G.
CONTROL_PAIRS = [("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]
CROSS_INVERTER_CONTROL = ("eu_1", "eu_3")

# THE mid band, defined once. Both the retired baseline (fit_pair/rolling_cap) and the
# vat-v1 slope (bench.theta_roll) estimate their slope here, and the papers' Algorithm 1
# quotes it as [rho_-, rho_+]. It used to be a literal in this module AND in bench.py;
# bench now imports this one, so a single edit moves every consumer and the frozen
# configuration record has one source to read from.
BAND = (0.35, 0.60)


# --------------------------------------------------------------------------- #
# load + preprocess                                                           #
# --------------------------------------------------------------------------- #
def load() -> pd.DataFrame:
    df = pd.read_csv(CSV, parse_dates=["time"]).set_index("time").sort_index()
    return df


def run_lengths(cond: pd.Series) -> pd.Series:
    """Length of the contiguous True run each sample belongs to (0 where False)."""
    grp = cond.ne(cond.shift()).cumsum()
    return cond.groupby(grp).transform("sum")


def preprocess(df: pd.DataFrame, reliable=None) -> dict:
    """Everything the notebook cell 6 does, returned as a dict of aligned objects.

    `reliable` lets the caller hold units out of the irradiance proxy (used by the
    injection benchmark so an injected pair never helps define its own reference).
    """
    RELIABLE_ = list(reliable) if reliable is not None else RELIABLE
    norm_rel = df[RELIABLE_].div(df[RELIABLE_].quantile(0.999))
    ref = norm_rel.median(axis=1)

    midday = (df.index.hour >= 9) & (df.index.hour <= 15)
    site_outage = ((df[RELIABLE_] < 50).mean(axis=1) > 0.8) & midday

    ref_varies = ref.diff().abs().rolling(6).sum() > 0.02
    unit_off = pd.DataFrame(False, index=df.index, columns=EU)
    stale = pd.DataFrame(False, index=df.index, columns=EU)
    for c in EU:
        off = (df[c] <= 1) & (ref > 0.4)
        unit_off[c] = off & (run_lengths(off) >= 12)
        same = df[c].eq(df[c].shift()) & df[c].gt(0)
        stale[c] = same & (run_lengths(same) >= 12) & ref_varies
    negative = df[EU] < 0

    daily = df[EU].resample("1D").sum()
    dailyN = daily.div(daily.quantile(0.95).replace(0, np.nan))
    dayidx = df.index.normalize()
    active = dailyN.loc[dayidx, EU].reset_index(drop=True).set_index(df.index) > 0.2

    return dict(ref=ref, site_outage=site_outage, unit_off=unit_off, stale=stale,
                negative=negative, active=active, dayidx=dayidx, dailyN=dailyN)


# --------------------------------------------------------------------------- #
# the ceiling + expected-output model (notebook cell 12)                      #
# --------------------------------------------------------------------------- #
def rolling_cap(s: pd.Series, act: pd.Series, ref: pd.Series, dayidx,
                window=31, min_periods=5, band=BAND):
    """Per-day p99 of the pair sum over active+daylight rows, then centred rolling median."""
    ok = act & (ref > 0.3)
    per_day = s[ok].groupby(dayidx[ok]).quantile(0.99)
    return per_day.rolling(window, center=True, min_periods=min_periods).median()


def fit_pair(df, s, act, ref, dayidx, band=BAND, **cap_kw):
    """Return cap_t (per-row), g0, gain, deficit for one pair."""
    cap_day = rolling_cap(s, act, ref, dayidx, **cap_kw)
    cap_t = cap_day.reindex(dayidx).to_numpy()
    gain = s / (cap_t * ref.replace(0, np.nan))
    mid = act & (ref >= band[0]) & (ref <= band[1])
    g0 = gain[mid].median()
    expected = g0 * cap_t * ref
    deficit = 1 - s / expected
    return dict(cap_day=cap_day, cap_t=cap_t, g0=g0, gain=gain,
                expected=expected, deficit=deficit)


# --------------------------------------------------------------------------- #
# the detector (notebook cell 15)                                             #
# --------------------------------------------------------------------------- #
P_DEFAULT = dict(REF_ON=0.70, DEF_MOD=0.15, DEF_SEV=0.30,
                 NEAR_CAP=0.60, PERSIST_MOD=3, PERSIST_SEV=6)


def dq_mask(fe, a, b):
    return (fe["unit_off"][a] | fe["unit_off"][b] | fe["stale"][a] | fe["stale"][b]
            | fe["negative"][a] | fe["negative"][b] | fe["site_outage"])


def detect(df, fe, a, b, p=None):
    """Exactly the notebook's `detect`, but usable for any pair (shared or control)."""
    p = p or P_DEFAULT
    s = df[a] + df[b]
    act = fe["active"][a] & fe["active"][b]
    fit = fit_pair(df, s, act, fe["ref"], fe["dayidx"])
    base = ((fe["ref"] >= p["REF_ON"]) & (s > p["NEAR_CAP"] * fit["cap_t"])
            & act & ~dq_mask(fe, a, b))
    out = {}
    for tier, thr, pers in [("moderate", p["DEF_MOD"], p["PERSIST_MOD"]),
                            ("severe", p["DEF_SEV"], p["PERSIST_SEV"])]:
        raw = base & (fit["deficit"] > thr)
        out[tier] = raw & (run_lengths(raw) >= pers)
    return dict(s=s, act=act, **fit, **out, base=base)
