"""
canon_metrics - THE single source of truth for the numbers the canonical artefact locks.

Why this module exists
----------------------
The review, the canvas, the deck and the manifest all quote the same headline figures. If
each recomputes them, they drift. So every one of them imports from here:

    step24_canonical_manifest.py   writes sat_work/canonical/CANONICAL_vat-v1.json
    step21_canonical_summary.py    prints the human-readable summary
    step23_metric_audit.py         audits definitions/denominators against this
    tests/test_manifest.py         fails if the live artefact stops matching the manifest

Definitions are stated once, in the docstrings, and are not restated elsewhere. In
particular every rate shares ONE denominator: `domain`, the same expression `evaluate()`
computes AUC over.

Nothing here chooses a threshold, a persistence rule or a topology. It only measures the
frozen vat-v1 artefact.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
CANONICAL_CSV = os.path.join(ROOT, "saturation_flags.csv")

import bench as B  # noqa: E402  (path is set up by the caller)

SAMPLES_PER_HOUR = 12.0        # verified modal 5-min grid; see step23 section 0
GATE_DEF = ("ref >= 0.70 AND s > 0.6*expected AND both active AND no DQ "
            "(the detector gate)")
DOMAIN_DEF = "both units active AND ref >= 0.70 AND no DQ (the evaluate() test set)"


# --------------------------------------------------------------------------- #
# the one domain definition
# --------------------------------------------------------------------------- #
def domain_mask(fe, a, b):
    """The decision domain as a boolean Series aligned to the source index.

    Byte-for-byte the `test` set `bench.evaluate` computes AUC over, and the set the
    canonical deficit column is masked to. One definition, one denominator.
    """
    return (fe["active"][a] & fe["active"][b] & (fe["ref"] >= B.REF_ON)
            & ~B.dq_mask(fe, a, b))


def load_canonical(path: str | None = None) -> pd.DataFrame:
    return pd.read_csv(path or CANONICAL_CSV, parse_dates=["time"])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def flagged_counts(can: pd.DataFrame | None = None) -> dict:
    """Flagged rows and sampled hours, published vs canonical.

    Denominator for the *hours* conversion is `SAMPLES_PER_HOUR`; acquisition gaps
    contribute no rows, so these are sampled hours present in the record, not wall-clock
    elapsed time.

    Pooled totals are summed from EXACT row counts. Summing per-pair hours that were
    already rounded to 1 dp drifts (78.7 vs the true 78.8 severe hours).
    """
    can = load_canonical() if can is None else can
    df, fe = B.raw_df(), B.raw_fe()
    per = {}
    for a, b in B.PAIRS:
        k = f"{a}_{b}"
        pub = B.detect(df, fe, a, b)
        per[k] = dict(
            published_rows=int(pub["moderate"].sum()),
            canonical_rows=int(can[f"{k}_sat_moderate"].astype(bool).sum()),
            published_severe_rows=int(pub["severe"].sum()),
            canonical_severe_rows=int(can[f"{k}_sat_severe"].astype(bool).sum()),
        )
    tot = {f"{n}_{t}": sum(v[f"{n}_{t}"] for v in per.values())
           for n in ("published", "canonical") for t in ("rows", "severe_rows")}
    out = dict(per_pair=per)
    for n in ("published", "canonical"):
        out[f"{n}_rows"] = tot[f"{n}_rows"]
        out[f"{n}_hours"] = round(tot[f"{n}_rows"] / SAMPLES_PER_HOUR, 1)
        out[f"{n}_severe_rows"] = tot[f"{n}_severe_rows"]
        out[f"{n}_severe_hours"] = round(tot[f"{n}_severe_rows"] / SAMPLES_PER_HOUR, 1)
    out["row_change_pct"] = round(100 * (out["canonical_rows"] / out["published_rows"] - 1), 1)
    out["severe_inflation_x"] = round(out["published_severe_rows"]
                                      / out["canonical_severe_rows"], 2)
    return out


def control_flag_counts(can: pd.DataFrame | None = None) -> dict:
    """Flags on the three legacy control pairs.

    These pairs have independent MPPT channels, so every flag is a false alarm. The
    cross-inverter pair (eu_1+eu_3) is the only honest flag-level null: same-inverter
    controls carry a co-movement common mode worth roughly 2x tail flags.
    """
    can = load_canonical() if can is None else can
    df, fe = B.raw_df(), B.raw_fe()
    out = {}
    for a, b in B.CONTROL_PAIRS:
        out[f"{a}_{b}"] = dict(
            published=int(B.detect(df, fe, a, b)["moderate"].sum()),
            canonical=int(B.det_rolling(df, fe, a, b)[1].sum()),
        )
    return out


def _loss(a, b, mode, fe=None) -> float:
    """Apparent lost energy in kWh over the GATE domain, for one pair.

    `sum(max(expected - measured, 0)) * 5/60 / 1000` over the gate domain. `mode='pub'`
    rebuilds the published expectation from `B.detect`; `mode='vat'` uses the vat-v1
    baseline. The earlier label "at ref >= 0.7" understated this: the domain also requires
    `s > 0.6*expected`, both units active, and no DQ flag.
    """
    df = B.raw_df()
    fe = B.raw_fe() if fe is None else fe
    ref = fe["ref"]
    d = B.pair_deficit(df, fe, a, b)
    gate = ((ref >= B.REF_ON) & (d["s"] > B.NEAR_CAP * d["expected"])
            & d["act"] & ~B.dq_mask(fe, a, b)).to_numpy()
    expected = d["expected"] if mode == "vat" else \
        pd.Series(B.detect(df, fe, a, b)["expected"], index=df.index)
    return float(np.maximum(expected - d["s"], 0).to_numpy()[gate].sum()) * 5 / 60 / 1000


def energy() -> dict:
    """Shared-pair apparent loss, control-pair bias floor, and the floor as a share."""
    fe = B.raw_fe()
    sh_p = sum(_loss(a, b, "pub", fe) for a, b in B.PAIRS)
    sh_v = sum(_loss(a, b, "vat", fe) for a, b in B.PAIRS)
    ct_p = sum(_loss(a, b, "pub", fe) for a, b in B.CONTROL_PAIRS)
    ct_v = sum(_loss(a, b, "vat", fe) for a, b in B.CONTROL_PAIRS)
    return dict(
        shared_published_kwh=round(sh_p), shared_canonical_kwh=round(sh_v),
        control_published_kwh=round(ct_p), control_canonical_kwh=round(ct_v),
        bias_floor_published_pct=round(100 * ct_p / sh_p),
        bias_floor_canonical_pct=round(100 * ct_v / sh_v),
    )


def domain_sizes() -> dict:
    """Rows in the decision domain, per pair and as a share of the record."""
    fe = B.raw_fe()
    n = len(fe["ref"])
    return {f"{a}_{b}": int(domain_mask(fe, a, b).sum()) for a, b in B.PAIRS}, n


def deficit_ranges(can: pd.DataFrame | None = None) -> dict:
    """In-domain min/max per pair, and how many values leak outside the domain.

    The leak must be 0: that is the downstream-safety guarantee. The in-domain range must
    stay inside [-1, 1], the physical range of `1 - s/(theta*ref)`.
    """
    can = load_canonical() if can is None else can
    fe = B.raw_fe()
    out, leaks = {}, 0
    for a, b in B.PAIRS:
        dom = domain_mask(fe, a, b).to_numpy()
        full = can[f"{a}_{b}_deficit"].to_numpy()
        inside = full[dom]
        inside = inside[~np.isnan(inside)]
        leaks += int((~np.isnan(full[~dom])).sum())
        out[f"{a}_{b}"] = [round(float(inside.min()), 4), round(float(inside.max()), 4)]
    return out, leaks


def benchmark() -> dict:
    """Ground-truth benchmark: specificity and discrimination, episodic regime.

    Episodic = 35 % of days clipped at q=0.75, the realistic regime. FPs are pooled over
    the two scenarios; per-scenario ratios span 48x-820x, so the pooled figure must not be
    quoted alone.
    """
    EPI = (("eu_1", "eu_3"), ("eu_15", "eu_23"))
    pub = can = 0
    auc_pub = auc_can = 0.0
    for p in EPI:
        s = B.scenario(p, duty=0.35, q=0.75, pattern="episodic")
        a = B.score_scenario(s, B.det_published)
        b = B.score_scenario(s, B.det_rolling)
        pub += a["fp_rows"]
        can += b["fp_rows"]
        auc_pub += a["auc00"] / len(EPI)
        auc_can += b["auc00"] / len(EPI)
    return dict(episodic_fp_published=int(pub), episodic_fp_canonical=int(can),
                specificity_gain_x=round(pub / max(can, 1)),
                episodic_auc_published=round(auc_pub, 3),
                episodic_auc_canonical=round(auc_can, 3))


def method() -> dict:
    """The frozen vat-v1 configuration. Read from code, never re-typed."""
    return dict(
        version="vat-v1",
        baseline="rolling median of s/ref over a 31-day centred window",
        expectation="theta(t) * ref, theta in W",
        definition="deficit = 1 - s / (theta(t) * ref)",
        ref_on=B.REF_ON, near_cap=B.NEAR_CAP,
        deficit_threshold_moderate=B.THR_MOD, deficit_threshold_severe=B.THR_SEV,
        persistence_moderate_samples=B.PERS_MOD,
        persistence_severe_samples=B.PERS_SEV,
        persistence_bridging_samples=B.PERS_GAP,
        persistence_bridging_samples_severe=B.PERS_GAP_SEV,
        null_bins=[round(float(B.BINS[0]), 2), round(float(B.BINS[-1]), 2)],
        theta_calibration_band=[0.35, 0.60],
        null_calibration="empirical control-pair quantile by ref bin",
        domain=DOMAIN_DEF,
        energy_gate=GATE_DEF,
        excluded_units=["eu_7"],
        mppt_channels=sorted(f"{a}+{b}" for a, b in B.PAIRS),
    )


def summarise(can: pd.DataFrame | None = None) -> dict:
    """Everything the manifest locks, in one deterministic dict."""
    can = load_canonical() if can is None else can
    dsz, n = domain_sizes()
    rng, leaks = deficit_ranges(can)
    return dict(
        flagged=flagged_counts(can),
        controls=control_flag_counts(can),
        energy=energy(),
        domain_rows=dsz,
        record_rows=n,
        deficit_in_domain_range=rng,
        deficit_values_outside_domain=leaks,
        benchmark=benchmark(),
        method=method(),
    )


def sha256(path: str | None = None) -> str:
    import hashlib
    with open(path or CANONICAL_CSV, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()
