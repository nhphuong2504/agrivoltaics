"""
Contract tests for the saturation FLAGS DATA PRODUCT.

These are not about detector accuracy. They guard the interface that downstream work
consumes: the canonical `saturation_flags.csv` ships a continuous `deficit` column and
per-pair flag columns, and a consumer will happily .mean(), .sum() or regress on them.

The canonical artefact has been the vat-v1 export since 2026-09 (review section 5). Two
defects are recorded here:

  * the ORIGINAL published artefact contained 5,373 `-inf` values in the deficit columns,
    from `1 - s/(g0*cap*ref)` when ref == 0 and s > 0 (dawn/dusk). Flags are 0 there, so
    the defect was invisible to any flag-level check but poisoned any aggregate. Fixed by
    the vat-v1 regeneration (review section 3.9).

  * even under vat-v1 the deficit column carries a large NEGATIVE tail below the gate,
    where `theta(t)*ref` approaches zero, reaching about -260. It is confined to
    ref < 0.3 and can never influence a flag, but it is the reason the physical-bound
    test below is scoped to the decision domain rather than the whole column.

Refactoring note: the export is built by `bench.build_export()`, a pure function, so most
tests here never depend on a file on disk. The ones that DO assert on the shipped artefact
say so explicitly.
"""
from __future__ import annotations

import os.path

import numpy as np
import pandas as pd

from _helpers import CANONICAL_CSV, PUBLISHED_V1_COLUMNS, bench

PAIR_KEYS = [f"{a}_{b}" for a, b in bench.PAIRS]
TIERS = ("moderate", "severe", "conservative")


def _canonical():
    """The canonical flags artefact ON DISK -- the file a consumer actually reads."""
    if not os.path.exists(CANONICAL_CSV):
        raise AssertionError(
            f"canonical flags file not found: {CANONICAL_CSV}\n"
            f"regenerate it with: python sat_work/research/recommended.py")
    return pd.read_csv(CANONICAL_CSV)


# --------------------------------------------------------------------------- #
# schema / backwards compatibility
# --------------------------------------------------------------------------- #
def test_export_is_a_schema_superset_of_the_published_v1_schema():
    """Existing consumers must keep working: every published v1 column must survive.

    Asserted against the FROZEN column list, not against the file on disk. Now that the
    canonical file is produced by a different method, deriving "the published schema" from
    it would be circular -- the contract would hold by construction and test nothing.
    """
    exported = set(bench.build_export().columns)
    missing = set(PUBLISHED_V1_COLUMNS) - exported
    assert not missing, f"export drops published v1 columns: {sorted(missing)}"
    # and the artefact on disk must actually be the export it claims to be
    assert set(_canonical().columns) == exported, (
        "canonical file columns differ from build_export() -- the file is stale; "
        "re-run sat_work/research/recommended.py")


def test_export_covers_every_pair_and_has_the_right_length():
    ex = bench.build_export()
    assert len(ex) == len(bench.raw_df()), "export is not row-aligned with the dataset"
    for k in PAIR_KEYS:
        for suffix in ("deficit", "null_d", "excess_vs_controls"):
            assert f"{k}_{suffix}" in ex.columns, f"missing {k}_{suffix}"
        for tier in TIERS:
            assert f"{k}_sat_{tier}" in ex.columns, f"missing {k}_sat_{tier}"


def test_export_has_a_time_column_matching_the_input_index():
    ex = bench.build_export()
    assert "time" in ex.columns, "published schema has a `time` column"
    assert pd.to_datetime(ex["time"]).equals(
        pd.Series(bench.raw_df().index)), "export `time` is not the input index"


# --------------------------------------------------------------------------- #
# the -inf defect
# --------------------------------------------------------------------------- #
def test_export_contains_no_infinities():
    """The recommended export must be safe for .mean() / .sum() / regression."""
    num = bench.build_export().select_dtypes("number")
    n_inf = int(np.isinf(num.to_numpy()).sum())
    assert n_inf == 0, f"{n_inf} infinite values in the export"


def test_export_deficit_is_finite_or_nan():
    ex = bench.build_export()
    for k in PAIR_KEYS:
        col = ex[f"{k}_deficit"]
        bad = col[np.isinf(col)]
        assert len(bad) == 0, f"{k}_deficit has {len(bad)} non-finite values"


def test_export_deficit_is_nan_outside_the_decision_domain():
    """The same safety property, asserted on the pure function rather than the file."""
    ex = bench.build_export()
    df, fe = bench.raw_df(), bench.raw_fe()
    ref = fe["ref"]
    for k, (a, b) in zip(PAIR_KEYS, bench.PAIRS):
        domain = (fe["active"][a] & fe["active"][b] & (ref >= bench.REF_ON)
                  & ~bench.dq_mask(fe, a, b)).to_numpy()
        col = ex[f"{k}_deficit"].to_numpy()
        assert np.isnan(col[~domain]).all(), f"{k}_deficit populated outside the domain"


def test_masking_changes_no_flag_so_the_safety_fix_is_free():
    """The downstream-safety fix must move NO number a consumer would report.

    Masking the deficit column to NaN outside the decision domain is a *presentation*
    change to a continuous column. Every `*_sat_*` flag is decided by `gate`, and the gate
    is a strict subset of the domain, so no flag can be touched. This asserts the
    byte-identity directly instead of trusting the subset argument, because "this fix is
    free" is exactly the kind of claim that rots: if someone ever widens the domain without
    widening the gate, the masking would start deleting severity information and the
    headline counts would drift silently.
    """
    masked = bench.build_export(mask_below_gate=True)
    unmasked = bench.build_export(mask_below_gate=False)
    assert masked.shape == unmasked.shape
    flag_cols = [c for c in masked.columns if "_sat_" in c]
    assert flag_cols, "expected flag columns in the export"
    for c in flag_cols + ["time", "dq_site_outage"]:
        assert masked[c].equals(unmasked[c]), f"masking altered flag column {c}"
    # and the non-flag payload really did change, so this is not vacuously true
    for k in PAIR_KEYS:
        c = f"{k}_deficit"
        assert not masked[c].equals(unmasked[c]), f"{c} was not actually masked"
        assert int(masked[c].notna().sum()) < int(unmasked[c].notna().sum())


def test_canonical_flags_csv_contains_no_infinities():
    """The canonical artefact must be safe for .mean() / .sum() / regression.

    The published artefact shipped 5,373 `-inf` values (review section 3.9). The vat-v1
    export cannot reproduce them: `theta(t)` is a rolling median of `s/ref` and is
    strictly positive wherever it is defined, so `1 - s/(theta*ref)` stays finite.
    """
    num = _canonical().select_dtypes("number")
    n_inf = int(np.isinf(num.to_numpy()).sum())
    assert n_inf == 0, f"{n_inf} infinite values in {os.path.basename(CANONICAL_CSV)}"


def test_canonical_deficit_is_well_formed_inside_the_decision_domain():
    """Inside the domain the deficit must be a sane, bounded severity.

    `deficit = 1 - s/(theta*ref)` is >= 1 only when s <= 0 and <= 1 by construction, so
    inside the decision domain it belongs to [-1, 1]. It has to be checked on the DECISION
    DOMAIN rather than the in-gate region because the domain additionally excludes inactive
    and DQ-flagged rows, where the pair sum is not a measurement of anything.

    Measured in-domain: min -0.141, max 1.0000 across the three pairs. The published
    formulation violated this even in-domain (its control-pair December deficit reached
    -0.49, review section 3.10).
    """
    can = _canonical()
    for key in PAIR_KEYS:
        col = can[f"{key}_deficit"].dropna()
        assert len(col) > 1_000, f"{key}_deficit is empty inside the domain"
        assert col.min() >= -1.0, (
            f"{key}_deficit is {col.min():.3f} inside the domain -- below the physical "
            f"floor of -1, so the baseline is under-predicting there")
        assert col.max() <= 1.0 + 1e-9, (
            f"{key}_deficit is {col.max():.3f} inside the domain -- above the ceiling of 1")


def test_canonical_deficit_is_nan_outside_the_decision_domain():
    """The downstream-safety guarantee: no meaningless value is ever published.

    Before this masking the column carried large *finite* negatives at low irradiance
    (down to about -260, where `theta*ref -> 0`), which are strictly better than the
    published `-inf` but still make an unfiltered `.mean()` meaningless. The column is now
    NaN wherever the detector would not decide, so the naive aggregate equals the
    in-domain aggregate.

    The domain must be exactly `act & ref >= REF_ON & ~dq_mask` -- the same one the AUC is
    computed over, so every published rate shares a denominator.
    """
    can = _canonical()
    df, fe = bench.raw_df(), bench.raw_fe()
    ref = fe["ref"]
    for key, (a, b) in zip(PAIR_KEYS, bench.PAIRS):
        domain = (fe["active"][a] & fe["active"][b] & (ref >= bench.REF_ON)
                  & ~bench.dq_mask(fe, a, b)).to_numpy()
        col = can[f"{key}_deficit"].to_numpy()
        assert np.isnan(col[~domain]).all(), (
            f"{key}_deficit is populated on {int((~np.isnan(col[~domain])).sum())} rows "
            f"outside the decision domain")
        assert not np.isnan(col[domain]).all(), "domain masking threw away the signal"
        # and the naive aggregate is therefore the in-domain aggregate
        naive, indomain = np.nanmean(col), np.nanmean(col[domain])
        assert abs(naive - indomain) < 1e-9, (
            f"{key}: unfiltered mean {naive:.4f} != in-domain mean {indomain:.4f}")


# --------------------------------------------------------------------------- #
# below-gate masking
# --------------------------------------------------------------------------- #
def test_null_d_is_nan_outside_the_gate_domain():
    """The empirical null is only meaningful where the detector actually decides.

    Publishing the bin-1 null (ref < 0.3) on a pre-dawn row at ref == 0 is misleading.
    """
    ex = bench.build_export()
    ref = ex["ref"]
    below = (ref < bench.REF_ON).to_numpy()
    assert below.sum() > 1000, "expected a substantial below-gate region"
    for k in PAIR_KEYS:
        col = ex[f"{k}_null_d"].to_numpy()
        assert np.isnan(col[below]).all(), f"{k}_null_d is populated below the gate"


def test_excess_vs_controls_is_nan_outside_the_gate_domain():
    ex = bench.build_export()
    below = (ex["ref"] < bench.REF_ON).to_numpy()
    for k in PAIR_KEYS:
        col = ex[f"{k}_excess_vs_controls"].to_numpy()
        assert np.isnan(col[below]).all(), f"{k}_excess_vs_controls populated below gate"


def test_null_d_is_populated_inside_the_gate_domain():
    """Masking must not have thrown away the signal too.

    Note `null_d` is masked where the pair is inactive as well as below the gate, so the
    populated count is checked rather than assumed. `.between` returns False for NaN, so
    the values must be dropped before the range check.
    """
    ex = bench.build_export()
    inside = ex["ref"] >= bench.REF_ON
    for k in PAIR_KEYS:
        vals = ex.loc[inside, f"{k}_null_d"].dropna()
        assert len(vals) > 1000, f"{k}_null_d is empty inside the gate"
        assert vals.abs().max() < 1.0, (
            f"{k}_null_d has implausible values inside the gate: "
            f"min={vals.min():.3f} max={vals.max():.3f}"
        )


# --------------------------------------------------------------------------- #
# flag column integrity
# --------------------------------------------------------------------------- #
def test_flags_are_binary_integers():
    ex = bench.build_export()
    for k in PAIR_KEYS:
        for tier in TIERS:
            col = ex[f"{k}_sat_{tier}"]
            assert pd.api.types.is_integer_dtype(col), f"{k}_sat_{tier} is {col.dtype}"
            assert set(col.unique()) <= {0, 1}, f"{k}_sat_{tier} is not binary"


def test_flags_are_never_set_where_the_deficit_is_undefined():
    """A flag with no score behind it is a bug, not a detection."""
    ex = bench.build_export()
    for k in PAIR_KEYS:
        undef = ex[f"{k}_deficit"].isna()
        for tier in TIERS:
            col = ex[f"{k}_sat_{tier}"]
            assert int((col[undef] == 1).sum()) == 0, \
                f"{k}_sat_{tier} flags {int((col[undef] == 1).sum())} undefined rows"


def test_flags_are_never_set_below_the_irradiance_gate():
    """The ref >= 0.70 gate is a hard scope limit and must not leak."""
    ex = bench.build_export()
    below = ex["ref"] < bench.REF_ON
    for k in PAIR_KEYS:
        for tier in TIERS:
            n = int((ex.loc[below, f"{k}_sat_{tier}"] == 1).sum())
            assert n == 0, f"{k}_sat_{tier} flags {n} rows below the gate"


def test_severe_tier_is_a_subset_of_moderate():
    """Severity must nest: a 0.30 deficit implies a 0.15 deficit."""
    ex = bench.build_export()
    for k in PAIR_KEYS:
        mod = ex[f"{k}_sat_moderate"] == 1
        sev = ex[f"{k}_sat_severe"] == 1
        leaked = int((sev & ~mod).sum())
        assert leaked == 0, f"{k}: {leaked} severe rows are not also moderate"


def test_persistence_breaks_runs_at_acquisition_gaps():
    """Unit test of the gap-aware rule on synthetic data.

    The defect this guards: `run_lengths` is computed over the whole series, so two
    flagged samples before a 40-day hole and one after used to be counted together as a
    run of three and all three were published as a 15-minute episode.
    """
    idx = pd.to_datetime(["2026-01-01 12:00", "2026-01-01 12:05", "2026-01-01 12:10",
                          "2026-02-10 12:00"])   # 40-day acquisition hole
    # 2 before the gap + 1 after == 3 samples, but not a contiguous 15 minutes
    raw = pd.Series([False, True, True, True], index=idx)
    assert bench.persist(raw, 3, index=idx).sum() == 0, \
        "gap-aware persistence still accepted a run split by a gap"

    # the naive rule is what produced the bug -- document it so nobody reverts
    assert int(bench.persist(raw, 3).sum()) == 3, \
        "the non-gap-aware rule no longer reproduces the defect it was fixed for"

    # without a gap the same four samples form a legitimate run and must survive
    idx2 = pd.to_datetime(["2026-01-01 12:00", "2026-01-01 12:05", "2026-01-01 12:10",
                           "2026-01-01 12:15"])
    raw2 = pd.Series([True, True, True, False], index=idx2)
    assert bench.persist(raw2, 3, index=idx2).sum() == 3


def test_every_flag_sits_in_a_contiguous_run_of_minimum_length():
    """Re-derived from the contract rather than by calling persist().

    For each pair and tier, every flagged sample must belong to a block of at least `k`
    consecutive flagged samples that are also contiguous in time. This independently
    checks the gap-aware persistence rule against the shipped export.
    """
    ex = bench.build_export()
    step = pd.to_datetime(ex["time"]).diff().dt.total_seconds().div(60).to_numpy()
    step = np.nan_to_num(step, nan=1e9)        # the first row is always a boundary
    seg = np.cumsum(step > 6)                  # increments at each acquisition gap
    need = {"moderate": 3, "severe": 6, "conservative": 3}

    for key in PAIR_KEYS:
        for tier, k in need.items():
            f = ex[f"{key}_sat_{tier}"].to_numpy().astype(bool)
            if not f.any():
                continue
            brk = np.empty(len(f), dtype=bool)
            brk[0] = True
            brk[1:] = (seg[1:] != seg[:-1]) | (f[1:] != f[:-1])
            block_len = pd.Series(f).groupby(np.cumsum(brk)).transform("sum").to_numpy()
            offenders = f & (block_len < k)
            assert not offenders.any(), (
                f"{key}_sat_{tier}: {int(offenders.sum())} flagged samples sit in a "
                f"contiguous run shorter than {k}"
            )
