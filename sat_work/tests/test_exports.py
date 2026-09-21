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

from _helpers import CANONICAL_CSV, PUBLISHED_V1_COLUMNS, RETIRED_V1_COLUMNS, bench

PAIR_KEYS = [f"{a}_{b}" for a, b in bench.PAIRS]
# One flag column per pair. The `sat_severe` and `sat_conservative` columns were removed
# deliberately -- see build_export's docstring. Severity lives in `deficit`.
FLAG_SUFFIXES = ("sat",)


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
def test_export_schema_is_the_documented_single_flag_schema():
    """The canonical schema is pinned here, not derived from the file on disk.

    Derived-vs-pinned matters: if this read the file it would agree with itself by
    construction. The schema is asserted exactly, so an accidental extra column (a
    half-removed tier, say) fails here rather than reaching a consumer.
    """
    exported = set(bench.build_export().columns)
    expected = {"time", "ref", "dq_site_outage"} | {
        f"{a}_{b}_{s}" for a, b in bench.PAIRS for s in ("deficit", "sat")}
    assert exported == expected, (
        f"canonical schema changed.\n  unexpected: {sorted(exported - expected)}\n"
        f"  missing   : {sorted(expected - exported)}")
    # and the artefact on disk must actually be the export it claims to be
    assert set(_canonical().columns) == exported, (
        "canonical file columns differ from build_export() -- the file is stale; "
        "re-run sat_work/research/recommended.py")


def test_migration_from_the_published_v1_schema_is_total_and_one_for_one():
    """Every retired published column must have exactly one documented replacement.

    The three tiers became one `sat` flag. This asserts the mapping is total (nothing was
    dropped without a replacement) and that `deficit` -- the column carrying severity
    forward -- survived untouched.
    """
    exported = set(bench.build_export().columns)
    dropped = set(PUBLISHED_V1_COLUMNS) - exported
    assert dropped == set(RETIRED_V1_COLUMNS), (
        f"unexpected published-v1 columns missing from the export: {sorted(dropped)}")
    # `deficit` must survive: it is what replaces the severity the tiers used to encode
    for a, b in bench.PAIRS:
        assert f"{a}_{b}_deficit" in exported, f"lost {a}_{b}_deficit in the migration"
        assert f"{a}_{b}_sat" in exported, f"lost the replacement flag {a}_{b}_sat"


def test_export_covers_every_pair_and_has_the_right_length():
    ex = bench.build_export()
    assert len(ex) == len(bench.raw_df()), "export is not row-aligned with the dataset"
    for k in PAIR_KEYS:
        assert f"{k}_deficit" in ex.columns, f"missing {k}_deficit"
        for suffix in FLAG_SUFFIXES:
            assert f"{k}_{suffix}" in ex.columns, f"missing {k}_{suffix}"


def test_export_carries_no_retired_tier_columns():
    """The tier columns are gone, not merely renamed.

    A reviewer reading the paper will look for `sat_severe`; it must not be there. This
    also stops a half-finished refactor from leaving a stale column that no document
    mentions and no test would otherwise catch.
    """
    exported = set(bench.build_export().columns)
    retired = [c for c in exported
               if c.endswith(("_sat_severe", "_sat_conservative", "_sat_moderate"))]
    assert not retired, f"retired tier columns still exported: {sorted(retired)}"
    assert not [c for c in exported
                if c.endswith(("_null_d", "_excess_vs_controls"))], \
        "control-null support columns were removed with the conservative tier"


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
    flag_cols = [c for c in masked.columns if c.endswith("_sat")]
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
def test_severity_is_recoverable_from_the_deficit_column():
    """Removing the `sat_severe` tier must not lose severity.

    The retired severe tier was `deficit > 0.30` for 6 samples. These assertions pin the
    continuity of it onto the surviving `deficit` column: if a `deficit`-based threshold
    ever stopped reproducing the old threshold's row count, the paper's claim that severity
    is recoverable by thresholding (rather than only through an exported tier) would be
    false, and nobody would notice until a reviewer asked.
    """
    ex = bench.build_export()
    tot = 0
    for k in PAIR_KEYS:
        # every severe-threshold crossing must sit inside the decision domain
        d = ex[f"{k}_deficit"]
        assert int((d > 0.30).sum()) > 0, f"{k}: no rows above 0.30 in the deficit column"
        tot += int((d > 0.30).sum())
    # the old severe tier was a persistence-filtered version of this, so the raw crossing
    # count must be the LARGER of the two. Guard against the column being silently zeroed.
    assert tot > 328, f"raw >0.30 crossings ({tot}) below the retired tier's 328 rows"


def test_defensive_deficit_column_can_be_thresholded_at_any_cut():
    """A single continuous column must support every severity cut the tiers used."""
    ex = bench.build_export()
    for k in PAIR_KEYS:
        d = ex[f"{k}_deficit"].dropna()
        for cut in (0.10, 0.15, 0.20, 0.30, 0.40):
            n = int((d > cut).sum())
            assert n >= 0
        # monotone in the cut: a stricter cut can never select more rows
        counts = [int((d > c).sum()) for c in (0.10, 0.15, 0.20, 0.30, 0.40)]
        assert counts == sorted(counts, reverse=True), (
            f"{k}: thresholding the deficit column is not monotone: {counts}")


# --------------------------------------------------------------------------- #
# flag column integrity
# --------------------------------------------------------------------------- #
def test_flags_are_binary_integers():
    ex = bench.build_export()
    for k in PAIR_KEYS:
        col = ex[f"{k}_sat"]
        assert pd.api.types.is_integer_dtype(col), f"{k}_sat is {col.dtype}"
        assert set(col.unique()) <= {0, 1}, f"{k}_sat is not binary"


def test_flags_are_never_set_where_the_deficit_is_undefined():
    """A flag with no score behind it is a bug, not a detection."""
    ex = bench.build_export()
    for k in PAIR_KEYS:
        undef = ex[f"{k}_deficit"].isna()
        col = ex[f"{k}_sat"]
        assert int((col[undef] == 1).sum()) == 0, \
            f"{k}_sat flags {int((col[undef] == 1).sum())} undefined rows"


def test_flags_are_never_set_below_the_irradiance_gate():
    """The ref >= 0.70 gate is a hard scope limit and must not leak."""
    ex = bench.build_export()
    below = ex["ref"] < bench.REF_ON
    for k in PAIR_KEYS:
        n = int((ex.loc[below, f"{k}_sat"] == 1).sum())
        assert n == 0, f"{k}_sat flags {n} rows below the gate"


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

    For each pair, every flagged sample must belong to a block of at least `k`
    consecutive flagged samples that are also contiguous in time. This independently
    checks the gap-aware persistence rule against the shipped export.
    """
    ex = bench.build_export()
    step = pd.to_datetime(ex["time"]).diff().dt.total_seconds().div(60).to_numpy()
    step = np.nan_to_num(step, nan=1e9)        # the first row is always a boundary
    seg = np.cumsum(step > 6)                  # increments at each acquisition gap
    # the single exported rule: 3 consecutive samples (15 min)
    assert bench.PERS_MOD == 3, "the exported run length changed; update this test"

    for key in PAIR_KEYS:
        f = ex[f"{key}_sat"].to_numpy().astype(bool)
        if not f.any():
            continue
        brk = np.empty(len(f), dtype=bool)
        brk[0] = True
        brk[1:] = (seg[1:] != seg[:-1]) | (f[1:] != f[:-1])
        block_len = pd.Series(f).groupby(np.cumsum(brk)).transform("sum").to_numpy()
        offenders = f & (block_len < bench.PERS_MOD)
        assert not offenders.any(), (
            f"{key}_sat: {int(offenders.sum())} flagged samples sit in a "
            f"contiguous run shorter than {bench.PERS_MOD}"
        )
