"""
Regression tests for the MPPT saturation DETECTORS, scored on injected ground truth.

The point of this file
----------------------
The published validation compares shared pairs against independent control pairs. It
cannot catch the failure that matters, because its control pairs are never clipped: a
detector whose baseline is mis-calibrated on *intermittently* clipped pairs passes
cleanly. Every test below therefore runs against manufactured ground truth.

One test is deliberately written to FAIL today:
  * `test_published_detector_is_biased_on_control_pairs` documents the published defect
It is marked `known_failure`, so it reports as XFAIL and will flip to a pass when the
underlying issue is fixed. Do not delete it -- it is a guard rail.

The two data-product defects that used to be recorded in test_exports are both resolved:
the `-inf` values by the vat-v1 regeneration, and the seasonal mis-scaling of the
continuous column by the fact that the canonical file is no longer produced by the
published method (see `test_published_baseline_is_seasonally_mis_scaled_...` below, which
keeps the published defect itself under test).

Numbers quoted in assertions are the values measured when this suite was frozen. Each
has headroom; they are regression bands, not exact-value checks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _helpers import (EPISODIC_PAIRS, bench, chronic, intermittent, known_failure,
                      noclip, score)

D0 = bench.det_published
D1 = bench.det_rolling
D2 = bench.det_envelope
D4 = bench.det_ceiling_pinned
D5 = bench.det_control_calibrated


# --------------------------------------------------------------------------- #
# 0. the fixture itself
# --------------------------------------------------------------------------- #
def test_fixture_has_positives_and_negatives():
    """A silently-empty fixture would make every other test vacuous.

    Note the chronic scenario legitimately has few negatives: when the limit binds every
    day, nearly every high-irradiance row is clamped, so only the low-sun rows inside the
    gate remain unclamped.
    """
    for sc in (intermittent(), intermittent(("eu_15", "eu_23")), chronic()):
        e = score(sc, D1)
        assert e["n_pos00"] > 300, f"too few positives: {e['n_pos00']}"
        assert e["n_neg"] > 250, f"too few negatives: {e['n_neg']}"
        # the evaluation domain partitions exactly into labelled positives and negatives
        assert e["n_eval"] == e["n_pos00"] + e["n_neg"], "domain accounting is inconsistent"
    for pair in EPISODIC_PAIRS:
        assert score(intermittent(pair), D1)["n_neg"] > 500, \
            "an episodic scenario should be dominated by unclipped days"


def test_evaluation_domain_is_excluded_from_flags():
    """Flags outside the ref gate must not be able to influence the metrics."""
    sc = intermittent()
    _, flag = D1(sc["df"], sc["fe"], sc["a"], sc["b"])
    ref = sc["fe"]["ref"]
    outside = ref < bench.REF_ON
    assert not bool(flag[outside].any()), "detector flagged below the ref gate"


# --------------------------------------------------------------------------- #
# 1. THE core test: specificity when the limit is intermittent
# --------------------------------------------------------------------------- #
def test_published_detector_overflags_when_clipping_is_intermittent():
    """D0 fires on unclipped days whenever the limit is not chronic.

    Measured on the 2026-09 export: 986 rows on eu_1+eu_3 and 708 on eu_15+eu_23, 1,694
    pooled. This is the defect the published control-pair validation cannot see.

    The band was ``>= 5000`` while the export was the incomplete ``..._jul2026`` file,
    which was missing four whole months (2025-08, 2025-09, 2026-03, 2026-04). Those gaps
    stretched ``cap(t)``'s 31-present-day rolling window to as much as 92 calendar days,
    mixing seasons inside one window and depressing the published expectation; the phantom
    deficit that followed accounted for most of the old count. The defect is real and
    survives the completed record -- it is simply 6x smaller. See SATURATION_REVIEW.md 3.3.
    """
    fp = sum(score(intermittent(pair), D0)["fp_rows"]
             for pair in (("eu_1", "eu_3"), ("eu_15", "eu_23")))
    assert fp >= 1000, (
        f"published detector produced only {fp} false-positive rows on intermittent "
        f"clips; if this dropped, the published defect may have been fixed -- update "
        f"SATURATION_REVIEW.md and relax this band rather than deleting the test"
    )


def test_rolling_slope_is_specific_when_clipping_is_intermittent():
    """D1 == the recommended detector's primary tier, same rules, corrected baseline.

    Measured: 1 and 101 false-positive rows on the same two scenarios (vs ~10,200).
    """
    fp = 0
    for pair in (("eu_1", "eu_3"), ("eu_15", "eu_23")):
        fp += score(intermittent(pair), D1)["fp_rows"]
    assert fp <= 250, f"D1 produced {fp} false-positive rows on intermittent clips"


def test_recommended_detector_is_more_specific_than_published():
    """The specificity margin, as a machine-checked ratio -- with its band justified.

    Measured on the 2026-09 export: published 1,694 vs D1 189 = 9.0x pooled over the two
    episodic scenarios. Per scenario the ratios are 58x (eu_1+eu_3) and 4x (eu_15+eu_23),
    so the pooled figure hides a wide spread; the absolute FP row counts are reported
    alongside it everywhere it is quoted.

    This was asserted at ``>= 20x`` against a published count of 10,202 that the incomplete
    export had inflated roughly 6x. The corrected margin is real but modest, and it is NOT
    the load-bearing evidence for vat-v1 -- the benchmark AUC (0.961 vs 0.803 episodic) and
    the bias floor (7 % vs 39 %) are. The band is ``>= 5x`` so the test still fails if the
    margin genuinely collapses, without encoding a number that depended on a broken record.
    """
    d0 = sum(score(intermittent(p), D0)["fp_rows"]
             for p in (("eu_1", "eu_3"), ("eu_15", "eu_23")))
    d1 = sum(score(intermittent(p), D1)["fp_rows"]
             for p in (("eu_1", "eu_3"), ("eu_15", "eu_23")))
    assert d1 > 0 or d0 > 0
    ratio = d0 / max(d1, 1)
    assert ratio >= 5, f"specificity margin collapsed to {ratio:.1f}x (published={d0}, D1={d1})"
    assert d1 <= 400, f"D1's absolute false-positive count rose to {d1}"


def test_scenario_day_selection_is_invariant_to_record_length():
    """The benchmark must not re-roll when the record is completed or truncated.

    ``scenario`` used to draw one uniform per POSITION in the day list, via
    ``default_rng(seed).random(len(days))``. Completing the 2026-09 export grew the day
    list from 328 to 487 entries, every position shifted, and all six scenarios silently
    moved onto different days -- the pooled episodic false-positive count then changed by
    more than 30x while no detector code had changed at all.

    ``bench.day_uniform`` keys the draw on the DATE instead, so a given day's fate is a
    function of (seed, date) alone and cannot be perturbed by the rest of the record. This
    guards that property: truncating the record must leave every surviving day's draw
    bit-identical.
    """
    days = pd.date_range("2025-05-01", "2026-08-31", freq="D")
    full = bench.day_uniform(days, seed=7)
    for cut in (120, 328, 430):
        truncated = bench.day_uniform(days[:cut], seed=7)
        assert np.array_equal(full[:cut], truncated), (
            f"the scenario draw for the first {cut} days changed when the record was "
            f"truncated -- day selection is position-dependent again"
        )
    # a different seed must move the selection, or the fixture would be degenerate
    assert not np.array_equal(full, bench.day_uniform(days, seed=8))
    # and the selection must be a genuine mix, not all-on or all-off
    picked = full < 0.35
    assert 0.2 < picked.mean() < 0.5, f"duty is degenerate: {picked.mean():.2f} of days on"


def test_chronic_clipping_hides_the_published_defect():
    """Why the published validation missed it.

    When the limit binds EVERY day, `cap(t)` adapts to it and the published detector
    looks almost as good as the corrected one (false positives collapse from thousands
    to about ten). The published control pairs were effectively in this regime. Any
    future validation must therefore include an intermittent scenario.
    """
    fp_inter = sum(score(intermittent(p), D0)["fp_rows"]
                   for p in (("eu_1", "eu_3"), ("eu_15", "eu_23")))
    fp_chronic = score(chronic(), D0)["fp_rows"]
    assert fp_chronic <= 60, f"chronic-regime false positives rose to {fp_chronic}"
    assert fp_inter / max(fp_chronic, 1) >= 50, (
        f"the regime contrast collapsed: intermittent={fp_inter}, chronic={fp_chronic}"
    )


# --------------------------------------------------------------------------- #
# 2. threshold-free ranking
# --------------------------------------------------------------------------- #
def test_rolling_slope_beats_published_on_auc():
    """Mean AUC over episodic scenarios: measured 0.962 (D1) vs 0.780 (D0)."""
    a1 = sum(score(intermittent(p), D1)["auc00"]
             for p in (("eu_1", "eu_3"), ("eu_15", "eu_23"))) / 2
    a0 = sum(score(intermittent(p), D0)["auc00"]
             for p in (("eu_1", "eu_3"), ("eu_15", "eu_23"))) / 2
    assert a1 >= 0.94, f"D1 AUC fell to {a1:.3f}"
    assert a1 - a0 >= 0.10, f"AUC advantage collapsed: D1={a1:.3f} D0={a0:.3f}"


def test_deep_clips_are_almost_perfectly_detected():
    """Measured 0.997-0.999 for D1 at >20 % severity."""
    for s in ("auc20",):
        vals = [score(intermittent(p), D1)[s] for p in (("eu_1", "eu_3"), ("eu_15", "eu_23"))]
        assert min(vals) >= 0.99, f"D1 {s} fell to {min(vals):.3f}"


def test_shallow_clips_are_harder_than_deep_ones():
    """Encodes that AUC rises monotonically with clip depth, i.e. the detector is a
    severity measure near the boundary. Measured: 0.964 -> 0.997 -> 0.999."""
    e = score(intermittent(), D1)
    assert e["auc00"] < e["auc10"] < e["auc20"], (
        f"severity stratification broke: {e['auc00']:.3f} {e['auc10']:.3f} {e['auc20']:.3f}"
    )


def test_ceiling_pinned_is_brilliant_for_chronic_and_useless_for_intermittent():
    """Documents a negative result so it is not rediscovered.

    A plateau cannot be established from days that are not clipped: measured AUC
    1.000 for a chronic limit vs 0.487-0.500 for an intermittent one.
    """
    assert score(chronic(), D4)["auc00"] >= 0.95
    for pair in (("eu_1", "eu_3"), ("eu_15", "eu_23")):
        assert score(intermittent(pair), D4)["auc00"] <= 0.65, "D4 unexpectedly works"


def test_control_calibrated_null_trades_sensitivity_for_specificity():
    """D5 is the most specific but the least sensitive -- measured recall 0.23-0.29
    against D1's 0.28-0.34 at ~1/10th the false positives of D0, not the best of both."""
    d5 = sum(score(intermittent(p), D5)["fp_rows"]
             for p in (("eu_1", "eu_3"), ("eu_15", "eu_23")))
    d1 = sum(score(intermittent(p), D1)["fp_rows"]
             for p in (("eu_1", "eu_3"), ("eu_15", "eu_23")))
    assert d5 <= 500, f"D5 false positives rose to {d5}"
    assert d1 <= 250, f"D1 false positives rose to {d1}"


# --------------------------------------------------------------------------- #
# 3. zero-false-positive guarantee on a pair that cannot saturate
# --------------------------------------------------------------------------- #
def test_no_flags_on_an_unconstrained_independent_pair():
    """No constraint was injected at all, so a flag is unambiguously a false alarm.

    Measured on the 2026-09 export (76,933 rows): published 28, D1 0, D2 4, D5 3, D4 1,723.
    D1 -- the canonical detector -- is held to exactly zero, which is the guarantee that
    matters. D5 is held to a small band rather than zero: it is a scored competitor, not
    the exported method, and on this export its control-derived null picks up 3 rows.
    """
    sc = noclip()
    assert score(sc, D1)["fp_rows"] == 0, "corrected baseline flagged an intact pair"
    d5 = score(sc, D5)["fp_rows"]
    assert d5 <= 5, f"control-calibrated tier flagged an intact pair ({d5} rows)"
    assert score(sc, D4)["fp_rows"] >= 100, "D4's small-sample artefact disappeared"


@known_failure("published detector flags intact independent pairs (review 3.1/3.2)")
def test_published_detector_is_biased_on_control_pairs():
    """KNOWN FAILURE - the published detector flags an intact independent pair.

    Measured 55 rows on a pair with no constraint injected at all. Its own model puts the
    deficit at ~+0.088 at full sun against a 0.15 threshold, which is the margin this
    exploits. Delete only if the published detector is replaced.
    """
    fp = score(noclip(), D0)["fp_rows"]
    assert fp < 20, f"published detector flagged {fp} rows on a pair that cannot saturate"


def test_published_baseline_is_seasonally_mis_scaled_and_the_corrected_one_is_not():
    """The mechanism behind review sections 3.2 / 3.10, as a machine-checked contrast.

    Control pairs cannot saturate, so their true deficit is ~0 and any non-zero value is
    pure baseline error. In December the published `g0*cap(t)*ref` falls off a seasonal
    cliff: `cap(t)` tracks the physical peak (10.3 MW in Dec vs 16.5 MW in Jun) while the
    true slope `theta` barely moves, so `expected` collapses and the published deficit
    reports an intact pair as roughly 45 % short. `theta(t)` tracks the level, so it does
    not. Measured at ref >= 0.5 (n=285, the most robust December slice available):

        published  -0.44 / -0.48 / -0.44        vat-v1  ~ 0.00

    This is the continuous-column defect that the published flag-level validation could
    not see. Relocated here from test_exports: once the canonical file became the vat-v1
    output it no longer contained a published column to assert against, and the subject
    of this test is the METHOD, not the data product.
    """
    df = bench.raw_df()
    fe = bench.raw_fe()
    ref = fe["ref"]
    dec = pd.Series(df.index.month == 12, index=df.index) & (ref >= 0.5)
    assert int(dec.sum()) >= 100, "December slice is too small to guard anything"

    for a, b in bench.CONTROL_PAIRS:
        pub = pd.Series(np.asarray(bench.detect(df, fe, a, b)["deficit"], dtype=float),
                        index=df.index)
        vat = bench.pair_deficit(df, fe, a, b)["deficit"]
        pm, vm = pub[dec].median(), vat[dec].median()
        assert pm < -0.35, (
            f"published December deficit on {a}+{b} is {pm:+.3f} -- the seasonal bias has "
            f"gone; if the baseline really was replaced, retire this test")
        assert vm > -0.05, (
            f"vat-v1 December deficit on {a}+{b} is {vm:+.3f} -- the seasonal bias is back")
        assert abs(vm) < 0.2 * abs(pm), (
            f"{a}+{b}: vat-v1 error {vm:+.4f} is not clearly better than published {pm:+.4f}")


# --------------------------------------------------------------------------- #
# 4. determinism and the persistence decision
# --------------------------------------------------------------------------- #
def test_scenarios_are_deterministic():
    """Ground truth must not drift between runs, or the bands above are meaningless."""
    key = ("scenario", ("eu_1", "eu_3"), 0.35, 0.75, "episodic", 7)
    first = score(intermittent(), D1)
    bench._STATE.pop(key, None)
    second = score(intermittent(), D1)
    assert first == second, "scenario rebuild changed the metrics"


def test_gap_tolerant_persistence_buys_recall_cheaply():
    """The adopted `gap` rule: bridging 1-sample gaps inside the gate.

    The default IS gap=1 now (bench.PERS_GAP), so this must compare gap=0 explicitly
    against gap=1. Measured on the intermittent scenario: recall 0.340 -> 0.372 for a
    precision cost of 0.9992 -> 0.9966. Asserted as a direction plus a precision floor so
    the trade-off is visible if it ever changes.

    Note this is a real trade-off, not a free gain: guarding the filled samples was tested
    (step18) and does not dominate. See SATURATION_REVIEW.md section 3.13.
    """
    strict = score(intermittent(), D1, gap=0)
    bridged = score(intermittent(), D1, gap=1)
    assert bridged["tp_rows"] > strict["tp_rows"], "gap bridging recovered no recall"
    assert bridged["prec"] >= strict["prec"] - 0.01, (
        f"gap bridging cost too much precision: {strict['prec']:.4f} -> {bridged['prec']:.4f}"
    )


def test_bridging_never_flags_below_the_gate():
    """The adopted rule fills only samples that satisfy the gate.

    A fill that lands below `ref` 0.70 breaks the export contract that no flag is set
    below the gate. This is what `bridge(..., within=gate)` exists for, and the suite
    caught the ungated version doing exactly that.
    """
    sc = intermittent()
    d = bench.pair_deficit(sc["df"], sc["fe"], sc["a"], sc["b"])
    gate = ((sc["fe"]["ref"] >= bench.REF_ON) & (d["s"] > bench.NEAR_CAP * d["expected"])
            & d["act"] & ~bench.dq_mask(sc["fe"], sc["a"], sc["b"]))
    raw = gate & (d["deficit"] > bench.THR_MOD)
    ungated = bench.bridge(raw, 1)
    gated = bench.bridge(raw, 1, within=gate)
    assert bool((ungated & ~gate).any()), "fixture cannot detect the ungated fill"
    assert not bool((gated & ~gate).any()), "gated bridging still filled outside the gate"


def test_persistence_rule_is_actually_applied():
    """A 3-sample minimum must remove short runs; the raw rule should flag more."""
    from bench import (NEAR_CAP, PERS_MOD, REF_ON, THR_MOD, dq_mask, pair_deficit,
                       persist)
    sc = intermittent()
    d = pair_deficit(sc["df"], sc["fe"], sc["a"], sc["b"])
    gate = ((sc["fe"]["ref"] >= REF_ON) & (d["s"] > NEAR_CAP * d["expected"])
            & d["act"] & ~dq_mask(sc["fe"], sc["a"], sc["b"]))
    raw = gate & (d["deficit"] > THR_MOD)
    persisted = persist(raw, PERS_MOD)
    assert int(persisted.sum()) < int(raw.sum()), "persistence removed nothing"
    assert not bool((persisted & ~raw).any()), "persistence added rows"


# --------------------------------------------------------------------------- #
# 5. the detector contract itself
# --------------------------------------------------------------------------- #
def test_detectors_return_aligned_score_and_flag():
    """Every detector must return a score and flag aligned to the frame index."""
    sc = intermittent()
    for name, fn in bench.DETECTORS.items():
        s, f = fn(sc["df"], sc["fe"], sc["a"], sc["b"])
        assert len(s) == len(sc["df"]), f"{name}: score length mismatch"
        assert len(f) == len(sc["df"]), f"{name}: flag length mismatch"
        assert str(f.dtype) == "bool", f"{name}: flag dtype {f.dtype} is not bool"


def test_recommended_primary_tier_is_the_benchmark_winner():
    """vat-v1's `moderate` tier must BE D1, not merely resemble it.

    This is the invariant that makes the review's claim ('one function changes') true.
    If someone edits one without the other, this fails.
    """
    from bench import (NEAR_CAP, PERS_GAP, PERS_MOD, REF_ON, THR_MOD, bridge, dq_mask,
                       pair_deficit, persist)
    sc = intermittent()
    _, d1_flag = D1(sc["df"], sc["fe"], sc["a"], sc["b"])
    d = pair_deficit(sc["df"], sc["fe"], sc["a"], sc["b"])
    gate = ((sc["fe"]["ref"] >= REF_ON) & (d["s"] > NEAR_CAP * d["expected"])
            & d["act"] & ~dq_mask(sc["fe"], sc["a"], sc["b"]))
    documented = persist(bridge(gate & (d["deficit"] > THR_MOD), PERS_GAP, within=gate),
                         PERS_MOD, index=sc["df"].index)
    assert bool((d1_flag == documented).all()), \
        "det_rolling no longer matches the documented vat-v1 primary tier"
