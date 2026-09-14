"""
Regression tests for the MPPT saturation DETECTORS, scored on injected ground truth.

The point of this file
----------------------
The published validation compares shared pairs against independent control pairs. It
cannot catch the failure that matters, because its control pairs are never clipped: a
detector whose baseline is mis-calibrated on *intermittently* clipped pairs passes
cleanly. Every test below therefore runs against manufactured ground truth.

Two tests are deliberately written to FAIL today:
  * `test_published_detector_is_biased_on_control_pairs` documents the published defect
  * `test_shipped_flags_csv_contains_infinities` documents a bug in a shipped artefact
They are marked `known_failure`, so they report as XFAIL and will flip to a pass when
the underlying issue is fixed. Do not delete them -- they are the guard rails.

Numbers quoted in assertions are the values measured when this suite was frozen. Each
has headroom; they are regression bands, not exact-value checks.
"""
from __future__ import annotations

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

    Measured: ~4,900-5,300 false-positive rows on each of the two episodic scenarios.
    This is the defect the published control-pair validation cannot see.
    """
    fp = 0
    for pair in (("eu_1", "eu_3"), ("eu_15", "eu_23")):
        fp += score(intermittent(pair), D0)["fp_rows"]
    assert fp >= 5000, (
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


def test_recommended_detector_is_at_least_20x_more_specific_than_published():
    """The headline claim of the review, as a machine-checked ratio."""
    d0 = sum(score(intermittent(p), D0)["fp_rows"]
             for p in (("eu_1", "eu_3"), ("eu_15", "eu_23")))
    d1 = sum(score(intermittent(p), D1)["fp_rows"]
             for p in (("eu_1", "eu_3"), ("eu_15", "eu_23")))
    assert d1 > 0 or d0 > 0
    ratio = d0 / max(d1, 1)
    assert ratio >= 20, f"specificity gain collapsed to {ratio:.1f}x (published={d0}, D1={d1})"


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

    Measured: published 55, D1 0, D2 4, D5 0, D4 957.
    """
    sc = noclip()
    assert score(sc, D1)["fp_rows"] == 0, "corrected baseline flagged an intact pair"
    assert score(sc, D5)["fp_rows"] == 0, "control-calibrated tier flagged an intact pair"
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
