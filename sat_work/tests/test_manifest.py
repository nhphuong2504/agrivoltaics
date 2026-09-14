"""Regression tests for the canonical version lock.

`saturation_flags.csv` is gitignored, so the *manifest* is the committed record of what the
canonical artefact contained. If these fail, either the artefact was regenerated with a
different code path, or a headline number moved without the manifest being updated.

Deliberately NOT asserted here: the sha256. A byte hash depends on the pandas version that
wrote the CSV, so pinning it would make the suite fail on a dependency bump while the
numbers were identical. The hash is provenance; the metrics are the contract.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402
import canon_metrics as CM  # noqa: E402
from _helpers import CANONICAL_CSV  # noqa: E402

MANIFEST = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "sat_work", "canonical", "CANONICAL_vat-v1.json")


def _locked():
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


def test_manifest_exists_and_names_the_live_artefact():
    """A lock that points somewhere else is not a lock."""
    assert os.path.exists(MANIFEST), f"missing version lock {MANIFEST}"
    m = _locked()
    assert m["version"] == "vat-v1", "the canonical version string must be vat-v1"
    assert m["artefact"]["path"] == "saturation_flags.csv"
    assert m["artefact"]["rows"] == len(CM.load_canonical())
    assert os.path.abspath(CANONICAL_CSV) == os.path.abspath(CM.CANONICAL_CSV)


def test_live_artefact_still_matches_the_locked_headline_metrics():
    """The substance of the lock: every headline number, re-derived from the live file.

    This is what makes the numbers quoted in SATURATION_REVIEW.md, the canvas and the deck
    checkable rather than merely asserted: they come from `canon_metrics.summarise()`, and
    this test proves that summary still describes the artefact on disk.
    """
    m = _locked()
    live = CM.summarise()
    for group in ("flagged", "controls", "energy", "domain_rows",
                  "deficit_in_domain_range", "benchmark"):
        assert m["metrics"][group] == live[group], (
            f"{group} drifted from the manifest:\n  locked {m['metrics'][group]}\n"
            f"  live   {live[group]}")


def test_locked_method_configuration_matches_the_code():
    """The manifest must describe the code that exists, not a remembered configuration.

    This is the guard against a silent re-tune: changing REF_ON, a deficit threshold or the
    persistence rule in bench.py makes this fail until the manifest is regenerated, which
    forces the change to be deliberate and visible.
    """
    m = _locked()
    assert m["method"] == CM.method(), "bench.py parameters no longer match the lock"
    # and spot-check the values that define the frozen method
    assert m["method"]["ref_on"] == bench.REF_ON == 0.70
    assert m["method"]["deficit_threshold_moderate"] == bench.THR_MOD == 0.15
    assert m["method"]["deficit_threshold_severe"] == bench.THR_SEV == 0.30
    assert m["method"]["persistence_bridging_samples"] == bench.PERS_GAP
    assert m["method"]["persistence_bridging_samples_severe"] == bench.PERS_GAP_SEV
    assert m["method"]["excluded_units"] == ["eu_7"]


def test_manifest_defines_its_denominators():
    """Every rate must ship with the set it is a rate over."""
    d = _locked()["definitions"]
    for key in ("domain", "energy_gate", "hours", "flagged_rows", "controls", "bias_floor"):
        assert d.get(key), f"definition {key} is missing from the lock"
    assert "evaluate()" in d["domain"], "the domain must be tied to the AUC test set"


def test_the_lock_is_not_vacuous():
    """A manifest that records zeros would 'pass' every equality check above."""
    m = _locked()["metrics"]
    assert m["flagged"]["published_rows"] > 10_000
    assert m["flagged"]["canonical_rows"] > 10_000
    assert 0 < m["energy"]["shared_canonical_kwh"]
    assert m["benchmark"]["episodic_fp_published"] > 1_000
    assert all(v > 0 for v in m["domain_rows"].values())
    assert m["deficit_in_domain_range"], "in-domain range must be recorded per pair"
