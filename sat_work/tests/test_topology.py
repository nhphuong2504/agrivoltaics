"""
Fleet-topology contracts.

The whole saturation analysis rests on which units share a tracker and which are
independent. That used to be inferred from the data; it is now recorded from the site
engineer's mapping. These tests enforce the recording, so a future edit cannot quietly
put a shared unit back into a control pair, or a faulty unit back into the reference.

`eu_7` is confirmed faulty by the site engineer (2026-09) and must be ignored everywhere:
it is not in the reference proxy, not in any pair, and not in any control.
"""
from __future__ import annotations

import numpy as np
import satsim
from satsim import (EU, PAIRS, SHARED, CONTROL_PAIRS, RELIABLE, INVERTERS, INV_OF,
                    MPPT_CHANNELS)
import bench


def test_topology_partitions_every_unit():
    """The inverter map must cover all 24 units exactly once, with no strays."""
    listed = [u for units in INVERTERS.values() for u in units]
    assert len(listed) == len(set(listed)), f"a unit appears on two inverters: {listed}"
    assert set(listed) == set(EU), (
        f"inverter map does not cover the fleet; "
        f"missing {sorted(set(EU) - set(listed))}, extra {sorted(set(listed) - set(EU))}")
    assert set(INV_OF) == set(EU)


def test_recorded_topology_agrees_with_the_pairs_in_use():
    """The recorded MPPT channels must be exactly the pairs the analysis uses."""
    assert set(MPPT_CHANNELS) == set(PAIRS), (
        f"MPPT_CHANNELS {MPPT_CHANNELS} != PAIRS {PAIRS}")
    assert sorted(SHARED) == sorted(u for p in PAIRS for u in p)


def test_shared_tracker_implies_shared_inverter():
    """Physical consistency: two units on one MPPT channel sit on one inverter."""
    for a, b in MPPT_CHANNELS:
        assert INV_OF[a] == INV_OF[b], (
            f"{a}+{b} is a recorded MPPT channel but spans inverters "
            f"{INV_OF[a]}/{INV_OF[b]}")


def test_eu_7_is_excluded_everywhere():
    """eu_7 has a confirmed technical fault: it must not appear in any analysis role."""
    assert "eu_7" not in SHARED
    assert all("eu_7" not in p for p in PAIRS)
    assert all("eu_7" not in p for p in CONTROL_PAIRS)
    assert "eu_7" not in RELIABLE, "eu_7 must not help define the irradiance proxy"
    assert "eu_7" not in [u for p in MPPT_CHANNELS for u in p]
    # it may still appear in the topology map itself - it is a real, mapped unit
    assert INV_OF["eu_7"] == 5


def test_reference_is_invariant_to_eu_7():
    """The strongest form of 'ignore eu_7': corrupt it and `ref` must not move.

    `ref` is the median of the RELIABLE units, so a faulty eu_7 cannot leak into the
    irradiance proxy, the site-outage rule, or any downstream detector.
    """
    df = bench.raw_df()
    base = satsim.preprocess(df)["ref"]
    corrupted = df.copy()
    corrupted["eu_7"] = 12345.0                      # absurd but finite
    ref2 = satsim.preprocess(corrupted)["ref"]
    assert np.array_equal(base.to_numpy(), ref2.to_numpy()), (
        "changing eu_7 moved the irradiance proxy - it has leaked into the reference")
    outage = satsim.preprocess(corrupted)
    assert np.array_equal(
        satsim.preprocess(df)["site_outage"].to_numpy(),
        outage["site_outage"].to_numpy()), "eu_7 leaked into the site-outage rule"


def test_controls_are_independent_units():
    """A control pair must contain no shared unit, or every flag on it is meaningless."""
    for a, b in CONTROL_PAIRS:
        assert a not in SHARED and b not in SHARED, f"{a}+{b} contains a shared unit"
    assert len(set(CONTROL_PAIRS)) == len(CONTROL_PAIRS), "duplicate control pair"


def test_at_least_one_cross_inverter_control_exists():
    """Flag-level nulls need a pair that shares neither tracker nor inverter.

    Same-inverter independent pairs carry a co-movement common mode and are ~2x more
    likely to trip the threshold (see step17_groundtruth.py), so at least one
    cross-inverter control must remain available for flag-level comparisons.
    """
    cross = [(a, b) for a, b in CONTROL_PAIRS if INV_OF[a] != INV_OF[b]]
    assert cross, (
        "every control pair shares an inverter; there is no clean flag-level null. "
        f"controls={CONTROL_PAIRS}, inverters={[INV_OF[a] for a, b in CONTROL_PAIRS]}")
    assert ("eu_1", "eu_3") in cross, "the known cross-inverter control has gone"
