"""Why did the published detector's benchmark false positives collapse?

Hypothesis: `satsim.rolling_cap` smooths with

    per_day.rolling(31, center=True)

over the DAILY INDEX AS IT APPEARS. With four months missing from the record, that
"31-day" window spans far more than 31 calendar days near a gap, distorting cap(t) and
therefore expected = g0*cap(t)*ref. The spurious deficit that distortion creates is what
the benchmark was scoring as false positives.

This script measures (a) the calendar span of the window and (b) the resulting phantom
deficit on the control pairs, which cannot saturate and so show model error only.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import bench as B  # noqa: E402
import satsim  # noqa: E402

OLD = satsim.ROOT + r"\combined_may2025_jul2026_area1.csv"
NEW = satsim.ROOT + r"\combined_may2025_aug2026_area1.csv"

CONTROLS = [("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]


def window_span(df, window=31):
    """Calendar days spanned by `rolling(window, center=True)` on the present-day index."""
    days = pd.Series(df.index.normalize().unique()).sort_values().reset_index(drop=True)
    spans = []
    for i in range(len(days)):
        lo = max(0, i - window // 2)
        hi = min(len(days) - 1, i + window // 2)
        spans.append((days[hi] - days[lo]).days)
    return pd.Series(spans, index=days)


def report(csv_path, label):
    satsim.CSV = csv_path
    B._STATE.clear()
    df = B.raw_df()
    fe = B.raw_fe()
    r = fe["ref"]

    sp = window_span(df)
    print("=" * 92)
    print(f"{label}   ({os.path.basename(csv_path)})")
    print(f"  rows {len(df):,}   dates {df.index.normalize().nunique()}   "
          f"span {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  rolling_cap window: nominal 31 days, actual calendar span "
          f"median {sp.median():.0f}, p90 {sp.quantile(.90):.0f}, max {sp.max()}")

    print(f"  {'control pair':14s} {'in-gate rows':>12s} {'fp-like rows':>13s} "
          f"{'median d':>9s} {'p95 d':>8s} {'max d':>8s}")
    for a, b in CONTROLS:
        d = B.detect(df, fe, a, b)
        gate = (r >= B.REF_ON) & (d["s"] > B.NEAR_CAP * d["cap_t"]) & d["act"] \
               & ~B.dq_mask(fe, a, b)
        dev = pd.Series(d["deficit"], index=df.index)[gate]
        # "fp-like" = a control-pair row the published rule would flag: pure model error
        fp_like = int((dev > B.THR_MOD).sum())
        print(f"  {a}+{b:9s} {int(gate.sum()):12d} {fp_like:13d} "
              f"{dev.median():9.3f} {dev.quantile(.95):8.3f} {dev.max():8.3f}")

    # the seasonal mis-scaling, which the retired detector is documented to suffer from
    dec = (r >= 0.5) & (pd.Series(df.index.month, index=df.index) == 12)
    print("  December control-pair deficit at ref>=0.5 (documented mis-scaling):")
    for a, b in CONTROLS:
        d = B.detect(df, fe, a, b)
        v = pd.Series(d["deficit"], index=df.index)[dec & d["act"]]
        print(f"     {a}+{b:9s} n={len(v):4d}  median {v.median():+.3f}")
    print()


if __name__ == "__main__":
    report(OLD, "OLD EXPORT")
    report(NEW, "NEW EXPORT")
