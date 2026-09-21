"""Attribute the benchmark change: day-selection vs data completeness.

`bench.scenario` picks its clipped days with

    days   = <every day present in the record>
    day_on = rng(seed=7).random(len(days)) < duty

so the SAME seed and duty select a DIFFERENT set of days once the record changes. The
87x -> 1.6x movement could therefore be (a) pure scenario re-roll, (b) the completed
record changing cap(t), or (c) both.

Decisive test: inject on the OLD day set while running on the NEW data.
  * if the published FP returns to ~10,200 -> the number was pure scenario re-roll
  * if it stays near 300             -> the completed record genuinely recalibrated it
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
PAIRS = (("eu_1", "eu_3"), ("eu_15", "eu_23"))


def load_df(csv_path):
    satsim.CSV = csv_path
    B._STATE.clear()
    return B.raw_df()


def old_day_on(base, a, duty=0.35, seed=7):
    """Reproduce the day selection the OLD record produced, as a set of dates."""
    days = base[a].groupby(base.index.normalize()).count().pipe(lambda x: x[x > 0]).index
    r = pd.Series(np.random.default_rng(seed).random(len(days)) < duty, index=days)
    return set(r[r].index.normalize())


def build(base, pair, day_dates, q=0.75):
    """A scenario on `base` with exactly `day_dates` clipped."""
    a, b = pair
    fe0 = B.preprocess(base, [c for c in B.RELIABLE if c not in (a, b)])
    Ptyp = np.nanpercentile((base[a] + base[b])[fe0["ref"] >= 0.85], 95)
    C = q * Ptyp
    dayidx = base.index.normalize()
    day_on = pd.Series(dayidx.isin(day_dates), index=base.index).to_numpy()
    d2, gt = B.inject(base, a, b, C, day_on)
    fe = B.preprocess(d2, [c for c in B.RELIABLE if c not in (a, b)])
    th_true = B.theta_roll(base[a] + base[b], fe0["ref"], fe0["active"][a] & fe0["active"][b],
                           fe0["dayidx"], q=0.5)
    with np.errstate(invalid="ignore", divide="ignore"):
        sev = np.clip(1 - C / (th_true * fe0["ref"]), 0, 1)
    return dict(pair=pair, a=a, b=b, df=d2, fe=fe, gt=gt,
                sev=pd.Series(sev, index=base.index).fillna(0),
                C=C, fe0=fe0, ref0=fe0["ref"])


if __name__ == "__main__":
    old_base = load_df(OLD)
    old_days = {p: old_day_on(old_base, p[0]) for p in PAIRS}
    old_rows = dict(n=len(old_base), dates=int(old_base.index.normalize().nunique()))

    new_base = load_df(NEW)
    new_rows = dict(n=len(new_base), dates=int(new_base.index.normalize().nunique()))

    print("=" * 96)
    print("benchmark FP: published (D0) vs canonical (D1), pooled over the 2 episodic pairs")
    print("=" * 96)

    for tag, base, days in (("A. OLD data, OLD day set", old_base, old_days),
                            ("B. NEW data, NEW day set", new_base, None),
                            ("C. NEW data, OLD day set", new_base, old_days)):
        satsim.CSV = OLD if tag.startswith("A") else NEW
        B._STATE.clear()
        f0 = f1 = 0
        for pair in PAIRS:
            if days is None:
                sc = B.scenario(pair, duty=0.35, q=0.75, pattern="episodic")
            else:
                sc = build(base, pair, days[pair])
            f0 += B.score_scenario(sc, B.det_published)["fp_rows"]
            f1 += B.score_scenario(sc, B.det_rolling)["fp_rows"]
        print(f"  {tag:26s}  D0 fp {f0:6d}   D1 fp {f1:5d}   ratio {f0 / max(f1, 1):6.1f}x")

    print()
    print(f"  old record: {old_rows['n']:,} rows / {old_rows['dates']} dates")
    print(f"  new record: {new_rows['n']:,} rows / {new_rows['dates']} dates")
