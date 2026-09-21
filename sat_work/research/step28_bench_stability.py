"""Does the clip-injection benchmark survive a data refresh?

The scenarios pick their clipped days with

    days   = <every day present in the record>
    day_on = rng(seed=7).random(len(days)) < duty

so the selected day set is a function of the record length as well as the seed. Adding
four months changes len(days) from ~328 to ~487, which re-rolls the whole selection. This
script measures how much of the published-vs-canonical benchmark contrast is therefore a
property of the detector and how much is a property of which days happened to be picked.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bench as B  # noqa: E402
import satsim  # noqa: E402

OLD = satsim.ROOT + r"\combined_may2025_jul2026_area1.csv"
NEW = satsim.ROOT + r"\combined_may2025_aug2026_area1.csv"

PAIRS = (("eu_1", "eu_3"), ("eu_15", "eu_23"))


def run(csv_path, label):
    satsim.CSV = csv_path
    B._STATE.clear()
    df = B.raw_df()
    print("=" * 96)
    print(f"{label}")
    print(f"  csv        : {os.path.basename(csv_path)}")
    print(f"  record     : {len(df):,} rows, {df.index.min()} -> {df.index.max()}")
    print(f"  dates      : {df.index.normalize().nunique()}")
    print("-" * 96)
    print(f"  {'scenario':22s} {'days_on':>8s} {'D0 fp':>8s} {'D1 fp':>8s} "
          f"{'D0 auc':>7s} {'D1 auc':>7s}")
    tots = {"D0": 0, "D1": 0}
    for pair in PAIRS:
        for label2, kw in (("episodic", dict(duty=0.35, q=0.75, pattern="episodic")),
                           ("chronic", dict(duty=1.00, q=0.75, pattern="chronic"))):
            sc = B.scenario(pair, **kw)
            s0 = B.score_scenario(sc, B.det_published)
            s1 = B.score_scenario(sc, B.det_rolling)
            ndays = int(sc["gt"].groupby(sc["gt"].index.normalize()).any().sum())
            name = f"{pair[0]}+{pair[1][-2:]} {label2}"
            print(f"  {name:22s} {ndays:8d} {s0['fp_rows']:8d} {s1['fp_rows']:8d} "
                  f"{s0['auc00']:7.3f} {s1['auc00']:7.3f}")
            if label2 == "episodic":
                tots["D0"] += s0["fp_rows"]
                tots["D1"] += s1["fp_rows"]
    print("-" * 96)
    print(f"  pooled episodic: D0 {tots['D0']} fp rows, D1 {tots['D1']} fp rows, "
          f"ratio {tots['D0'] / max(tots['D1'], 1):.1f}x")
    return tots


if __name__ == "__main__":
    old = run(OLD, "OLD EXPORT (through 2026-07-25)")
    print()
    new = run(NEW, "NEW EXPORT (through 2026-08-31)")
    print()
    print("=" * 96)
    print("summary")
    print("=" * 96)
    print(f"  published (D0) pooled episodic FP : {old['D0']}  ->  {new['D0']}")
    print(f"  canonical (D1) pooled episodic FP : {old['D1']}  ->  {new['D1']}")
    print(f"  specificity ratio                 : {old['D0'] / max(old['D1'], 1):.1f}x  ->  "
          f"{new['D0'] / max(new['D1'], 1):.1f}x")
