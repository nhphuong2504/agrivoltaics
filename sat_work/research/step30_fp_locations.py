"""Where are the published detector's benchmark false positives, old vs new?

`rolling_cap` stretches to ~91 calendar days across the four missing months in the old
export (measured in step29). Those stretched windows sit next to the gaps. If the old
scenario's false positives cluster in the months flanking a gap, then the 10,202-row
figure the review quotes was an artefact of the incomplete record rather than a property
of the published detector.
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

GAP_MONTHS = {"2025-08", "2025-09", "2026-03", "2026-04"}


def ym(ix):
    return pd.Series([f"{d.year}-{d.month:02d}" for d in ix], index=ix)


def report(csv_path, label):
    satsim.CSV = csv_path
    B._STATE.clear()
    df = B.raw_df()
    print("=" * 92)
    print(f"{label}   ({os.path.basename(csv_path)})")
    print("=" * 92)
    for pair in (("eu_1", "eu_3"), ("eu_15", "eu_23")):
        sc = B.scenario(pair, duty=0.35, q=0.75, pattern="episodic")
        score, flag = B.det_published(sc["df"], sc["fe"], sc["a"], sc["b"])
        fe, gt = sc["fe"], sc["gt"]
        act = fe["active"][sc["a"]] & fe["active"][sc["b"]]
        test = (act & (fe["ref"] >= B.REF_ON) & ~B.dq_mask(fe, sc["a"], sc["b"])).to_numpy()
        # a detector is only scored where its score is finite, exactly as evaluate() does
        ok = test & np.isfinite(np.asarray(score, dtype=float))
        fl = np.asarray(flag, dtype=bool) & ok
        fp = pd.Series(fl & ~gt.to_numpy(), index=sc["df"].index)
        fp = fp[fp]
        tot = len(fp)
        print(f"\n  {pair[0]}+{pair[1]}: {tot} fp rows")
        if tot:
            per = ym(fp.index).value_counts().sort_index()
            print("    by month:")
            for m, c in per.items():
                tag = "   <-- a month the OLD export was missing" if m in GAP_MONTHS else ""
                print(f"      {m}: {c:6d}{tag}")
    print()


if __name__ == "__main__":
    report(OLD, "OLD EXPORT")
    report(NEW, "NEW EXPORT")
