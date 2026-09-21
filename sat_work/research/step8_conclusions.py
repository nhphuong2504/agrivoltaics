"""Does the corrected detector change the published *conclusions*?

Compares the published detector against the CANONICAL artefact (repo-root
`saturation_flags.csv`, vat-v1 since 2026-09). The canonical file is written by
`recommended.py`; the published method is reproduced live from `satsim.detect`, so this
script never depends on a retired copy of the old output.
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
from satsim import *

CANONICAL = os.path.join(_ROOT, "saturation_flags.csv")
f2 = pd.read_csv(CANONICAL, parse_dates=["time"]).set_index("time")
df = load(); FE = preprocess(df); dayidx = FE["dayidx"]

print("=" * 78)
print("MONTHLY SATURATION HOURS:  published vs vat-v1")
print("=" * 78)
for a, b in PAIRS:
    k = f"{a}_{b}"
    pub = detect(df, FE, a, b)["moderate"]
    vat = f2[f"{k}_sat"].astype(bool)
    m = pd.DataFrame({"published": pub.groupby(dayidx).sum() / 12,
                      "vat": vat.groupby(dayidx).sum() / 12}).resample("1MS").sum().round(1)
    m.index = m.index.strftime("%Y-%m")
    print("\n" + k)
    print(m.to_string())
    pd_ = pub.groupby(dayidx).sum(); vd_ = vat.groupby(dayidx).sum()
    print(f"  worst single-day disagreement: {abs(pd_ - vd_).max()/12:.1f} h  "
          f"(max published day {pd_.max()/12:.1f} h, vat {vd_.max()/12:.1f} h)")
    pdset, vdset = set(pub[pub].index.normalize()), set(vat[vat].index.normalize())
    print(f"  affected days: published {len(pdset)}, vat {len(vdset)}, "
          f"overlap {len(pdset & vdset)}, published-only {len(pdset - vdset)}, vat-only {len(vdset - pdset)}")

print()
print("=" * 78)
print("SEASONAL SHAPE (both detectors): share of hours falling in each month")
print("=" * 78)
for a, b in PAIRS:
    k = f"{a}_{b}"
    pub = detect(df, FE, a, b)["moderate"]
    vat = f2[f"{k}_sat"].astype(bool)
    pv = pub[pub].index.month.value_counts(normalize=True).sort_index() * 100
    vv = vat[vat].index.month.value_counts(normalize=True).sort_index() * 100
    t = pd.DataFrame({"published": pv, "vat": vv}).round(1)
    print(f"\n{k}  (% of flagged hours per calendar month)")
    print(t.to_string())
