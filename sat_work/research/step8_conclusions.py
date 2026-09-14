"""Does the corrected detector change the published *conclusions*?"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *

f2 = pd.read_csv(r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research\saturation_flags_v2.csv",
                 parse_dates=["time"]).set_index("time")
df = load(); FE = preprocess(df); dayidx = FE["dayidx"]

print("=" * 78)
print("MONTHLY SATURATION HOURS:  published vs vat-v1")
print("=" * 78)
for a, b in PAIRS:
    k = f"{a}_{b}"
    pub = detect(df, FE, a, b)["moderate"]
    vat = f2[f"{k}_sat_moderate"].astype(bool)
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
    vat = f2[f"{k}_sat_moderate"].astype(bool)
    pv = pub[pub].index.month.value_counts(normalize=True).sort_index() * 100
    vv = vat[vat].index.month.value_counts(normalize=True).sort_index() * 100
    t = pd.DataFrame({"published": pv, "vat": vv}).round(1)
    print(f"\n{k}  (% of flagged hours per calendar month)")
    print(t.to_string())
