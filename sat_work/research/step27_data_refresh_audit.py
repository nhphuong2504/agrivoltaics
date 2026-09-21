"""Compare the old and new combined exports month by month."""
from __future__ import annotations

import pandas as pd

ROOT = r"c:\Users\nhphuong\Desktop\Solar\all_data"
OLD = ROOT + r"\combined_may2025_jul2026_area1.csv"
NEW = ROOT + r"\combined_may2025_aug2026_area1.csv"

old = pd.read_csv(OLD, parse_dates=["time"]).set_index("time").sort_index()
new = pd.read_csv(NEW, parse_dates=["time"]).set_index("time").sort_index()


def ym(ix):
    return pd.Series([f"{d.year}-{d.month:02d}" for d in ix], index=ix)


o = ym(old.index).value_counts()
n = ym(new.index).value_counts()

print("month      old rows  new rows     delta")
print("-" * 44)
for m in sorted(set(o.index) | set(n.index)):
    a, b = int(o.get(m, 0)), int(n.get(m, 0))
    tag = ""
    if a == 0:
        tag = "   <== ENTIRELY MISSING before"
    elif b != a:
        tag = f"   (+{b - a})"
    print(f"{m:9s} {a:9d} {b:9d} {b - a:9d}{tag}")

print()
absent_old = 451 - len(set(old.index.date))
cal_new = (pd.Timestamp("2026-08-31").date() - pd.Timestamp("2025-05-01").date()).days + 1
absent_new = cal_new - len(set(new.index.date))
print(f"calendar days old : 451   dates present {len(set(old.index.date))}   absent {absent_old}")
print(f"calendar days new : {cal_new}   dates present {len(set(new.index.date))}   absent {absent_new}")
