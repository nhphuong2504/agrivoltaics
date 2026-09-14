"""
STEP 13 - the one piece of the July gap that is the detector's own fault.

step10 section C left 74 disputed rows unexplained. They are rows where vat-v1's
deficit DID exceed 0.15 but the strict "3 consecutive samples" rule broke the run -
i.e. real signal discarded by a brittle persistence test.

Test: bridge runs separated by a short gap (<= g samples) before applying the run
length rule. Measure hours recovered on the shared pairs and new false positives on
the control pairs, which cannot saturate.

Run:  ./venv/Scripts/python.exe sat_work/research/step13_persistence.py
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *

pd.set_option("display.width", 200)
df = load(); FE = preprocess(df); ref = FE["ref"]
FULLCAL = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
BAND = (0.35, 0.60)


def broadcast(per_day, dayidx, labels, window="31D", minp=5):
    roll = per_day.reindex(FULLCAL).rolling(window, center=True, min_periods=minp).median()
    return pd.Series(roll.reindex(dayidx).to_numpy(), index=labels)


def theta(s, act):
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= BAND[0]) & (ref <= BAND[1]) & u.notna()
    return broadcast(u[sel].groupby(FE["dayidx"][sel]).median(), FE["dayidx"], s.index)


def bridge(raw, g):
    """Fill False gaps of length <= g that sit between two True samples."""
    r = raw.copy()
    for k in range(1, g + 1):
        r = r | (raw.shift(k, fill_value=False) & raw.shift(-k, fill_value=False))
    return r


def flags(a, b):
    s = df[a] + df[b]
    act = FE["active"][a] & FE["active"][b]
    d = (1 - s / (theta(s, act) * ref)).replace([np.inf, -np.inf], np.nan)
    base = (ref >= 0.70) & (s > 0.6 * theta(s, act) * ref) & act & ~dq_mask(FE, a, b)
    raw = base & (d > 0.15)
    # NOTE: filled samples are restricted to the gate domain, matching the adopted rule in
    # bench.bridge(within=...). Without that restriction a fill can land below ref 0.70 and
    # break the export contract. The ungated variant reported here previously overstated
    # both the gain (+203.2 h) and the cost (+18 control rows); see step18 for the adopted
    # numbers, which supersede this script's decision.
    return {g: (bridge(raw, g) & base & (run_lengths(bridge(raw, g) & base) >= 3))
            for g in (0, 1, 2, 3)}


print("=" * 104)
print("EFFECT OF GAP-TOLERANT PERSISTENCE   (moderate tier, 3 samples, bridge gaps <= g)")
print("=" * 104)
print("SHARED pairs - extra hours recovered is real saturation the strict rule threw away")
rows = []
for a, b in PAIRS:
    f = flags(a, b)
    for g in (0, 1, 2, 3):
        rows.append(dict(pair=f"{a}+{b}", gap=g, hours=f[g].sum() / 12,
                         recovered_h=(f[g].sum() - f[0].sum()) / 12,
                         days=f[g][f[g]].index.normalize().nunique()))
S = pd.DataFrame(rows)
print(S.pivot_table(index="pair", columns="gap", values=["hours", "recovered_h", "days"]).round(1).to_string())
print(f"\n   recovered across the three shared pairs: "
      f"g=1 {S[S.gap==1].recovered_h.sum():.1f} h, "
      f"g=2 {S[S.gap==2].recovered_h.sum():.1f} h, "
      f"g=3 {S[S.gap==3].recovered_h.sum():.1f} h")

print()
print("CONTROL pairs - EVERY extra row here is a false positive, so this is the cost")
rows = []
for a, b in CONTROL_PAIRS:
    f = flags(a, b)
    for g in (0, 1, 2, 3):
        rows.append(dict(pair=f"{a}+{b}", gap=g, rows=int(f[g].sum()),
                         extra_rows=int(f[g].sum() - f[0].sum())))
Cc = pd.DataFrame(rows)
print(Cc.pivot_table(index="pair", columns="gap", values=["rows", "extra_rows"]).astype(int).to_string())
print(f"\n   total new false-positive rows: g=1 {Cc[Cc.gap==1].extra_rows.sum()}, "
      f"g=2 {Cc[Cc.gap==2].extra_rows.sum()}, g=3 {Cc[Cc.gap==3].extra_rows.sum()}")

print()
print("=" * 104)
print("VERDICT")
print("=" * 104)
rec1, fp1 = S[S.gap == 1].recovered_h.sum(), Cc[Cc.gap == 1].extra_rows.sum()
rec2, fp2 = S[S.gap == 2].recovered_h.sum(), Cc[Cc.gap == 2].extra_rows.sum()
print(f"   g=1 bridges 10-minute dropouts: +{rec1:.1f} h of shared-pair signal "
      f"for +{fp1} control-pair rows")
print(f"   g=2 bridges 15-minute dropouts: +{rec2:.1f} h of shared-pair signal "
      f"for +{fp2} control-pair rows")
print("\n   A gap-tolerant rule is only worth adopting if the control-pair cost stays at")
print("   zero or near zero.")
