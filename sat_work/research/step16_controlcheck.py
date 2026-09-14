"""
STEP 16 - is eu_15+eu_23 really an independent control, or is something wrong with it?

Section 3.12 of the review flagged this: eu_15+eu_23 is used as a pair that "cannot
saturate", yet both detectors flag ~107 rows on it, against 16 and 0 on the other two
control pairs. That matters because almost every ground-truth claim in the review rests
on "control pairs cannot saturate":
  * the 3,356 kWh bias floor
  * D5's empirical null (it is built FROM these pairs)
  * the peer-ratio truth normalisation

Two hypotheses:

  (a) The pair is not independent (a shared MPPT, or one unit mirroring the other), so a
      shared limit is physically possible and the flags are real.
  (b) One or both units is individually degraded, so the pair sum rolls off at high
      irradiance for a unit-level reason. The flags are then real but the pair is a bad
      control: it cannot distinguish a shared limit from a sick panel.

Both are testable. (b) is visible per UNIT: a degraded unit shows the same high-irradiance
roll-off on its own that a clipped pair shows in sum. This script computes that roll-off
for EVERY unit, which also gives a fleet health overview worth having.

Run:  ./venv/Scripts/python.exe sat_work/research/step16_controlcheck.py
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *
import bench as B

pd.set_option("display.width", 200)
df = B.raw_df()
FE = B.raw_fe()
ref = FE["ref"]
X = df[EU].div(df[EU].quantile(0.999))
mo = df.index.to_period("M").astype(str)

print("=" * 104)
print("A. PER-UNIT HIGH-IRRADIANCE ROLL-OFF  (a shared-MPPT signature at the level of ONE unit)")
print("=" * 104)
print("For each unit on its own: theta_u from the mid band, then deficit_u = 1 - X_u/(theta_u*ref).")
print("A healthy unit sits near 0 at high ref. The six shared units are marked *.\n")
rows = []
for u in EU:
    s = df[u]
    act = FE["active"][u]
    th = B.theta_roll(s, ref, act, FE["dayidx"])
    d = (1 - s / (th * ref)).replace([np.inf, -np.inf], np.nan)
    clean = act & ~FE["unit_off"][u] & ~FE["stale"][u] & ~FE["negative"][u] & ~FE["site_outage"]
    hi = clean & (ref >= 0.75)
    mid = clean & (ref >= 0.35) & (ref <= 0.60)
    rows.append(dict(unit=u, shared="*" if u in SHARED else "",
                     n_hi=int(hi.sum()),
                     deficit_hi=round(float(d[hi].median()), 3) if hi.sum() else np.nan,
                     theta_mid=round(float(th[mid].median()), 1) if mid.sum() else np.nan))
T = pd.DataFrame(rows).set_index("unit")
print(T.to_string())

print()
print("ranked by high-irradiance deficit  (top = the units that look constrained):")
print(T.sort_values("deficit_hi", ascending=False).head(12).to_string())

print()
print("=" * 104)
print("B. THE SPECIFIC QUESTION: does eu_15 or eu_23 roll off on its own?")
print("=" * 104)
for u in ["eu_15", "eu_23", "eu_1", "eu_3", "eu_4", "eu_12"]:
    d15 = T.loc[u, "deficit_hi"]
    print(f"   {u}: deficit at ref>=0.75 = {d15:+.3f}   "
          f"({'SUSPICIOUS' if d15 > 0.05 else 'healthy'})")
print()
print("   for reference, the six units inside the shared pairs:")
for u in SHARED:
    print(f"   {u}: deficit at ref>=0.75 = {T.loc[u, 'deficit_hi']:+.3f}")

# --------------------------------------------------------------------------- #
print()
print("=" * 104)
print("C. DO eu_15+eu_23's FLAGS CO-OCCUR WITH THE SHARED PAIRS' FLAGS?")
print("=" * 104)
print("If they fire on the same timestamps, the cause is shared (site-wide or seasonal).")
print("If they fire on different days, something is specific to that pair.\n")
_, flag_ctl = B.det_rolling(df, FE, "eu_15", "eu_23")
shared_flags = {f"{a}+{b}": B.det_rolling(df, FE, a, b)[1] for a, b in PAIRS}
any_shared = pd.concat(shared_flags.values(), axis=1).any(axis=1)
sub = pd.DataFrame({"ctl": flag_ctl, "shared": any_shared})[flag_ctl | any_shared]
tab = pd.crosstab(sub["ctl"], sub["shared"])
tab.index = ["control_only(0)", "control_flagged(1)"]
tab.columns = ["shared_clean(0)", "shared_flagged(1)"]
print(tab.to_string())
only_ctl = int((flag_ctl & ~any_shared).sum())
both = int((flag_ctl & any_shared).sum())
print(f"\n   control rows flagged: {int(flag_ctl.sum())}  "
      f"(with a shared-pair flag: {both}, on their own: {only_ctl})")
if flag_ctl.sum():
    print(f"   shared-pair flag present on {100*both/max(int(flag_ctl.sum()),1):.0f}% of them")
print(f"   base rate of a shared-pair flag in the whole record: {100*any_shared.mean():.1f}%")

# --------------------------------------------------------------------------- #
print()
print("=" * 104)
print("D. IS THE PAIR INDEPENDENT?  spread / co-movement diagnostics")
print("=" * 104)
print("Units on one MPPT typically move together more tightly than independent units.\n")


def corr_stats(a, b, label):
    x, y = X[a].to_numpy(), X[b].to_numpy()
    ok = np.isfinite(x) & np.isfinite(y) & (ref.to_numpy() > 0.5)
    r_lev = np.corrcoef(x[ok], y[ok])[0, 1]
    dx, dy = np.diff(x[ok]), np.diff(y[ok])
    r_dif = np.corrcoef(dx, dy)[0, 1]
    ratio = pd.Series(x[ok] / np.where(y[ok] > 0, y[ok], np.nan))
    print(f"   {label:22s} corr(level)={r_lev:.4f}  corr(diff)={r_dif:.4f}  "
          f"ratio cv={ratio.std()/ratio.mean():.3f}")
    return r_lev, r_dif


pairs_to_test = [("eu_15", "eu_23", "CONTROL eu_15+eu_23"),
                 ("eu_1", "eu_3", "CONTROL eu_1+eu_3"),
                 ("eu_4", "eu_12", "CONTROL eu_4+eu_12"),
                 ("eu_8", "eu_16", "SHARED eu_8+eu_16"),
                 ("eu_10", "eu_18", "SHARED eu_10+eu_18"),
                 ("eu_15", "eu_1", "cross: eu_15+eu_1"),
                 ("eu_23", "eu_12", "cross: eu_23+eu_12"),
                 ("eu_23", "eu_1", "cross: eu_23+eu_1")]
res = [corr_stats(a, b, lab) for a, b, lab in pairs_to_test]

print()
print("=" * 104)
print("E. WHERE IN THE YEAR DOES THE CONTROL PAIR FIRE?")
print("=" * 104)
fdf = pd.DataFrame({"m": mo, "f": flag_ctl})
by_m = fdf[fdf.f].groupby("m").size()
print(by_m.to_string() if len(by_m) else "   (no flags)")
print(f"\n   ref range of the flagged rows: "
      f"{ref[flag_ctl].min():.2f} to {ref[flag_ctl].max():.2f}")
s_col = B.pair_deficit(df, FE, "eu_15", "eu_23")["deficit"]
print(f"   deficit on those rows: min {s_col[flag_ctl].min():.3f} "
      f"median {s_col[flag_ctl].median():.3f} max {s_col[flag_ctl].max():.3f}")
print("   (shallow values hugging the 0.15 threshold mean a calibration boundary,")
print("    not a hard constraint.)")
