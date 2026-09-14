"""
STEP 21 - the canonical artefact's headline numbers, printed from the ONE source of truth.

`saturation_flags.csv` became the vat-v1 export in 2026-09 (recommended.py writes it; the
old published output is retired rather than kept as a parallel file). This script prints the
numbers the review (section 5), the canvas and the deck quote.

Every number here comes from `canon_metrics.summarise()`, which is also what the committed
version lock (`sat_work/canonical/CANONICAL_vat-v1.json`) records and what
`tests/test_manifest.py` re-derives from the live file. Nothing in this script recomputes a
headline figure independently, because that is how two documents end up disagreeing.

It also reports the data-product properties that make the file safe to publish, all enforced
by test_exports.py:
  * no infinities anywhere
  * `deficit` is NaN outside the decision domain, so an unfiltered .mean() means what a
    consumer will assume it means
  * inside the domain `deficit` is physically bounded (-1 <= d <= 1)

Run:  ./venv/Scripts/python.exe sat_work/research/step21_canonical_summary.py
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import bench as B
import canon_metrics as CM

CANON = CM.CANONICAL_CSV
df = B.raw_df(); fe = B.raw_fe()
can = CM.load_canonical()
S = CM.summarise(can)
pd.set_option("display.width", 200)
F, E, CT, BM = S["flagged"], S["energy"], S["controls"], S["benchmark"]

# --------------------------------------------------------------------------- #
print("=" * 100)
print("A. THE CANONICAL ARTEFACT")
print("=" * 100)
num = can.select_dtypes("number")
print(f"   {CANON}")
print(f"   {len(can):,} rows x {can.shape[1]} cols   "
      f"infinite values {int(np.isinf(num.to_numpy()).sum())}")
print(f"   sha256 {CM.sha256()}")
print(f"   definition of the decision domain: {CM.DOMAIN_DEF}")
print("\n   deficit column inside the domain, and the leak outside it:")
print(f"      {'pair':14s} {'n in domain':>12s} {'min':>9s} {'max':>7s}")
for a, b in B.PAIRS:
    k = f"{a}_{b}"
    lo, hi = S["deficit_in_domain_range"][k]
    print(f"      {a}+{b:6s} {S['domain_rows'][k]:12,} {lo:+9.4f} {hi:+7.4f}")
print(f"      populated values outside the domain: {S['deficit_values_outside_domain']}"
      f"   <- must be 0")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("B. FLAGGED HOURS: published vs canonical (the review's section 5 table)")
print("=" * 100)
print(f"   hours = flagged rows / {CM.SAMPLES_PER_HOUR:.0f} (sampled hours, not wall-clock)")
rows = []
for a, b in B.PAIRS:
    k = f"{a}_{b}"
    p = F["per_pair"][k.replace("+", "_")]
    rows.append(dict(pair=f"{a}+{b}",
                     pub_rows=p["published_rows"], can_rows=p["canonical_rows"],
                     pub_h=round(p["published_rows"] / CM.SAMPLES_PER_HOUR, 1),
                     can_h=round(p["canonical_rows"] / CM.SAMPLES_PER_HOUR, 1),
                     pub_sev_rows=p["published_severe_rows"],
                     can_sev_rows=p["canonical_severe_rows"],
                     pub_sev_h=round(p["published_severe_rows"] / CM.SAMPLES_PER_HOUR, 1),
                     can_sev_h=round(p["canonical_severe_rows"] / CM.SAMPLES_PER_HOUR, 1)))
print(pd.DataFrame(rows).set_index("pair").to_string())
print(f"\n   TOTALS  published {F['published_rows']:,} rows / {F['published_hours']:,.1f} h"
      f"   ->   canonical {F['canonical_rows']:,} rows / {F['canonical_hours']:,.1f} h"
      f"   ({F['row_change_pct']:+.1f} %)")
print(f"   SEVERE  published {F['published_severe_hours']:,.1f} h ({F['published_severe_rows']}"
      f" rows) -> canonical {F['canonical_severe_hours']:,.1f} h "
      f"({F['canonical_severe_rows']} rows)   ({F['severe_inflation_x']:.2f}x removed)")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("C. THE CONTROL PAIRS (every canonical flag on them would be a false alarm)")
print("=" * 100)
print(f"   {'pair':14s} {'published':>10s} {'canonical':>10s}")
for a, b in B.CONTROL_PAIRS:
    v = CT[f"{a}_{b}"]
    print(f"   {a}+{b:6s} {v['published']:10d} {v['canonical']:10d}")
tp = sum(v["published"] for v in CT.values())
tc = sum(v["canonical"] for v in CT.values())
print(f"\n   totals: published {tp}   canonical {tc}")
print(f"   only eu_1+eu_3 crosses inverters; it is the honest flag-level null and it")
print(f"   goes {CT['eu_1_eu_3']['published']} -> {CT['eu_1_eu_3']['canonical']}.")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("D. ENERGY (apparent lost energy over the DETECTOR GATE domain, kWh)")
print("=" * 100)
print(f"   {CM.GATE_DEF}")
print("   (NOT 'ref >= 0.70' alone -- the earlier label understated the domain)")
print(f"   shared pairs  total   published {E['shared_published_kwh']:,}   "
      f"canonical {E['shared_canonical_kwh']:,} kWh")
print(f"   control pairs total   published {E['control_published_kwh']:,}   "
      f"canonical {E['control_canonical_kwh']:,} kWh")
print(f"   bias floor as a share of the shared figure: "
      f"published {E['bias_floor_published_pct']} %   "
      f"canonical {E['bias_floor_canonical_pct']} %")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("E. BENCHMARK (ground truth, episodic regime = the realistic one)")
print("=" * 100)
print(f"   false-positive rows on unclipped days, pooled over the 2 episodic scenarios:")
print(f"      published {BM['episodic_fp_published']:,}  ->  canonical "
      f"{BM['episodic_fp_canonical']:,}   = {BM['specificity_gain_x']}x")
print(f"      (per-scenario ratios span 48x-820x; do not quote the pooled figure alone)")
print(f"   AUC: published {BM['episodic_auc_published']:.3f}  ->  canonical "
      f"{BM['episodic_auc_canonical']:.3f}")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("F. REPRODUCE AND VERIFY")
print("=" * 100)
print("   python sat_work/research/recommended.py                  # rewrites the artefact")
print("   python sat_work/research/step24_canonical_manifest.py    # re-lock the version")
print("   python sat_work/research/step24_canonical_manifest.py --check")
print("   python sat_work/tests/run_tests.py")
print("\n   This script's numbers ARE canon_metrics.summarise(), so they cannot drift from")
print("   the lock that tests/test_manifest.py enforces.")
