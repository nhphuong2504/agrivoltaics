"""
STEP 21 - the canonical artefact's headline numbers.

`saturation_flags.csv` became the vat-v1 export in 2026-09 (recommended.py writes it, and
the old published output is retired rather than kept as a parallel file). This script
prints the exact numbers the review (section 5) and the deck quote, from the canonical
file and from the frozen published detector, so both documents can be regenerated rather
than hand-copied.

It also reports the two data-product properties that make the file safe to publish, both
now enforced by test_exports.py:
  * no infinities anywhere
  * the deficit column is physically bounded (-1 <= d <= 1) inside the decision domain,
    with the residual out-of-range tail confined to ref < 0.5

Run:  ./venv/Scripts/python.exe sat_work/research/step21_canonical_summary.py
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import bench as B

CANON = os.path.join(_ROOT, "saturation_flags.csv")
df = B.raw_df(); fe = B.raw_fe(); ref = fe["ref"]
can = pd.read_csv(CANON, parse_dates=["time"])
pd.set_option("display.width", 200)


def days_of(mask_series):
    return pd.to_datetime(mask_series[mask_series].index).normalize().nunique()


# --------------------------------------------------------------------------- #
print("=" * 100)
print("A. THE CANONICAL ARTEFACT")
print("=" * 100)
num = can.select_dtypes("number")
print(f"   {CANON}")
print(f"   {len(can):,} rows x {can.shape[1]} cols   infinite values {int(np.isinf(num.to_numpy()).sum())}")
extra = [c for c in can.columns
         if c.endswith(("_null_d", "_excess_vs_controls", "_sat_conservative"))]
print(f"   published v1 columns all present: "
      f"{set(B.build_export().columns) >= set(can.columns)}")
print(f"   columns beyond the published v1 schema: {len(extra)} "
      f"(null_d, excess_vs_controls, sat_conservative per pair)")

inside = can["ref"] >= B.REF_ON
print(f"\n   deficit column, inside the decision domain (ref >= {B.REF_ON}):")
for a, b in B.PAIRS:
    c = can.loc[inside, f"{a}_{b}_deficit"].dropna()
    print(f"      {a}+{b}: n={len(c):5,}  min {c.min():+.4f}  max {c.max():+.4f}  "
          f"median {c.median():+.4f}")
beyond = {f"{a}_{b}": int((can[f"{a}_{b}_deficit"] < -1).sum()) for a, b in B.PAIRS}
worst_ref = max((can.loc[can[f"{a}_{b}_deficit"] < -1, "ref"].max()
                 for a, b in B.PAIRS if (can[f"{a}_{b}_deficit"] < -1).any()), default=float("nan"))
print(f"\n   rows with deficit < -1 (out of physical range): {beyond}")
print(f"   worst ref among them: {worst_ref:.3f}  (gate is {B.REF_ON}) -> confined below gate")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("B. FLAGGED HOURS: published vs canonical (this is the review's section 5 table)")
print("=" * 100)
rows = []
for a, b in B.PAIRS:
    k = f"{a}_{b}"
    pub = B.detect(df, fe, a, b)
    pm = pub["moderate"]; ps = pub["severe"]
    cm = can[f"{k}_sat_moderate"].astype(bool); cs = can[f"{k}_sat_severe"].astype(bool)
    cm.index = df.index
    pdm = days_of(pm); cdm = days_of(cm)
    rows.append(dict(pair=f"{a}+{b}",
                     pub_rows=int(pm.sum()), can_rows=int(cm.sum()),
                     pub_h=round(pm.sum() / 12, 1), can_h=round(cm.sum() / 12, 1),
                     pub_sev_h=round(ps.sum() / 12, 1), can_sev_h=round(cs.sum() / 12, 1),
                     pub_days=pdm, can_days=cdm,
                     shared_days=len(set(pm[pm].index.normalize()) & set(cm[cm].index.normalize()))))
T = pd.DataFrame(rows).set_index("pair")
print(T.to_string())
# totals from EXACT row counts -- rebuilding them from hours rounded to 1 dp drifts by a row
print(f"\n   TOTALS  published {int(T.pub_rows.sum()):,} rows / {T.pub_h.sum():,.1f} h"
      f"   ->   canonical {int(T.can_rows.sum()):,} rows / {T.can_h.sum():,.1f} h"
      f"   ({100*(T.can_rows.sum()/T.pub_rows.sum()-1):+.1f} %)")
print(f"   SEVERE  published {T.pub_sev_h.sum():,.1f} h  ->  canonical {T.can_sev_h.sum():,.1f} h"
      f"   ({(T.pub_sev_h.sum()/T.can_sev_h.sum()):.1f}x inflation removed)")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("C. THE CONTROL PAIRS (every canonical flag on them would be a false alarm)")
print("=" * 100)
rows = []
for a, b in B.CONTROL_PAIRS:
    rows.append(dict(pair=f"{a}+{b}",
                     published=int(B.detect(df, fe, a, b)["moderate"].sum()),
                     canonical=int(B.det_rolling(df, fe, a, b)[1].sum())))
C = pd.DataFrame(rows).set_index("pair")
print(C.to_string())
print(f"\n   totals: published {int(C.published.sum())}   canonical {int(C.canonical.sum())}")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("D. ENERGY (apparent lost energy at ref >= 0.70, kWh)")
print("=" * 100)


def loss(a, b, mode):
    d = B.pair_deficit(df, fe, a, b)
    gate = ((ref >= B.REF_ON) & (d["s"] > B.NEAR_CAP * d["expected"])
            & d["act"] & ~B.dq_mask(fe, a, b)).to_numpy()
    expected = d["expected"] if mode == "vat" else \
        pd.Series(B.detect(df, fe, a, b)["expected"], index=df.index)
    return float(np.maximum(expected - d["s"], 0).to_numpy()[gate].sum()) * 5 / 60 / 1000


sh_p = sum(loss(a, b, "pub") for a, b in B.PAIRS)
sh_v = sum(loss(a, b, "vat") for a, b in B.PAIRS)
ct_p = sum(loss(a, b, "pub") for a, b in B.CONTROL_PAIRS)
ct_v = sum(loss(a, b, "vat") for a, b in B.CONTROL_PAIRS)
print(f"   shared pairs  total   published {sh_p:,.0f}   canonical {sh_v:,.0f} kWh")
print(f"   control pairs total   published {ct_p:,.0f}   canonical {ct_v:,.0f} kWh")
print(f"   bias floor as a share of the shared figure: "
      f"published {100*ct_p/sh_p:.0f} %   canonical {100*ct_v/sh_v:.0f} %")

print()
print("=" * 100)
print("E. REPRODUCE")
print("=" * 100)
print("   python sat_work/research/recommended.py        # rewrites the canonical file")
print("   python sat_work/research/step21_canonical_summary.py")
print("   python sat_work/tests/run_tests.py")
