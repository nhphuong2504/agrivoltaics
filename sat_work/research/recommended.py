"""
The recommended detector (vat-v1), end to end.

This is now a thin consumer of the frozen harness in `bench.py` so that the detector
used to produce the shipped artefacts cannot drift from the detector the regression
suite scores. Before this refactor the two files each had their own copy of `persist`,
and only the harness copy broke runs at acquisition gaps -- exactly the kind of silent
divergence the test suite exists to prevent.

vat-v1:
  baseline     theta(t)  = 31-day centred rolling median of the DAILY MEDIAN of s/ref
                           over active samples with ref in [0.35, 0.60]
  deficit      d(t)      = 1 - s(t) / (theta(t)*ref(t))
  gate                    ref >= 0.70, s > 0.6*theta*ref, both units active, no DQ flag
  flag         saturated = d > 0.15 for 3 consecutive samples, 1-sample gate-respecting
                           gap bridging

ONE FLAG, ONE CONTINUOUS SCORE. Earlier revisions also exported `severe` (a strict subset
of the flag) and `conservative` (a different, control-null-calibrated rule). Both are gone:
severity is recovered by thresholding `deficit`, and the control-null detector remains
available as `bench.det_control_calibrated` for the benchmark. See `bench.build_export`.

Outputs:
  <repo root>/saturation_flags.csv   THE CANONICAL FLAGS ARTEFACT (vat-v1)
  energy_estimate.csv                a research artefact, written beside this file

CANONICAL AS OF 2026-09. The repo-root `saturation_flags.csv` is now this detector's
output. The previous published method is deliberately NOT kept as a parallel result file:
it stays reproducible from `bench.det_published()` (frozen, tested) and from the committed
`saturation_detection.ipynb`, both of which are in git history. Keeping two result files
would leave two silent sources of truth for the same numbers.

Run:  ./venv/Scripts/python.exe sat_work/research/recommended.py
"""
import io
import os
import sys
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import bench as B  # noqa: E402  (frozen harness)

ROOT = os.path.dirname(os.path.dirname(_HERE))       # the repo root
CANONICAL_CSV = os.path.join(ROOT, "saturation_flags.csv")

df = B.raw_df()
fe = B.raw_fe()
ref = fe["ref"]

CONTROLS = B.CONTROLS
PAIRS = B.PAIRS
CONTROL_PAIRS = B.CONTROL_PAIRS

# --------------------------------------------------------------------------- #
# 1. the export
# --------------------------------------------------------------------------- #
out = B.build_export()
out.to_csv(CANONICAL_CSV, index=False)


def flags_for(pair):
    k = f"{pair[0]}_{pair[1]}"
    return out[f"{k}_sat"].astype(bool)


# --------------------------------------------------------------------------- #
# 2. head-to-head on the three real shared pairs
# --------------------------------------------------------------------------- #
rows = []
for a, b in PAIRS:
    pub = B.detect(df, fe, a, b)
    mod = flags_for((a, b))
    rows.append(dict(pair=f"{a}_{b}",
                     pub_rows=int(pub["moderate"].sum()), vat_rows=int(mod.sum()),
                     pub_hours=pub["moderate"].sum() / 12, vat_hours=mod.sum() / 12,
                     vat_days=pd.to_datetime(out.loc[mod, "time"]).dt.normalize().nunique()))
cmp_ = pd.DataFrame(rows).set_index("pair")

print("=" * 96)
print("RECOMMENDED DETECTOR (vat-v1) vs PUBLISHED, on the three real shared pairs")
print("=" * 96)
print(cmp_.round(1).to_string())

print()
print("--- same detectors on the three independent control pairs (every flag is a false alarm) ---")
cf = []
for a, b in CONTROL_PAIRS:
    _, vat = B.det_rolling(df, fe, a, b)
    pub = B.detect(df, fe, a, b)
    cf.append(dict(pair=f"{a}+{b}", published_rows=int(pub["moderate"].sum()),
                   vat_rows=int(vat.sum())))
print(pd.DataFrame(cf).set_index("pair").to_string())

# --------------------------------------------------------------------------- #
# 3. energy: apparent lost energy at ref >= 0.70, and the control-pair bias floor
# --------------------------------------------------------------------------- #
print()
print("=" * 96)
print("ENERGY: apparent lost energy at ref>=0.7, kWh, and the control-pair bias floor")
print("=" * 96)


def loss(a, b, mode):
    d = B.pair_deficit(df, fe, a, b)
    gate = ((ref >= B.REF_ON) & (d["s"] > B.NEAR_CAP * d["expected"])
            & d["act"] & ~B.dq_mask(fe, a, b)).to_numpy()
    if mode == "vat":
        expected = d["expected"]
    else:
        expected = pd.Series(B.detect(df, fe, a, b)["expected"], index=df.index)
    shortfall = np.maximum(expected - d["s"], 0).to_numpy()[gate]
    return float(shortfall.sum()) * 5 / 60 / 1000


L = pd.DataFrame({f"{a}+{b}": {"published": loss(a, b, "pub"), "vat": loss(a, b, "vat")}
                  for a, b in list(PAIRS) + list(CONTROL_PAIRS)}).T
L["group"] = ["SHARED"] * 3 + ["control"] * 3
print(L.round(0).to_string())
shared, ctrl = L[L.group == "SHARED"], L[L.group == "control"]
print(f"\ntotals  shared  published={shared['published'].sum():.0f}  "
      f"vat={shared['vat'].sum():.0f} kWh   |   "
      f"control-floor  published={ctrl['published'].sum():.0f}  "
      f"vat={ctrl['vat'].sum():.0f} kWh")
print(f"bias as a share of the shared figure: "
      f"published {100*ctrl['published'].sum()/shared['published'].sum():.0f} %   "
      f"vat-v1 {100*ctrl['vat'].sum()/shared['vat'].sum():.0f} %")
L.to_csv(os.path.join(_HERE, "energy_estimate.csv"))

# --------------------------------------------------------------------------- #
# 4. data-product sanity, reported rather than asserted (the suite asserts it)
# --------------------------------------------------------------------------- #
num = out.select_dtypes("number")
print()
print("=" * 96)
print("EXPORT SANITY")
print("=" * 96)
print(f"  shape {out.shape};  infinite values {int(np.isinf(num.to_numpy()).sum())}")
print(f"  deficit columns finite-or-NaN: "
      f"{all(not np.isinf(out[f'{a}_{b}_deficit']).any() for a, b in PAIRS)}")
print(f"  flag columns binary: "
      f"{all(set(out[f'{a}_{b}_sat'].unique()) <= {0, 1} for a, b in PAIRS)}")

# --------------------------------------------------------------------------- #
# 5. the canonical artefact round-trips
# --------------------------------------------------------------------------- #
#    Read back what was actually written, rather than trusting the frame in memory:
#    the file is the deliverable, and a silent dtype/NaN regression on the way to disk
#    would leave a good frame in memory and a bad artefact on disk.
print()
print("=" * 96)
print("CANONICAL ARTEFACT")
print("=" * 96)
back = pd.read_csv(CANONICAL_CSV)
back_num = back.select_dtypes("number")
n_inf = int(np.isinf(back_num.to_numpy()).sum())
ok = len(back) == len(df) and list(back.columns) == list(out.columns) and n_inf == 0
print(f"  {CANONICAL_CSV}")
print(f"  rows {len(back):,} (dataset {len(df):,})   cols {back.shape[1]}   "
      f"infinite values {n_inf}")
print(f"  round-trip clean: {ok}")
for a, b in PAIRS:
    k = f"{a}_{b}"
    n = int(back[f"{k}_sat"].sum())
    print(f"    {k}: {n:5,} rows = {n/12:7.1f} h")
assert ok, "canonical artefact did not round-trip cleanly"
print("\nwrote", CANONICAL_CSV)
