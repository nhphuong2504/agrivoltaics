"""Probe the frozen benchmark for the exact values the regression tests will assert."""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
import bench as B

pd.set_option("display.width", 220)

print("=" * 100)
print("EXPORT CONTRACT")
print("=" * 100)
ex = B.build_export()
print(f"shape {ex.shape}")
dtypes = ex.dtypes.astype(str).to_frame("dtype").T
print(f"dtypes: {set(dtypes.iloc[0])}")
num = ex.select_dtypes("number")
inf_count = int(np.isinf(num.to_numpy()).sum())
print(f"inf values in numeric columns: {inf_count}")

ref = ex["ref"]
below = ref < B.REF_ON
for k in ("eu_8_eu_16", "eu_10_eu_18", "eu_13_eu_21"):
    nd = ex[f"{k}_null_d"]
    exc = ex[f"{k}_excess_vs_controls"]
    print(f"  {k}: null_d NaN below gate {int(nd[below].isna().sum())}/{int(below.sum())}, "
          f"not-NaN at/above gate {int(nd[~below].notna().sum())}/{int((~below).sum())}, "
          f"excess NaN below gate {int(exc[below].isna().sum())}/{int(below.sum())}")
    for tier in ("moderate", "severe", "conservative"):
        c = ex[f"{k}_sat_{tier}"]
        print(f"     {tier:13s} dtype={c.dtype} values={sorted(c.unique())} "
              f"sum={int(c.sum())} hours={c.sum()/12:.1f}")

print()
print("published column schema (for the backwards-compatibility test):")
pub_cols = pd.read_csv(r"c:\Users\nhphuong\Desktop\Solar\all_data\saturation_flags.csv",
                       nrows=0).columns.tolist()
print(f"  {pub_cols}")
miss = [c for c in pub_cols if c not in ex.columns]
print(f"  published columns missing from the export: {miss}")

print()
print("=" * 100)
print("KNOWN FAILURE - the SHIPPED published csv")
print("=" * 100)
shipped = pd.read_csv(r"c:\Users\nhphuong\Desktop\Solar\all_data\saturation_flags.csv")
sc = shipped.select_dtypes("number")
print(f"  inf values: {int(np.isinf(sc.to_numpy()).sum())}")
print(f"  columns affected: "
      f"{[c for c in sc.columns if np.isinf(sc[c].to_numpy()).any()]}")

print()
print("=" * 100)
print("BENCHMARK SCENARIO NUMBERS THE TESTS WILL USE")
print("=" * 100)
SC = {
    "noclip":      (("eu_1", "eu_3"), 1.00, None, "chronic"),
    "intermittent": (("eu_1", "eu_3"), 0.35, 0.75, "episodic"),
    "chronic":     (("eu_1", "eu_3"), 1.00, 0.75, "chronic"),
    "inter2":      (("eu_15", "eu_23"), 0.35, 0.75, "episodic"),
}
rows = []
for name, (pair, duty, q, pat) in SC.items():
    sc = B.scenario(pair, duty, q, pat)
    for dname, fn in B.DETECTORS.items():
        e = B.score_scenario(sc, fn)
        e.update(scen=name, det=dname)
        rows.append(e)
T = pd.DataFrame(rows)
print(T[["scen", "det", "auc00", "auc10", "auc20", "tp_rows", "fp_rows", "rec", "prec"]]
      .round(3).to_string(index=False))

print()
print("D1 severity stratification (single scenario, intermittent):")
sc = B.scenario(("eu_1", "eu_3"), 0.35, 0.75, "episodic")
e = B.score_scenario(sc, B.det_rolling)
print(f"   auc00={e['auc00']:.3f} auc10={e['auc10']:.3f} auc20={e['auc20']:.3f}  "
      f"n_pos00={e['n_pos00']} n_pos10={e['n_pos10']} n_pos20={e['n_pos20']} n_neg={e['n_neg']}")

print()
print("gap-tolerant persistence (bridge=1) vs strict, on the intermittent scenario:")
for gap in (0, 1):
    def det(df, fe, a, b, _g=gap):
        return B.det_rolling(df, fe, a, b, gap=_g)
    e = B.score_scenario(sc, det)
    print(f"   gap={gap}: tp={e['tp_rows']} fp={e['fp_rows']} prec={e['prec']:.4f} rec={e['rec']:.4f}")

print()
print("determinism check:")
a1 = B.score_scenario(B.scenario(("eu_1", "eu_3"), 0.35, 0.75, "episodic"), B.det_rolling)
B._STATE.pop(("scenario", ("eu_1", "eu_3"), 0.35, 0.75, "episodic", 7), None)
a2 = B.score_scenario(B.scenario(("eu_1", "eu_3"), 0.35, 0.75, "episodic"), B.det_rolling)
print(f"   rebuilt scenario identical: {a1 == a2}")
