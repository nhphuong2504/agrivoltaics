"""
STEP 18 - settle the persistence rule.

The last open decision before regenerating the data product was strict vs gap-tolerant
persistence. step13/step14 measured two options; this script searches the space between
them, because the diagnostics suggested a rule that beats both.

Evidence that motivated the search (step14): of the 4,498 rows bridging recovers, 3,958
are genuine clip rows but 540 are spurious. The two groups separate cleanly on deficit:

    genuine recovered rows : median deficit 0.161
    spurious recovered rows: median deficit 0.017

So the spurious rows are gap-INTERIOR samples whose own evidence is far too weak to flag.
Bridging accepts them purely because they sit between two real samples. Guarding the fill
on the filled sample's own deficit should keep the recall and drop the false positives.

Variants scored here, on injected ground truth AND on the real fleet:
    strict          k=3, no bridging
    bridged         k=3, fill gaps <= 1 sample
    bridged+guard   fill gaps <= 1 sample only where the filled sample's deficit >= s

Run:  ./venv/Scripts/python.exe sat_work/research/step18_persist_decision.py
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from satsim import *
import bench as B

pd.set_option("display.width", 200)
df = B.raw_df()
FE = B.raw_fe()
ref = FE["ref"]

GUARDS = [None, 0.03, 0.05, 0.075, 0.10]


def bridge_guarded(raw, deficit, g, guard, within=None):
    """bridge(), but a filled sample must carry its own evidence (deficit >= guard).

    `within` is passed through to the gate guard, so the fill never lands below the ref
    gate (that is the contract-preserving part, and it is on every variant here).
    """
    out = raw.copy()
    for k in range(1, g + 1):
        fill = raw.shift(k, fill_value=False) & raw.shift(-k, fill_value=False)
        if within is not None:
            fill = fill & within
        if guard is not None:
            fill = fill & (deficit >= guard)
        out = out | fill
    return out


def make_det(g, guard, thr=B.THR_MOD, k=B.PERS_MOD):
    """The recommended moderate tier, with a pluggable persistence rule."""
    def det(df_, fe, a, b, **_):
        d = B.pair_deficit(df_, fe, a, b, q=0.5)
        base = ((fe["ref"] >= B.REF_ON) & (d["s"] > B.NEAR_CAP * d["expected"])
                & d["act"] & ~B.dq_mask(fe, a, b))
        raw = base & (d["deficit"] > thr)
        if g:
            raw = bridge_guarded(raw, d["deficit"], g, guard, within=base)
        return d["deficit"], B.persist(raw, k, index=df_.index)
    return det


VARIANTS = [("strict", 0, None)] + \
           [(f"bridged g=1" + ("" if s is None else f" guard={s}"), 1, s) for s in GUARDS]

print("=" * 100)
print("A. GROUND TRUTH - 6 injected scenarios (3 independent pairs, chronic + episodic)")
print("=" * 100)
scen = [B.scenario(p, duty=d, q=q, pattern=pat) for p, d, q, pat in B.DEFAULT_SCENARIOS]
rows = []
for name, g, guard in VARIANTS:
    det = make_det(g, guard)
    tp = fp = fn = 0
    for sc in scen:
        m = B.score_scenario(sc, det)
        tp += m["tp_rows"]; fp += m["fp_rows"]
        fn += int(round(m["n_pos00"] * (1 - m["rec"])))
    rows.append(dict(variant=name, tp=tp, fp=fp, fn=fn,
                     rec=tp / max(tp + fn, 1), prec=tp / max(tp + fp, 1)))
T = pd.DataFrame(rows)
base_rec, base_fp = T.loc[0, "rec"], T.loc[0, "fp"]
T["d_rec_pp"] = 100 * (T.rec - base_rec)
T["fp_x"] = T.fp / max(base_fp, 1)
print(T.round(4).to_string(index=False))

print()
print("   The question is whether any guarded variant matches 'bridged' on recall while")
print("   staying near 'strict' on false positives. Pareto check (recall up, FP down):")
best = T.iloc[0]
for i in range(1, len(T)):
    r = T.iloc[i]
    frontier = (r.rec >= T.rec.max() - 1e-9) or \
               not (T.rec > r.rec).any() and False
dom = []
for i, r in T.iterrows():
    dominated = ((T.rec >= r.rec) & (T.fp <= r.fp) &
                 ((T.rec > r.rec) | (T.fp < r.fp))).any()
    dom.append(dominated)
T["dominated"] = dom
print(T[["variant", "rec", "fp", "prec", "dominated"]].round(4).to_string(index=False))
frontier_names = T[~T.dominated].variant.tolist()
print(f"\n   Pareto frontier: {frontier_names}")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("B. REAL FLEET - hours claimed, and control-pair rows (every one a false alarm)")
print("=" * 100)
print("Controls are split by class: cross-inverter (a provably clean null) and the two")
print("same-inverter pairs, which carry a co-movement common mode (see step17).\n")
rows = []
for name, g, guard in VARIANTS:
    det = make_det(g, guard)
    h = {f"{a}+{b}": det(df, FE, a, b)[1].sum() / 12 for a, b in PAIRS}
    shared_h = sum(h.values())
    fp = {}
    for a, b in CONTROL_PAIRS:
        fp[f"{a}+{b}"] = int(det(df, FE, a, b)[1].sum())
    rows.append(dict(variant=name, shared_h=shared_h,
                     cross_inv_fp=fp["eu_1+eu_3"],
                     same_inv_fp=fp["eu_4+eu_12"] + fp["eu_15+eu_23"],
                     **{k: v for k, v in h.items()}))
R = pd.DataFrame(rows)
R["d_hours_pct"] = 100 * (R.shared_h / R.shared_h.iloc[0] - 1)
print(R.round(1).to_string(index=False))

print()
print("=" * 100)
print("C. VERDICT")
print("=" * 100)
strict = T.iloc[0]
bridge_plain = T[T.variant.str.startswith("bridged g=1") & (T.variant == "bridged g=1")]
if len(bridge_plain):
    bp = bridge_plain.iloc[0]
    print(f"   strict        : recall {strict.rec:.4f}, FP {int(strict.fp)}")
    print(f"   bridged       : recall {bp.rec:.4f}, FP {int(bp.fp)}   "
          f"(+{100*(bp.rec-strict.rec):.1f} pp recall, {bp.fp/max(strict.fp,1):.2f}x FP)")
for _, r in T[T.variant.str.contains("guard")].iterrows():
    print(f"   {r.variant:22s}: recall {r.rec:.4f}, FP {int(r.fp)}   "
          f"({100*(r.rec-strict.rec):+.1f} pp recall, {r.fp/max(strict.fp,1):.2f}x FP)")
print("\n   Real-data cost of the cross-inverter null (must stay 0):")
for _, r in R.iterrows():
    flag = "OK " if r.cross_inv_fp == 0 else "BAD"
    print(f"      {flag} {r.variant:22s} cross-inv FP {int(r.cross_inv_fp):3d}   "
          f"same-inv FP {int(r.same_inv_fp):3d}   hours {r.shared_h:7.1f} "
          f"({r.d_hours_pct:+.1f}%)")
