"""
STEP 11 - two correctness checks on my own review.

Check 1 ("3.2 is it really the ceiling being dragged DOWN?")
------------------------------------------------------------------
The review's defect-B story (and step4's docstring) says: `cap(t)` is dragged down
toward the clip level, so `expected = g0*cap*ref` is too SMALL, so `deficit` is
inflated, so healthy days get flagged.

But deficit = 1 - s/expected, so a SMALLER expected makes the deficit SMALLER, not
larger. A dragged-down ceiling should cause MISSES, not false positives. And an FP
row by construction has s < 0.85*expected, i.e. expected is too BIG.

So measure it instead of arguing: on rows the published detector flags while the
ground truth is "not clipped", is exp_pub above or below the true unsaturated slope?

Check 2 ("is the published bias seasonal?")
------------------------------------------------------------------
On the control pairs - which cannot saturate - the published baseline reports a
positive deficit at high ref (0.074-0.089). If that bias is seasonal (driven by
`cap(t)`, which tracks the physical peak), it should peak in summer, which would
explain why July 2026 is the largest detector disagreement without appealing to any
clipping interaction at all.

Run:  ./venv/Scripts/python.exe sat_work/research/step11_fpmech.py
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *

pd.set_option("display.width", 200)
df_raw = load()
FULLCAL = pd.date_range(df_raw.index.min().normalize(), df_raw.index.max().normalize(), freq="D")
BAND = (0.35, 0.60)


def broadcast(per_day, dayidx, labels, window="31D", minp=5):
    roll = per_day.reindex(FULLCAL).rolling(window, center=True, min_periods=minp).median()
    return pd.Series(roll.reindex(dayidx).to_numpy(), index=labels)


def inject(df, a, b, C, day_on=None):
    d = df.copy()
    s = (d[a] + d[b]).to_numpy(dtype=float)
    Cv = np.asarray(C) if np.ndim(C) else np.full(len(d), float(C))
    if day_on is not None:
        Cv = np.where(day_on, Cv, np.inf)
    f = np.where(s > 0, np.minimum(s, Cv) / np.where(s > 0, s, 1.0), 1.0)
    d[a] = d[a] * f; d[b] = d[b] * f
    return d, pd.Series(s > Cv, index=d.index)


def slope(s, ref, act, dayidx, band=BAND):
    """theta(t): the true unsaturated slope proxy, from the (un-clipped) pair."""
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= band[0]) & (ref <= band[1]) & u.notna()
    return broadcast(u[sel].groupby(dayidx[sel]).median(), dayidx, s.index)


# --------------------------------------------------------------------------- #
print("=" * 100)
print("CHECK 2. IS THE PUBLISHED HIGH-IRRADIANCE BIAS SEASONAL?")
print("=" * 100)
print("Control pairs cannot saturate, so their deficit is pure model error.")
print("If that error is driven by cap(t) - which tracks the physical peak - it must")
print("be largest in summer and smallest in winter.\n")
fe = preprocess(df_raw)
ref = fe["ref"]
mo = df_raw.index.to_period("M").astype(str)
rows = []
for a, b in CONTROL_PAIRS:
    r = detect(df_raw, fe, a, b)
    cap = pd.Series(r["cap_t"], index=df_raw.index)
    gain = pd.Series(r["gain"], index=df_raw.index)
    th = slope(df_raw[a] + df_raw[b], ref, fe["active"][a] & fe["active"][b], fe["dayidx"])
    hi = (ref >= 0.7) & fe["active"][a] & fe["active"][b] & ~dq_mask(fe, a, b)
    gg = pd.DataFrame({"m": mo, "d": pd.Series(r["deficit"], index=df_raw.index),
                       "cap": cap, "th": th, "hi": hi})
    for m, g in gg[gg.hi].groupby("m"):
        rows.append(dict(pair=f"{a}+{b}", month=m, deficit_pub=g["d"].median(),
                         cap_MW=g["cap"].median() / 1000, cap_over_th=(g["cap"] / g["th"]).median()))
C = pd.DataFrame(rows)
print("mean over the three control pairs:")
piv = C.pivot_table(index="month", values=["deficit_pub", "cap_MW", "cap_over_th"], aggfunc="mean")
print(piv.round(3).to_string())
sub = piv.dropna()
print(f"\ncorrelation(deficit_pub, cap_over_th) across months = "
      f"{sub['deficit_pub'].corr(sub['cap_over_th']):+.3f}")
print(f"correlation(deficit_pub, cap_MW)      across months = "
      f"{sub['deficit_pub'].corr(sub['cap_MW']):+.3f}")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("CHECK 1. ON A FALSE POSITIVE, IS `expected` TOO BIG OR TOO SMALL?")
print("=" * 100)
print("Inject a hard clip at q=0.75 into an independent pair, 35% of days, exactly as")
print("step3/step4 do. Then compare the published `expected` with the TRUE unsaturated")
print("slope (measured on the un-injected pair).\n")
a, b = "eu_1", "eu_3"
rel = [c for c in RELIABLE if c not in (a, b)]
fe0 = preprocess(df_raw, rel)
Ptyp = np.nanpercentile((df_raw[a] + df_raw[b])[fe0["ref"] >= 0.85], 95)
days = df_raw[a].groupby(df_raw.index.normalize()).count().pipe(lambda x: x[x > 0]).index
day_on = (pd.Series(np.random.default_rng(3).random(len(days)) < 0.35, index=days)
          .reindex(df_raw.index.normalize()).fillna(False).to_numpy())
d2, gt = inject(df_raw, a, b, 0.75 * Ptyp, day_on)
fe2 = preprocess(d2, rel)
ref2 = fe2["ref"]
act2 = fe2["active"][a] & fe2["active"][b]
r2 = detect(d2, fe2, a, b)

s2 = d2[a] + d2[b]
th_true = slope(df_raw[a] + df_raw[b], fe0["ref"],           # TRUE slope: un-injected
                fe0["active"][a] & fe0["active"][b], fe0["dayidx"])
exp_true = th_true * ref2
exp_pub = pd.Series(r2["expected"], index=d2.index)
d_pub = pd.Series(r2["deficit"], index=d2.index)
err = exp_pub / exp_true.replace(0, np.nan)                   # >1 == expected too big

test = act2 & (ref2 >= 0.70) & ~dq_mask(fe2, a, b)
samp = test & gt                      # ground truth: clip IS binding
uncl = test & ~gt                     # ground truth: clip is NOT binding
grp = {
    "TP  flagged, clip binding": samp & pd.Series(r2["moderate"], index=d2.index),
    "FN  missed, clip binding": samp & ~pd.Series(r2["moderate"], index=d2.index),
    "FP  flagged, NOT binding": uncl & pd.Series(r2["moderate"], index=d2.index),
    "TN  not flagged, NOT binding": uncl & ~pd.Series(r2["moderate"], index=d2.index),
}
tab = []
for lab, m in grp.items():
    if m.sum() == 0:
        tab.append(dict(group=lab, rows=0)); continue
    tab.append(dict(group=lab, rows=int(m.sum()),
                    s=round(float(s2[m].median()) / 1000, 2),
                    exp_pub=round(float(exp_pub[m].median()) / 1000, 2),
                    exp_true=round(float(exp_true[m].median()) / 1000, 2),
                    err_exp_pub=round(float(err[m].median()), 3),
                    deficit_pub=round(float(d_pub[m].median()), 3),
                    ref=round(float(ref2[m].median()), 3)))
T = pd.DataFrame(tab)
print(T.to_string(index=False))
print("\n   err_exp_pub > 1  =>  the published expected value EXCEEDS the true")
print("   unsaturated slope, which is what can manufacture a positive deficit.")
print(f"\n   median err_exp_pub on FP rows  = {float(err[grp['FP  flagged, NOT binding']].median()):.3f}")
print(f"   median err_exp_pub on TN rows  = {float(err[grp['TN  not flagged, NOT binding']].median()):.3f}")
print(f"   median err_exp_pub on TP rows  = {float(err[grp['TP  flagged, clip binding']].median()):.3f}")

print()
print("   ... and the DRAG claim itself, measured on this injected pair:")
cap2 = pd.Series(r2["cap_t"], index=d2.index)
th2 = slope(s2, ref2, act2, fe2["dayidx"])
ratio = (cap2 / th2)
onclip, offclip = ratio[samp], ratio[uncl]
print(f"   median cap/theta on CLIPPED days (clip binding)   : {onclip.median():.3f}")
print(f"   median cap/theta on UNClipped days                : {offclip.median():.3f}")
print(f"   -> clipping changed cap/theta by {100*(onclip.median()/offclip.median()-1):+.1f}%")
print("\n   If clipping dragged the ceiling down, the clipped-day ratio would sit far")
print("   BELOW the unclipped-day ratio here.")
