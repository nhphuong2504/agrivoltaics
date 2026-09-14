"""
STEP 14 - should gap-tolerant persistence be adopted?  Decide it on ground truth.

step13 says bridging 1-sample gaps recovers +203 h on the shared pairs for only +18
extra control-pair rows. That looks like free signal - but it also lets a run that
flickers around the 0.15 boundary qualify, which is just a lower effective threshold.
The benchmark settles it: inject a known clip, then ask what the recovered rows
actually are.

Run:  ./venv/Scripts/python.exe sat_work/research/step14_persist_bench.py
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 220)
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


def theta(s, ref, act, dayidx):
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= BAND[0]) & (ref <= BAND[1]) & u.notna()
    return broadcast(u[sel].groupby(dayidx[sel]).median(), dayidx, s.index)


def bridge(raw, g):
    r = raw.copy()
    for k in range(1, g + 1):
        r = r | (raw.shift(k, fill_value=False) & raw.shift(-k, fill_value=False))
    return r


def d1(adf, fe, a, b, g):
    s = adf[a] + adf[b]
    act = fe["active"][a] & fe["active"][b]
    th = theta(s, fe["ref"], act, fe["dayidx"])
    exp_ = th * fe["ref"]
    d = (1 - s / exp_).replace([np.inf, -np.inf], np.nan)
    base = (fe["ref"] >= 0.70) & (s > 0.6 * exp_) & act & ~dq_mask(fe, a, b)
    raw = base & (d > 0.15)
    br = bridge(raw, g)
    return d, (br & (run_lengths(br) >= 3))


print("=" * 112)
print("GAP-TOLERANT PERSISTENCE ON INJECTED GROUND TRUTH   (D1 rolling-slope, q=0.75 clip)")
print("=" * 112)
print("The question is not 'how much does it recover' but 'what does it recover'.\n")
rows, rec_rows = [], []
for frac in (0.35, 0.65, 1.0):
    for (a, b) in [("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]:
        rel = [c for c in RELIABLE if c not in (a, b)]
        base = df_raw.copy()
        fe0 = preprocess(base, rel)
        ref0 = fe0["ref"]
        act0 = fe0["active"][a] & fe0["active"][b]
        Ptyp = np.nanpercentile((base[a] + base[b])[ref0 >= 0.85], 95)
        days = base[a].groupby(base.index.normalize()).count().pipe(lambda x: x[x > 0]).index
        day_on = (pd.Series(np.random.default_rng(7).random(len(days)) < frac, index=days)
                  .reindex(base.index.normalize()).fillna(False).to_numpy())
        d2, gt = inject(base, a, b, 0.75 * Ptyp, day_on)
        fe = preprocess(d2, rel)
        ref, act = fe["ref"], fe["active"][a] & fe["active"][b]
        # true clip severity, on the un-injected slope
        th_true = theta(base[a] + base[b], ref0, act0, fe0["dayidx"])
        sev = pd.Series(np.clip(1 - 0.75 * Ptyp / (th_true * ref), 0, 1), index=base.index).fillna(0)
        gtv = gt.to_numpy()
        test = (act & (ref >= 0.70) & ~dq_mask(fe, a, b)).to_numpy()
        f0 = d1(d2, fe, a, b, 0)[1]
        f1 = d1(d2, fe, a, b, 1)[1]
        rec = f1 & ~f0
        neg = test & ~gtv
        srows = pd.Series(sev.to_numpy(), index=base.index)
        rows.append(dict(duty=frac, pair=f"{a}+{b}",
                         tp0=int((f0 & gt).sum()), fp0=int((f0 & neg).sum()),
                         tp1=int((f1 & gt).sum()), fp1=int((f1 & neg).sum()),
                         rec_rows=int(rec.sum()),
                         rec_true_clip=int((rec & gt).sum()),
                         rec_med_sev=round(float(srows[rec].median()), 3) if rec.sum() else np.nan))
        print(f"  duty={frac:.2f} {a}+{b}: done")
R = pd.DataFrame(rows)
print()
print(R.to_string(index=False))
print()
print("totals:")
print(f"   strict  (g=0): TP {R.tp0.sum():5d}   FP {R.fp0.sum():5d}   precision "
      f"{R.tp0.sum()/max(R.tp0.sum()+R.fp0.sum(),1):.3f}")
print(f"   bridged (g=1): TP {R.tp1.sum():5d}   FP {R.fp1.sum():5d}   precision "
      f"{R.tp1.sum()/max(R.tp1.sum()+R.fp1.sum(),1):.3f}")
print(f"\n   the {R.rec_rows.sum()} recovered rows: {R.rec_true_clip.sum()} are genuine clip rows "
      f"({100*R.rec_true_clip.sum()/max(R.rec_rows.sum(),1):.0f}%)")
print(f"   median true clip severity of the recovered rows: {R.rec_med_sev.median():.3f}")

# what do the recovered rows look like in deficit terms - are they boundary cases?
print()
print("   margin above the 0.15 threshold, genuine vs spurious recovered rows:")
allrec, allgt, allsev = [], [], []
for frac in (0.35, 0.65, 1.0):
    for (a, b) in [("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]:
        rel = [c for c in RELIABLE if c not in (a, b)]
        base = df_raw.copy()
        fe0 = preprocess(base, rel)
        ref0 = fe0["ref"]; act0 = fe0["active"][a] & fe0["active"][b]
        Ptyp = np.nanpercentile((base[a] + base[b])[ref0 >= 0.85], 95)
        days = base[a].groupby(base.index.normalize()).count().pipe(lambda x: x[x > 0]).index
        day_on = (pd.Series(np.random.default_rng(7).random(len(days)) < frac, index=days)
                  .reindex(base.index.normalize()).fillna(False).to_numpy())
        d2, gt = inject(base, a, b, 0.75 * Ptyp, day_on)
        fe = preprocess(d2, rel)
        ref, act = fe["ref"], fe["active"][a] & fe["active"][b]
        th_true = theta(base[a] + base[b], ref0, act0, fe0["dayidx"])
        sev = pd.Series(np.clip(1 - 0.75 * Ptyp / (th_true * ref), 0, 1), index=base.index).fillna(0)
        d, f0 = d1(d2, fe, a, b, 0)
        _, f1 = d1(d2, fe, a, b, 1)
        rec = (f1 & ~f0).to_numpy()          # use positions, not labels: this loop
        allrec.append(d.to_numpy()[rec])     # reuses one index across 9 scenarios, so
        allgt.append(gt.to_numpy()[rec])     # label alignment would explode to 9x rows
        allsev.append(sev.to_numpy()[rec])
dd = np.concatenate(allrec); gg = np.concatenate(allgt).astype(bool)
print(f"     genuine clip rows  (n={int(gg.sum())}): median deficit {np.median(dd[gg]):.3f}")
print(f"     spurious rows      (n={int((~gg).sum())}): median deficit {np.median(dd[~gg]):.3f}")
print(f"     share of recovered rows with deficit < 0.17: {100*(dd < 0.17).mean():.0f}%")
print("\n   If both sit just above 0.15 the rule is a threshold shift, not a better")
print("   persistence test.")
