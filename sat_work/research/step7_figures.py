"""STEP 7 - figures for the research review."""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = load(); FE = preprocess(df); ref = FE["ref"]
OUT = r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research"
FULLCAL = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 10, "font.size": 9})
C = dict(shared="#c0392b", ctrl="#2471a3", pub="#c0392b", roll="#1e8449", env="#8e44ad", did="#2c3e50")

def broadcast(per_day, dayidx, labels, window="31D", minp=5):
    roll = per_day.reindex(FULLCAL).rolling(window, center=True, min_periods=minp).median()
    return pd.Series(roll.reindex(dayidx).to_numpy(), index=labels)

def theta_roll(s, act, q=0.5, band=(0.35, 0.6)):
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= band[0]) & (ref <= band[1]) & u.notna()
    g = u[sel].groupby(FE["dayidx"][sel])
    per_day = g.quantile(q) if q != 0.5 else g.median()
    return broadcast(per_day, FE["dayidx"], s.index)

# ---------------------------------------------------------------- FIG 1: DiD
X = df[EU].div(df[EU].quantile(0.999))
BINS = np.linspace(0.30, 1.05, 21); CTR = 0.5 * (BINS[1:] + BINS[:-1])
fig, ax = plt.subplots(figsize=(7.2, 4.4))
for a, b, lab, col in [("eu_8", "eu_16", "shared  eu_8+eu_16", C["pub"]),
                       ("eu_10", "eu_18", "shared  eu_10+eu_18", "#e67e22"),
                       ("eu_13", "eu_21", "shared  eu_13+eu_21", "#a93226"),
                       ("eu_4", "eu_12", "control eu_4+eu_12", C["ctrl"]),
                       ("eu_15", "eu_23", "control eu_15+eu_23", "#5dade2")]:
    P = X[a] + X[b]; Q = X["eu_1"] + X["eu_3"]
    rr = (P / Q.replace(0, np.nan))
    cal = (ref >= 0.35) & (ref <= 0.6)
    rr = rr / rr[cal].median()
    idx = np.digitize(ref, BINS).clip(1, len(BINS) - 1)
    ok = (ref > 0.3) & rr.notna()
    med = rr[ok].groupby(idx[ok]).median().reindex(range(1, len(BINS)))
    ls = "-" if lab.startswith("shared") else "--"
    ax.plot(CTR, med.values, ls, color=col, lw=1.9, marker="o", ms=2.6, label=lab)
ax.axhline(1, color="k", lw=0.7)
ax.axvline(0.7, color="grey", ls=":", lw=1)
ax.set_xlabel("irradiance proxy  ref  (fleet median of unit / p99.9)")
ax.set_ylabel("pair output ÷ independent peer output\n(normalised to 1 at ref = 0.35–0.6)")
ax.set_title("FIG 1 — Normalisation-free test: shared pairs fall away from peers at high light,\n"
             "control pairs do not (the roll-off is real, not an artefact of ref or cap)")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(OUT + r"\fig1_did.png"); plt.close(fig)

# ------------------------------------------------- FIG 2: baseline calibration
ALLP = [("eu_8", "eu_16"), ("eu_10", "eu_18"), ("eu_13", "eu_21"),
        ("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]
def by_ref(x):
    idx = np.digitize(ref, BINS).clip(1, len(BINS) - 1)
    ser = pd.Series(x, index=df.index)
    ok = (ref > 0.3) & np.isfinite(ser)
    return ser[ok].groupby(idx[ok]).median().reindex(range(1, len(BINS))).to_numpy()

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True)
for ax, (a, b), kind in [(axes[0], ("eu_8", "eu_16"), "shared"),
                         (axes[1], ("eu_1", "eu_3"), "control (cannot saturate)")]:
    s = df[a] + df[b]; act = FE["active"][a] & FE["active"][b]
    d_pub = pd.Series(detect(df, FE, a, b)["deficit"], index=df.index)
    th = theta_roll(s, act, q=0.5); d_roll = 1 - s / (th * ref)
    th9 = theta_roll(s, act, q=0.9); d_env = 1 - s / (th9 * ref)
    p = df["eu_1"] + df["eu_3"]; k = (s / p)[act & (ref >= 0.35) & (ref <= 0.6)].median()
    d_did = 1 - s / (k * p)
    ax.plot(CTR, by_ref(d_pub), color=C["pub"], lw=2, label="published  g0·cap·ref")
    ax.plot(CTR, by_ref(d_roll), color=C["roll"], lw=2, label="rolling    θ(t)·ref")
    ax.plot(CTR, by_ref(d_env), color=C["env"], lw=1.5, ls="-.", label="envelope   θ₉₀(t)·ref")
    ax.plot(CTR, by_ref(d_did), color=C["did"], lw=1.5, ls=":", label="truth      peer ratio")
    ax.axhline(0.15, color="grey", ls="--", lw=1)
    ax.text(0.31, 0.157, "flag threshold 0.15", fontsize=7.5, color="grey")
    ax.set_title(f"{a}+{b}  [{kind}]")
    ax.set_xlabel("ref")
axes[0].set_ylabel("median deficit")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("FIG 2 — The published baseline reports a growing deficit on a pair that cannot saturate.\n"
             "The rolling baseline is flat at 0 on the same pair.", y=1.0, fontsize=10)
fig.tight_layout(); fig.savefig(OUT + r"\fig2_baseline.png"); plt.close(fig)

# ------------------------------------------------- FIG 3: detector scoreboard
# Recomputed from the FROZEN harness (bench.DEFAULT_SCENARIOS + bench.DETECTORS) rather than
# read from the stale bench_results.csv of step3's older 13-scenario matrix. Review section
# 3.5 quotes this figure, so it and the table must come from the same scenarios, the same
# detector implementations and the same evaluation domain. step23_metric_audit.py prints the
# same table in text form.
import bench as _B

SCEN = _B.DEFAULT_SCENARIOS
ORDER = ["D1 rolling-slope", "D2 envelope-q90", "D5 control-calibrated",
         "D0 published", "D4 ceiling-pinned"]
AUC = {k: [] for k in ORDER}
FP = {k: [] for k in ORDER}
for name, fn in _B.DETECTORS.items():
    if name not in AUC:
        continue
    for s in SCEN:
        r = _B.score_scenario(_B.scenario(*s), fn)
        AUC[name].append(r["auc00"])          # None where the scenario has no positives
        FP[name].append(r["fp_rows"])
AUCm = {k: np.nanmean(np.array(v, dtype=float)) for k, v in AUC.items()}
FPm = {k: np.nanmean(v) for k, v in FP.items()}
N_POS = int(np.isfinite(np.array(AUC[ORDER[0]], dtype=float)).sum())

fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
COLS = {"D1 rolling-slope": C["roll"], "D2 envelope-q90": C["env"],
        "D5 control-calibrated": "#16a085", "D4 ceiling-pinned": "#7f8c8d",
        "D0 published": C["pub"]}
v = [AUCm[k] for k in ORDER]
axes[0].bar(range(len(ORDER)), v, 0.6, color=[COLS[k] for k in ORDER])
axes[0].axhline(0.5, color="k", lw=0.8, ls=":")
for i, val in enumerate(v):
    axes[0].text(i, val + 0.008, f"{val:.3f}", ha="center", fontsize=8)
axes[0].set_xticks(range(len(ORDER)))
axes[0].set_xticklabels([k.split(" ", 1)[0] for k in ORDER])
axes[0].set_ylim(0.4, 1.05)
axes[0].set_ylabel("AUC, any clip (mean)")
axes[0].set_title(f"Discrimination — mean over the {N_POS} scenarios with positives")
axes[1].bar(range(len(ORDER)), [FPm[k] for k in ORDER], 0.6,
            color=[COLS[k] for k in ORDER])
axes[1].set_xticks(range(len(ORDER)))
axes[1].set_xticklabels([k.split(" ", 1)[0] for k in ORDER])
axes[1].set_yscale("symlog")
axes[1].set_ylabel("false-positive rows at ref >= 0.7 (mean)")
axes[1].set_title("False alarms on unclipped days (symlog)")
for i, k in enumerate(ORDER):
    axes[1].text(i, FPm[k] * 1.25, f"{FPm[k]:,.0f}", ha="center", fontsize=8)
fig.suptitle("FIG 3 — Ground-truth benchmark (clipping injected into independent pairs). "
             f"Mean over {'all' if N_POS == len(SCEN) else str(N_POS) + ' of ' + str(len(SCEN))} "
             "scenarios; the no-clip scenario has no positives and is excluded.", y=1.0, fontsize=10)
fig.tight_layout(); fig.savefig(OUT + r"\fig3_scoreboard.png"); plt.close(fig)
print("fig3 AUC  :", {k: round(AUCm[k], 3) for k in ORDER})
print("fig3 FP   :", {k: round(FPm[k]) for k in ORDER})

print("wrote fig1_did.png fig2_baseline.png fig3_scoreboard.png")
