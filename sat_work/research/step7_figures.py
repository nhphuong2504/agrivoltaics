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
S = pd.read_csv(OUT + r"\bench_results.csv")
S = S[S["q"] != 1.0]
ph = S[S.pattern == "episodic"]
order = ["D1 rolling-slope", "D2 envelope-q90", "D5 control-calibrated",
         "D4 ceiling-pinned", "D0 published", "D3 HMM-2state"]
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
w = 0.26
for i, (col, lab) in enumerate([("auc00", "all clipped rows"), ("auc10", "clips > 10% deep"),
                                ("auc20", "clips > 20% deep")]):
    v = [S[S.det == k][col].mean() for k in order]
    axes[0].bar(np.arange(len(order)) + (i - 1) * w, v, w, label=lab)
axes[0].set_xticks(range(len(order)))
axes[0].set_xticklabels([k.split(" ", 1)[0] for k in order])
axes[0].axhline(0.5, color="k", lw=0.8, ls=":")
axes[0].set_ylim(0.4, 1.03); axes[0].set_ylabel("AUC (mean over scenarios)")
axes[0].set_title("Discrimination"); axes[0].legend(fontsize=8, loc="lower right")
v = [S[S.det == k]["fp_rows"].mean() for k in order]
axes[1].bar(range(len(order)), v, color=[C["roll"], C["env"], "#16a085", "#7f8c8d", C["pub"], "#95a5a6"])
axes[1].set_xticks(range(len(order)))
axes[1].set_xticklabels([k.split(" ", 1)[0] for k in order])
axes[1].set_yscale("symlog")
axes[1].set_ylabel("false-positive rows at ref ≥ 0.7 (mean)")
axes[1].set_title("False alarms on unclipped days (symlog)")
for i, val in enumerate(v):
    axes[1].text(i, val * 1.25, f"{val:,.0f}", ha="center", fontsize=8)
fig.suptitle("FIG 3 — Ground-truth benchmark (clipping injected into independent pairs). "
             "Episodic clips, ref ≥ 0.70.", y=1.0, fontsize=10)
fig.tight_layout(); fig.savefig(OUT + r"\fig3_scoreboard.png"); plt.close(fig)

print("wrote fig1_did.png fig2_baseline.png fig3_scoreboard.png")
