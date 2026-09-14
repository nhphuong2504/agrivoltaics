"""
STEP 26 - the figures for METHODS_RESULTS.md (the audience-facing write-up).

These are deliberately SEPARATE from step7_figures.py and step25_validated_evidence.py.
Those two plot the retired EDA baseline (D0) as a comparator, because their job is the audit
in SATURATION_REVIEW.md. The write-up presents the finding on its own terms, so every figure
here shows only this work:

  methods_fig1_rolloff.png      the roll-off, normalisation-free, with internal controls
  methods_fig2_calibration.png  the detector's model vs a model-free truth, per class
  methods_fig3_scoreboard.png   candidate detectors scored on injected ground truth (no D0)
  methods_fig4_evidence.png     the four-panel evidence chain, ROC without D0

D0 stays in sat_work/research/bench.py so the regression suite keeps its documented
baseline, and test_detectors.py keeps asserting against it. It is simply not shown.

Run:  ./venv/Scripts/python.exe sat_work/research/step26_methods_figures.py
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from satsim import EU, PAIRS, CONTROL_PAIRS, INVERTERS  # noqa: E402
import bench as B  # noqa: E402

df = B.raw_df()
fe = B.raw_fe()
ref = fe["ref"]
INV_OF = {u: k for k, us in INVERTERS.items() for u in us}

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.22,
                     "font.size": 9, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})
C_SHARED, C_CTRL, C_OURS = "#c0392b", "#2471a3", "#1e8449"
SHARED_UNITS = [c for p in PAIRS for c in p]

BINS = np.linspace(0.30, 1.05, 21)
CTR = 0.5 * (BINS[1:] + BINS[:-1])
X = df[EU].div(df[EU].quantile(0.999))          # normalised unit output, 0..1


def binmed(x, lo=0.3):
    """Median of `x` per ref bin, as a numpy array aligned to CTR."""
    idx = np.digitize(ref, BINS).clip(1, len(BINS) - 1)
    ser = pd.Series(np.asarray(x, dtype=float), index=df.index)
    ok = (ref > lo) & np.isfinite(ser)
    return ser[ok].groupby(idx[ok]).median().reindex(range(1, len(BINS))).to_numpy()


def peer_truth(a, b):
    """Model-free deficit for a pair: 1 - pair / (k * peer), k fixed in the mid band.

    Nothing here uses theta, cap, g0 or any fitted parameter. The peer is a cross-inverter
    independent pair chosen so it is never the pair under test.
    """
    peer = ("eu_4", "eu_12") if (a, b) == ("eu_1", "eu_3") else ("eu_1", "eu_3")
    P = X[a] + X[b]
    Q = X[peer[0]] + X[peer[1]]
    ratio = P / Q.replace(0, np.nan)
    cal = (ref >= 0.35) & (ref <= 0.60)
    k = ratio[cal].median()
    return 1 - ratio / k


# =========================================================================== #
# FIG 1 - the roll-off is real, and it is not a normalisation artefact
# =========================================================================== #
fig, ax = plt.subplots(figsize=(7.6, 4.6))
for (a, b), col in zip(PAIRS, [C_SHARED, "#e67e22", "#a93226"]):
    P = X[a] + X[b]
    Q = X["eu_1"] + X["eu_3"]
    rr = P / Q.replace(0, np.nan)
    cal = (ref >= 0.35) & (ref <= 0.60)
    ax.plot(CTR, binmed((rr / rr[cal].median())), "-", color=col, lw=2.0, marker="o", ms=3,
            label=f"shared MPPT  {a[3:]}+{b[3:]}")
for a, b in CONTROL_PAIRS:
    P = X[a] + X[b]
    Q = X["eu_1"] + X["eu_3"]
    rr = P / Q.replace(0, np.nan)
    cal = (ref >= 0.35) & (ref <= 0.60)
    ax.plot(CTR, binmed((rr / rr[cal].median())), "--", color=C_CTRL, lw=1.6, marker="o", ms=3,
            label=f"independent  {a[3:]}+{b[3:]}")
ax.axhline(1, color="k", lw=0.8)
ax.axvline(B.REF_ON, color="grey", ls=":", lw=1.2)
ax.text(B.REF_ON + 0.012, 0.545, "detection gate", fontsize=7.5, color="grey", rotation=90,
        va="bottom")
ax.set_xlabel("irradiance proxy  ref")
ax.set_ylabel("pair output ÷ independent peer pair\n(normalised to 1 at ref 0.35–0.60)")
ax.set_title("The roll-off is real and normalisation-free\n"
             "shared-tracker pairs fall away from their peers at high light; "
             "independent pairs do not")
ax.legend(fontsize=7.6, loc="lower left")
fig.tight_layout()
fig.savefig(os.path.join(_HERE, "methods_fig1_rolloff.png"))
plt.close(fig)

# =========================================================================== #
# FIG 2 - the detector's model agrees with a model-free truth, on both classes
# =========================================================================== #
# Model vs truth per ref bin, averaged across each class. The two curves coincide
# almost exactly -- that IS the result -- so they are drawn as line + open markers
# rather than two lines, and the disagreement is broken out explicitly in panel (b).
curves = {}
for grp, lab in [(PAIRS, "shared"), (CONTROL_PAIRS, "control")]:
    m = np.full(len(CTR), np.nan)
    t = np.full(len(CTR), np.nan)
    for a, b in grp:
        d = B.pair_deficit(df, fe, a, b)["deficit"]
        m = np.nanmean([m, binmed(d.to_numpy())], axis=0)
        t = np.nanmean([t, binmed(peer_truth(a, b).to_numpy())], axis=0)
    curves[lab] = (m, t)

fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
ax = axes[0]
COL = {"shared": C_SHARED, "control": C_CTRL}
for lab in ("shared", "control"):
    m, t = curves[lab]
    ax.plot(CTR, t, "o", mfc="none", mec=COL[lab], mew=1.6, ms=6.5, ls="none",
            label=f"{lab} pairs — model-free truth (peer ratio)")
    ax.plot(CTR, m, "-", color=COL[lab], lw=2.1,
            label=f"{lab} pairs — detector  θ(t)·ref")
ax.axhline(0, color="k", lw=0.8)
ax.axhline(B.THR_MOD, color="grey", ls="--", lw=1)
ax.text(0.31, B.THR_MOD + 0.008, "moderate threshold 0.15", fontsize=7.5, color="grey")
ax.axvline(B.REF_ON, color="grey", ls=":", lw=1.2)
ax.text(B.REF_ON + 0.012, 0.006, "detection gate", fontsize=7.5, color="grey", rotation=90,
        va="bottom")
ax.set_xlabel("irradiance proxy  ref")
ax.set_ylabel("median deficit")
ax.set_title("(a) Model against a model-free measurement", loc="left")
ax.legend(fontsize=7.4, loc="upper left")

ax = axes[1]
w = 0.38
xp = np.arange(len(CTR))
for k, lab in enumerate(("shared", "control")):
    m, t = curves[lab]
    ax.bar(xp + (k - 0.5) * w, m - t, w, color=COL[lab],
           label=f"{lab} pairs   mean |residual| {np.nanmean(np.abs(m - t)[CTR >= B.REF_ON]):.3f} "
                 f"over the gate")
ax.axhline(0, color="k", lw=0.9)
ax.axvline(np.interp(B.REF_ON, CTR, xp), color="grey", ls=":", lw=1.2)
ax.set_xticks(xp[::3])
ax.set_xticklabels([f"{c:.2f}" for c in CTR[::3]])
ax.set_xlabel("irradiance proxy  ref")
ax.set_ylabel("detector deficit − model-free truth")
ax.set_title("(b) The disagreement, drawn to scale", loc="left")
ax.legend(fontsize=7.4, loc="upper left")

fig.suptitle("The detector's model reproduces an independent measurement of the same quantity\n"
             "≈0 on pairs that cannot clip; a growing deficit on pairs that do",
             y=1.04)
fig.tight_layout()
fig.savefig(os.path.join(_HERE, "methods_fig2_calibration.png"))
plt.close(fig)

print("fig2 model-vs-truth residual, mean over the gate (ref >= 0.70):")
for lab in ("shared", "control"):
    m, t = curves[lab]
    g = CTR >= B.REF_ON
    print(f"   {lab:8s} model {np.nanmean(m[g]):+.3f}  truth {np.nanmean(t[g]):+.3f}  "
          f"residual {np.nanmean(m[g]-t[g]):+.4f}  max|bin| {np.nanmax(np.abs(m-t)[g]):.3f}")

# =========================================================================== #
# FIG 3 - candidate detectors on injected ground truth (D0 excluded)
# =========================================================================== #
ORDER = ["D1 rolling-slope", "D2 envelope-q90", "D5 control-calibrated", "D4 ceiling-pinned"]
CAND = ["θ(t)·ref  (adopted)", "clear-sky θ₉₀(t)", "control-null calibrated",
        "ceiling-pinned"]
EPI = (("eu_1", "eu_3"), ("eu_15", "eu_23"))

auc_any, auc_epi, fp = {}, {}, {}
for k in ORDER:
    a_, e_, f_ = [], [], []
    for s in B.DEFAULT_SCENARIOS:
        r = B.score_scenario(B.scenario(*s), B.DETECTORS[k])
        a_.append(r["auc00"])
        f_.append(r["fp_rows"])
    for p in EPI:
        e_.append(B.score_scenario(B.scenario(p, 0.35, 0.75, "episodic"),
                                   B.DETECTORS[k])["auc00"])
    auc_any[k] = float(np.nanmean(np.array(a_, dtype=float)))
    auc_epi[k] = float(np.mean(e_))
    fp[k] = float(np.nanmean(f_))
N_POS = int(np.isfinite(np.array(
    [B.score_scenario(B.scenario(*s), B.DETECTORS[ORDER[0]])["auc00"]
     for s in B.DEFAULT_SCENARIOS], dtype=float)).sum())

fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))
xp = np.arange(len(ORDER))
cols = [C_OURS, "#8e44ad", "#16a085", "#7f8c8d"]
for ax, vals, title, ylab in [
        (axes[0], [auc_epi[k] for k in ORDER], "Episodic regime  (the realistic one)",
         "AUC, 35 %-duty scenarios"),
        (axes[1], [auc_any[k] for k in ORDER], f"All {N_POS} scenarios with positives",
         "AUC, any clip (mean)"),
        (axes[2], [fp[k] for k in ORDER], "False alarms on unclipped days",
         "false-positive rows (mean)")]:
    ax.bar(xp, vals, 0.62, color=cols)
    if "AUC" in ylab:
        ax.axhline(0.5, color="k", lw=0.8, ls=":")
        ax.set_ylim(0.4, 1.05)
    else:
        ax.set_yscale("symlog")
    for i, v in enumerate(vals):
        ax.text(i, v * 1.06 if "AUC" in ylab else v * 1.35,
                f"{v:.3f}" if "AUC" in ylab else f"{v:,.0f}", ha="center", fontsize=8)
    ax.set_xticks(xp)
    ax.set_xticklabels(["θ(t)·ref\n(adopted)", "clear-sky\nθ₉₀(t)", "control-null\ncalibrated",
                        "ceiling-\npinned"], fontsize=8.0)
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylab)
fig.suptitle("Candidate detectors, scored on clip-injection ground truth "
             "(positives are labels, not model output)", y=1.03)
fig.tight_layout()
fig.savefig(os.path.join(_HERE, "methods_fig3_scoreboard.png"))
plt.close(fig)

# =========================================================================== #
# FIG 4 - the evidence chain, four independent validations
# =========================================================================== #
def unit_deficit_hi(u, thresh=0.75):
    t = B.theta_roll(df[u], ref, fe["active"][u], fe["dayidx"])
    d = (1 - df[u] / (t * ref)).replace([np.inf, -np.inf], np.nan)
    clean = (fe["active"][u] & ~fe["unit_off"][u] & ~fe["stale"][u]
             & ~fe["negative"][u] & ~fe["site_outage"])
    return float(d[clean & (ref >= thresh)].median())


def roc_points(sc, detector):
    score, _ = detector(sc["df"], sc["fe"], sc["a"], sc["b"])
    a, b = sc["a"], sc["b"]
    test = (sc["fe"]["active"][a] & sc["fe"]["active"][b] & (sc["fe"]["ref"] >= B.REF_ON)
            & ~B.dq_mask(sc["fe"], a, b)).to_numpy()
    s = pd.Series(np.asarray(score, dtype=float), index=sc["fe"]["ref"].index).to_numpy()
    ok = test & np.isfinite(s)
    pos = ok & sc["gt"].to_numpy().astype(bool) & (np.asarray(sc["sev"]) >= 0.10)
    s, pos = s[ok], pos[ok]
    o = np.argsort(-s)
    pos = pos[o]
    tpr = np.concatenate([[0], np.cumsum(pos) / max(pos.sum(), 1)])
    fpr = np.concatenate([[0], np.cumsum(~pos) / max((~pos).sum(), 1)])
    npos, nneg = pos.sum(), (~pos).sum()
    y = pos.astype(float)
    r = np.arange(1, len(y) + 1)
    return fpr, tpr, float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

# (a) roll-off
ax = axes[0, 0]
for (a, b), col in zip(PAIRS, [C_SHARED, "#e67e22", "#a93226"]):
    P = X[a] + X[b]
    Q = X["eu_1"] + X["eu_3"]
    rr = P / Q.replace(0, np.nan)
    cal = (ref >= 0.35) & (ref <= 0.60)
    ax.plot(CTR, binmed((rr / rr[cal].median())), "-", color=col, lw=2.0, marker="o", ms=3,
            label=f"shared MPPT  {a[3:]}+{b[3:]}")
for a, b in CONTROL_PAIRS:
    P = X[a] + X[b]
    Q = X["eu_1"] + X["eu_3"]
    rr = P / Q.replace(0, np.nan)
    cal = (ref >= 0.35) & (ref <= 0.60)
    ax.plot(CTR, binmed((rr / rr[cal].median())), "--", color=C_CTRL, lw=1.6, marker="o", ms=3,
            label=f"independent  {a[3:]}+{b[3:]}")
ax.axhline(1, color="k", lw=0.8)
ax.axvline(B.REF_ON, color="grey", ls=":", lw=1.2)
ax.set_xlabel("irradiance proxy  ref")
ax.set_ylabel("pair output ÷ independent peer\n(normalised to 1 at ref 0.35–0.6)")
ax.set_title("(a) The constraint is real — and normalisation-free\n"
             "shared pairs fall away from peers at high light", loc="left")
ax.legend(fontsize=7.2, loc="lower left")

# (b) natural experiment
ax = axes[0, 1]
labels, clipped, mates = [], [], []
for a, b in PAIRS:
    inv = INV_OF[a]
    mu = [u for u in INVERTERS[inv] if u not in (a, b) and u != "eu_7"]
    labels.append(f"inv {inv}\n{a[3:]}+{b[3:]}")
    clipped.append(np.mean([unit_deficit_hi(a), unit_deficit_hi(b)]))
    mates.append(np.mean([unit_deficit_hi(u) for u in mu]) if mu else np.nan)
xp = np.arange(len(labels))
ax.bar(xp - 0.19, clipped, 0.38, color=C_SHARED,
       label="units sharing an MPPT channel  (clipped)")
ax.bar(xp + 0.19, mates, 0.38, color=C_CTRL,
       label="their inverter mates  (not clipped)")
for i, (c, m) in enumerate(zip(clipped, mates)):
    ax.text(i - 0.19, c + 0.006, f"{100*c:+.0f}%", ha="center", fontsize=8, color=C_SHARED)
    ax.text(i + 0.19, m + 0.006, f"{100*m:+.1f}%", ha="center", fontsize=8, color=C_CTRL)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xp)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("median deficit at high irradiance")
ax.set_title("(b) It is the MPPT, not the inverter\n"
             "same inverter, same conditions — only the shared tracker rolls off", loc="left")
ax.legend(fontsize=7.8, loc="upper right")

# (c) nameplate
ax = axes[1, 0]
th, name = [], []
for u in EU:
    if u == "eu_7":
        continue
    t = B.theta_roll(df[u], ref, fe["active"][u], fe["dayidx"])
    mid = fe["active"][u] & (ref >= 0.35) & (ref <= 0.60)
    th.append(100.0 * (float(t[mid].median()) / 9000.0 - 1.0))
    name.append(u.replace("eu_", ""))
order = np.argsort(th)
th = np.array(th)[order]
name = np.array(name)[order]
cols = [C_SHARED if f"eu_{n}" in SHARED_UNITS else "#7f8c8d" for n in name]
ax.bar(np.arange(len(th)), th, color=cols, width=0.72)
ax.axhline(0, color="k", lw=1.2)
env = float(abs(th).max())
ax.fill_between([-0.5, len(th) - 0.5], -env, env, color="k", alpha=0.07, zorder=0)
ax.text(len(th) - 0.4, env + 0.15, f"all within {env:.1f} %", ha="right", fontsize=7.5,
        color="#555555")
ax.set_xticks(np.arange(len(th)))
ax.set_xticklabels(name, fontsize=6.6, rotation=90)
ax.set_ylim(-env - 1.4, env + 2.1)
ax.set_xlim(-0.5, len(th) - 0.5)
ax.set_ylabel("detector slope  θ  vs 9.0 kW nameplate  (%)")
ax.set_title("(c) The detector's slope IS the hardware rating\n"
             f"nothing was tuned to this; median {9000*(1+np.median(th)/100)/1000:.2f} kW, "
             f"all {len(th)} healthy units within {env:.1f} %", loc="left")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_SHARED),
                   plt.Rectangle((0, 0), 1, 1, color="#7f8c8d")],
          labels=["shares an MPPT channel (clipped)", "independent"],
          fontsize=7.5, loc="lower left", ncol=2)

# (d) ROC
ax = axes[1, 1]
ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1.2, label="chance")
aucs = []
for i, p in enumerate(EPI):
    sc = B.scenario(p, 0.35, 0.75, "episodic")
    f, t, a_ = roc_points(sc, B.det_rolling)
    aucs.append(a_)
    ax.plot(f, t, lw=2.0, color=C_OURS, alpha=1.0 if i == 0 else 0.5,
            label=f"detector, mean AUC {np.mean(aucs):.3f}" if i == 0 else None)
ax.set_xlabel("false-positive rate  (unclipped rows)")
ax.set_ylabel("true-positive rate  (clipped rows, > 10 % deep)")
ax.set_xlim(-0.01, 1.0)
ax.set_ylim(0, 1.01)
ax.set_title("(d) The detector works on labelled ground truth\n"
             "clip-injection benchmark, 35 % of days, both episodic scenarios", loc="left")
ax.legend(fontsize=7.8, loc="lower right")

fig.suptitle("MPPT saturation: the constraint is real, it is at the tracker, and the "
             "detector that finds it is validated",
             fontsize=13.5, fontweight="bold", y=0.985)
fig.tight_layout(rect=(0, 0, 1, 0.972))
fig.savefig(os.path.join(_HERE, "methods_fig4_evidence.png"))
plt.close(fig)

# =========================================================================== #
print("methods_fig1_rolloff.png      written")
print("methods_fig2_calibration.png  written")
print("methods_fig3_scoreboard.png   written")
print("methods_fig4_evidence.png     written")
print()
print("scoreboard numbers quoted in METHODS_RESULTS.md (D0 excluded):")
print(f"  {'candidate':32s} {'episodic':>9s} {'any clip':>9s} {'FP rows':>9s}")
for k, c in zip(ORDER, CAND):
    print(f"  {c:32s} {auc_epi[k]:9.3f} {auc_any[k]:9.3f} {fp[k]:9.1f}")
