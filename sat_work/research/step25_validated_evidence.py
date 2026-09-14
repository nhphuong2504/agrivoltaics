"""
STEP 25 - the validated evidence figure (the one figure the deck leads with).

Four panels, four *independent* validations. The point of putting them together is that no
single one of them is a model output checking another model output:

  (a) real data, normalisation-free  - a difference-in-differences roll-off. Shared-MPPT
      pairs fall away from independent peer pairs at high irradiance; control pairs do not.
      Validation: an internal control, so nothing here depends on the detector.
  (b) real data, grounded in hardware - the natural experiment the inverter topology gives
      us. The clipped pairs roll off, their inverter mates do not. Validation: it is the
      MPPT that clips, not the inverter, which rules out the most obvious confound.
  (c) physical cross-check           - the detector's slope theta against the documented
      9.0 kW string rating. Validation: an external hardware spec the analysis never used.
  (d) injected ground truth          - ROC on the clip-injection benchmark. Validation:
      labelled positives, so the detector is scored rather than asserted.

Panels (a)-(c) say the problem is real and is where we think it is. Panel (d) says the
detector that finds it works. That is the whole argument in one image.

Run:  ./venv/Scripts/python.exe sat_work/research/step25_validated_evidence.py
Writes: sat_work/research/validated_evidence.png
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

OUT = os.path.join(_HERE, "validated_evidence.png")
df = B.raw_df()
fe = B.raw_fe()
ref = fe["ref"]
INV = INVERTERS
INV_OF = {u: k for k, us in INV.items() for u in us}

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.22,
                     "font.size": 9, "axes.titlesize": 10.5,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})
C_SHARED, C_CTRL, C_PUB, C_VAT = "#c0392b", "#2471a3", "#d35400", "#1e8449"

fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

# ========================================================================== #
# (a) DiD roll-off - real data, no normalisation to the detector
# ========================================================================== #
ax = axes[0, 0]
BINS = np.linspace(0.30, 1.05, 21)
CTR = 0.5 * (BINS[1:] + BINS[:-1])
X = df[EU].div(df[EU].quantile(0.999))
peer = X["eu_1"] + X["eu_3"]                      # cross-inverter independent peer
for (a, b), col, ls in [(("eu_8", "eu_16"), C_SHARED, "-"),
                        (("eu_10", "eu_18"), "#e67e22", "-"),
                        (("eu_13", "eu_21"), "#a93226", "-")]:
    P = X[a] + X[b]
    rr = P / peer.replace(0, np.nan)
    cal = (ref >= 0.35) & (ref <= 0.6)
    rr = rr / rr[cal].median()
    idx = np.digitize(ref, BINS).clip(1, len(BINS) - 1)
    ok = (ref > 0.3) & rr.notna()
    ax.plot(CTR, rr[ok].groupby(idx[ok]).median().reindex(range(1, len(BINS))).values,
            ls, color=col, lw=2.0, marker="o", ms=3,
            label=f"shared MPPT  {a.replace('eu_','')}+{b.replace('eu_','')}")
for a, b in CONTROL_PAIRS:
    P = X[a] + X[b]
    rr = P / peer.replace(0, np.nan)
    cal = (ref >= 0.35) & (ref <= 0.6)
    rr = rr / rr[cal].median()
    idx = np.digitize(ref, BINS).clip(1, len(BINS) - 1)
    ok = (ref > 0.3) & rr.notna()
    ax.plot(CTR, rr[ok].groupby(idx[ok]).median().reindex(range(1, len(BINS))).values,
            "--", color=C_CTRL, lw=1.6, marker="o", ms=3,
            label=f"independent  {a.replace('eu_','')}+{b.replace('eu_','')}")
ax.axhline(1, color="k", lw=0.8)
ax.axvline(B.REF_ON, color="grey", ls=":", lw=1.2)
ax.text(B.REF_ON + 0.012, ax.get_ylim()[0] + 0.03, "detection gate", fontsize=7.5,
        color="grey", rotation=90, va="bottom")
ax.set_xlabel("irradiance proxy  ref")
ax.set_ylabel("pair output ÷ independent peer\n(normalised to 1 at ref 0.35–0.6)")
ax.set_title("(a) The constraint is real — and normalisation-free\n"
             "shared pairs fall away from peers at high light", loc="left")
ax.legend(fontsize=7.2, loc="lower left")

# ========================================================================== #
# (b) the natural experiment - is it the MPPT or the inverter?
# ========================================================================== #
ax = axes[0, 1]


def unit_deficit_hi(u, thresh=0.75):
    s = df[u]
    act = fe["active"][u]
    t = B.theta_roll(s, ref, act, fe["dayidx"])
    d = (1 - s / (t * ref)).replace([np.inf, -np.inf], np.nan)
    clean = (act & ~fe["unit_off"][u] & ~fe["stale"][u]
             & ~fe["negative"][u] & ~fe["site_outage"])
    return float(d[clean & (ref >= thresh)].median())


labels, clipped, mates = [], [], []
for a, b in PAIRS:
    inv = INV_OF[a]
    mate_units = [u for u in INV[inv] if u not in (a, b) and u != "eu_7"]
    labels.append(f"inv {inv}\n{a[3:]}+{b[3:]}")
    clipped.append(np.mean([unit_deficit_hi(a), unit_deficit_hi(b)]))
    mates.append(np.mean([unit_deficit_hi(u) for u in mate_units]) if mate_units else np.nan)
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

# ========================================================================== #
# (c) physical cross-check against the 9.0 kW nameplate
# ========================================================================== #
ax = axes[1, 0]
th, name = [], []
for u in EU:
    if u == "eu_7":
        continue
    t = B.theta_roll(df[u], ref, fe["active"][u], fe["dayidx"])
    mid = fe["active"][u] & (ref >= 0.35) & (ref <= 0.60)
    th.append(100.0 * (float(t[mid].median()) / 9000.0 - 1.0))   # % vs the 9.0 kW rating
    name.append(u.replace("eu_", ""))
order = np.argsort(th)
th = np.array(th)[order]
name = np.array(name)[order]
SHARED_UNITS = [c for p in PAIRS for c in p]
cols = [C_SHARED if f"eu_{n}" in SHARED_UNITS else "#7f8c8d" for n in name]
# Plotted as DEVIATION, not as absolute kW. Absolute bars on a truncated axis would all be
# clipped to the same visible height and hide the very variation the panel is about; a
# zero-baseline deviation plot shows it honestly and reads directly as "within +/-4 %".
ax.bar(np.arange(len(th)), th, color=cols, width=0.72)
ax.axhline(0, color="k", lw=1.2)
# The band IS the observed envelope, not a round number the data happens to cross: the
# worst unit deviates 4.14 %, so shading +/-4 % would draw a boundary one bar pokes
# through. Shade the measured maximum and label it, which is the claim being made.
env = float(abs(th).max())
ax.fill_between([-0.5, len(th) - 0.5], -env, env, color="k", alpha=0.07, zorder=0)
ax.text(len(th) - 0.4, env + 0.15, f"all within {env:.1f} %", ha="right", fontsize=7.5,
        color="#555555")
ax.set_xticks(np.arange(len(th)))
ax.set_xticklabels(name, fontsize=6.6, rotation=90)
ax.set_ylim(-env - 1.4, env + 2.1)
ax.set_xlim(-0.5, len(th) - 0.5)
ax.set_ylabel("detector slope  θ  vs 9.0 kW nameplate  (%)")
ax.set_title(f"(c) The detector's slope IS the hardware rating\n"
             f"nothing was tuned to this; median {9000*(1+np.median(th)/100)/1000:.2f} kW, "
             f"all {len(th)} healthy units within {env:.1f} %", loc="left")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C_SHARED),
                   plt.Rectangle((0, 0), 1, 1, color="#7f8c8d")],
          labels=["shares an MPPT channel (clipped)", "independent"],
          fontsize=7.5, loc="lower left", ncol=2)

# ========================================================================== #
# (d) ROC on injected ground truth
# ========================================================================== #
ax = axes[1, 1]


def roc_points(sc, detector):
    score, flag = detector(sc["df"], sc["fe"], sc["a"], sc["b"])
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
    return fpr, tpr, roc_auc(pos, s)


def roc_auc(y, s):
    o = np.argsort(s)
    y = y[o].astype(float)
    r = np.arange(1, len(y) + 1)
    npos, nneg = y.sum(), (1 - y).sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


EPI = [("eu_1", "eu_3"), ("eu_15", "eu_23")]
series = {C_PUB: ("published  g0·cap(t)·ref", B.det_published, []),
          C_VAT: ("vat-v1  θ(t)·ref", B.det_rolling, [])}
for col, (lab, det, acc) in series.items():
    for p in EPI:
        f, t, _ = roc_points(B.scenario(p, 0.35, 0.75, "episodic"), det)
        acc.append((f, t))
ax.plot([0, 1], [0, 1], ls=":", color="grey", lw=1.2, label="chance")
for col, (lab, det, acc) in series.items():
    aucs = []
    for p in EPI:
        sc = B.scenario(p, 0.35, 0.75, "episodic")
        score, _ = det(sc["df"], sc["fe"], sc["a"], sc["b"])
        x, y, a_ = roc_points(sc, det)
        aucs.append(a_)
    # pool the two scenarios by concatenating their curves' underlying rates is not valid,
    # so draw each and label the mean AUC -- the same convention step23 uses
    for i, (f, t) in enumerate(acc):
        ax.plot(f, t, lw=2.0, color=col,
                alpha=1.0 if i == 0 else 0.55,
                label=f"{lab}   mean AUC {np.mean(aucs):.3f}" if i == 0 else None)
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
fig.savefig(OUT)
plt.close(fig)
print(f"wrote {OUT}")
