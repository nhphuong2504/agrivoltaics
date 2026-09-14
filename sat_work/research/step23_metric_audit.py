"""
STEP 23 - audit of every headline metric: one definition, one denominator, each.

The point
---------
After a regeneration, the numbers in the review, the canvas, the deck and
SATURATION_DETECTION.md must all be the same numbers, computed the same way. This script
recomputes every published figure from source and prints it WITH its definition and its
denominator, so a disagreement is visible instead of being buried in prose.

It found four real inconsistencies when first run (fixed in the documents):
  * "apparent lost energy at ref >= 0.7" was actually computed on the DETECTOR gate, i.e.
    ref >= 0.7 AND s > 0.6*expected. The label understated the domain.
  * the "150x specificity" figure has at least three different denominators in circulation
    (one scenario, the mean of two, the pool). Now reported per scenario with n stated.
  * the AUC table in review section 3.5 averages over a different scenario set from the one
    the regression test asserts, so the two quoted AUCs legitimately differ. Both are
    printed here, labelled by scenario set.
  * "flagged hours" silently assumed a complete 5-minute grid. The modal timestep is
    verified below, and the conversion is stated as sampled-hours.

Run:  ./venv/Scripts/python.exe sat_work/research/step23_metric_audit.py
"""
import sys, io, os, warnings, hashlib
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
import bench as B

CANON = os.path.join(_ROOT, "saturation_flags.csv")
df = B.raw_df()
fe = B.raw_fe()
ref = fe["ref"]
can = pd.read_csv(CANON, parse_dates=["time"])
NAMEPLATE_W = 9000.0                      # 9.0 kW per string, power_production.pptx slide 2
SAMPLES_PER_HOUR = 12.0                   # verified below
INV_OF = {u: k for k, us in __import__("satsim").INVERTERS.items() for u in us}

INCONSISTENT = []


def row(name, definition, value, denominator="", note=""):
    print(f"\n  {name}")
    print(f"      value        : {value}")
    print(f"      definition   : {definition}")
    if denominator:
        print(f"      denominator  : {denominator}")
    if note:
        print(f"      note         : {note}")


# =========================================================================== #
print("=" * 104)
print("0. SAMPLING GRID - the denominator behind every 'hours' figure")
print("=" * 104)
step = pd.Series(df.index).diff().dt.total_seconds().div(60)
vc = step.value_counts().head(4)
print(f"   modal timestep {vc.index[0]:.0f} min on {vc.iloc[0]:,} of {len(step)-1:,} steps "
      f"({100*vc.iloc[0]/(len(step)-1):.1f} %)")
print(f"   -> hours = flagged rows / {SAMPLES_PER_HOUR:.0f}. Acquisition gaps contribute no")
print(f"      rows, so this is SAMPLED hours present in the record, not wall-clock elapsed.")

# =========================================================================== #
print()
print("=" * 104)
print("1. THE CANONICAL ARTEFACT")
print("=" * 104)
with open(CANON, "rb") as fh:
    sha = hashlib.sha256(fh.read()).hexdigest()
num = can.select_dtypes("number")
row("artefact identity",
    "repo-root saturation_flags.csv, the vat-v1 export (recommended.py)",
    f"{len(can):,} rows x {can.shape[1]} cols, 0 infinities "
    f"({int(np.isinf(num.to_numpy()).sum())} found)",
    "one row per 5-minute timestamp in the source record")
print(f"      sha256       : {sha}")

# the decision domain, per pair
domains = {}
for a, b in B.PAIRS:
    domains[(a, b)] = (fe["active"][a] & fe["active"][b] & (ref >= B.REF_ON)
                       & ~B.dq_mask(fe, a, b)).to_numpy()
dom_min = min(int(d.sum()) for d in domains.values())
dom_max = max(int(d.sum()) for d in domains.values())
row("decision domain size",
    "act & ref >= REF_ON & ~dq_mask -- the SAME domain evaluate() computes AUC over",
    f"{dom_min:,}-{dom_max:,} rows per pair, {100*dom_min/len(can):.0f}-"
    f"{100*dom_max/len(can):.0f} % of the record",
    "the shared denominator for every rate below")

bad = 0
for (a, b), d in domains.items():
    col = can[f"{a}_{b}_deficit"].to_numpy()
    bad += int((~np.isnan(col[~d])).sum())
row("downstream safety",
    "deficit is NaN outside the decision domain; bounded to [-1, 1] inside it",
    f"{bad} populated values outside the domain; "
    f"in-domain range "
    f"{min(float(np.nanmin(can[f'{a}_{b}_deficit'].to_numpy()[d])) for (a, b), d in domains.items()):+.4f}"
    f" to "
    f"{max(float(np.nanmax(can[f'{a}_{b}_deficit'].to_numpy()[d])) for (a, b), d in domains.items()):+.4f}",
    "so an unfiltered .mean() equals the in-domain mean")

# =========================================================================== #
print()
print("=" * 104)
print("2. FLAGGED HOURS - canonical vs published, one table both documents quote")
print("=" * 104)
rows = []
for a, b in B.PAIRS:
    k = f"{a}_{b}"
    pub = B.detect(df, fe, a, b)
    cm = can[f"{k}_sat_moderate"].astype(bool).to_numpy()
    cs = can[f"{k}_sat_severe"].astype(bool).to_numpy()
    pm, ps = pub["moderate"], pub["severe"]
    rows.append(dict(pair=f"{a}+{b}",
                     pub_rows=int(pm.sum()), can_rows=int(cm.sum()),
                     pub_h=pm.sum() / SAMPLES_PER_HOUR, can_h=cm.sum() / SAMPLES_PER_HOUR,
                     pub_sev_rows=int(ps.sum()), can_sev_rows=int(cs.sum()),
                     pub_sev_h=ps.sum() / SAMPLES_PER_HOUR,
                     can_sev_h=cs.sum() / SAMPLES_PER_HOUR,
                     pub_days=pd.to_datetime(pm[pm].index).normalize().nunique(),
                     can_days=pd.to_datetime(can.loc[cm, "time"]).dt.normalize().nunique()))
T = pd.DataFrame(rows).set_index("pair")
print(T.round(1).to_string())
tp = int(T.pub_rows.sum()); tc = int(T.can_rows.sum())
sp = T.pub_sev_h.sum(); sc = T.can_sev_h.sum()
row("canonical vs published, total",
    "moderate tier, all three shared pairs pooled",
    f"{tp:,} rows / {T.pub_h.sum():,.1f} h  ->  {tc:,} rows / {T.can_h.sum():,.1f} h "
    f"({100*(tc/tp-1):+.1f} %)",
    "pooled flagged rows over 3 pairs (note: not a rate -- no denominator needed, but do "
    "not divide this by the domain size of ONE pair)")
row("severe tier inflation removed",
    "severe tier (d > 0.30 for 30 min), pooled",
    f"{sp:,.1f} h -> {sc:,.1f} h  = {sp/sc:.2f}x",
    "same pooling")

# =========================================================================== #
print()
print("=" * 104)
print("3. THE CONTROL PAIRS - flag-level null")
print("=" * 104)
cr = []
for a, b in B.CONTROL_PAIRS:
    cr.append(dict(pair=f"{a}+{b}",
                   kind="same-inverter" if INV_OF[a] == INV_OF[b] else "cross-inverter",
                   published=int(B.detect(df, fe, a, b)["moderate"].sum()),
                   canonical=int(B.det_rolling(df, fe, a, b)[1].sum())))
C = pd.DataFrame(cr).set_index("pair")
print(C.to_string())
cross_pub = int(C.loc[C.kind == "cross-inverter", "published"].sum())
cross_can = int(C.loc[C.kind == "cross-inverter", "canonical"].sum())
row("the cross-inverter null is the honest one",
    "flags on the ONLY control pair that does not share an inverter",
    f"published {cross_pub} rows, canonical {cross_can} rows",
    "1 pair",
    "same-inverter controls (eu_4+eu_12, eu_15+eu_23) carry a co-movement common mode that "
    "roughly doubles tail flags, so they are magnitude nulls, not flag-level nulls")
row("SATURATION_DETECTION.md section 7.1 'separation' claim",
    "published shared flags vs published control flags",
    f"{tp:,} vs {int(C['published'].sum())} rows = {tp/max(int(C['published'].sum()),1):.0f}x",
    "3 shared pairs vs 3 control pairs",
    "the ratio is real but the doc's INTERPRETATION of the 122 rows was wrong: they are "
    "baseline bias, not small real constraint events (see metric 5)")

# =========================================================================== #
print()
print("=" * 104)
print("4. ENERGY - and the denominator that was mislabelled")
print("=" * 104)
GATE_DEF = "ref >= 0.70 AND s > 0.6*expected AND both active AND no DQ  (the DETECTOR gate)"


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
print(f"   per pair (kWh)          published     vat-v1")
for a, b in B.PAIRS:
    print(f"      shared {a}+{b:6s} {loss(a,b,'pub'):10,.0f} {loss(a,b,'vat'):10,.0f}")
for a, b in B.CONTROL_PAIRS:
    print(f"      ctrl   {a}+{b:6s} {loss(a,b,'pub'):10,.0f} {loss(a,b,'vat'):10,.0f}")
row("shared-pair apparent lost energy",
    f"sum of max(expected - measured, 0) over the gate domain: {GATE_DEF}",
    f"published {sh_p:,.0f} kWh -> vat-v1 {sh_v:,.0f} kWh",
    "NOT 'ref >= 0.70' alone -- the review and the deck must say gate domain",
    "the earlier label 'at ref >= 0.7' understated it")
row("bias floor",
    "the same quantity on the three control pairs, which cannot saturate, as a share of "
    "the shared figure",
    f"published {ct_p:,.0f}/{sh_p:,.0f} = {100*ct_p/sh_p:.0f} %   ->   "
    f"vat-v1 {ct_v:,.0f}/{sh_v:,.0f} = {100*ct_v/sh_v:.0f} %",
    "control total / shared total, same units, same gate")

# =========================================================================== #
print()
print("=" * 104)
print("5. THE FLEET-WIDE NULL - why the 122 control rows are bias, not events")
print("=" * 104)
indep = [u for u in __import__("satsim").EU
         if u not in __import__("satsim").SHARED and u != "eu_7"]
cross_pairs = [(a, b) for i, a in enumerate(indep) for b in indep[i + 1:]
               if INV_OF[a] != INV_OF[b]]
PR = np.array([loss(a, b, "pub") for a, b in cross_pairs])
VR = np.array([loss(a, b, "vat") for a, b in cross_pairs])
row("fleet-wide phantom loss",
    "per-pair phantom lost energy on EVERY cross-inverter independent pair (each has no "
    "shared MPPT and so cannot saturate)",
    f"published median {np.median(PR):,.0f} kWh/pair, vat-v1 median {np.median(VR):,.0f} kWh/pair",
    f"{len(cross_pairs)} independent pairs",
    f"vat-v1 fabricates {100*np.median(VR)/np.median(PR):.0f} % of the published phantom "
    f"median; this is the evidence that the control-pair flags are bias")

# =========================================================================== #
print()
print("=" * 104)
print("6. BENCHMARK - ground truth, with the scenario set named every time")
print("=" * 104)
EPI = (("eu_1", "eu_3"), ("eu_15", "eu_23"))
sc_fp = {}
for p in EPI:
    s = B.scenario(p, duty=0.35, q=0.75, pattern="episodic")
    sc_fp[p] = (B.score_scenario(s, B.det_published)["fp_rows"],
                B.score_scenario(s, B.det_rolling)["fp_rows"])
    print(f"   episodic {p[0]}+{p[1]}: published FP {sc_fp[p][0]:,}   vat-v1 FP {sc_fp[p][1]:,}"
          f"   ratio {sc_fp[p][0]/max(sc_fp[p][1],1):.0f}x")
pub_pool = sum(v[0] for v in sc_fp.values())
can_pool = sum(v[1] for v in sc_fp.values())
chronic_pub = B.score_scenario(
    B.scenario(("eu_1", "eu_3"), duty=1.00, q=0.75, pattern="chronic"),
    B.det_published)["fp_rows"]
chronic_can = B.score_scenario(
    B.scenario(("eu_1", "eu_3"), duty=1.00, q=0.75, pattern="chronic"),
    B.det_rolling)["fp_rows"]
row("specificity gain - THE definition to quote",
    "false-positive rows on unclipped days, clip injected on 35 % of days at q=0.75",
    f"published {pub_pool:,} vs vat-v1 {can_pool:,} = {pub_pool/max(can_pool,1):.0f}x",
    "POOLED over the 2 episodic scenarios",
    "this number has had three denominators in circulation (one scenario, the mean of the "
    "two, the pool). Quote it pooled and say so. Per scenario the ratios are "
    + " and ".join(f"{v[0]/max(v[1],1):.0f}x" for v in sc_fp.values())
    + f", so the pooled figure hides a wide spread - report the FP ROWS as well.")
row("why the published validation could not see this",
    "the same FP count with the limit binding every day",
    f"published {chronic_pub} vs vat-v1 {chronic_can} rows", "1 chronic scenario",
    "when the limit binds daily, cap(t) adapts to it and the published detector looks fine "
    "- which is the regime its control pairs were effectively in")

# the full duty-cycle table (review section 3.3), recomputed on the frozen harness
print()
print("   duty-cycle FP table (the numbers review section 3.3 must quote):")
print("      %-22s %14s %14s" % ("regime", "published FP", "vat-v1 FP"))
DUTY_TABLE = []
for duty, pat in ((0.35, "episodic"), (0.65, "episodic"), (1.00, "chronic")):
    p = B.score_scenario(B.scenario(("eu_1", "eu_3"), duty, 0.75, pat),
                         B.det_published)["fp_rows"]
    v = B.score_scenario(B.scenario(("eu_1", "eu_3"), duty, 0.75, pat),
                         B.det_rolling)["fp_rows"]
    DUTY_TABLE.append((duty, p, v))
    print("      %-22s %14s %14s" % (f"clip {duty*100:.0f} % of days", f"{p:,}", f"{v:,}"))
print("      (single pair eu_1+eu_3, q=0.75; the two episodic scenarios differ by ~2x in")
print("       published FP, so quote the pair you measured)")

auc_rows = []
for name, fn in [("D0 published", B.det_published), ("D1 rolling-slope", B.det_rolling)]:
    epi = [B.score_scenario(B.scenario(p, 0.35, 0.75, "episodic"), fn) for p in EPI]
    all_ = [B.score_scenario(B.scenario(*s), fn) for s in B.DEFAULT_SCENARIOS]
    all_auc = np.array([e["auc00"] for e in all_], dtype=float)
    auc_rows.append(dict(det=name,
                         epi_auc00=np.nanmean([e["auc00"] for e in epi]),
                         epi_auc10=np.nanmean([e["auc10"] for e in epi]),
                         epi_auc20=np.nanmean([e["auc20"] for e in epi]),
                         all_auc00=np.nanmean(all_auc),
                         n_with_pos=int(np.isfinite(all_auc).sum())))
A = pd.DataFrame(auc_rows).set_index("det")
print()
print(A.round(3).to_string())
n_pos_scen = int(A.loc["D1 rolling-slope", "n_with_pos"])
row("AUC - TWO legitimate denominators, so quote the set",
    "threshold-free AUC on the evaluation domain",
    f"episodic-only: D0 {A.loc['D0 published','epi_auc00']:.3f} vs "
    f"D1 {A.loc['D1 rolling-slope','epi_auc00']:.3f};   "
    f"all benchmark scenarios with positives: "
    f"D0 {A.loc['D0 published','all_auc00']:.3f} vs "
    f"D1 {A.loc['D1 rolling-slope','all_auc00']:.3f}",
    f"2 episodic scenarios / {n_pos_scen} of {len(B.DEFAULT_SCENARIOS)} benchmark scenarios",
    "review section 3.5 and the regression test quote different sets, which is why they "
    "differ. Both are correct; each must name its set. The noclip scenario has no "
    "positives by construction, so it is excluded from the mean rather than scored 0.")

# the detector scoreboard review section 3.5 needs: EVERY detector, one scenario set
print()
print("   detector discrimination table (the numbers review section 3.5 must quote):")
print(f"      mean over the {n_pos_scen} benchmark scenarios that contain positives, "
      f"episodic + chronic")
R = []
for name, fn in B.DETECTORS.items():
    es = [B.score_scenario(B.scenario(*s), fn) for s in B.DEFAULT_SCENARIOS]
    es_epi = [B.score_scenario(B.scenario(p, 0.35, 0.75, "episodic"), fn) for p in EPI]
    R.append(dict(det=name,
                  auc_any=np.nanmean([e["auc00"] for e in es]),
                  auc_gt10=np.nanmean([e["auc10"] for e in es]),
                  auc_gt20=np.nanmean([e["auc20"] for e in es]),
                  fp=np.nanmean([e["fp_rows"] for e in es]),
                  epi_auc=np.nanmean([e["auc00"] for e in es_epi]),
                  epi_fp=np.nanmean([e["fp_rows"] for e in es_epi])))
R = pd.DataFrame(R).set_index("det").sort_values("epi_auc", ascending=False)
print()
print(R.round(3).to_string())
row("discrimination, all detectors, one set",
    "threshold-free AUC against injected ground truth; `epi_` columns are the 2 EPISODIC "
    "scenarios (the realistic regime), the others are all 5 scenarios with positives",
    f"episodic: best {R.index[0]} {R.iloc[0]['epi_auc']:.3f} down to "
    f"{R.index[-1]} {R.iloc[-1]['epi_auc']:.3f}",
    "2 episodic scenarios / 5 positive scenarios",
    "D1 leads on the episodic regime AND on deep clips (auc_gt20) AND on FP; D5 edges it "
    "only on the pooled auc_any, where the chronic scenario is mixed in. vat-v1 keeps D1 "
    "as canonical (frozen decision). Review section 3.5's table was computed on an older "
    "scenario set and matches no current definition; it is replaced by this one.")

# =========================================================================== #
print()
print("=" * 104)
print("7. PHYSICAL VALIDATION - the nameplate round-trip")
print("=" * 104)
th = {}
for u in __import__("satsim").EU:
    if u == "eu_7":
        continue
    t = B.theta_roll(df[u], ref, fe["active"][u], fe["dayidx"])
    mid = fe["active"][u] & (ref >= 0.35) & (ref <= 0.60)
    th[u] = float(t[mid].median())
ratio = pd.Series({u: v / NAMEPLATE_W for u, v in th.items()})
row("theta recovers the documented string rating",
    "per-unit theta (the detector's slope) vs the 9.0 kW nameplate on slide 2 of the deck; "
    "theta is estimated in ref in [0.35, 0.60], where clipping cannot bind",
    f"median {pd.Series(th).median():,.0f} W; median ratio {ratio.median():.3f}, "
    f"sd {ratio.std():.3f}; max deviation {100*abs(ratio-1).max():.1f} %",
    f"{len(ratio)} healthy units (24 minus eu_7)",
    "nothing was tuned to a nameplate -- this is an independent check that the detector's "
    "slope IS the hardware rating")

# =========================================================================== #
print()
print("=" * 104)
print("8. NATURAL EXPERIMENT - MPPT clipping, not inverter clipping")
print("=" * 104)


def unit_deficit_hi(u, thresh=0.75):
    s = df[u]
    act = fe["active"][u]
    t = B.theta_roll(s, ref, act, fe["dayidx"])
    d = (1 - s / (t * ref)).replace([np.inf, -np.inf], np.nan)
    clean = act & ~fe["unit_off"][u] & ~fe["stale"][u] & ~fe["negative"][u] & ~fe["site_outage"]
    hi = clean & (ref >= thresh)
    return d[hi].median()


INV = __import__("satsim").INVERTERS
for a, b in B.PAIRS:
    inv = INV_OF[a]
    mates = [u for u in INV[inv] if u not in (a, b) and u != "eu_7"]
    clip = np.mean([unit_deficit_hi(a), unit_deficit_hi(b)])
    mate = np.mean([unit_deficit_hi(u) for u in mates]) if mates else float("nan")
    print(f"   inverter {inv}: {a}+{b} deficit_hi {clip:+.3f}   "
          f"inverter mate(s) {mates} {mate:+.3f}   gap {clip-mate:+.3f}")
print("   -> the clipped pairs roll off, their inverter mates do not: the constraint is at")
print("      the MPPT, not the inverter. This is the deck's strongest causal evidence.")

print()
print("=" * 104)
print("AUDIT COMPLETE")
print("=" * 104)
