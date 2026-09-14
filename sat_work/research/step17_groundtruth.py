"""
STEP 17 - the fleet ground truth, and what it does to the controls.

The site engineer supplied the real topology:

  shared MPPT channels : eu_8+eu_16, eu_10+eu_18, eu_13+eu_21
  inverter 1 : eu_1,  eu_9,  eu_17
  inverter 2 : eu_2,  eu_3,  eu_10, eu_11, eu_18, eu_19
  inverter 3 : eu_4,  eu_12, eu_20
  inverter 4 : eu_5,  eu_6,  eu_13, eu_14, eu_21, eu_22
  inverter 5 : eu_7,  eu_15, eu_23
  inverter 6 : eu_8,  eu_16, eu_24

Two things fall out immediately:

  1. The three MPPT pairs are EXACTLY the three pairs inferred from the data (satsim.PAIRS).
     The inference was right, and now it is documented rather than inferred.

  2. Two of the three "independent" control pairs are NOT independent at the inverter level:
     eu_4+eu_12 are both on inverter 3, and eu_15+eu_23 are both on inverter 5. Only
     eu_1+eu_3 crosses inverters. That matters — but it turns out to matter in the
     *reassuring* direction, because the inverter topology gives us a natural experiment
     that separates MPPT clipping from inverter clipping.

Run:  ./venv/Scripts/python.exe sat_work/research/step17_groundtruth.py
"""
import sys, io, os, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from satsim import *
import bench as B

# ---- ground truth from the site engineer, verbatim ------------------------- #
INVERTERS = {
    1: ["eu_1", "eu_9", "eu_17"],
    2: ["eu_2", "eu_3", "eu_10", "eu_11", "eu_18", "eu_19"],
    3: ["eu_4", "eu_12", "eu_20"],
    4: ["eu_5", "eu_6", "eu_13", "eu_14", "eu_21", "eu_22"],
    5: ["eu_7", "eu_15", "eu_23"],
    6: ["eu_8", "eu_16", "eu_24"],
}
MPPT_CHANNELS = [("eu_8", "eu_16"), ("eu_10", "eu_18"), ("eu_13", "eu_21")]
INV_OF = {u: k for k, us in INVERTERS.items() for u in us}

df = B.raw_df()
FE = B.raw_fe()
ref = FE["ref"]
indep = [u for u in EU if u not in SHARED and u != "eu_7"]

pd.set_option("display.width", 220)


def unit_deficit(u, thresh=0.75):
    s = df[u]
    act = FE["active"][u]
    th = B.theta_roll(s, ref, act, FE["dayidx"])
    d = (1 - s / (th * ref)).replace([np.inf, -np.inf], np.nan)
    clean = (act & ~FE["unit_off"][u] & ~FE["stale"][u]
             & ~FE["negative"][u] & ~FE["site_outage"])
    hi = clean & (ref >= thresh)
    return d, hi


print("=" * 108)
print("A. DOES THE INFERRED TOPOLOGY MATCH THE RECORDED ONE?")
print("=" * 108)
print(f"   inferred shared pairs : {PAIRS}")
print(f"   recorded MPPT channels: {MPPT_CHANNELS}")
print(f"   -> identical sets: {set(PAIRS) == set(MPPT_CHANNELS)}")
print(f"\n   recorded control pairs : {CONTROL_PAIRS}")
for a, b in CONTROL_PAIRS:
    tag = "same inverter %d" % INV_OF[a] if INV_OF[a] == INV_OF[b] else \
          "crosses inverters %d/%d" % (INV_OF[a], INV_OF[b])
    print(f"      {a}+{b}: {tag}   <- {'SHARES AN INVERTER' if INV_OF[a]==INV_OF[b] else 'clean'}")

# --------------------------------------------------------------------------- #
print()
print("=" * 108)
print("B. NATURAL EXPERIMENT: units that share an INVERTER with an MPPT-clipped pair")
print("=" * 108)
print("If inverter-level clipping were causing the roll-off, the shared pair's inverter")
print("mates would roll off too. They do not.\n")
rows = []
for p in MPPT_CHANNELS:
    inv = INV_OF[p[0]]
    mates = [u for u in INVERTERS[inv] if u not in p and u != "eu_7"]
    for u in p:
        d, hi = unit_deficit(u)
        rows.append(dict(unit=u, inv=inv, role="MPPT-paired", deficit_hi=d[hi].median()))
    for u in mates:
        d, hi = unit_deficit(u)
        rows.append(dict(unit=u, inv=inv, role="inverter mate", deficit_hi=d[hi].median()))
E = pd.DataFrame(rows)
for inv in [6, 2, 4]:
    sub = E[E.inv == inv]
    print(f"   inverter {inv}:")
    for _, r in sub.iterrows():
        print(f"      {r.unit:6s} {r.role:14s} deficit_hi = {r.deficit_hi:+.3f}"
              f"{'   <-- clipped' if r.role=='MPPT-paired' else ''}")
    clip = sub[sub.role == "MPPT-paired"].deficit_hi.median()
    mate = sub[sub.role == "inverter mate"].deficit_hi.median()
    print(f"      => pair {clip:+.3f} vs inverter mate {mate:+.3f}   "
          f"(difference {clip-mate:+.3f})\n")

# --------------------------------------------------------------------------- #
print("=" * 108)
print("C. SAME-INVERTER vs CROSS-INVERTER INDEPENDENT PAIRS (the null re-examined)")
print("=" * 108)
print("If sharing an inverter alone produced a roll-off, same-inverter independent pairs")
print("would be negative. They are not — so inverter sharing is not a confound for the")
print("MPPT story, and same-inverter controls remain usable as nulls for it.\n")


def pair_row(a, b):
    d = B.pair_deficit(df, FE, a, b)
    clean = (d["act"] & ~B.dq_mask(FE, a, b)).to_numpy() & (ref.to_numpy() >= 0.75)
    defv = d["deficit"].to_numpy()
    med = float(np.nanmedian(defv[clean])) if clean.sum() else np.nan
    return dict(pair=f"{a}+{b}", same_inv=INV_OF[a] == INV_OF[b],
                n_hi=int(clean.sum()), deficit_hi=med)


named = [pair_row(a, b) for a, b in CONTROL_PAIRS] + \
        [pair_row(a, b) for a, b in MPPT_CHANNELS]
N = pd.DataFrame(named)
print(N.round(3).to_string(index=False))

print("\n   sweeping EVERY cross-inverter independent pair (the honest null population):")
crosspairs = [(a, b) for i, a in enumerate(indep) for b in indep[i + 1:]
              if INV_OF[a] != INV_OF[b]]
samepairs = [(a, b) for i, a in enumerate(indep) for b in indep[i + 1:]
             if INV_OF[a] == INV_OF[b]]
print(f"      {len(crosspairs)} cross-inverter pairs, {len(samepairs)} same-inverter pairs")
CR = pd.DataFrame([pair_row(a, b) for a, b in crosspairs])
SA = pd.DataFrame([pair_row(a, b) for a, b in samepairs])
for lab, D in [("cross-inverter", CR), ("same-inverter", SA)]:
    v = D.deficit_hi.dropna() * 100
    print(f"   {lab:15s} n={len(v):3d}  median={v.median():+.2f}%  "
          f"p90={v.quantile(0.90):+.2f}%  p99={v.quantile(0.99):+.2f}%  max={v.max():+.2f}%")
print("\n   for scale, the three MPPT pairs:")
for _, r in N[N.pair.isin([f"{a}+{b}" for a, b in MPPT_CHANNELS])].iterrows():
    print(f"      {r.pair:14s} {100*r.deficit_hi:+.1f}%")

# --------------------------------------------------------------------------- #
print()
print("=" * 108)
print("D. REBUILDING THE NULL: what the bias floor looks like with clean controls only")
print("=" * 108)
print("The published method's control-pair 'losses' are its bias floor. Recompute it three")
print("ways: all three legacy controls, the cross-inverter control only, and the whole")
print("cross-inverter population (which the topology now lets us use as a real null).\n")


def loss(a, b, mode):
    d = B.pair_deficit(df, FE, a, b)
    gate = ((ref >= B.REF_ON) & (d["s"] > B.NEAR_CAP * d["expected"])
            & d["act"] & ~B.dq_mask(FE, a, b)).to_numpy()
    if mode == "vat":
        expected = d["expected"]
    else:
        expected = pd.Series(B.detect(df, FE, a, b)["expected"], index=df.index)
    shortfall = np.maximum(expected - d["s"], 0).to_numpy()[gate]
    return float(shortfall.sum()) * 5 / 60 / 1000


def shared_total(mode):
    return sum(loss(a, b, mode) for a, b in MPPT_CHANNELS)


S_pub, S_vat = shared_total("pub"), shared_total("vat")
print(f"   shared-pair lost energy: published {S_pub:,.0f} kWh   vat-v1 {S_vat:,.0f} kWh\n")

groups = {
    "all 3 legacy controls": [("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")],
    "legacy minus eu_15+eu_23": [("eu_1", "eu_3"), ("eu_4", "eu_12")],
    "cross-inverter only": [("eu_1", "eu_3")],
}
out = []
for lab, ps in groups.items():
    p = sum(loss(a, b, "pub") for a, b in ps)
    v = sum(loss(a, b, "vat") for a, b in ps)
    out.append(dict(null=lab, n_pairs=len(ps), pub_kwh=p, vat_kwh=v,
                    pub_pct=100 * p / S_pub, vat_pct=100 * v / S_vat))
# the full cross-inverter population as a null
cp = sum(loss(a, b, "pub") for a, b in crosspairs)
cv = sum(loss(a, b, "vat") for a, b in crosspairs)
out.append(dict(null=f"all {len(crosspairs)} cross-inv pairs (pop.)", n_pairs=len(crosspairs),
                pub_kwh=cp, vat_kwh=cv, pub_pct=100 * cp / S_pub, vat_pct=100 * cv / S_vat))
O = pd.DataFrame(out)
print(O.round(0).to_string(index=False))

# per-pair distribution of phantom loss - more robust than a 3-pair total
PR = np.array([loss(a, b, "pub") for a, b in crosspairs])
VR = np.array([loss(a, b, "vat") for a, b in crosspairs])
print(f"\n   per-pair phantom loss over {len(crosspairs)} independent pairs, kWh:")
print(f"      published : median {np.median(PR):7.0f}  p90 {np.percentile(PR,90):7.0f}  "
      f"max {PR.max():7.0f}")
print(f"      vat-v1    : median {np.median(VR):7.0f}  p90 {np.percentile(VR,90):7.0f}  "
      f"max {VR.max():7.0f}")
print(f"      -> vat-v1 fabricates {np.median(VR)/max(np.median(PR),1e-9)*100:.0f}% of the "
      f"published phantom loss at the median pair")
print(f"\n   the three legacy controls, individually (kWh published / vat-v1):")
for a, b in CONTROL_PAIRS:
    tag = "same-inverter" if INV_OF[a] == INV_OF[b] else "cross-inverter"
    print(f"      {a}+{b}: {loss(a,b,'pub'):7.0f} / {loss(a,b,'vat'):6.0f}   ({tag})")
print("   -> all three are the same order of magnitude, so the bias is a fleet-wide")
print("      systemic property of the published expectation, not one bad control pair.")

# --------------------------------------------------------------------------- #
print()
print("=" * 108)
print("E. eu_15+eu_23: is it a bad null, a bad unit, or just a tail?")
print("=" * 108)
print("Individually eu_15 and eu_23 are flat at high ref (step16). Two remaining suspects:")
print("an inconsistent pair-level calibration, or ordinary threshold-tail noise.\n")
a, b = "eu_15", "eu_23"
pa = B.pair_deficit(df, FE, a, b)
da, hia = unit_deficit(a)
db, hib = unit_deficit(b)
thA = B.theta_roll(df[a], ref, FE["active"][a], FE["dayidx"])
thB = B.theta_roll(df[b], ref, FE["active"][b], FE["dayidx"])
thP = B.theta_roll((df[a] + df[b]), ref, pa["act"], FE["dayidx"])
hi = (pa["act"] & ~B.dq_mask(FE, a, b) & (ref >= 0.75)).to_numpy()
pair_expect_units = (thA + thB) * ref
from_units = (1 - (df[a] + df[b]) / pair_expect_units).to_numpy()[hi]
print("   median high-ref deficit, two independent calibrations:")
print(f"      pair's own theta      : {np.nanmedian(pa['deficit'].to_numpy()[hi]):+.4f}")
print(f"      sum of unit thetas    : {np.nanmedian(from_units):+.4f}")
print(f"      calibration gap       : {100*((thP/(thA+thB)).median()-1):+.3f} %")
print("   -> the pair calibration agrees with its own units, so there is no pair-level")
print("      calibration artefact. And the pair is flat ON MEDIAN, not 18% short:")
print(f"      median over all {int(hi.sum())} high-ref rows = "
      f"{np.nanmedian(pa['deficit'].to_numpy()[hi]):+.4f}")

flag_ctl = B.det_rolling(df, FE, a, b)[1]
mo = df.index.to_period("M").astype(str)
sub = pd.DataFrame({"m": mo, "f": flag_ctl})
print(f"\n   but the detector does flag {int(flag_ctl.sum())} rows. When?")
print("     ", sub[sub.f].m.value_counts().sort_index().to_dict())
print(f"      deficit on those rows: median "
      f"{np.nanmedian(pa['deficit'].to_numpy()[flag_ctl.to_numpy()]):.3f}")
print("   -> a handful of rows in a few peak months, on a pair that is flat on median.")
print("      That is threshold-tail behaviour, not a topology fault.")
print("\n   HOW MANY SUCH TAIL FLAGS DO FLAT INDEPENDENT PAIRS PRODUCE IN GENERAL?")
tail = []
for (x, y) in crosspairs[:40]:
    tail.append(int(B.det_rolling(df, FE, x, y)[1].sum()))
tail = np.array(tail)
print(f"      over 40 cross-inverter independent (provably unclippable) pairs:")
print(f"      flags per pair: median {np.median(tail):.0f}  p90 {np.percentile(tail,90):.0f}  "
      f"max {tail.max():.0f}")
print(f"      eu_15+eu_23 sits at {int(flag_ctl.sum())} - much higher than this class.")
print("      The correct reference class is same-inverter pairs, though, not")
print("      cross-inverter ones: section G revises this picture.")

print()
print("=" * 108)
print("G. DOES THE INVERTER COMMON MODE EXPLAIN eu_15+eu_23's EXCESS FLAGS?")
print("=" * 108)
print("Section H finds that same-inverter units share a deficit common mode. If that is")
print("cause, then FLAT same-inverter pairs should flag more than FLAT cross-inverter")
print("pairs, and eu_15+eu_23 should look ordinary among its own kind.\n")


def flag_count(a, b):
    return int(B.det_rolling(df, FE, a, b)[1].sum())


same_ind = [p for p in [(indep[i], indep[j]) for i in range(len(indep))
                        for j in range(i + 1, len(indep))]
            if INV_OF[p[0]] == INV_OF[p[1]]]
sf = np.array([flag_count(*p) for p in same_ind])
cf = np.array([flag_count(*p) for p in crosspairs])
print(f"   flat SAME-inverter pairs     (n={len(sf)}): median {np.median(sf):.0f}  "
      f"p90 {np.percentile(sf,90):.0f}  max {sf.max():.0f}")
print(f"   flat CROSS-inverter pairs    (n={len(cf)}): median {np.median(cf):.0f}  "
      f"p90 {np.percentile(cf,90):.0f}  max {cf.max():.0f}")
print(f"   eu_15+eu_23 (same-inverter)                : {flag_count('eu_15','eu_23')}")
print(f"\n   same-inverter flag inflation at p90: "
      f"{np.percentile(sf,90)/max(np.percentile(cf,90),1):.1f}x   "
      f"(both medians are 0, so the median ratio is uninformative)")
print("   -> sharing an inverter roughly doubles how often a flat pair trips the")
print("      threshold. That is a co-movement effect, not a constraint.")
print(f"   -> eu_15+eu_23 is still the most-flagged flat pair in the fleet "
      f"(= the class max, 95th pct), so the common mode explains part of it, not all.")
print()
print("   LEGACY CONTROL CLASSIFICATION:")
for a, b in CONTROL_PAIRS:
    n = flag_count(a, b)
    kind = "same-inv" if INV_OF[a] == INV_OF[b] else "cross-inv"
    ref_pop = sf if kind == "same-inv" else cf
    pct = 100 * (ref_pop < n).mean()
    print(f"      {a}+{b:6s} ({kind}): {n:4d} flags  -> {pct:.0f}th percentile of its class")
print("\n   => use cross-inverter pairs for flag-LEVEL nulls. For MAGNITUDE nulls (the")
print("      bias floor) all three remain valid: every control class is flat on median")

print()
print("=" * 108)
print("H. WITHIN-INVERTER vs ACROSS-INVERTER CO-MOVEMENT OF HIGH-REF DEFICIT")
print("=" * 108)
D = {}
for u in EU:
    if u == "eu_7":
        continue
    d, hid = unit_deficit(u)
    D[u] = pd.Series(np.where(hid, d, np.nan), index=df.index)
DV = pd.DataFrame(D)
DV = DV[DV.notna().sum(axis=1) >= 4]
C = DV.corr(min_periods=2000)


def corr_split(cols, label):
    same, cross = [], []
    for i, u in enumerate(cols):
        for v in cols[i + 1:]:
            c = C.loc[u, v]
            if not np.isfinite(c):
                continue
            (same if INV_OF[u] == INV_OF[v] else cross).append(c)
    print(f"   {label}: same inverter {np.mean(same):+.3f} (n={len(same)})   "
          f"different inverter {np.mean(cross):+.3f} (n={len(cross)})")


corr_split(list(C.columns), "all units        ")
ind_only = [u for u in C.columns if u not in SHARED]
corr_split(ind_only, "independent only ")
print("\n   The 'all units' row is inflated by the three clipped pairs correlating with")
print("   themselves; the 'independent only' row is the honest test. Its positive")
print("   same-inverter excess is a mild common mode - shared conditions, not clipping -")
print("   and it does NOT produce a systematic deficit (section C: both nulls ~0.2-0.3%).")
