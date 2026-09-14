"""
STEP 10 - forensic on the single largest disagreement between the published
detector and vat-v1.

step8 flagged it: for eu_8+eu_16, July 2026 is 78.8 h published vs 36.2 h vat-v1.
Everything else agrees. Two possible stories:

  (a) the published detector is over-firing in July, because the adaptive ceiling
      `cap(t)` is dragged down by the very clipping it is trying to measure, which
      inflates the deficit (this is "defect B" - the ceiling-drag mechanism);
  (b) vat-v1 is MISSING real clipping in July, because its clear-sky anchor `theta`
      is too high or its persistence rule is breaking up genuine episodes.

This script decides between them with the one test that is independent of BOTH
baselines: the normalisation-free peer ratio from step2.

    r(t) = (X_a + X_b) / (X_peer1 + X_peer3),   X = raw unit power / p99.9
           rescaled so median(r) over ref in [0.35,0.60] is 1

r needs no cap, no g0 and no ref in the denominator. For a healthy pair r ~ 1 at
every irradiance. So for the disputed rows (published flags, vat-v1 does not) ask
what r is:

    r ~ 1     -> no physical shortfall -> (a). the published detector is over-firing
    r << 1    -> a real shortfall      -> (b). vat-v1 is missing real saturation

Run:  ./venv/Scripts/python.exe sat_work/research/step10_july.py
"""
import sys, io, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0, r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research")
from satsim import *

pd.set_option("display.width", 200)

df = load(); FE = preprocess(df); ref = FE["ref"]
FULLCAL = pd.date_range(df.index.min().normalize(), df.index.max().normalize(), freq="D")
BINS = np.linspace(0.30, 1.05, 21)
BAND = (0.35, 0.60)
CONTROLS = [("eu_1", "eu_3"), ("eu_4", "eu_12"), ("eu_15", "eu_23")]

# --------------------------------------------------------------------------- #
# vat-v1, identical to recommended.py
# --------------------------------------------------------------------------- #
def broadcast(per_day, dayidx, labels, window="31D", minp=5):
    roll = per_day.reindex(FULLCAL).rolling(window, center=True, min_periods=minp).median()
    return pd.Series(roll.reindex(dayidx).to_numpy(), index=labels)


def persist(raw, k):
    return raw & (run_lengths(raw) >= k)


def theta(s, act):
    u = s / ref.replace(0, np.nan)
    sel = act & (ref >= BAND[0]) & (ref <= BAND[1]) & u.notna()
    per_day = u[sel].groupby(FE["dayidx"][sel]).median()
    return broadcast(per_day, FE["dayidx"], s.index)


def vat(a, b):
    """Return everything vat-v1 computes for one pair, including the reasons."""
    s = df[a] + df[b]
    act = FE["active"][a] & FE["active"][b]
    th = theta(s, act)
    exp_ = th * ref
    d = (1 - s / exp_).replace([np.inf, -np.inf], np.nan)
    gate = (ref >= 0.70) & (s > 0.6 * exp_) & act & ~dq_mask(FE, a, b)
    raw = gate & (d > 0.15)
    return dict(s=s, act=act, theta=th, expected=exp_, deficit=d, gate=gate,
                raw=raw, mod=persist(raw, 3), sev=persist(gate & (d > 0.30), 6))


PUB, VAT, D = {}, {}, {}
for p in PAIRS + CONTROL_PAIRS:
    a, b = p
    PUB[p] = detect(df, FE, a, b)
    VAT[p] = vat(a, b)

# --------------------------------------------------------------------------- #
# A. where the two detectors disagree, by month
# --------------------------------------------------------------------------- #
print("=" * 100)
print("A. MONTHLY FLAGGED HOURS - published vs vat-v1 (shared pairs, moderate tier)")
print("=" * 100)
mo = df.index.to_period("M").astype(str)
rows = []
for p in PAIRS:
    pub, v = PUB[p]["moderate"], VAT[p]["mod"]
    for m in sorted(set(mo)):
        k = mo == m
        rows.append(dict(pair="+".join(p), month=m, pub_h=pub[k].sum() / 12,
                         vat_h=v[k].sum() / 12, only_pub_h=(pub & ~v)[k].sum() / 12))
A = pd.DataFrame(rows)
A["diff_h"] = A.pub_h - A.vat_h
piv = A.pivot_table(index="month", columns="pair", values="diff_h", aggfunc="sum")
tot = A.groupby("month")[["pub_h", "vat_h", "only_pub_h", "diff_h"]].sum()
print(tot[tot.pub_h + tot.vat_h > 0].round(1).to_string())
print("\nmonth with the largest excess of published-only hours:")
print(A.nlargest(6, "only_pub_h")[["pair", "month", "pub_h", "vat_h", "only_pub_h"]].round(1).to_string(index=False))

# --------------------------------------------------------------------------- #
# B. mechanism - is the ceiling dragged down in July?
# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("B. MECHANISM - the published expected value vs the clear-sky anchor, by month")
print("=" * 100)
print("For each midday high-ref row, exp_pub = g0*cap(t)*ref and exp_vat = theta(t)*ref.")
print("Ratio exp_pub/exp_vat < 1 means the published ceiling is sitting BELOW the")
print("clear-sky anchor - i.e. it has been dragged down by clipping - which inflates")
print("every deficit the published detector reports.\n")
for p in PAIRS:
    a, b = p
    k = "".join(p)
    ep, ev = PUB[p]["expected"], VAT[p]["expected"]
    ok = VAT[p]["gate"] | (PUB[p]["base"] & VAT[p]["act"])
    r = (pd.Series(ep, index=df.index) / pd.Series(ev, index=df.index))
    t = pd.DataFrame({"m": mo, "r": r, "ok": ok})
    med = t[t.ok].groupby("m")["r"].median()
    print(f"{k}:  median exp_pub/exp_vat   overall {med.median():.3f}   "
          f"min-month {med.idxmin()} = {med.min():.3f}   July2026 = {med.get('2026-07', np.nan):.3f}")

print()
print("... and the same ratio on the control pairs, which cannot saturate:")
for p in CONTROL_PAIRS:
    ep, ev = PUB[p]["expected"], VAT[p]["expected"]
    r = (pd.Series(ep, index=df.index) / pd.Series(ev, index=df.index))
    print(f"   {'+'.join(p)}  median exp_pub/exp_vat = {r.median():.3f}")

# --------------------------------------------------------------------------- #
# C. why vat-v1 rejects each row the published detector flags (July, worst pair)
# --------------------------------------------------------------------------- #
PAIR = ("eu_8", "eu_16")
a, b = PAIR
pub, v = PUB[PAIR]["moderate"], VAT[PAIR]["mod"]
jul = (df.index.year == 2026) & (df.index.month == 7)
dis = pub & ~v & jul
print()
print("=" * 100)
print(f"C. WHY vat-v1 REJECTS THE {int(dis.sum())} DISPUTED ROWS  ({a}+{b}, July 2026)")
print("=" * 100)
V = VAT[PAIR]
reasons = [
    ("unit inactive or DQ-flagged (masking, not a threshold)", ~V["act"] | dq_mask(FE, a, b)),
    ("ref < 0.70 (irradiance gate)", ref < 0.70),
    ("pair decoupled: s <= 0.6*theta*ref", V["s"] <= 0.6 * V["expected"]),
    ("deficit_vat <= 0.15 (threshold)", V["deficit"] <= 0.15),
    ("failed 3-sample persistence", ~V["raw"]),
]
claimed = pd.Series(False, index=df.index)
for lab, m in reasons:
    c = (dis & m & ~claimed).sum()
    print(f"   {c:4d} rows   {lab}")
    claimed |= m
left = (dis & ~claimed)
if left.any():
    print(f"   {int(left.sum()):4d} rows   (unexplained)")

print()
print(f"   ... of the {int(dis.sum())} disputed rows, median values:")
print(f"       deficit published = {(PUB[PAIR]['deficit'][dis]).median():+.3f}"
      f"    deficit vat = {(V['deficit'][dis]).median():+.3f}")
print(f"       exp_pub = {(pd.Series(PUB[PAIR]['expected'], index=df.index)[dis]).median():,.0f} W"
      f"   exp_vat = {(V['expected'][dis]).median():,.0f} W"
      f"   s = {(V['s'][dis]).median():,.0f} W     ref = {ref[dis].median():.3f}")

# --------------------------------------------------------------------------- #
# D. the decisive test - peer ratio on the disputed rows
# --------------------------------------------------------------------------- #
X = df[EU].div(df[EU].quantile(0.999))
PEER = ("eu_1", "eu_3")
peerP = X[PEER[0]] + X[PEER[1]]
peer_act = FE["active"][PEER[0]] & FE["active"][PEER[1]]
peer_dq = dq_mask(FE, *PEER)


def peer_ratio(p, mask):
    """Median peer ratio on `mask`, with both the pair and the peer clean."""
    a_, b_ = p
    s = X[a_] + X[b_]
    act = FE["active"][a_] & FE["active"][b_]
    r = s / peerP.replace(0, np.nan)
    cal = act & peer_act & (ref >= BAND[0]) & (ref <= BAND[1])
    r = r / r[cal].median()
    m = mask & act & peer_act & ~dq_mask(FE, a_, b_) & ~peer_dq
    if m.sum() == 0:
        return np.nan, 0
    return float(r[m].median()), int(m.sum())


print()
print("=" * 100)
print("D. THE DECISIVE TEST - is there a real physical shortfall on the disputed rows?")
print("=" * 100)
print("Peer ratio r = (X_pair / X_peer), normalised to 1 at mid irradiance. Needs no")
print("cap, no g0 and no ref in the denominator. r ~ 1 == the pair is delivering exactly")
print("what an independent pair delivers at the same irradiance.\n")
groups = [("both detectors flag", pub & v & jul),
          ("PUBLISHED ONLY  (disputed)", pub & ~v & jul),
          ("vat-v1 only", ~pub & v & jul),
          ("neither flags", ~pub & ~v & jul)]
tab = []
for lab, m in groups:
    r_, n = peer_ratio(PAIR, m)
    tab.append(dict(group=lab, rows=n, peer_ratio=round(r_, 3) if n else np.nan))
T = pd.DataFrame(tab)
print(f"{a}+{b}, July 2026, all ref >= 0.70 rows split by flag agreement:\n")
print(T.to_string(index=False))

print()
print("reference levels for the same peer ratio:")
r_mid, n_mid = peer_ratio(PAIR, jul & (ref >= BAND[0]) & (ref <= BAND[1]))
print(f"   {a}+{b} at mid irradiance  [calibration, =1 by construction] : {r_mid:.3f}  ({n_mid} rows)")
r_ctl, n_ctl = peer_ratio(("eu_4", "eu_12"), jul & (ref >= 0.70))
print(f"   control eu_4+eu_12 at ref>=0.70 (cannot saturate)            : {r_ctl:.3f}  ({n_ctl} rows)")

print()
print("same split, but over the WHOLE record instead of July alone:")
tab2 = []
for lab, m in [("both flag", pub & v), ("PUBLISHED ONLY", pub & ~v),
               ("vat-v1 only", ~pub & v), ("neither", ~pub & ~v)]:
    r_, n = peer_ratio(PAIR, m & (ref >= 0.70))
    tab2.append(dict(group=lab, rows=n, peer_ratio=round(r_, 3) if n else np.nan))
print(pd.DataFrame(tab2).to_string(index=False))

# --------------------------------------------------------------------------- #
# E. day-level July table
# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("E. DAY-LEVEL JULY 2026  (published-only hours are the disputed ones)")
print("=" * 100)
day = df.index.normalize()
rows = []
for d_, g in pd.Series(1, index=df.index)[jul].groupby(day[jul]):
    k = jul & (day == d_)
    pk, vk = pub[k], v[k]
    r_, n = peer_ratio(PAIR, k & (ref >= 0.70))
    rows.append(dict(day=str(d_.date()), rows=int(k.sum()),
                     ref_max=round(float(ref[k].max()), 2),
                     pub_h=round(pk.sum() / 12, 1), vat_h=round(vk.sum() / 12, 1),
                     disputed_h=round((pk & ~vk).sum() / 12, 1),
                     peer_r_hi=round(r_, 3) if n else np.nan,
                     exp_ratio=round(float((pd.Series(PUB[PAIR]["expected"], index=df.index)[k]
                                            / V["expected"][k]).median()), 3)))
E = pd.DataFrame(rows)
print(E.to_string(index=False))
print(f"\nJuly totals: published {pub[jul].sum()/12:.1f} h, vat-v1 {v[jul].sum()/12:.1f} h, "
      f"disputed {(pub & ~v)[jul].sum()/12:.1f} h")
E.to_csv(r"c:\Users\nhphuong\Desktop\Solar\all_data\sat_work\research\july_forensic.csv", index=False)
print("wrote sat_work/research/july_forensic.csv")
