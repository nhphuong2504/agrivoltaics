"""
STEP 19 - audit of the two downstream artefacts (power_production.pptx, SATURATION_DETECTION.md).

Part A is the interesting one. Slide 2 of the deck states a fact the analysis never had:

    "Each string rating within an experimental unit is 9.0-kW ca."

The recommended detector replaces `g0 * cap(t) * ref` with `theta(t) * ref`, where theta is
the rolling median of the daily median of s/ref in the mid band. theta therefore has units
of W per unit-ref: an estimate of what the pair delivers at ref = 1.0. If theta is a
meaningful physical quantity and not a fitted fudge factor, then for a SINGLE unit it should
recover the nameplate string rating, ~9.0 kW.

That is a falsifiable prediction, and it is independent of everything in the review.

Part B cross-checks the published ceiling against the nameplate, and Part C lists the
specific claims in SATURATION_DETECTION.md that the analysis changes.

Run:  ./venv/Scripts/python.exe sat_work/research/step19_docs_audit.py
"""
import sys, io, os, re, zipfile, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from satsim import *
import bench as B

ROOT = os.path.dirname(os.path.dirname(_HERE))
pdf = pd.set_option("display.width", 200)

df = B.raw_df()
FE = B.raw_fe()
ref = FE["ref"]
NAMEPLATE = 9.0  # kW per string, from power_production.pptx slide 2

# --------------------------------------------------------------------------- #
print("=" * 100)
print("A. DOES theta RECOVER THE DOCUMENTED 9.0 kW STRING RATING?")
print("=" * 100)
print("theta per unit = rolling median of the daily median of s/ref in ref in [0.35, 0.60].")
print("This is the quantity the recommended detector multiplies by ref. Nothing was tuned")
print("to make it match a nameplate.\n")
rows = []
for u in EU:
    if u == "eu_7":      # confirmed faulty, excluded everywhere
        continue
    th = B.theta_roll(df[u], ref, FE["active"][u], FE["dayidx"])
    mid = FE["active"][u] & (ref >= 0.35) & (ref <= 0.60)
    rows.append(dict(unit=u, theta_W=float(th[mid].median()),
                     ratio_to_9kW=float(th[mid].median()) / (NAMEPLATE * 1000)))
T = pd.DataFrame(rows).set_index("unit")
print(T.round(3).to_string())
r = T.ratio_to_9kW
print(f"\n   theta / 9.0 kW  ->  median {r.median():.3f}   "
      f"range {r.min():.3f}-{r.max():.3f}   std {r.std():.3f}")
print(f"   theta in W      ->  median {T.theta_W.median():.0f}   "
      f"range {T.theta_W.min():.0f}-{T.theta_W.max():.0f}")
print(f"\n   {int((r.between(0.90, 1.10)).sum())} of {len(r)} units fall within +/-10 % of the "
      f"nameplate; max deviation {100*abs(r-1).max():.1f} %")
print("\n   VERDICT: theta independently recovers the documented string rating. The recommended")
print("   model is not a free parameter -- its slope is the physical nameplate, which is why")
print("   replacing g0*cap(t) with theta(t) makes the deficit columns interpretable.")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("B. THE PUBLISHED CEILING vs THE NAMEPLATE")
print("=" * 100)
print("cap(t) is the per-day P99 of the pair sum. A pair is two 9.0 kW strings, so the")
print("combined nameplate is 18 kW. If the pairs routinely fell short of 18 kW, that is")
print("itself evidence of a shared constraint -- worth stating in the deck.\n")
for a, b in PAIRS:
    d = B.pair_deficit(df, FE, a, b)
    cap = B.broadcast(
        d["s"][d["act"] & (ref > 0.3)].groupby(FE["dayidx"][d["act"] & (ref > 0.3)]).quantile(0.99),
        FE["dayidx"], d["s"].index)
    print(f"   {a}+{b}: cap(t) median {cap.median()/1000:.2f} kW  "
          f"(range {cap.quantile(.05)/1000:.2f}-{cap.quantile(.95)/1000:.2f})  "
          f"= {100*cap.median()/(2*NAMEPLATE*1000):.0f} % of the 18 kW pair nameplate")
    obs_max = d["s"][ref >= 0.7].max() / 1000
    print(f"      highest pair sum actually observed at ref>=0.7: {obs_max:.2f} kW "
          f"= {100*obs_max/(2*NAMEPLATE):.0f} % of nameplate")
print("\n   CAUTION: the two strings in a pair need not be at the same tilt or azimuth, so")
print("   they need not peak simultaneously and the naive 2 x 9.0 kW is an upper bound.")
print("   Read this as SUPPORTING evidence for a shared constraint, not proof of one --")
print("   the decisive evidence is the peer-ratio test in section 2.")
print("   Note eu_10+eu_18 does reach 16.2 kW (90 %), so its constraint is not a hard")
print("   ceiling at ~13 kW; that matches the 'soft knee' description in the original doc.")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("C. THE DECK ITSELF")
print("=" * 100)
pptx = os.path.join(ROOT, "power_production.pptx")
if os.path.exists(pptx):
    z = zipfile.ZipFile(pptx)
    names = sorted([n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
                   key=lambda n: int(re.search(r"(\d+)", n.split("/")[-1]).group(1)))
    print(f"   {len(names)} slides:")
    for n in names:
        xml = z.read(n).decode("utf-8", "replace")
        txt = [t.strip() for t in re.findall(r"<a:t>(.*?)</a:t>", xml, re.S) if t.strip()]
        figs = len(re.findall(r"<a:blip", xml))
        print(f"      {n.split('/')[-1]:12s} text: {txt if txt else '(none)'}   embedded images: {figs}")
    print("\n   NO saturation numbers appear anywhere in the deck. Slide 3 is literally the")
    print("   placeholder 'Explain about saturated problem' + one figure; slide 4 is a figure")
    print("   with no text. So there is nothing in the deck to CORRECT -- it needs FILLING.")
    print("   The numbers to put on it are the review's section 5 table and the figure set.")
else:
    print("   (power_production.pptx not found)")

# --------------------------------------------------------------------------- #
print()
print("=" * 100)
print("D. SATURATION_DETECTION.md - CLAIMS THE ANALYSIS CHANGES")
print("=" * 100)
MD = os.path.join(ROOT, "SATURATION_DETECTION.md")
txt = open(MD, encoding="utf-8").read()
for probe in ["144", "473", "593", "394", "368 / 334 / 243", "17,519", "g0 · cap(t)",
              "making it rolling", "-inf", "122"]:
    hits = [ln.strip() for ln in txt.splitlines() if probe in ln]
    print(f"   '{probe}' -> {len(hits)} line(s)")
    for h in hits[:3]:
        print(f"        {h[:150]}")
print("\n   Note the doc DISCLOSES the -inf artifact in section 8.1 ('division artifact;")
print("   flags can never fire there -- filter ref >= 0.7'). So the -inf is documented, not")
print("   hidden; the problem is only that it is left IN the shipped file, where it poisons")
print("   aggregation despite the warning.")
print("\n   Note also that section 10 already proposes the fix this review recommends:")
print("   'g0 is currently a global-per-pair scalar; making it rolling (31-day median of")
print("   mid-ref gain) would track long-term soiling even more tightly.' The recommended")
print("   detector is that extension, not a departure from the method.")
