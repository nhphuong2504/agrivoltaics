"""Full four-detector scoreboard on the six labelled scenarios, with the duplicate flagged.

`bench.scenario` passes `day_on` to `inject` only for `pattern == "episodic"`, and `inject`
clamps every row when `day_on is None`. `duty` therefore does nothing for a chronic scenario:
scenario 1 (35 % chronic) and scenario 3 (100 % chronic) are the same experiment.

This prints the scoreboard both ways -- as the harness currently enumerates it (five
"positive" scenarios, two of them duplicates) and deduplicated (four distinct) -- so the
size of the effect on the pooled columns is visible before anything is changed.
"""
import sys
sys.path.insert(0, "sat_work/research")
import numpy as np
import bench as B

SC = [
    (1, ("eu_1", "eu_3"), 0.35, 0.75, "chronic"),
    (2, ("eu_1", "eu_3"), 0.35, 0.75, "episodic"),
    (3, ("eu_1", "eu_3"), 1.00, 0.75, "chronic"),
    (4, ("eu_1", "eu_3"), 1.00, None, "chronic"),
    (5, ("eu_4", "eu_12"), 0.35, 0.75, "chronic"),
    (6, ("eu_15", "eu_23"), 0.35, 0.75, "episodic"),
]
DET = [("D1", B.det_rolling), ("D2", B.det_envelope),
       ("D5", B.det_control_calibrated), ("D4", B.det_ceiling_pinned)]

rows = []
for i, p, d, q, pat in SC:
    sc = B.scenario(p, duty=d, q=q, pattern=pat)
    rec = {"i": i, "pos": q is not None}
    for nm, fn in DET:
        s = B.score_scenario(sc, fn)
        rec[nm] = s["auc00"]
    rows.append(rec)

print("scenario   D1     D2     D5     D4")
for r in rows:
    print("  #{:<2d}      {:.3f}  {:.3f}  {:.3f}  {:.3f}".format(
        r["i"], r["D1"], r["D2"], r["D5"], r["D4"]))

pos = [r for r in rows if r["pos"]]
# scenario 3 duplicates scenario 1; keep the first occurrence
dedup = []
seen = set()
for r in pos:
    key = (r["i"] not in (3,),)
    if r["i"] == 3:
        continue
    dedup.append(r)

print()
print("pooled 'any clip' as enumerated ({} scenarios incl. the duplicate):".format(len(pos)))
for nm, _ in DET:
    print("   {}  {:.3f}".format(nm, float(np.mean([r[nm] for r in pos]))))
print("pooled 'any clip' deduplicated ({} distinct scenarios):".format(len(dedup)))
for nm, _ in DET:
    print("   {}  {:.3f}".format(nm, float(np.mean([r[nm] for r in dedup]))))
print()
print("scenario 1 and scenario 3 are byte-identical, so their AUCs are equal:",
      rows[0]["D1"] == rows[2]["D1"])
