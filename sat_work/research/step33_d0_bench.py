"""Recompute the published-vs-canonical benchmark table on the completed record.

The review's §3.3 specificity claim rested on a 10,202-row pooled false-positive count for
the published detector. On the complete record that count collapses, so the claim has to be
re-derived rather than carried over.
"""
import sys
sys.path.insert(0, "sat_work/research")
import bench as B

SC = [
    (("eu_1", "eu_3"), 0.35, 0.75, "chronic"),
    (("eu_1", "eu_3"), 0.35, 0.75, "episodic"),
    (("eu_1", "eu_3"), 1.00, 0.75, "chronic"),
    (("eu_1", "eu_3"), 1.00, None, "chronic"),
    (("eu_4", "eu_12"), 0.35, 0.75, "chronic"),
    (("eu_15", "eu_23"), 0.35, 0.75, "episodic"),
]

t0 = t1 = 0
for i, (p, d, q, pat) in enumerate(SC, 1):
    sc = B.scenario(p, duty=d, q=q, pattern=pat)
    s0 = B.score_scenario(sc, B.det_published)
    s1 = B.score_scenario(sc, B.det_rolling)
    tag = "{}+{}".format(p[0], p[1])
    print("{:d} {:14s} duty={:4.2f} q={} {:9s} | D0 fp {:>7d} auc {:.3f} | D1 fp {:>6d} auc {:.3f}".format(
        i, tag, d, q, pat, s0["fp_rows"], s0["auc00"], s1["fp_rows"], s1["auc00"]))
    if pat == "episodic":
        t0 += s0["fp_rows"]
        t1 += s1["fp_rows"]
print("POOLED episodic: D0 {} vs D1 {}  =  {:.1f}x".format(t0, t1, t0 / max(t1, 1)))
