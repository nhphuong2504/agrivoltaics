"""Does `duty` actually reach the injector for chronic scenarios?

`bench.scenario` passes `day_on` to `inject` only when `pattern == "episodic"`, and `inject`
clamps every row when `day_on is None`. If that reading is right, scenario 1 (35 % chronic)
and scenario 3 (100 % chronic) are the same experiment under two labels, and the benchmark's
"duty cycle" axis is not actually spanned by the six scenarios.
"""
import sys
sys.path.insert(0, "sat_work/research")
import numpy as np
import bench as B

p = ("eu_1", "eu_3")
a = B.scenario(p, duty=0.35, q=0.75, pattern="chronic")
b = B.scenario(p, duty=1.00, q=0.75, pattern="chronic")
e = B.scenario(p, duty=0.35, q=0.75, pattern="episodic")

da = a["df"][["eu_1", "eu_3"]].to_numpy()
db = b["df"][["eu_1", "eu_3"]].to_numpy()
de = e["df"][["eu_1", "eu_3"]].to_numpy()

print("chronic 35 %  vs chronic 100 % : identical =", np.array_equal(da, db, equal_nan=True))
print("  ground-truth rows  :", int(a["gt"].sum()), "vs", int(b["gt"].sum()))
print("episodic 35 % vs chronic 100 % : identical =", np.array_equal(de, db, equal_nan=True))
print("  ground-truth rows  :", int(e["gt"].sum()), "vs", int(b["gt"].sum()))
print()
print("days clipped, chronic 35 % :", int(a["gt"].groupby(a["gt"].index.normalize()).any().sum()))
print("days clipped, chronic 100% :", int(b["gt"].groupby(b["gt"].index.normalize()).any().sum()))
print("days clipped, episodic 35% :", int(e["gt"].groupby(e["gt"].index.normalize()).any().sum()))
print()
print("inject() signature default day_on = None  ->  clamp applied on every row")
