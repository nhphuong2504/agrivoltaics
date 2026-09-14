# Review — MPPT Saturation Detection (Area 1)

**Reviewer:** research review, 2026-09-14
**Subject:** `SATURATION_DETECTION.md` / `saturation_detection.ipynb` (2026-08-27)
**Evidence:** `sat_work/research/` — 9 scripts, reproducible end-to-end
**Verdict:** the *science* is sound and the saturation is real. The *detector* has a
measurable calibration bias that inflates every count in the report, and its specificity
collapses in exactly the regime that matters (intermittent clipping). A simpler,
better-calibrated detector is proposed and verified.

---

## 1. What I did

| Step | Script | Purpose |
|---|---|---|
| 1 | `step1_verify.py` | Re-implement the notebook from scratch and reproduce every published number |
| 2 | `step2_identify.py` | Normalisation-free test of the core claim (difference-in-differences vs peers) |
| 3 | `step3_benchmark.py` | Clip-injection benchmark: 6 detectors × 13 scenarios with ground truth |
| 4 | `step4_synthesis.py` | Mechanism of the failure + a synthesised replacement |
| 5 | `step5_calibration.py` | Measure the baseline's error on pairs that *cannot* saturate |
| 6 | `step6_robustness.py` | Ranking vs hard/soft and constant/drifting limits |
| 7 | `step7_figures.py` | Figures 1–3 |
| 8 | `step8_conclusions.py` | Does the correction change any published conclusion? |
| – | `recommended.py` | Final detector, end to end, with exports |

> **Reproduction is exact.** All 51,005 rows, all three `g0`, all flag counts
> (5,680 / 7,111 / 4,728 rows; 473.3 / 592.6 / 394.0 h; 118 / 149 / 106 days;
> 368 / 334 / 243 severe), the control separation (122 rows, 144×) and the whole
> §7.3 sensitivity table reproduce to the decimal. Nothing in the report is
> hand-waved — that is a real credit to the original work.

---

## 2. What is solid (keep this)

1. **The saturation is real, and the roll-off is not an artefact of `ref` or `cap`.**
   The published validation is self-referential (`ref`, `cap` and `g0` all come from the
   same fleet), so I re-tested it with a *ratio of raw normalised measurements* — no
   ceiling, no `g0`, no `ref` in the denominator (Fig 1). Shared pairs sit at 1.000 vs
   peers in the mid band and fall to 0.57–0.63 at `ref ≈ 1.0`; control pairs are flat
   (0.98–1.02) over the whole range. Three different peer references give the same curve.
   At `ref ≥ 0.7` the shared pairs run **20.5 %, 21.7 % and 23.3 % below peers**.

   ![Fig 1](sat_work/research/fig1_did.png)

2. **The adaptive-ceiling idea is the right idea.** `expected = g0·cap(t)·ref` is
   algebraically a *time-varying slope*: `g0·cap(t)` is the pair's potential output at
   `ref = 1`. Tracking soiling/seasonal drift through a rolling term is exactly right —
   the implementation of that term is what needs fixing (§3.2), not the concept.

3. **The data-quality screening is well done.** `unit_off`, `negative` (correctly
   identified as cross-talk that cancels within the pair), per-day activity masking,
   and the decision that *no DQ-flagged timestamp is ever eligible* are all correct and
   should be carried over unchanged. `stale = 0` and `negative = 188` rows reproduce.

4. **Using independent pseudo-pairs as controls is the right instinct.** The conclusion
   that the detector answers *"constrained vs peers"* is the correct one.

5. **A per-unit test separates the six shared units from the other eighteen almost
   perfectly** (`step16`). Every unit was scored alone — `deficit_u = 1 − X_u/(θ_u·ref)`
   at `ref ≥ 0.75`, no pair sums involved:

   | group | high-irradiance deficit |
   |---|---|
   | eu_8, eu_10, eu_18, eu_16, eu_21, eu_13 | **+0.169 to +0.200** |
   | all 18 other units | **−0.005 to +0.011** |

   The six highest-scoring units in the fleet are *exactly* the six inside the shared
   pairs, and the next-best unit is 15× lower. That is an independent confirmation of the
   shared-MPPT hypothesis that uses neither `ref`, nor `cap`, nor any pair sum — and it
   validates the shared/independent labelling the whole review relies on. `eu_7` also
   stands out (`θ_mid` ≈ 39 W against ≈ 8,800 W for every other unit) — now **confirmed
   faulty by the site engineer**, so it is excluded from every analysis role and guarded by
   `test_topology.py`.

6. **The recorded topology matches the inference, and it rules out inverter-level
   clipping** (§3.12). The site engineer's unit → inverter → MPPT mapping gives *exactly*
   the three inferred MPPT channels. Each clipped pair has independent units as inverter
   mates, and those mates show ~0 % deficit while the paired units show ~20 %: the
   constraint is at the MPPT channel, not the inverter. `eu_24`, flat at −0.001 % while
   sharing an inverter with the clipped `eu_8+eu_16`, is the sharpest single control in
   the review.

7. **`θ(t)` independently recovers the documented hardware nameplate** (`step19`). The
   maintenance deck states *"each string rating within an experimental unit is 9.0-kW ca."*
   Nothing in this analysis was tuned to a nameplate, yet `θ_u` — the rolling mid-band
   median of `s/ref`, the quantity `vat-v1` multiplies by `ref` — comes out at
   **median 8,847 W, all 23 healthy units within 4.1 % of 9.0 kW** (median ratio 0.983,
   sd 0.017).

   This is the strongest single validation in the review. It means `vat-v1`'s slope is the
   *physical string rating*, not a fitted fudge factor — which is exactly why replacing
   `g0·cap(t)` with `θ(t)` makes the `deficit` columns interpretable as a fraction of
   nameplate. It also independently corroborates `eu_7`'s exclusion and the whole
   shared/independent labelling. Supporting check: the published `cap(t)` sits at
   12.8–13.1 kW ≈ **71–73 %** of the 18 kW combined pair nameplate, and the highest pair
   sum ever observed at `ref ≥ 0.7` is 13.6–16.2 kW (75–90 %). Two strings need not be at
   the same tilt, so treat that as supporting rather than decisive.

---

## 3. What is wrong, with numbers

### 3.1 The baseline has a systematic high-irradiance bias (~7–9 %)

The control pairs **cannot** saturate, so any deficit they show is model error. At the
top of the irradiance range the published baseline reports:

| pair | published `g0·cap·ref` | rolling `θ(t)·ref` | peer-ratio truth |
|---|---|---|---|
| eu_1+eu_3 | **+0.088** | +0.016 | 0.000 |
| eu_4+eu_12 | **+0.074** | 0.000 | −0.022 |
| eu_15+eu_23 | **+0.089** | −0.011 | −0.005 |

*(median deficit in the top `ref` bin)*

The model calls ~9 % saturation on a pair that is physically incapable of it, and the
moderate threshold is only 0.15. That is a **margin of 6 points** between the method's
own null and its decision boundary. Note this is a summer-dominated average: §3.2 shows
the bias is *seasonally signed*, running +0.08 in midsummer and **−0.45** in December.

![Fig 2](sat_work/research/fig2_baseline.png)

### 3.2 `g0·cap(t)` carries the whole seasonal swing, and that is the real mechanism

> **Correction (2026-09-14, `step11`).** My first version of this section said the
> ceiling is *dragged down* by clipping, so `expected` is too small and the deficit
> inflates. That is backwards: `deficit = 1 − s/expected`, so a smaller `expected` makes
> the deficit *smaller*, which would cause **misses**, not false positives. Every FP row
> has `s < 0.85·expected` by definition, i.e. `expected` is too **large** there. I
> measured it instead of arguing, and the measured mechanism is below.

`cap(t)` is the rolling median of the pair's **own daily 99th percentile**, so it tracks
the physical peak, which is strongly seasonal (10.3 MW in Dec → 16.5 MW in Jun). `g0` is
one global scalar fitted at mid band, so `g0·cap(t)` inherits that entire seasonal swing
while the true unsaturated slope `θ(t)` barely moves. The consequence is a
seasonally-signed error in `expected`:

| month (control pairs, cannot saturate) | `cap` | `cap/θ` | published `deficit` |
|---|---|---|---|
| 2025-06 | 16.1 MW | 0.909 | **+0.082** |
| 2025-07 | 16.1 MW | 0.908 | **+0.080** |
| 2026-02 | 14.5 MW | 0.809 | −0.008 |
| 2025-12 | 10.3 MW | 0.576 | **−0.452** |

Correlation between the control-pair deficit and `cap(t)` across months is **r = +0.991**
— the bias is not a clipping interaction at all, it is the model's seasonal scaling error.
In summer `expected` runs **12–15 % too high** (`exp_pub/exp_true` = 1.15 on FP rows vs
1.12 on true-negative rows), which manufactures positive deficit on healthy rows; in
winter it under-predicts badly (−0.45), which suppresses flags.

Direct test of the old "drag" story (`step11`, clip at q=0.75 injected into a control
pair): `cap/θ` on clipped days is **0.859** vs **0.863** on unclipped days — clipping
moves the ceiling by **−0.5 %**. The ceiling is essentially not dragged; the seasonal
scaling is the whole effect.

### 3.3 Consequence: specificity collapses for intermittently-clipped pairs

Clip-injection benchmark (`step3`, `step4`) — clipping injected into independent pairs,
so positives are labelled ground truth and unclipped days give labelled negatives:

| duty cycle | published (D0) FP rows | rolling-slope (D1) FP rows |
|---|---|---|
| clip 35 % of days | **5,251** | **34** |
| clip 65 % of days | 1,297 | 34 |
| clip every day | 42 | 25 |

**150× more false positives** in the realistic intermittent regime. On unclipped days
the published model simply claims saturation. Critically, **the published validation
cannot see this**: its control pairs are never clipped, so the failure mode is never
exercised.

### 3.4 The published energy-loss estimate is ~50 % bias

Apparent lost energy at `ref ≥ 0.7` (`step5`, `recommended.py`):

| | shared pairs | control pairs (**all bias**) | bias / signal |
|---|---|---|---|
| published | 7,131 kWh | **3,356 kWh** | **47 %** |
| recommended | 6,120 kWh | 409 kWh | 7 % |

(Control pairs here are the three legacy ones. The topology now lets us measure this on the
whole flat population instead of three pairs — see §3.12 part 4: the published expectation
fabricates a median 803 kWh/pair across 117 independent pairs versus 92 kWh/pair for vat-v1,
so the bias is systemic and not a property of these three.)

The published estimator assigns almost **half as much "lost energy" to pairs that cannot
lose energy** as it does to the saturating pairs. Any downstream yield or ROI figure
built on it inherits that bias.

### 3.5 `deficit` is a poor discriminator of "is the constraint active"

AUC against ground truth, stratified by how deep the clip is (threshold-free, so this is
not a tuning artefact):

| detector | any clip | >10 % | >20 % |
|---|---|---|---|
| **D1 rolling-slope** | **0.849** | **0.924** | **0.998** |
| D2 envelope-q90 | 0.846 | 0.923 | 0.997 |
| D5 control-calibrated | 0.782 | 0.909 | 0.966 |
| D0 published | 0.782 | 0.878 | 0.979 |
| D4 ceiling-pinned | 0.794 | 0.686 | 0.612 |

A relative-shortfall score is dominated by cloud dips at moderate `ref`, so it cannot
cleanly answer "is the limit binding". At shallow clips the published detector is close
to its null. This is the honest reading of the published tiering: it is a **severity**
detector, not a **presence** detector, which the report does not say.

![Fig 3](sat_work/research/fig3_scoreboard.png)

### 3.6 Blind spot below the calibration band

`g0` is calibrated on `ref ∈ [0.35, 0.60]`. Any constraint that binds *below* `ref ≈ 0.6`
is absorbed into the calibration and becomes invisible. In the benchmark at `q = 0.65`,
21 % of clipped samples were unreachable by the `ref ≥ 0.7` gate. Worth stating explicitly
as a scope limit.

### 3.7 Smaller items

| item | finding |
|---|---|
| "31-day" rolling window | built on an **irregular** daily index (230 of 451 calendar days present), so the median window really spans **48 calendar days**, and up to **110 days** across the acquisition gaps. Numerically minor (median ceiling change 21 W; max 610 W ≈ 4.5 %) but it should be reindexed to a full calendar. |
| persistence | `run_lengths` runs over the whole series, so a run can straddle a gap or a day boundary. **FIXED** — two bugs, both caught by the suite. (1) Runs could straddle multi-month acquisition gaps; `persist` now breaks at any gap > 6 min. (2) **Adopted** gap-tolerant persistence (`step13`, `step14`, `step18`): bridge 1-sample gaps before the run-length rule, restricted to the gate domain. +5.0 pp recall on injected ground truth (0.349 → 0.399) for +72 FP rows; on real data +5.3 % hours with **zero** extra control-pair flags. A guard on the filled samples was tested and does **not** dominate — this is a real trade-off, not a free gain. Not applied to the severe tier (§3.13). |
| run straddling acquisition gaps | **fixed** — a run could span a multi-month hole in the record. See §3.11. |
| "no predefined threshold" | the method contains six fixed constants (`REF_ON`, `DEF_MOD`, `DEF_SEV`, `NEAR_CAP`, `PERSIST_MOD`, `PERSIST_SEV`). Only `cap` and `g0` adapt. The framing overstates it. |
| uncertainty | no confidence intervals anywhere; hard 0/1 output. |
| `eu_7` | excluding it is well justified and I reproduced the r ≈ 0.63 / scale argument. |

---

### 3.8 Case study — July 2026 is a threshold artefact, not an anomaly

§5 flags July 2026 as the one place the two detectors tell materially different stories
(eu_8+eu_16: 78.8 h → 46.0 h). I chased it because it is the obvious candidate for "the
new detector is missing something". It is not (`step10`–`step12`).

Measured at matched irradiance (`ref ≥ 0.75`), month by month, against the
normalisation-free peer ratio — for **eu_8+eu_16**:

| month | truth (peer ratio) | published | vat-v1 | published gap |
|---|---|---|---|---|
| 2025-05 | 0.226 | 0.258 | 0.233 | +0.032 |
| 2025-06 | 0.194 | 0.237 | 0.203 | +0.043 |
| 2025-07 | 0.178 | 0.223 | 0.193 | +0.045 |
| 2026-02 | 0.179 | 0.200 | 0.183 | +0.021 |
| 2026-06 | 0.215 | 0.250 | 0.216 | +0.034 |
| **2026-07** | **0.136** | **0.186** | **0.136** | **+0.050** |

1. **vat-v1's deficit tracks the independent physical truth to within 0.001–0.010 in
   every month, on all three pairs.** That is real-data validation, not a benchmark
   artefact — and it is the strongest evidence in this review for the recommended model.
2. The published detector over-states the deficit by a roughly constant **+0.02 to
   +0.05** in *every* month, not just July.
3. **July 2026 has the mildest true shortfall of the whole record (0.136)** — the pair was
   less clipped then, not more. So the 0.15 threshold lands mid-population: the truth
   (0.136) sits just under it, the published deficit (0.186) sits just over it, and 42.5 h
   of rows change side. In the loud months the true deficit is 0.19–0.23, clear of the
   boundary, so the two detectors agree. The "spike" is a boundary slicing through a
   population, not a physical event.

The clear-sky anchor is **not** contaminated: the pair's mid-band efficiency relative to
peers is 0.998–1.033 in every month including July 2026 (`step12`), so `θ` is clip-safe.
This also validates the recommended detector's central assumption on real data.

Two by-products of the same dig:

* **The published detector also misses badly in winter** — its `expected` is ~40 % too low
  (control-pair deficit −0.452 in Dec 2025), so it reports large negative deficits that can
  never flag. Its failure is two-sided; only the summer side has been visible.
* 74 of the 510 disputed July rows were not about thresholds at all: vat-v1's deficit did
  exceed 0.15 and the strict 3-sample rule broke the run (see §3.7).

### 3.9 The shipped flags file contained 5,373 `-inf` values — **now repaired**

`saturation_flags.csv` publishes `eu_*_deficit` as a continuous column. It contained
**5,373 infinite values** (all `-inf`) across **2,807 rows**, in all three deficit columns:

```
2025-05-01 06:35:00,0.0,0,-inf,0,0,-inf,0,0,-inf,0,0
2025-05-01 20:10:00,0.0,0,-inf,0,0,-inf,0,0,-inf,0,0
```

Mechanism: `d = 1 − s/(g0·cap(t)·ref)`. At dawn/dusk `ref = 0`, so `expected = 0` while the
pair sum is still positive, giving `−inf`. The flags are correctly `0` on those rows — the
`ref ≥ 0.70` gate excludes them — which is exactly why no flag-level check ever caught it.
`pandas.read_csv` parses `-inf` as `float('-inf')`, so any downstream `.mean()`, `.sum()`,
regression or plot on that column is poisoned. I tripped over it in my own aggregate.

**Repaired 2026-09** (`step20`): the values are now `NaN`. Every affected row sits at
`ref = 0.0000`, seven hundred times below the gate, so no flag could ever have fired there —
verified that the `*_sat_moderate` / `*_sat_severe` columns are **byte-identical** after the
repair. Only the continuous deficit columns change, and only on rows the original document
already told readers to exclude. The original file is preserved as
`saturation_flags.published_backup.csv` for provenance, and the regression test that
documented the defect now asserts the fixed condition and passes.

One caveat, now closed: the *notebook* is the generator that would reintroduce `-inf`.
Its export cell (cell 25) has been patched at the source, so
`out[f'{k}_deficit'] = …replace([np.inf, -np.inf], np.nan).round(4)`. A fresh run now
produces a clean file, and the notebook is tracked by git so the fix is versioned. Note
that the notebook's saved outputs are now stale relative to its source until it is re-run;
re-running is safe (it reproduces the same flags and a `-inf`-free deficit column).

### 3.10 The continuous `deficit` column is seasonally mis-scaled by up to 0.50

Same seasonal mechanism as §3.2, but the continuous column is worse than the flags suggest,
because the annual aggregate the report quotes partly cancels the signs:

| pair | median abs error, published | median abs error, vat-v1 | worst month |
|---|---|---|---|
| shared eu_8+eu_16 | 0.040 | **0.004** | 2025-12 (0.242) |
| control eu_4+eu_12 | **0.090** | **0.014** | 2025-12 (**0.496**) |
| control eu_15+eu_23 | **0.114** | **0.009** | 2025-12 (0.452) |

In December the published column reports **−0.50** on a control pair whose true deficit is
**+0.003**. Anyone treating `deficit` as a severity measure — which is what §6.5 invites —
inherits a half-unit error that no flag-level check would surface. vat-v1 is 6–12× more
accurate on the same column, and maps infinities to `NaN`.

### 3.11 A bug in my own recommended detector, found by the test suite

`run_lengths` was applied over the whole series, so two flagged samples before an
acquisition gap plus one after counted as a run of three, and all three were published as a
15-minute episode. On the real data this affected **9 runs**, including `eu_10+eu_18`. Fixed
by breaking runs at any step longer than 6 minutes (`bench.persist(..., index=...)`). The
effect on the benchmark is 0.15 % of true positives, so no earlier conclusion moves.

### 3.12 The control pairs: what the fleet topology actually says

The site engineer supplied the true layout (now recorded in `satsim.py`):

```
MPPT channels : eu_8+eu_16, eu_10+eu_18, eu_13+eu_21
inverter 1 : eu_1,  eu_9,  eu_17          inverter 4 : eu_5,  eu_6,  eu_13, eu_14, eu_21, eu_22
inverter 2 : eu_2,  eu_3,  eu_10, eu_11,  inverter 5 : eu_7,  eu_15, eu_23
             eu_18, eu_19                 inverter 6 : eu_8,  eu_16, eu_24
inverter 3 : eu_4,  eu_12, eu_20
```

**The inferred topology was exactly right.** The three MPPT channels are precisely the
three pairs inferred from the data in §2 — the shared/independent labelling the whole
review rests on is now documented rather than inferred.

**But two of the three control pairs share an inverter.** `eu_4+eu_12` are both on
inverter 3 and `eu_15+eu_23` are both on inverter 5; only `eu_1+eu_3` crosses inverters.
That is worth knowing, and it turns out to *strengthen* the case in two ways.

**1. A natural experiment rules out inverter-level clipping.** Each clipped pair has
independent units as inverter mates. If the roll-off were an inverter limit, those mates
would roll off too. They do not — scored individually at `ref ≥ 0.75`:

| inverter | MPPT-paired units | inverter mates |
|---|---|---|
| 2 | eu_10 +0.199, eu_18 +0.193 | eu_2 +0.000, eu_3 +0.011, eu_11 +0.001, eu_19 −0.005 |
| 4 | eu_13 +0.169, eu_21 +0.187 | eu_5 +0.004, eu_6 +0.006, eu_14 −0.001, eu_22 +0.001 |
| 6 | eu_8 +0.200, eu_16 +0.190 | eu_24 −0.001 |

A ~20 % deficit on the paired units with ~0 % on their inverter mates is as clean a
separation as the data could give: the constraint is at the **MPPT channel**, not the
inverter. `eu_24` sharing an inverter with the clipped `eu_8+eu_16` while staying flat is
the single sharpest control in the review.

**2. Same-inverter independent pairs are flat on median.** Sweeping the whole fleet:

| null population | pairs | median | p90 | p99 | max |
|---|---|---|---|---|---|
| cross-inverter independent | 117 | +0.22 % | +0.74 % | +1.06 % | +1.14 % |
| same-inverter independent | 19 | +0.32 % | +0.67 % | +0.84 % | +0.87 % |
| **the three MPPT pairs** | 3 | — | — | — | **+18.0 to +19.8 %** |

So sharing an inverter adds nothing to the deficit — the nulls stay under 1.2 % at the
99th percentile while the MPPT pairs sit at ~19 %. Sharing an inverter is not a confound
for the saturation story, and the legacy controls remain valid as magnitude nulls.

**3. `eu_15+eu_23` is a warm null, not a broken one.** It flags 108 rows, the most of any
flat pair. Four diagnostics:

* both units are individually healthy (+0.007 / −0.002);
* the pair's own calibration agrees with its own units to **+0.08 %**, so there is no
  pair-level calibration artefact — my earlier guess that these were shallow boundary
  cases was wrong;
* the pair is **flat on median** (+0.003 over all 9,981 high-ref rows) — the 108 flags are
  a tail, not a systematic deficit, with a median deficit of 0.184 on the flagged rows;
* flat **same-inverter** pairs are inherently hotter than cross-inverter ones (p90 32 vs
  15 flags, ≈2×), because same-inverter units share a deficit common mode
  (mean correlation **+0.50** vs **+0.04** across inverters). `eu_15+eu_23` is nonetheless
  still the most-flagged flat pair in the fleet, so the common mode explains part of the
  excess, not all of it.

The leading hypothesis was a mild condition on **inverter 5**. That hypothesis is now
**withdrawn**: the site engineer confirms `eu_7` has a unit-level technical fault, so its
degradation says nothing about inverter 5's health and cannot explain `eu_15+eu_23`. What
remains is the honest position — the excess is consistent with the same-inverter
co-movement common mode plus threshold-tail behaviour, and no cause has been identified
beyond that.

`eu_7` is confirmed faulty and must be ignored everywhere. It is excluded from `RELIABLE`
(so it cannot enter `ref` or the site-outage rule), from `SHARED`, from `PAIRS`, from
`CONTROL_PAIRS` and from the inverter-mate sweeps. `test_topology.py` now enforces this,
including a behavioural guard: corrupting `eu_7` must leave `ref` bit-identical. Verified
to have teeth — forcing `eu_7` into `RELIABLE` moves `ref` by up to **0.39** on a 0–1
scale, so the test would catch a regression.

**Practical consequence.** For **magnitude** work (the bias floor, the empirical null) all
three controls are valid — every control class is flat on median. For **flag-level**
comparisons, use cross-inverter pairs only.

**4. The published bias is fleet-wide, not one bad control.** Over 117 independent pairs
the published expectation fabricates a median of **803 kWh/pair** of phantom lost energy
(p90 1,704; max 2,989), against **92 kWh/pair** for vat-v1 — **11 %**. The three legacy
controls individually (published / vat-v1): `eu_1+eu_3` 1,071 / 145, `eu_4+eu_12`
964 / 87, `eu_15+eu_23` 1,321 / 176. All the same order of magnitude, so the bias floor is
a systemic property of the published `expected`, not an artefact of which control you pick.
That is a stronger and simpler statement than the one this section made before the
topology arrived.

**Honest scope note on §3.3:** the 150× specificity gain is a *benchmark* number. On the
three real control pairs the flag-level improvement is modest (122 → 108 rows total). The
real-data win is in the magnitude of what gets claimed, and it is now measurable on a real
null rather than three hand-picked pairs: across 117 independent pairs the published
expectation fabricates a **median 803 kWh/pair** of phantom loss against **92 kWh/pair** for
vat-v1 (§3.12). Do not quote the 150× as a real-data result.

### 3.13 The persistence rule: two failed ideas, then a rule with no real-data cost

The last open modelling decision is strict vs gap-tolerant persistence. `step13`/`step14`
suggested bridging 1-sample gaps recovers genuine signal, and `step14`'s diagnostics
suggested an obvious refinement. Of the rows bridging recovers, most are genuine and a
minority are spurious, and the two groups separate cleanly on deficit:

```
genuine recovered rows : median deficit 0.161
spurious recovered rows: median deficit 0.017
```

The spurious rows are gap-*interior* samples whose own evidence is far too weak to flag;
bridging accepts them only because they sit between two real samples. So **idea 1** was to
guard the fill: accept a bridged-in sample only if its own deficit clears a floor `s`.

**It does not work.** Every guard setting lands on the Pareto frontier (`step18`):

| variant | recall | FP rows | precision |
|---|---|---|---|
| strict | 0.3491 | 108 | 0.9925 |
| bridged | **0.3991** | 180 | 0.9891 |
| bridged + guard 0.03 | 0.3949 | 151 | 0.9907 |
| bridged + guard 0.05 | 0.3924 | 145 | 0.9910 |
| bridged + guard 0.075 | 0.3872 | 137 | 0.9914 |
| bridged + guard 0.10 | 0.3803 | 128 | 0.9918 |

No variant dominates another; the guard sheds false positives and genuine recall at broadly
comparable rates. **There is no free lunch here, and I was wrong to expect one.**

**Idea 2 came from the regression suite, and it worked.** The suite failed the first
implementation with `flag below the ref gate` — 236 rows on `eu_8+eu_16`. Bridging a gap
whose missing sample dropped out because a cloud edge pushed `ref` under 0.70 was setting a
flag *below the gate*, violating the contract in §5. The fix is to restrict fills to samples
that satisfy the gate: **bridge only where the sample was genuinely evaluable.** That is
strictly better-motivated than the original rule and it changes the real-data picture
completely:

| variant | hours on shared pairs | cross-inverter FP | same-inverter FP |
|---|---|---|---|
| strict | 1,205.0 | **0** | 108 |
| bridged (ungated, rejected) | 1,408.2 | **0** | 126 |
| **bridged within the gate (adopted)** | **1,269.2 (+5.3 %)** | **0** | **108** |

The ungated rule's entire real-data cost — the +18 rows on `eu_15+eu_23` reported earlier —
was an artefact of filling below the gate. With fills restricted to the gate domain, the
adopted rule adds **exactly zero** rows on *both* control classes while recovering +5.3 % of
hours and +5.0 pp of recall. On real data it is free; on the benchmark it costs 72 FP rows,
and that is the honest caveat.

**Adopted rule:** bridge 1-sample gaps **within the gate** before the run-length test, for
the 15-minute (`k=3`) tiers only.

**Deliberately not applied to the severe tier.** The benchmark's injected severity caps at
about 0.25, so a `>0.30` tier scores **zero true positives** — the benchmark cannot
adjudicate the rule there at all. Bridging would move severe from 7.1 to 21.0 h on
`eu_8+eu_16` (a 3× inflation) on the strength of an untested analogy, in the tier most
exposed to baseline bias. Severe stays strict until ground truth exists for it. This is a
limitation of the benchmark, added to §8.

---

## 4. Alternatives tested

| detector | result |
|---|---|
| **D1 rolling-slope** `expected = θ(t)·ref`, `θ` = 31-day rolling median of daily median `s/ref` in `[0.35,0.6]` | **winner**: best AUC, 15–200× fewer FP, unbiased on controls (≈0.00 at all `ref`) |
| **D2 envelope-q90** same but daily 90th pct | equal ranking, higher recall at the standard threshold, but a constant ≈ +0.03 offset (it estimates *potential*, not the median) so its threshold must be shifted |
| **D5 control-calibrated** empirical null from control pairs per `ref` bin | best specificity, but the control null carries the same model bias, so it costs real sensitivity at 10–20 % severity |
| **D4 ceiling-pinned** "is `s` sitting on its own plateau while `θ·ref` exceeds it?" | near-perfect (AUC 1.00) for a **stable** limit, but fails for intermittent clips (AUC 0.59–0.68). Useful as a *complementary* diagnostic, not a primary detector |
| **D3 ref-gated 2-state HMM** (EM, state-1 mean = `min(1, knee/ref)`) | **failed** (AUC ≈ 0.48). A cloud dip and a clipped sample are both "low utilisation"; a 2-state Gaussian emission cannot separate them without the plateau/pinning structure. Reported because negative results matter |

**Robustness** (`step6`, all four constraint shapes): D1/D2 hold AUC ≥ 0.993 with 6–42 FP
rows; D0 stays at 0.866–0.946 with 220–1,335 FP rows. The ranking does not depend on
hard vs soft clips, or constant vs drifting limits. (A drifting limit *helps* D0, because
its rolling cap partly tracks the drift — the constant clip is its worst case.)

---

## 5. Recommended approach (`vat-v1`)

```
θ(t)   = 31-day centred rolling median of the DAILY MEDIAN of s/ref
         over active samples with ref ∈ [0.35, 0.60]     <- clear-sky anchor, clip-safe
d(t)   = 1 - s(t) / (θ(t) · ref(t))                      <- continuous severity
gate   = ref ≥ 0.70  AND  s > 0.6·θ·ref  AND  both units active  AND  no DQ flag
q_n    = 99.5th pct of the same d on the independent control pairs, per ref bin
moderate      = gate AND d > 0.15 AND (bridge 1-sample gaps within gate) AND 3 consecutive samples
severe        = gate AND d > 0.30 AND 6 consecutive samples          <- strict, see §3.13
conservative  = gate AND d > q_n(ref) AND (bridge within gate) AND 3 consecutive samples
```

Implementation: `sat_work/research/recommended.py`; output
`sat_work/research/saturation_flags_v2.csv` (per-pair `deficit`, `excess_vs_controls`,
`null_d`, three flag columns). Nothing in the repo root was modified.

**What changes, and what does not** (`step8`):

| pair | published h | vat h | published severe h | vat severe h | days (pub → vat) |
|---|---|---|---|---|---|
| eu_8+eu_16 | 473.3 | **386.6** | 30.7 | **7.1** | 118 → 115, all 115 shared |
| eu_10+eu_18 | 592.6 | **539.8** | 27.8 | **13.0** | 149 → 147, 145 shared, 2 new |
| eu_13+eu_21 | 394.0 | **342.8** | 20.2 | **7.2** | 106 → 104, 102 shared, 2 new |

Total: **17,519 rows / 1,459.9 h → 15,230 rows / 1,269.2 h** (−13.1 %).

* **Every published conclusion survives.** The affected days are almost a subset of the
  published ones (0–2 new days; 3–4 published-only). The seasonal shape is unchanged:
  May–Jul peak plus the cold-February clear-day peak, zero in the acquisition gaps and in
  the pair-off blocks.
* **Magnitudes shrink 9–18 %** (−18.3 %, −8.9 %, −13.0 %), *after* the adopted
  gap-tolerant persistence rule has recovered +5.3 % of hours (§3.13). Without that rule
  the shrink is 13–22 %. **The severe tier shrinks 2.5–4×** — unchanged, because bridging
  is deliberately not applied there.
* **July 2026 (eu_8+eu_16: 78.8 h → 46.0 h) is now explained and is not a defect** — it
  is a threshold boundary landing mid-population in the mildest month of the record. See
  §3.8. On the same rows vat-v1's deficit matches the independent peer-ratio truth to
  within 0.001–0.010, which is the best real-data evidence for the recommended model.
* **Persistence: the adopted rule is now in the artefact.** `vat` numbers above include
  gap-tolerant persistence restricted to the gate domain — +5.0 pp recall on injected
  ground truth (0.349 → 0.399) and, on real data, +5.3 % hours with **zero** extra flags
  on any control pair (§3.13). Not applied to the severe tier, which the benchmark cannot
  adjudicate.

---

## 6. Prioritised actions

1. **Replace the ceiling.** Swap `g0·cap(t)·ref` for `θ(t)·ref`. One function, all
   downstream numbers become calibrated. Highest impact, lowest risk.
2. **Re-state the counts.** The headline 17,519 rows / 1,459.9 h and the implied kWh are
   inflated: 15,230 rows / 1,269.2 h after the fix (−13.1 %), and the severe tier is
   inflated 2.5–4×. The bias floor is 47 % of the published energy figure.
3. ~~**Delete the `-inf` values** (§3.9).~~ **DONE 2026-09** (`step20`): repaired to `NaN`
   in the shipped file (flag columns verified byte-identical, original kept as
   `saturation_flags.published_backup.csv`) **and fixed at the source** in the notebook's
   export cell, so a re-run cannot reintroduce them.
4. **Run the regression suite before publishing any detector change**
   (`sat_work/tests/run_tests.py`, 42 tests, ~4 s, no new dependencies). It encodes the
   ground-truth benchmark that the existing control-pair validation structurally cannot
   replace, and it has now caught three real defects — the persistence bug in §3.11, the
   below-gate fill in §3.13, and the `-inf` values in §3.9.
5. **Separate presence from severity.** Publish `d(t)` and `q_n(ref)` as continuous
   columns; let the user threshold. State that the moderate tier is a severity tier, and
   mask `q_n` outside the gate domain (§3.10).
6. **State the scope limits**: blind below `ref ≈ 0.6`; winter low-sun untestable;
   single-timestamp local shading indistinguishable from mild saturation (as the report
   already says).
7. **Reindex the rolling windows** to a complete daily calendar before `.rolling()`.
8. **Adopted: gap-tolerant persistence within the gate** (`step18`, §3.13), for the
   `k=3` tiers only. +5.0 pp recall (0.349 → 0.399) for +72 FP rows on injected ground
   truth, and on real data **zero** extra flags on any control pair for +5.3 % hours.
   Two caveats to quote, not bury: it is a genuine precision/recall trade-off rather than a
   free gain, and it is *not* applied to the severe tier because the benchmark's injected
   severity caps at ~0.25 and cannot adjudicate a `>0.30` tier.
9. **Restrict flag-level nulls to cross-inverter pairs** (§3.12). The topology is now
   known: `eu_4+eu_12` (inverter 3) and `eu_15+eu_23` (inverter 5) share an inverter and
   are ~2× more likely to trip the threshold through a co-movement common mode. Magnitude
   work can keep all three.
10. **Note the unexplained residue**: `eu_15+eu_23` remains the most-flagged flat pair in
    the fleet (the inverter-5 hypothesis was withdrawn once `eu_7` was confirmed as a
    unit-level fault). It is a warm null, not a broken one — the pair is flat on median
    and agrees with its own units to +0.08 % — but the excess flags are not fully
    accounted for.
11. **Optional**: keep the `q_n(ref)` control-calibrated tier as an alternative
    calibration — note it is *not* stricter than moderate in practice despite the
    `conservative` name — and the plateau-pinning test as a *chronicity* diagnostic.

---

## 7. Audit of the downstream artefacts

Requested audit of the two artefacts that consume this work (`step19`).

### 7.1 `power_production.pptx` — nothing to correct, everything to fill

4 slides, and **no saturation numbers appear anywhere in it**:

| slide | text | figures |
|---|---|---|
| 1 | "Analysis of Power Production" — Phuong Nguyen and Hugo Villegas | 1 |
| 2 | "Time-Series Power Measurements", `eu_11` (strawberry), `eu_15` (grass or control), `eu_11 − eu_15`, "Each string rating within an experimental unit is 9.0-kW ca." | 2 |
| 3 | **"Explain about saturated problem"** — a placeholder | 1 |
| 4 | *(no text)* | 1 |

So the deck does not need correcting — slide 3 is an empty heading waiting for the
analysis. **What it needs:** the §5 table (per-pair hours and severe hours), the severe-tier
shrink of 2.5–4×, the bias floor (47 % → 7 % of the claimed energy), and figures 1–3.
Slide 2's own numbers are consistent with the analysis and need no change.

Slide 2 also produced the review's best validation (§2.7): the 9.0 kW nameplate matches
`θ_u` to 4 % across all 23 units.

**One loose end the deck hands back.** It records `eu_15` as *"grass or control"* and
`eu_11` as *"strawberry"* — i.e. units differ in ground cover. That is a plausible
mechanism for the one unexplained result in this review (§3.12): `eu_15` and `eu_23` are
individually healthy yet their pair is the most-flagged flat pair in the fleet. Two units
with different ground cover can have different high-irradiance behaviour (temperature
coefficient, soiling), which a single pair-level `θ` cannot capture. It is testable — ask
which units share a cover type — and it is the most promising remaining lead.

### 7.2 `SATURATION_DETECTION.md` — specific claims that change

| § | claim as written | status |
|---|---|---|
| 7.1 | **"144× separation"**, 122 control rows vs 17,519 shared | separation is real but the 122 are **not** what the doc says |
| 7.1 | control flags "land on genuine few-percent midday sags — consistent with mild inverter AC-limit clipping, i.e. small *real* constraint events rather than detector noise" | **falsified.** They are baseline bias: the published expectation fabricates a median 803 kWh/pair across 117 independent pairs (§3.12). The comparison is also mismatched — 122 rows is the flag count, not the claimed energy |
| 7.4 | rows/hours/days: 5,680 / 473 / 118; 7,111 / 593 / 149; 4,728 / 394 / 106 | → **4,639 / 386.6 / 115; 6,478 / 539.8 / 147; 4,113 / 342.8 / 104** (−13.1 % overall) |
| 7.4 | "Severe events: 368 / 334 / 243 rows" | → **85 / 156 / 86 rows** (2.5–4× smaller; the tier most exposed to baseline bias) |
| 7.3 | threshold-sensitivity table (473 / 593 / 394 at 0.15) | **recomputed** — see below |
| 5.3 | `expected = g0 · cap(t) · ref` | → **`θ(t) · ref`** (§3.2). The doc's §10 already proposes this: *"`g0` is currently a global-per-pair scalar; making it rolling (31-day median of mid-ref gain) would track long-term soiling more tightly."* **`vat-v1` is that extension, not a departure from the method** |
| 8.1 | `-inf` at `ref = 0` "division artifact; flags can never fire there — filter `ref ≥ 0.7`" | **disclosed, not hidden** — good practice. The values have now been repaired to `NaN` in the shipped file (§3.9) |
| 2.1 | `eu_7` "on a different scale … treated as an odd/reference unit and excluded" | correct; now **confirmed by the site engineer as a technical fault** and guarded by `test_topology.py` |

**Recomputed §7.3 — flagged hours vs the deficit cutoff** (`step20`). The doc's conclusion
was *"monotonic, smooth, no cliff → the method does not hinge on a finely tuned cutoff"*.
That claim **survives** the corrected baseline:

| deficit > | doc (pub) | vat-v1 | vat-v1 relative drop |
|---|---|---|---|
| 0.10 | 557 / 714 / 508 | 499 / 677 / 463 | — |
| 0.15 | 473 / 593 / 394 | 387 / 540 / 343 | 22–26 % |
| 0.20 | 342 / 410 / 253 | 254 / 362 / 209 | 33–39 % |
| 0.30 | 61 / 58 / 34 | 31 / 50 / 24 | 86–89 % |
| 0.40 | 6 / 2 / 3 | 0 / 0 / 0 | 100 % |

(order is eu_8+eu_16 / eu_10+eu_18 / eu_13+eu_21.) Monotonic throughout with no discontinuity,
so the "not finely tuned" claim holds. Two honest qualifications the doc omits: the fall-off
between 0.20 and 0.30 is **steep** — ~88 % of the remaining hours disappear in that step, in
both detectors — and vat-v1 is **empty at 0.40** where the doc reports 6 / 2 / 3 hours. A
`>0.40` tier has no support under the corrected baseline, which is consistent with the
benchmark's finding that injected severity caps near 0.25 (§8).

Two things the doc gets right and the review preserves: the DQ screening is sound and
carries over unchanged (§2.3), and section 4's `RELIABLE` list is exactly right, including
`eu_7`'s exclusion.

---

## 8. Limitations of this review

* The benchmark injects a **clamp on the pair sum**. That is the correct observable
  signature of a shared MPPT limit, but it assumes the limit acts on total power; a
  limit that removes one unit entirely would look slightly different.
* Ground truth exists only for injected clips. On the real pairs I can validate the
  *baseline* (against control pairs) and the *deficit magnitude* month by month
  (`step12`); `d_vat` tracks the independent peer-ratio truth to within 0.001–0.010, which
  is stronger than expected. But the correctness of *individual timestamps* is still not
  verifiable without a shunt/pyranometer measurement. A pyranometer would close this gap
  and the report already flags that extension.
* The peer-ratio truth is a **cross-sectional** measurement: it compares the pair with
  independent peers, so it catches the shortfall of *that pair* but cannot detect a loss
  that hit the whole fleet at once (soiling, a site-wide inverter setting). A uniform fleet
  loss would be invisible to both the peer test and the irradiance proxy `ref`.
* The `D0` numbers here use the published parameters verbatim. Letting D0 re-tune its
  thresholds on the benchmark would raise its AUC slightly, but it cannot fix an
  *unbiasedness* problem with tuning — the FP counts are a calibration defect, not a
  threshold choice.
* **The benchmark cannot exercise the severe tier.** Injected severity caps at ≈0.25
  (the clamp sits at 75 % of the pair's P95 clear-sky peak), so a `deficit > 0.30` tier
  scores **zero true positives** and no persistence or threshold change can be scored
  there. This is why gap-tolerant persistence is applied to the `k=3` tiers only: the
  analogous change for the severe tier would inflate it ~3× (7.1 → 21.0 h on `eu_8+eu_16`)
  on an untested analogy, in the tier most exposed to baseline bias. Extending the
  benchmark to deeper clips is the single highest-value improvement to this test setup.
* **The adopted persistence rule trades recall for precision.** Bridging within the gate
  costs 72 false-positive rows on the benchmark (108 → 180) for +5.0 pp recall. It is
  free on the real control pairs, but the benchmark disagrees, and the benchmark is the
  only place with ground truth. Quote both numbers.

---

## 9. Files

```
sat_work/
  research/
    satsim.py              faithful re-implementation of the published pipeline
    bench.py               FROZEN regression harness: scenarios, detectors, metrics,
                           build_export(). Do not let research scripts drift from this.
    step1_verify.py        reproduction of every published number
    step2_identify.py      difference-in-differences vs peers
    step3_benchmark.py     6 detectors x 13 scenarios, ground truth
    step4_synthesis.py     the synthesised detector R (its 'drag' rationale needed fixing)
    step5_calibration.py   baseline error measured on non-saturable pairs
    step6_robustness.py    hard/soft, constant/drifting limits
    step7_figures.py       fig1_did.png, fig2_baseline.png, fig3_scoreboard.png
    step8_conclusions.py   published conclusions re-checked
    step9_canvasdata.py    chart series for the canvas
    step10_july.py         forensics on the July 2026 detector disagreement
    step11_fpmech.py       corrected FP mechanism (seasonal cap scaling, r = +0.991)
    step12_julycause.py    month x matched-irradiance truth vs both detectors
    step13_persistence.py  gap-tolerant persistence: real-data cost/benefit
    step14_persist_bench.py  the same rule scored on injected ground truth
    step15_deficitcol.py   audit of the shipped continuous `deficit` column
    step16_controlcheck.py per-unit high-irradiance deficit for every unit
    step17_groundtruth.py  fleet topology: natural experiment, null re-examination
    step18_persist_decision.py  persistence rule search: the guard hypothesis, tested
    step19_docs_audit.py   downstream-artefact audit + the 9.0 kW nameplate check
    step20_sensitivity.py  recomputed threshold sensitivity; repaired the -inf values
    recommended.py         vat-v1 end to end -> saturation_flags_v2.csv
    bench_results.csv, bench_synthesis.csv, bench_robustness.csv,
    baseline_calibration.csv, energy_estimate.csv, july_forensic.csv,
    saturation_flags_v2.csv
  tests/
    run_tests.py           dependency-free runner: 42 tests, ~4 s, exit code 0 when green
    test_detectors.py      detector accuracy on injected ground truth
    test_exports.py        data-product contracts (the -inf and gap guards live here)
    test_topology.py       fleet mapping; eu_7's exclusion; the cross-inverter null guard
    _helpers.py            scenario accessors + known_failure marker
    conftest.py            pytest support, if pytest is ever installed
```

Run the suite with:

```
./venv/Scripts/python.exe sat_work/tests/run_tests.py          # all
./venv/Scripts/python.exe sat_work/tests/run_tests.py -v       # with timings
./venv/Scripts/python.exe sat_work/tests/run_tests.py -k export
```

Status at hand-off: **40 passed, 0 failed, 2 xfailed, 0 xpassed**. The two expected failures
are the documented defects that are real but not yet fixed — the published detector's
control-pair bias, and the winter mis-scaling of the published continuous column. Both are
properties of the **published** method: fixing them means regenerating the artefact with
`vat-v1`, which is a modelling decision rather than a patch. They flip to passes
automatically once that happens; the runner prints an XPASS notice telling you to remove the
marker. (The third xfail — the `-inf` values — was a pure defect with no modelling content,
and has been repaired; see §3.9.)
