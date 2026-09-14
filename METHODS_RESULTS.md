# MPPT Saturation in the Area-1 Array — Method and Results

Method and results for the detection of maximum-power-point-tracker (MPPT) saturation in the
Area-1 agrivoltaic array, and for the evaluation of the previously published detector that
this replaces.

The detector described in §1.4–§1.7 is referred to throughout as **`vat-v1`**. It is frozen:
nothing in this document was produced by tuning it to the numbers reported. Every figure,
table and range quoted in §2 is reproduced by the scripts named in §3 from the canonical
artefact, and the artefact is locked by a committed manifest (§1.10).

---

## 1. Method

### 1.1 Site, instrumentation and record

Twenty-four experimental units (`eu_1` … `eu_24`) log DC power at 5-minute resolution.
The record spans **2025-05-01 06:15 to 2026-07-25 21:10 — 450 days, 51,005 rows, 21 columns**.
The 5-minute grid is modal on 50,561 of 51,004 consecutive steps (99.1 %); the remainder are
acquisition gaps, which contribute no rows.

All figures reported in units of hours are **sampled hours**, i.e. `rows / 12`. They are hours
present in the record, not wall-clock hours elapsed. Where a denominator matters it is stated
with the number.

The units are wired to six inverters, and a single inverter hosts units on more than one
tracker. The relevant fact is which units **share** a channel, because units on one channel
share a tracker and therefore a common power ceiling. This mapping was supplied by the site
engineer and is recorded in `sat_work/research/satsim.py`:

| MPPT channel | units | inverter | role in this analysis |
|---|---|---|---|
| shared | `eu_8` + `eu_16` | 6 | saturating pair |
| shared | `eu_10` + `eu_18` | 2 | saturating pair |
| shared | `eu_13` + `eu_21` | 4 | saturating pair |
| independent | the other 18 units | 1–6 | peers, controls, reference input |

Each saturating pair is therefore co-located with **independent** units on its own inverter
(inverter 6 also hosts `eu_24`; inverter 2 hosts `eu_2`, `eu_3`, `eu_11`, `eu_19`; inverter 4
hosts `eu_5`, `eu_6`, `eu_14`, `eu_22`). §2.2 exploits this.

`eu_7` is excluded from every role. It was confirmed faulty by the site engineer, and its
mid-band slope is 38.6 W against ≈8,800 W for every other unit — a factor of ~230. Its
exclusion is guarded by a behavioural test that corrupts its data and asserts the irradiance
proxy is bit-identical (`test_topology.py`).

### 1.2 Data-quality screening

Five conditions are evaluated per unit and per timestamp, and **any** of them renders a
timestamp ineligible for detection:

| flag | criterion | rows in this record |
|---|---|---|
| `unit_off` | `power ≤ 1 W` while `ref > 0.4`, for ≥12 consecutive samples | — |
| `stale` | value identical to its predecessor and non-zero, for ≥12 samples, while `ref` is varying | **0** |
| `negative` | `power < 0` (sensor cross-talk; cancels within a pair) | **188** |
| `site_outage` | >80 % of reference units below 50 W during 09:00–15:00 | — |
| *activity mask* | day-normalised daily energy ≤ 0.2 of the unit's 0.95 quantile | — |

The activity mask is per unit per day, so a unit that is offline for a whole day is excluded
for that day without affecting its pair-mate's other days.

### 1.3 Irradiance proxy

No pyranometer is available, so the available-resource signal `ref` is built from the fleet:

\[
\text{ref}(t) \;=\; \operatorname{median}_{u \in \mathcal{R}}
\frac{P_u(t)}{\operatorname{q}_{0.999}\!\left(P_u\right)},
\qquad
\mathcal{R} = \{\text{11 units never offline at midday}\}
\]

Each unit is normalised by its own 0.999 quantile, so every member of \(\mathcal{R}\) enters
`ref` as a dimensionless fraction of its own peak. The median across 11 units suppresses a
single unit's fault or shading. `ref` is therefore a **fleet-median irradiance proxy**: it is
proportional to irradiance over the range where the reference units are unclipped, and it is
deliberately built only from units that do not share a tracker with a saturating pair.

### 1.4 The saturation statistic

Measured output is the pair sum \(s(t) = P_a(t) + P_b(t)\). The detector's model of the
*saturation-free* output of that pair is a **time-varying slope**:

\[
\theta(t) \;=\; \text{31-day centred rolling median of}\;
\operatorname{median}_{d}\!\left\{\frac{s}{\text{ref}}\right\}_{d},
\qquad \text{ref} \in [0.35,\ 0.60]
\]

\[
\text{expected}(t) \;=\; \theta(t)\cdot\text{ref}(t)
\qquad\qquad
d(t) \;=\; 1 - \frac{s(t)}{\theta(t)\cdot\text{ref}(t)}
\]

Two design choices carry the method:

* **`θ` is estimated in `ref ∈ [0.35, 0.60]`.** A shared channel clips in the top of the
  range, so any slope fitted there is fitted to clipped data. The mid band is the highest
  range in which clipping provably cannot bind, and it is retained per day rather than
  pooled, so that days without mid-band data do not silently borrow θ from elsewhere.
* **`θ` is a slope in watts, not a ceiling in watts.** Because `expected = θ·ref`, the
  quantity θ is the pair's potential output at `ref = 1`. That makes it directly comparable
  to a hardware rating (§2.3) — a comparison that no ceiling-parameterised model admits.

The 31-day centred window lets θ follow soiling and seasonal drift while remaining slow
relative to the weather. `d(t)` is the continuous severity: the fraction of the saturation-free
output that the pair failed to deliver.

### 1.5 Decision domain and gating

Detection is applied only where the question is answerable:

\[
\text{gate} \;=\; \big(\text{ref} \ge 0.70\big)\;\wedge\;
\big(s > 0.6\cdot\text{expected}\big)\;\wedge\;
\text{both units active}\;\wedge\;\neg\,\text{DQ}
\]

\[
\text{decision domain} \;=\; \text{both units active}\;\wedge\;
\big(\text{ref} \ge 0.70\big)\;\wedge\;\neg\,\text{DQ}
\]

The first two conjuncts of the gate are physical: below `ref = 0.70` there is not enough
resource for a ceiling to bind, and `s > 0.6·expected` excludes the low-power rows where
`θ·ref → 0` makes `d` numerically meaningless.

The **decision domain** is the set over which every rate in §2 is computed, and it is the same
expression the benchmark's AUC is computed over. It contains **8,160–11,635 rows per pair, or
16–23 % of the record**. The `deficit` column of the released artefact is `NaN` outside it, so
an unfiltered `.mean()` over that column *is* the in-domain mean. Inside the domain the column
is bounded to `[-1, 1]` by construction.

### 1.6 Empirical null from independent pairs

The detector's own model can be biased — a residual slope error at high irradiance would
inflate `d` for a healthy pair. Rather than assume that away, the null is **measured** on
pairs that cannot saturate. Over the three independent control pairs,

\[
q_n(\text{ref}) \;=\; \text{99.5th percentile of } d
\;\text{ pooled over ref-bins of width 0.0375}
\]

bins spanning `ref ∈ [0.30, 1.05]` (21 edges), computed per bin. The control pairs carry no
shared tracker, so any non-zero `d` they show is baseline error by construction, and `q_n` is
an assumption-light empirical quantile of that error. It inherits whatever bias the model has,
trading sensitivity for specificity rather than removing the bias — which is why it defines a
*conservative* tier, not the primary one.

### 1.7 Decision rules and persistence

Clipping is a sustained condition, so a transient threshold crossing must not raise a flag.
Three tiers are exported, all inside the gate:

| tier | rule | persistence | gap bridging |
|---|---|---|---|
| **moderate** | `d > 0.15` | 3 consecutive samples (15 min) | 1-sample gaps bridged **within the gate** |
| **severe** | `d > 0.30` | 6 consecutive samples (30 min) | none (strict) |
| **conservative** | `d > q_n(ref)` | 3 consecutive samples (15 min) | 1-sample gaps bridged within the gate |

Gap bridging fills a single missing sample between two qualifying runs, but only where the
sample lies inside the gate — a fill can never place a flag on a row that the physical gate
would have excluded. This is the one place the method departs from the published rules, and it
is adopted for a measured reason: it recovers **+5.0 pp recall** on injected ground truth
(0.349 → 0.399) and **+5.3 % of flagged hours** on real data, at **zero** additional flags on
the cross-inverter control pair. It is not free — see §2.6 and §2.9.

Bridging is deliberately **not** applied to the severe tier. The benchmark's injected severity
caps at ≈0.25 (§1.9), so it produces zero true positives for a `d > 0.30` tier and cannot
adjudicate the rule there; the analogous change would inflate that tier roughly 3× on the
strength of an untested analogy. Severe therefore stays strict until ground truth exists for
it.

### 1.8 The comparator: the previously published detector

The method being evaluated here — referred to as `D0` or *published* — differs from `vat-v1` in
one term. It models

\[
\text{cap}(t) \;=\; \text{31-day centred rolling median of the per-day 99th percentile of } s
\]

\[
\text{expected}(t) \;=\; g_0 \cdot \text{cap}(t) \cdot \text{ref}(t),
\qquad
g_0 \;=\; \operatorname{median}\left\{\frac{s}{\text{cap}\cdot\text{ref}}\right\}
\;\text{ over ref} \in [0.35, 0.60]
\]

with `deficit = 1 − s / expected` and **identical** gating and decision rules, except that
persistence is strict (no bridging). Everything downstream of `expected` is shared, which is
what makes the comparison a controlled one: the two detectors differ in exactly one quantity.

`D0` is retained in the frozen harness as `bench.det_published` and is reproducible from code.
It is the source of every "published" column in §2.

**A-`priori` mechanism.** `g₀·cap(t)` is itself a time-varying slope. The published method
allocates the seasonal swing to `cap(t)` and then removes only one *global* multiplicative
factor `g₀`. Because `g₀` is a single number, it cannot track a ratio that moves through the
year; the residual reappears as a spurious deficit. Measured on the never-clipped control pair
`eu_1+eu_3`: `g₀·cap(t)/θ(t)` has median 1.065, standard deviation 0.136, and drifts from 0.71
to 1.10 across the record. `θ(t)` tracks the level directly, so it has no such residual.

### 1.9 Validation design

Five validations are used, chosen so that no two share a failure mode. The first four are
independent of the detector and of each other:

1. **Normalisation-free peer test** (§2.1). A difference-in-differences on *raw* normalised
   power. Each unit is divided by its own 0.999 quantile; the pair sum is divided by a peer
   pair's sum; the ratio is rescaled to have median 1 in `ref ∈ [0.35, 0.60]`. Nothing but a
   ratio of raw measurements appears — no `cap`, no `g₀`, no `ref` in the denominator. If the
   roll-off is a normalisation artefact of `ref` or `cap`, it cannot survive this test. Run
   against three different peer references.
2. **Per-unit test** (§2.1). Each unit is scored *alone*:
   `deficit_u = 1 − P_u/(θ_u·ref)` at `ref ≥ 0.75`, with `θ_u` from that unit's own mid band.
   No pair sums, no shared-parameter borrowing. This tests the *topology assumption* that at
   most the six units in shared pairs can saturate.
3. **Inverter-topology natural experiment** (§2.2). For each saturating pair, the same
   statistic is computed on its inverter mates — independent units on the same inverter. If the
   constraint were imposed by the inverter (a shared AC limit, or a shared DC bus), the mates
   would roll off with the pair. If it is imposed by the tracker, they will not.
4. **Nameplate cross-check** (§2.3). `θ` is compared against the documented **9.0 kW** string
   rating. `θ` is estimated in `ref ∈ [0.35, 0.60]`, where clipping cannot bind, and **nothing
   in the method was ever fitted to a nameplate** — so agreement is a genuine prediction, not
   a calibration.
5. **Clip-injection benchmark** (§2.4). Ground truth is manufactured on pairs that *cannot*
   saturate. For a chosen pair, an artificial ceiling is imposed by scaling output down on
   selected days, and both detectors are scored on that pair and its neighbours. Six scenarios
   span the two failure-relevant axes:

   | scenario | pair | clipped days | ceiling | pattern |
   |---|---|---|---|---|
   | 1 | `eu_1+eu_3` | 35 % | 75th pct | chronic |
   | 2 | `eu_1+eu_3` | 35 % | 75th pct | episodic |
   | 3 | `eu_1+eu_3` | 100 % | 75th pct | chronic |
   | 4 | `eu_1+eu_3` | 100 % | none (control) | chronic |
   | 5 | `eu_4+eu_12` | 35 % | 75th pct | chronic |
   | 6 | `eu_15+eu_23` | 35 % | 75th pct | episodic |

   Crucially, the injected pair is **held out of the `ref` construction**, so an injected pair
   never helps define the reference it is scored against. Two metrics are reported: threshold-free
   AUC on the decision domain (so the result cannot be a threshold artefact) and raw
   false-positive row counts on unclipped days (which is what a user actually experiences).

   The benchmark has a stated ceiling on severity: the clamp sits at 75 % of the pair's
   p95 clear-sky peak, so injected severity caps at ≈0.25. Consequences are carried through to
   §2.9 rather than hidden.

Finally, a **threshold-sensitivity sweep** (§2.7) and a **constraint-shape sweep** (§2.7) test
whether the ranking is an artefact of one threshold value or one assumed clip shape.

### 1.10 Reproducibility, version lock and regression tests

The method's output is the repo-root `saturation_flags.csv`: **51,005 × 21**, one row per
5-minute timestamp, carrying every column of the previously published export plus, per pair,
`excess_vs_controls`, `null_d` and `sat_conservative`.

* **Regeneration is byte-deterministic.** Two consecutive runs from `recommended.py` produce
  the same file, sha256 `19e9f5467a5a72dd403a9ddf47d754dff07250106cecb01bcdb9f65623593203`.
* The CSV is git-ignored, so reproducibility is locked by a **committed manifest**,
  `sat_work/canonical/CANONICAL_vat-v1.json`, recording the method parameters, every headline
  metric with its denominator, and the artefact's sha256. `step24_canonical_manifest.py --check`
  verifies it.
* The manifest reads its parameters **from the detector code**, not from a transcription.
  Silently re-tuning `REF_ON`, a threshold or the persistence rule therefore fails the test
  suite until the lock is regenerated deliberately.
* `sat_work/tests/test_manifest.py` re-derives every locked metric from the live file on each
  run, so the lock cannot become vacuous.
* All headline numbers quoted in §2 come from one module, `canon_metrics.py`, so the summary,
  the lock, the audit and the presentation cannot drift apart.

---

## 2. Results

### 2.1 Result 1 — the roll-off is real and is not an artefact of the reference

The concern that motivates this validation is that `ref`, `cap` and `g₀` are all derived from
the same fleet, so a roll-off in `s/(cap·ref)` could in principle be produced by the
normalisation rather than by the hardware.

![Fig 1: difference-in-differences against a peer pair](sat_work/research/fig1_did.png)

*Fig 1. Median normalised pair-output ratio against a peer pair, in `ref` bins, rescaled to 1 in
`ref ∈ [0.35, 0.60]`. No ceiling, no `g₀`, and no `ref` in the denominator.*

Shared pairs sit at **1.000 against peers in the mid band** and fall monotonically across the
whole high-irradiance half of the range, reaching **0.54–0.61 in the topmost bins**, while
control pairs are flat (**0.988–1.000**) across the entire range. Three different peer
references give the same curve. Averaged over the gate:

| pair | median ratio at `ref ≥ 0.7` | shortfall vs peers |
|---|---|---|
| `eu_8+eu_16` | 0.767 | **23.3 %** |
| `eu_10+eu_18` | 0.783 | **21.7 %** |
| `eu_13+eu_21` | 0.795 | **20.5 %** |
| control `eu_4+eu_12` | 1.014 | −1.4 % |
| control `eu_15+eu_23` | 1.001 | −0.1 % |

A companion sanity check rules out the specific fear that `cap` absorbs the clip it is meant to
detect. If `cap` were pinned at the clipping level, the published expectation would be clipped
too and the deficit would vanish. It does not, because `g₀` rescales: the implied
unsaturated-to-observed peak ratio is **1.41–1.45**, i.e. the published model does see the
shortfall even though its ceiling sits at the clipped level.

**The per-unit test** confirms the same effect with a different statistic and no pair sums at
all. Scoring each of the 24 units individually at `ref ≥ 0.75`:

| group | high-irradiance deficit |
|---|---|
| the six units in shared pairs | **+0.169 to +0.200** |
| the other 18 units | **−0.005 to +0.011** |
| `eu_7` (faulty, excluded) | −0.227, with `θ_mid` = 38.6 W |

The six highest-scoring units in the fleet are *exactly* the six inside shared pairs, and the
next-best unit is 15× lower. This validates the shared/independent labelling on which the whole
analysis depends, and it does so using neither `ref`, nor `cap`, nor any pair sum.

### 2.2 Result 2 — the constraint is at the tracker, not the inverter

Each saturating pair sits on an inverter that also hosts independent units. Computing the same
high-irradiance deficit for those mates isolates where the limit lives:

| inverter | saturating pair | pair deficit | inverter mates | mates' deficit | gap |
|---|---|---|---|---|---|
| 6 | `eu_8+eu_16` | +0.195 | `eu_24` | −0.001 | **+0.196** |
| 2 | `eu_10+eu_18` | +0.196 | `eu_2`, `eu_3`, `eu_11`, `eu_19` | +0.002 | **+0.195** |
| 4 | `eu_13+eu_21` | +0.178 | `eu_5`, `eu_6`, `eu_14`, `eu_22` | +0.002 | **+0.175** |

The clipped pairs roll off by 18–20 %; their inverter mates — which share AC cabling, the
inverter's DC bus and any grid-side limit — roll off by ~0 %. A shared AC or inverter-level
constraint would have to appear in both. **This locates the constraint at the MPPT channel**,
which is the mechanism the detector is designed to find, and it is the strongest causal
evidence in this analysis.

### 2.3 Result 3 — the detector's slope reproduces the hardware rating

`θ` is a slope in watts, so it is directly comparable to the documented string rating. Across
the 23 healthy units:

| statistic | value |
|---|---|
| median `θ` | **8,847 W** |
| median `θ` / nameplate | **0.983** |
| sd of the ratio | 0.017 |
| maximum deviation from 9.0 kW | **4.1 %** |

Nothing in the method was fitted to 9.0 kW: `θ` is estimated from mid-band irradiance, and the
nameplate entered the analysis only after the detector was frozen. The detector's central
parameter therefore *is* the hardware rating, recovered to within a few percent. This is the
single result that most strongly indicates the model is physically right rather than
numerically convenient.

![The validated evidence: (a) peer roll-off, (b) the inverter-mates natural experiment, (c) the nameplate round-trip, (d) ground-truth ROC](sat_work/research/validated_evidence.png)

*The four validations summarised: the first three establish that the problem is real and is
located at the tracker; the fourth scores the detector that finds it. Panel (c) plots the
percentage deviation from nameplate, so the 4.1 % envelope is visible rather than flattened by
the axis.*

### 2.4 Result 4 — the defect in the published baseline, and its two consequences

Because `g₀` cannot track a ratio that moves seasonally (§1.8), the published expectation
collapses in winter. On control pairs, which **cannot** saturate, the published deficit in
December is:

| December, `ref ≥ 0.5` (n = 285) | published | `vat-v1` |
|---|---|---|
| `eu_1+eu_3` | −0.44 | ~0.00 |
| `eu_4+eu_12` | −0.48 | ~0.00 |
| `eu_15+eu_23` | −0.44 | ~0.00 |

The published method reports an intact pair as roughly **45 % short** for a full month. Two
consequences follow, and both are measurable.

**Consequence A — specificity collapses for intermittently clipped pairs.** A clip-injection
benchmark on a pair that cannot saturate, sweeping the fraction of days on which the limit
binds:

| clipped days | published FP rows | `vat-v1` FP rows |
|---|---|---|
| 35 % of days (episodic) | 4,922 | **6** |
| 65 % of days | 1,285 | 17 |
| 100 % of days (chronic) | 12 | 25 |

The failure is *regime-dependent*, and in the direction that matters: the published detector is
adequate when the limit binds every day, because `cap(t)` then adapts to the clipped level —
exactly the regime its original control-pair validation happened to sit in. It degrades by more
than two orders of magnitude when clipping is intermittent, which is what real weather does.

Pooled over the two episodic scenarios, the false-positive count on unclipped days falls from
**10,202 to 117 rows — an 87× reduction**. Per scenario the ratios are 820× and 48×, so the
pooled figure understates the spread and row counts are quoted alongside it.

**Consequence B — the published energy-loss estimate is roughly half bias.** Integrating
`max(expected − s, 0)` over the gate domain:

| | published | `vat-v1` |
|---|---|---|
| shared pairs (real loss) | 7,131 kWh | **6,120 kWh** |
| control pairs (cannot saturate ⇒ pure bias) | 3,356 kWh | **409 kWh** |
| bias as a share of the shared figure | **47 %** | **7 %** |

The published estimator attributes almost half as much "lost energy" to pairs that *cannot lose
energy* as it does to the pairs that do. The bias is not a property of the three legacy control
pairs: across **all 117 cross-inverter independent pairs**, the published method fabricates a
median **803 kWh/pair** of phantom loss, against **92 kWh/pair** for `vat-v1` — 11 %.

### 2.5 Result 5 — discrimination against injected ground truth

![Fig 3: detector scoreboard](sat_work/research/fig3_scoreboard.png)

Threshold-free AUC on the decision domain, aggregated over the frozen harness. Denominators
are stated because they change the answer: the *episodic* column averages the two 35 %-duty
scenarios (the realistic regime); the other columns average the five scenarios containing
positives. The `q = 1.0` clear-well scenario has no positives by construction and is excluded
from the mean rather than scored 0.

| detector | episodic | any clip | `> 10 %` | `> 20 %` | mean FP rows |
|---|---|---|---|---|---|
| **D1 rolling-slope** (= `vat-v1`) | **0.962** | 0.934 | **0.994** | **0.999** | **30** |
| D2 envelope-q90 | 0.957 | 0.932 | 0.993 | 0.997 | 45 |
| D5 control-calibrated | 0.957 | **0.936** | 0.974 | 0.944 | 53 |
| D0 published | 0.780 | 0.861 | 0.934 | 0.979 | 1,714 |
| D4 ceiling-pinned | 0.493 | 0.797 | 0.556 | 0.584 | 785 |

On the episodic regime, discrimination rises from **0.780 to 0.962** — from marginally better
than chance-plus-trend to near-separable.

**Honest reading of the table.** D1 leads on the episodic regime, on deep clips, and on false
positives. D5 edges it on the pooled "any-clip" column, where the chronic scenario is mixed in
and where D5's score is a different quantity (excess over the control null, not a shortfall).
D2 and D5 are close peers, not losers. The defensible claim is that **D1 leads where it
matters**, not that it dominates everywhere.

Two negative results are worth recording so they are not rediscovered. A ref-gated 2-state HMM
(D3) failed for a physical reason: cloud dips and clips are both "low utilisation" states, so
it cannot separate them. And a ceiling-pinned detector (D4) is excellent at a stable limit
(AUC 0.999) and unusable at an intermittent one (0.49) — the same fragility as the published
method, arrived at from the opposite direction.

### 2.6 Corrected saturation results

Replacing the baseline changes the magnitudes but not the structure.

| pair | published | **`vat-v1`** | published severe | **`vat-v1` severe** | flagged days |
|---|---|---|---|---|---|
| `eu_8+eu_16` | 5,680 rows / 473.3 h | **4,639 / 386.6 h** | 30.7 h | **7.1 h** | 118 → 115 |
| `eu_10+eu_18` | 7,111 rows / 592.6 h | **6,478 / 539.8 h** | 27.8 h | **13.0 h** | 149 → 147 |
| `eu_13+eu_21` | 4,728 rows / 394.0 h | **4,113 / 342.8 h** | 20.2 h | **7.2 h** | 106 → 104 |
| **total** | 17,519 / 1,459.9 h | **15,230 / 1,269.2 h** (−13.1 %) | 78.8 h | **27.3 h** (2.88×) | — |

* **The moderate tier shrinks 9–18 %** (−18.3 %, −8.9 %, −13.0 %) *after* gap bridging has
  already recovered +5.3 % of hours. Without bridging the shrink is 13–22 %.
* **The severe tier shrinks 2.88×**, from 78.8 h to 27.3 h. It is unchanged by bridging, which
  is not applied there.
* **Every published conclusion survives.** The affected days are almost a subset of the
  published ones (0–2 new days, 3–4 published-only), and the seasonal shape is unchanged:
  May–July peak, the cold-February clear-day peak, zero in acquisition gaps and pair-off blocks.
  Only the magnitudes were inflated.

### 2.7 Robustness

**To the assumed shape of the constraint.** Three pseudo-pairs × four constraint shapes, with
the limit binding on 40 % of days:

| case | D1 AUC `> 10 %` | D1 FP | D0 AUC `> 10 %` | D0 FP |
|---|---|---|---|---|
| hard, constant | 0.995 | 26 | 0.866 | 1,335 |
| hard, drifting | 0.993 | 42 | 0.946 | 251 |
| soft, constant | 0.996 | 6 | 0.867 | 1,309 |
| soft, drifting | 0.996 | 6 | 0.941 | 220 |

The ranking is stable across hard/soft and constant/drifting limits; the published detector's
*rate* of false positives varies by 6× across these cases while D1's stays within a factor of
7 of an already-small number.

**To the deficit threshold.** Sweeping the cutoff:

| cutoff | published hours (three pairs) | `vat-v1` hours | relative drop |
|---|---|---|---|
| 0.10 | 557 / 714 / 508 | 499 / 677 / 463 | — |
| 0.15 | 473 / 593 / 394 | 387 / 540 / 343 | 22–26 % |
| 0.20 | 342 / 410 / 253 | 254 / 362 / 209 | 33–39 % |
| 0.30 | 61 / 58 / 34 | 31 / 50 / 24 | 86–89 % |
| 0.40 | 6 / 2 / 3 | **0 / 0 / 0** | 100 % |

The response is monotone and smooth with no cliff, so the result is not finely tuned to the
0.15 cutoff. Two features are worth stating rather than leaving implicit: the fall between 0.20
and 0.30 is steep (≈88 % of remaining hours vanish in that step, in both detectors), and
`vat-v1` has no support at all above `d = 0.40`.

### 2.8 Regression status

The frozen harness and the data-product contracts are covered by **51 tests: 50 passing, 1
expected failure** (the published method's control-pair bias — real, and unfixed by design).
Runtime ≈4 s. The tests include behavioural guards rather than only snapshot comparisons: that
`ref` is invariant to corrupting the excluded unit, that flags are never set below the physical
gate, that the released `deficit` column is `NaN` outside the decision domain, that masking it
changes **no** flag column (asserted, not argued), and that the live artefact still satisfies
the committed version lock.

### 2.9 Open questions

1. **`eu_15+eu_23` is unexplained.** It is the most-flagged flat pair (108 rows) despite both
   units being individually healthy, its flags are not marginal boundary cases, and they
   co-occur with the shared pairs' flags. Published ground cover differs across units (`eu_15`
   grass/control, `eu_11` strawberry), which would make a single pair-level `θ` inadequate at
   high irradiance. **This needs site information**: which units share a cover type.
2. **The severe tier cannot be benchmarked.** Injected severity caps at ≈0.25, so `d > 0.30`
   scores zero true positives and no threshold or persistence choice can be adjudicated there.
   The severe tier rests on real-data bias floors alone. Extending the benchmark to deeper
   clips is the highest-value improvement to the test setup.
3. **The benchmark and real data disagree on gap bridging.** On injected ground truth,
   bridging costs 72 additional false-positive rows (108 → 180) for +5.0 pp recall
   (0.349 → 0.399). On real data it recovers +5.3 % of flagged hours with the control-pair
   counts *bit-identical*: 0 rows cross-inverter and 108 rows on the same-inverter pairs, the
   same as the strict rule. The benchmark is the only place with ground truth, and it
   disagrees with the real-data null; both numbers should be quoted.
4. **The decision domain is narrow.** It covers 16–23 % of the record. The released `deficit`
   column is `NaN` elsewhere by design, so a consumer wanting a whole-day severity must compute
   one; the column will not silently supply it.

---

## 3. Reproduction

```
./venv/Scripts/python.exe sat_work/research/recommended.py                 # rewrite the artefact
./venv/Scripts/python.exe sat_work/research/step24_canonical_manifest.py   # re-lock the version
./venv/Scripts/python.exe sat_work/research/step24_canonical_manifest.py --check
./venv/Scripts/python.exe sat_work/research/step21_canonical_summary.py    # every headline number
./venv/Scripts/python.exe sat_work/research/step23_metric_audit.py         # definitions + denominators
./venv/Scripts/python.exe sat_work/tests/run_tests.py                      # 51 tests
```

| result | produced by |
|---|---|
| §2.1 peer test, energy shortfall | `step2_identify.py` |
| §2.1 per-unit test | `step16_controlcheck.py` |
| §2.2 natural experiment | `step17_groundtruth.py`, `step25_validated_evidence.py` |
| §2.3 nameplate | `step19_docs_audit.py`, `step25_validated_evidence.py` |
| §2.4 duty-cycle FPs, baseline bias | `step23_metric_audit.py`, `step11_fpmech.py` |
| §2.5 AUC scoreboard | `step7_figures.py` (fig 3), `step23_metric_audit.py` |
| §2.6 flagged hours | `step21_canonical_summary.py` |
| §2.7 robustness, sensitivity | `step6_robustness.py`, `step20_sensitivity.py` |

Method parameters live in exactly two places: `sat_work/research/satsim.py` (the published
comparator and the fleet topology) and `sat_work/research/bench.py` (the frozen thresholds and
persistence constants). `canon_metrics.method()` reads them from there.
