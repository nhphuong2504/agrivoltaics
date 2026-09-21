# MPPT Saturation in the Area-1 Array — Method and Results

## Finding

Three of the twenty-one MPPT channels in the Area-1 array are **saturated**: the two units on
each channel together stop producing what the available irradiance would allow, and the
shortfall is lost energy. The constraint is at the tracker, not the inverter. Over the record
the three affected pairs show **1,799.2 sampled hours** of saturation.

The detector that establishes this is referred to throughout as **`vat-v1`**. It models each
pair's saturation-free output as a *time-varying slope* against an irradiance proxy, and calls
a deficit when the measured output falls below that model while irradiance is high. The method
is frozen: no parameter was tuned to any number reported here, and the detector's central
parameter turns out to reproduce the documented **9.0 kW** string rating to within a few
percent without having been fitted to it.

The finding is supported by five validations that share no failure mode — a normalisation-free
peer comparison and a per-unit score on real data, a natural experiment provided by the inverter
topology, a cross-check against a hardware specification, and a clip-injection benchmark with
labelled ground truth.

![MPPT saturation: the constraint is real, it is at the tracker, and the detector that finds
it is validated](sat_work/research/methods_fig4_evidence.png)

*The evidence chain. (a) Normalisation-free peer comparison on real data. (b) The
inverter-topology natural experiment. (c) The 9.0 kW nameplate cross-check. (d) ROC against
injected ground truth. Panels (a)–(c) establish that the problem is real and is located at the
tracker; panel (d) scores the detector that finds it.*

---

## 1. Method

### 1.1 Site, instrumentation and record

Twenty-four experimental units (`eu_1` … `eu_24`) log DC power at 5-minute resolution. The
record spans **2025-05-01 06:15 to 2026-08-31 21:10 — 488 days, 76,933 rows, 25 columns**. The
5-minute grid is modal on 99.2 % of consecutive steps, and all but one of the 488 calendar dates
carry data.

All figures reported in units of hours are **sampled hours**, i.e. `rows / 12`. They are hours
present in the record, not wall-clock hours elapsed. Where a denominator matters it is stated
with the number.

The units are wired to six inverters, and a single inverter hosts units on more than one
tracker. The relevant fact is which units **share** an MPPT channel, because units on one
channel share a tracker and therefore a common power ceiling. The mapping was supplied by the
site engineer and is recorded in `sat_work/research/satsim.py`:

| MPPT channel | units | inverter | role in this analysis |
|---|---|---|---|
| shared | `eu_8` + `eu_16` | 6 | saturating pair |
| shared | `eu_10` + `eu_18` | 2 | saturating pair |
| shared | `eu_13` + `eu_21` | 4 | saturating pair |
| independent | the other 18 units | 1–6 | peers, controls, reference input |

Each saturating pair is therefore co-located with **independent** units on its own inverter
(inverter 6 also hosts `eu_24`; inverter 2 hosts `eu_2`, `eu_3`, `eu_11`, `eu_19`; inverter 4
hosts `eu_5`, `eu_6`, `eu_14`, `eu_22`). §2.3 exploits this.

`eu_7` is excluded from every role. It was confirmed faulty by the site engineer, and its
mid-band slope is 38.6 W against ≈8,800 W for every other unit — a factor of ≈230. Its
exclusion is guarded by a behavioural test that corrupts its data and asserts the irradiance
proxy is bit-identical (`test_topology.py`).

### 1.2 Data-quality screening

Four conditions are evaluated per unit and per timestamp, and **any** of them makes that
timestamp ineligible for detection. A fifth screen, the activity mask, is evaluated per unit
per day and enters the decision domain rather than the data-quality screen:

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
built only from units that do not share a tracker with a saturating pair.

### 1.4 The saturation statistic

Measured output is the pair sum \(s(t) = P_a(t) + P_b(t)\). The model of the *saturation-free*
output of that pair is a **time-varying slope**:

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

Three design choices carry the method.

* **`θ` is estimated in `ref ∈ [0.35, 0.60]`.** A shared channel clips in the top of the
  range, so any slope fitted there is fitted to clipped data. The mid band is the highest
  range in which clipping provably cannot bind. It is retained *per day* rather than pooled,
  so a day without mid-band data does not silently borrow θ from elsewhere.
* **`θ` is a slope in watts, not a ceiling in watts.** Because `expected = θ·ref`, θ is the
  pair's potential output at `ref = 1`, which makes it directly comparable to a hardware
  rating (§2.4). A ceiling-parameterised model does not admit that comparison, because a
  ceiling is capped by whatever the hardware was doing at peak.
* **The slope is time-varying, and the model applies no single global scale factor.** The
  quantity \(s/\text{ref}\) moves through the year — with soiling, temperature and the seasonal
  resource — and a *rolling* estimate follows it. A single global factor cannot track a moving
  ratio, and whatever it fails to track reappears as a spurious deficit. Tracking the level
  locally is what keeps the model calibrated in every month (§2.2).

`d(t)` is the continuous severity: the fraction of the saturation-free output the pair failed
to deliver.

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

The first two conjuncts are physical: below `ref = 0.70` there is not enough resource for a
ceiling to bind, and `s > 0.6·expected` excludes the low-power rows where `θ·ref → 0` makes `d`
numerically meaningless.

The **decision domain** is the set over which every rate in §2 is computed, and it is the same
expression the benchmark's AUC is computed over. It contains **14,650–19,080 rows per pair, or
19–25 % of the record**. The `deficit` column of the released artefact is `NaN` outside it, so
an unfiltered `.mean()` over that column *is* the in-domain mean. Inside the domain the column
is bounded to `[-1, 1]` by construction.

### 1.6 Empirical null from independent pairs

The model can carry residual error at high irradiance, which would inflate `d` for a healthy
pair. Rather than assume that away, the null is **measured** on pairs that cannot saturate.
Over the three independent control pairs,

\[
q_n(\text{ref}) \;=\; \text{99.5th percentile of } d
\;\text{ pooled over ref-bins of width 0.0375}
\]

bins spanning `ref ∈ [0.30, 1.05]` (21 edges), computed per bin. The control pairs have no
shared tracker, so any non-zero `d` they show is model error by construction, and `q_n` is an
assumption-light empirical quantile of that error. It inherits whatever error the model has,
trading sensitivity for specificity rather than removing the error — which is why it defines an
*alternative calibration*, not the adopted one. It is scored as a competing detector in §2.5 but
is **not** exported, and §2.7 measures how small that error actually is.

### 1.7 Decision rule and persistence

Saturation is a sustained condition, so a transient threshold crossing must not raise a flag.
A pair is reported **saturated** when `d > 0.15` holds for **3 consecutive samples (15 min)**,
inside the gate. That is the only decision the method makes, and the only flag exported.

| column | kind | definition |
|---|---|---|
| `deficit` | score | `d`, bounded to `[-1, 1]`, NaN outside the decision domain |
| `sat` | flag | `d > 0.15` for 3 consecutive samples (15 min) |

Gap bridging fills a single missing sample between two qualifying runs, but only where the
sample lies inside the gate — a fill can never place a flag on a row the physical gate would
have excluded. It is adopted for a measured reason: it recovers **+4.5 pp recall** on injected
ground truth (0.305 → 0.350) and **+6.8 % of flagged hours** on real data, with control-pair
counts *unchanged* (§2.6). It is not free, and §2.10 states the cost.

**No severity tier is exported.** Severity is carried by the continuous `deficit` column, not by
additional flags: any stricter cut is a threshold on the score the artefact already publishes.
A tier is not only a threshold, though — it also has to re-decide persistence, and that is what
makes a second tier redundant rather than merely inconvenient. A bare `d > 0.30` cut over the
in-domain rows selects **3,829** samples, **1,423** of which the flag rejects as isolated spikes;
as a decision it is therefore *looser* than the reported flag unless it re-implements the full
run-length rule, at which point it is the same method with a different constant. Of the rows the
flag does set, `8.8–13.0 %` exceed a 30 % deficit, and the deepest deficit reached on any flagged
row is `0.40` on all three pairs — the shortfall saturates at about 40 % rather than approaching
the 100 % a total loss would give, which is the signature of a power-limited tracker rather than
a failed unit. One exported decision and one continuous score keep the decision, and its
denominator, unambiguous.

### 1.8 Validation design

Five validations are used, chosen so that no two share a failure mode. The first four are
independent of the detector and of each other.

1. **Normalisation-free peer test** (§2.1). A difference-in-differences on *raw* normalised
   power. Each unit is divided by its own 0.999 quantile; the pair sum is divided by a peer
   pair's sum; the ratio is rescaled to have median 1 in `ref ∈ [0.35, 0.60]`. Nothing but a
   ratio of raw measurements appears — no `θ`, no `ref` in the denominator, no fitted
   parameter. If the roll-off were an artefact of the irradiance proxy it could not survive
   this test. Run against three different peer references.
2. **Per-unit test** (§2.1). Each unit is scored *alone*:
   `deficit_u = 1 − P_u/(θ_u·ref)` at `ref ≥ 0.75`, with `θ_u` from that unit's own mid band.
   No pair sums and no borrowing of shared parameters. This tests the *topology assumption*
   that only the six units in shared pairs can saturate.
3. **Inverter-topology natural experiment** (§2.3). For each saturating pair the same statistic
   is computed on its inverter mates — independent units on the same inverter. If the constraint
   were imposed by the inverter (a shared AC limit, or a shared DC bus) the mates would roll off
   with the pair. If it is imposed by the tracker, they will not.
4. **Nameplate cross-check** (§2.4). `θ` is compared against the documented **9.0 kW** string
   rating. `θ` is estimated in `ref ∈ [0.35, 0.60]`, where clipping cannot bind, and **nothing
   in the method was ever fitted to a nameplate** — so agreement is a prediction, not a
   calibration.
5. **Clip-injection benchmark** (§2.5). Ground truth is manufactured on pairs that *cannot*
   saturate. An artificial ceiling is imposed by scaling output down on selected days, and the
   detector is scored on that pair and its neighbours. Six scenarios span the two
   failure-relevant axes:

   | scenario | pair | clipped days | ceiling | pattern |
   |---|---|---|---|---|
   | 1 | `eu_1+eu_3` | 35 % | 75th pct | chronic |
   | 2 | `eu_1+eu_3` | 35 % | 75th pct | episodic |
   | 3 | `eu_1+eu_3` | 100 % | 75th pct | chronic |
   | 4 | `eu_1+eu_3` | 100 % | none (control) | chronic |
   | 5 | `eu_4+eu_12` | 35 % | 75th pct | chronic |
   | 6 | `eu_15+eu_23` | 35 % | 75th pct | episodic |

   The injected pair is **held out of the `ref` construction**, so an injected pair never helps
   define the reference it is scored against. Two metrics are reported: threshold-free AUC on
   the decision domain (so the result cannot be a threshold artefact) and raw false-positive
   row counts on unclipped days (what a user actually experiences).

   The benchmark has a stated ceiling on severity: the clamp sits at 75 % of the pair's p95
   clear-sky peak, so injected severity caps at ≈0.25. The consequence is carried through to
   §2.10 rather than hidden.

Two further sweeps test whether the result depends on an arbitrary choice: a
**constraint-shape sweep** (hard/soft × constant/drifting) and a **threshold-sensitivity
sweep** (§2.8).

### 1.9 Reproducibility, version lock and regression tests

The method's output is the repo-root `saturation_flags.csv`: **76,933 × 9**, one row per
5-minute timestamp, carrying the shared irradiance proxy, the site-outage flag, and per pair
exactly two columns — the continuous `deficit` and the single binary `sat` flag.

* **Regeneration is byte-deterministic.** Two consecutive runs from `recommended.py` produce
  the same file, sha256 `16db6510f59fa3c10a57ee765a158e96836ff80000f17b41ea0aea422a1fc1cc`.
* The CSV is git-ignored, so reproducibility is locked by a **committed manifest**,
  `sat_work/canonical/CANONICAL_vat-v1.json`, recording the method parameters, every headline
  metric with its denominator, and the artefact's sha256. `step24_canonical_manifest.py --check`
  verifies it.
* The manifest reads its parameters **from the detector code**, not from a transcription.
  Silently re-tuning `REF_ON`, a threshold or the persistence rule therefore fails the test
  suite until the lock is regenerated deliberately.
* `sat_work/tests/test_manifest.py` re-derives every locked metric from the live file on each
  run, so the lock cannot become vacuous.
* Every headline number quoted in §2 comes from one module, `canon_metrics.py`, so the summary,
  the lock, the audit and the presentation cannot drift apart.

---

## 2. Results

### 2.1 Result 1 — the roll-off is real and is not an artefact of the reference

`ref` is derived from the same fleet whose saturation is being tested, so a roll-off in
`s/(θ·ref)` could in principle be produced by the normalisation rather than by the hardware.
This validation removes that possibility.

![Fig 1: difference-in-differences against a peer pair](sat_work/research/methods_fig1_rolloff.png)

*Fig 1. Median normalised pair-output ratio against a peer pair, in `ref` bins, rescaled to 1
in `ref ∈ [0.35, 0.60]`. No `θ`, no fitted parameter, and no `ref` in the denominator.*

Shared pairs sit at **1.000 against peers in the mid band** and fall across the whole
high-irradiance half of the range, reaching **0.572–0.594 in the topmost bin** with roughly
half of the total fall above `ref ≈ 0.88`, while control pairs are flat (**0.991–1.020**)
across the entire range. Repeating the test against three different peer references moves the
shared-pair figures only between **0.840 and 0.896**, so no single choice of peer drives the
result. Taking the median ratio directly over every sample with
`ref ≥ 0.70` (no binning, so no bin-width choice enters):

| pair | median ratio at `ref ≥ 0.7` | vs peers | n |
|---|---|---|---|
| `eu_8+eu_16` | 0.863 | **13.7 % below** | 15,247 |
| `eu_10+eu_18` | 0.851 | **14.9 % below** | 19,080 |
| `eu_13+eu_21` | 0.896 | **10.4 % below** | 14,650 |
| control `eu_1+eu_3` | 0.993 | 0.7 % below | 15,247 |
| control `eu_4+eu_12` | 1.007 | 0.7 % above | 19,080 |
| control `eu_15+eu_23` | 1.000 | 0.0 % | 14,650 |

**The per-unit test** confirms the same effect with a different statistic and no pair sums at
all. Scoring each of the 24 units individually at `ref ≥ 0.75`:

| group | high-irradiance deficit |
|---|---|
| the six units in shared pairs | **+0.152 to +0.184** |
| the other 18 units | **−0.005 to +0.010** |
| `eu_7` (faulty, excluded) | −0.232 |

The six highest-scoring units in the fleet are *exactly* the six inside shared pairs, and the
next-best unit is 15× lower. This validates the shared/independent labelling the whole analysis
depends on, and it does so using neither `ref`, nor any pair sum, nor any shared parameter.

### 2.2 Result 2 — the model is calibrated against a model-free measurement

The detector's deficit is a model output. The peer ratio of §2.1 is not: it is a ratio of raw
measurements with no fitted parameter. Computing both per irradiance bin lets the model be
checked against an independent measurement of the same physical quantity.

![Fig 2: the detector's deficit against a model-free peer-ratio truth](sat_work/research/methods_fig2_calibration.png)

*Fig 2. Panel (a): median deficit per `ref` bin, averaged over each class of pairs. Open markers
are the model-free peer-ratio truth; solid lines are the detector. Panel (b) draws the
disagreement at its own scale.*

Averaged over the gate (`ref ≥ 0.70`):

| class | detector | model-free truth | residual | largest single-bin gap |
|---|---|---|---|---|
| shared pairs | **+0.230** | +0.215 | **+0.0142** | 0.023 |
| control pairs | +0.005 | −0.000 | +0.0050 | 0.010 |

Two things follow. On the pairs that cannot clip the model reports essentially nothing
(+0.005 against a truth of −0.000), so the deficit it reports elsewhere is not a coupling
artefact. And on the pairs that do clip it tracks the independent measurement to within 0.014
on average, so the *magnitude* is right, not just the sign.

**There is no seasonal residual.** In the low-sun month (December, `ref ≥ 0.5`) the
control-pair median deficit is −0.005, −0.001 and +0.002; in June it is +0.005, +0.002 and
+0.007. Every value is inside ±0.007, so the model does not manufacture a deficit when the
resource is low — the failure mode a level-tracking slope is specifically chosen to avoid
(§1.4).

### 2.3 Result 3 — the constraint is at the tracker, not the inverter

Each saturating pair sits on an inverter that also hosts independent units. Computing the same
high-irradiance deficit for those mates isolates where the limit lives:

| inverter | saturating pair | pair deficit | inverter mates | mates' deficit | gap |
|---|---|---|---|---|---|
| 6 | `eu_8+eu_16` | +0.176 | `eu_24` | +0.000 | **+0.176** |
| 2 | `eu_10+eu_18` | +0.181 | `eu_2`, `eu_3`, `eu_11`, `eu_19` | −0.001 | **+0.182** |
| 4 | `eu_13+eu_21` | +0.160 | `eu_5`, `eu_6`, `eu_14`, `eu_22` | +0.001 | **+0.159** |

The clipped pairs roll off by 16–18 %; their inverter mates — which share AC cabling, the
inverter's DC bus and any grid-side limit — roll off by ~0 %. A shared AC or inverter-level
constraint would have to appear in both. **The constraint is at the MPPT channel**, which is
the mechanism the detector is designed to find, and this is the strongest causal evidence in
the analysis.

### 2.4 Result 4 — the detector's slope reproduces the hardware rating

`θ` is a slope in watts, so it is directly comparable to the documented string rating. Across
the 23 healthy units:

| statistic | value |
|---|---|
| median `θ` | **8,787 W** |
| median `θ` / nameplate | **0.976** |
| sd of the ratio | 0.017 |
| maximum deviation from 9.0 kW | **4.8 %** |

Nothing in the method was fitted to 9.0 kW: `θ` is estimated from mid-band irradiance, and the
nameplate entered the analysis only after the detector was frozen. The detector's central
parameter therefore *is* the hardware rating, recovered to within a few percent. This is the
result that most strongly indicates the model is physically right rather than numerically
convenient. Panel (c) of the summary figure shows the full distribution.

### 2.5 Result 5 — performance against injected ground truth

![Fig 3: candidate detectors scored on injected ground truth](sat_work/research/methods_fig3_scoreboard.png)

*Fig 3. Four candidate detectors on the clip-injection benchmark. The positives are injected
labels, not model output, so this scores the detector rather than asserting it.*

Threshold-free AUC on the decision domain. Denominators are stated because they change the
answer: the *episodic* column averages the two 35 %-duty scenarios (the realistic regime); the
*any clip* column averages the five scenarios containing positives. The `q = 1.0` clear-well
scenario has no positives by construction and is excluded from the mean rather than scored 0.

| candidate | episodic | any clip | mean FP rows |
|---|---|---|---|
| **θ(t)·ref  (adopted)** | **0.961** | **0.933** | **49** |
| clear-sky θ₉₀(t) | 0.957 | 0.930 | 70 |
| control-null calibrated | 0.942 | 0.887 | 52 |
| ceiling-pinned | 0.515 | 0.806 | 1,641 |

On the realistic episodic regime the adopted detector reaches **0.961** against a chance level
of 0.5, and it leads on every column. The margin over the clear-sky envelope is small — **0.961
against 0.957** — and the two should be read as equivalent on discrimination; what separates
them is the false-alarm count, **49 rows against 70**. The control-null variant, whose score is
a different quantity (excess over the control null rather than a shortfall against an
expectation), trails at 0.942.

Two negative results are recorded so they are not rediscovered, and one of them is a genuine
trade-off rather than a defeat.

A ref-gated 2-state HMM fails for a physical reason: cloud dips and clips are both "low
utilisation" states, so it cannot separate them (AUC ≈0.48).

A **ceiling-pinned** detector — flag when output sits on a plateau while the resource is high —
is the mirror image of the adopted one. On the benchmark's chronic scenarios it is near-perfect
(**AUC 1.000 and 0.999**, 8 and 5 false-positive rows) against the adopted detector's
0.889–1.000. On the two episodic scenarios it collapses to **0.519 and 0.512** with **3,795 and
4,305** false-positive rows. It is not a weak detector; it is a detector that requires the
constraint to be present
*every day* in order to find its plateau. Since the real array clips intermittently, and since
the episodic regime is where a real deployment lives, the adopted slope-based detector is the
one carried forward — but the trade-off is explicit, not hidden.

**False alarms stay bounded as the regime hardens.** Sweeping the fraction of days on which an
injected limit binds, on a pair that cannot saturate:

| clipped days | adopted detector, FP rows |
|---|---|
| 35 % (episodic) | 17 |
| 65 % | 31 |
| 100 % (chronic) | 40 |

Across a duty cycle from 35 % to 100 %, false-positive rows stay below forty.
The detector does not depend on the constraint being rare in order to be specific.

### 2.6 Saturation results

| pair | saturated hours | flagged days |
|---|---|---|
| `eu_8+eu_16` | 6,584 rows / **548.7 h** | 186 |
| `eu_10+eu_18` | 8,932 rows / **744.3 h** | 226 |
| `eu_13+eu_21` | 6,075 rows / **506.2 h** | 169 |
| **total** | 21,591 rows / **1,799.2 h** | 244 (union) |

`eu_10+eu_18` is the most affected channel on every measure: most hours and most affected
days.

**The seasonal structure is coherent.** Flags appear in every high-irradiance month of both
summers — May–September 2025 (26/23/28/24/9 days) and May–August 2026 (26/26/23/18) — plus a
secondary group of 41 days from February to April 2026. No flag is raised between October and
January, when the resource never reaches the level at which a shared channel binds, and none is
raised inside an acquisition gap or while a pair is off. The pattern is what a resource-driven
constraint should produce.

### 2.7 Residual model error, measured three ways

The detector's own model is the one quantity in the chain that could be biased, so its error is
measured on pairs that cannot saturate rather than assumed to be zero.

**Apparent lost energy.** Integrating `max(expected − s, 0)` over the gate domain:

| | apparent lost energy |
|---|---|
| shared pairs (real loss) | **9,249 kWh** |
| control pairs (cannot saturate) | **661 kWh** |
| control share of the shared figure | **7 %** |

Per pair the shared losses are 2,871 / 3,699 / 2,680 kWh. The same estimator applied to three
pairs on independent trackers attributes 661 kWh to them — 7 % of the claimed loss. That is the
residual error of the energy estimate, and it is a fair statement of its precision.

**Fleet-wide.** The check is not limited to the three legacy control pairs. Across **all 117
cross-inverter independent pairs** (no shared tracker, so none can saturate), the median phantom
loss is **141 kWh/pair**. A per-pair bias of that size, on a population of that size, is the
model's error floor rather than a property of particular pairs.

**Seasonal.** As reported in §2.2, the control-pair median deficit is inside ±0.007 in both
December and June, so the error does not grow when the resource is low.

### 2.8 Robustness

**To the assumed shape of the constraint.** Three pseudo-pairs × four shapes (hard/soft ×
constant/drifting), with the limit binding on 40 % of days, give AUC `> 10 %` of
**0.993–0.994** with **42–55** false-positive rows. The detector does not depend on a
hard-edged or stationary limit.

**To the deficit threshold.** Sweeping the cutoff:

| cutoff | hours (`eu_8+eu_16` / `eu_10+eu_18` / `eu_13+eu_21`) | relative drop per step |
|---|---|---|
| 0.10 | 753 / 979 / 715 | — |
| 0.15 (adopted) | 549 / 744 / 506 | 24–29 % |
| 0.20 | 344 / 485 / 297 | 35–41 % |
| 0.30 | 41 / 69 / 34 | 86–89 % |
| 0.40 | **0 / 0 / 0** | 100 % |

The response is monotone and smooth with no cliff, so the reported hours are not finely tuned
to the adopted 0.15 cutoff. Two features are stated rather than left implicit: the fall between
0.20 and 0.30 is steep (≈88 % of remaining hours vanish in that step), and there is no support
at all above `d = 0.40`, which bounds how deep the measured clipping gets.

### 2.9 Regression status

The frozen harness and the data-product contracts are covered by **53 tests: 52 passing, 1
expected failure**, running in ≈4 s. (The one `xfail` documents a known bias in an EDA baseline
retained in the harness as a comparator; it is not part of this method.) The tests are
behavioural rather than only snapshot comparisons: that `ref` is invariant to corrupting the
excluded unit, that flags are never set below the physical gate, that the released `deficit`
column is `NaN` outside the decision domain, that masking it changes **no** flag column
(asserted, not argued), and that the live artefact still satisfies the committed version lock.

### 2.10 Open questions

1. **`eu_15+eu_23` is unexplained.** It is the only pair on independent trackers to carry any
   flags at all (161 rows on 7 distinct days) despite both units being individually healthy, its
   flags are not marginal boundary cases (median flagged deficit 0.24 against a 0.15 threshold),
   and they co-occur with the shared pairs' flags. Recorded ground cover differs across units (`eu_15`
   grass/control, `eu_11` strawberry), which would make a single pair-level `θ` inadequate at
   high irradiance. **This needs site information**: which units share a cover type.
2. **The benchmark's injected severity caps at ≈0.25.** That ceiling is above the adopted 0.15
   threshold, so the benchmark *can* adjudicate the threshold itself; what it cannot do is
   characterise behaviour on the deeper clips the real record contains, which reach `0.40`.
   Performance on those rows rests on real-data evidence alone. Raising the injected ceiling is
   the highest-value improvement to the test setup.
3. **Gap bridging is a real precision/recall trade.** On injected ground truth it costs 130
   additional false-positive rows (166 → 296) for +4.5 pp recall (0.305 → 0.350). On real data
   it recovers +6.8 % of flagged hours with the control-pair counts bit-identical (0, 0 and
   161). The benchmark is the only place with ground truth and it
   is less favourable than the real null; both numbers are reported.
4. **The decision domain is narrow.** It covers 19–25 % of the record. The released `deficit`
   column is `NaN` elsewhere by design, so a consumer wanting a whole-day severity must compute
   one; the column will not silently supply it.

---

## 3. Reproduction

```
./venv/Scripts/python.exe sat_work/research/recommended.py                 # rewrite the artefact
./venv/Scripts/python.exe sat_work/research/step24_canonical_manifest.py   # re-lock the version
./venv/Scripts/python.exe sat_work/research/step24_canonical_manifest.py --check
./venv/Scripts/python.exe sat_work/research/step26_methods_figures.py      # the figures above
./venv/Scripts/python.exe sat_work/research/step21_canonical_summary.py    # every headline number
./venv/Scripts/python.exe sat_work/research/step23_metric_audit.py         # definitions + denominators
./venv/Scripts/python.exe sat_work/tests/run_tests.py                      # 53 tests
```

| result | produced by |
|---|---|
| §2.1 peer test | `step26_methods_figures.py`, `step2_identify.py` |
| §2.1 per-unit test | `step16_controlcheck.py` |
| §2.2 calibration, seasonal residual | `step26_methods_figures.py` |
| §2.3 natural experiment | `step17_groundtruth.py`, `step26_methods_figures.py` |
| §2.4 nameplate | `step19_docs_audit.py`, `step26_methods_figures.py` |
| §2.5 benchmark, duty sweep | `step26_methods_figures.py`, `step23_metric_audit.py` |
| §2.6 flagged hours, energy | `step21_canonical_summary.py` |
| §2.7 fleet-wide null | `step23_metric_audit.py` |
| §2.8 robustness, sensitivity | `step6_robustness.py`, `step20_sensitivity.py` |

Method parameters live in exactly two places: `sat_work/research/satsim.py` (the fleet topology
and the data preparation) and `sat_work/research/bench.py` (the frozen thresholds and
persistence constants). `canon_metrics.method()` reads them from there.

Figures are generated rather than committed (`*.png` is git-ignored), so a fresh clone shows
broken image links here until `step26_methods_figures.py` is run.
