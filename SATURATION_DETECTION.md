# MPPT Saturation Detection — Implementation Document

**Project:** Solar Area 1 — shared-MPPT saturation analysis
**Dataset:** `combined_may2025_jul2026_area1.csv` (only input used)
**Implementation:** `saturation_detection.ipynb` (fully executed, self-contained)
**Outputs:** `saturation_flags.csv`, `data_quality_events.csv`
**Author of method:** automated analysis, 2026-08-27

---

> **Superseded as the spec for `saturation_flags.csv` (2026-09).**
>
> This document specifies the **published** method, `expected = g0 · cap(t) · ref`. It
> remains accurate as the record of that method and of `saturation_detection.ipynb`, and it
> is kept deliberately: it is the reproducible spec for the previous approach, which is
> still available as `bench.det_published()`.
>
> It is no longer the spec for the shipped flags file. `saturation_flags.csv` is now the
> canonical **vat-v1** artefact, written by `sat_work/research/recommended.py`, with
> `expected = theta(t) · ref`, and it is locked by
> `sat_work/canonical/CANONICAL_vat-v1.json`.
>
> - Corrected method and results: **`SATURATION_REVIEW.md`** §5.
> - The exact claims here that change: §7.2 of the review. In summary, the §7.4 counts, the
>   §7.1 interpretation of the control-pair flags, and the §8.1 deficit-column `-inf` note.
> - Two of those are now historical in a second sense: the canonical file has no `-inf` at
>   all, and its `deficit` column is additionally **masked to NaN outside the decision
>   domain** (`active ∧ ref ≥ 0.70 ∧ no DQ`), so the §8.1 caveat about aggregating the raw
>   column does not apply to the canonical file. It still applies here, to the published
>   column this document describes.
> - To interpret the current `saturation_flags.csv`, use the review. To reproduce or audit
>   the published outputs, this document is still the right one.

---

## 1. Background and Objective

The Area 1 array consists of 24 experimental units (`eu_1` … `eu_24`) logging power
\[W\] at 5-minute resolution. Most units are connected to **independent MPPTs**, but
three pairs share a single MPPT:

| Shared pair | Units |
|---|---|
| Pair 1 | `eu_8` + `eu_16` |
| Pair 2 | `eu_10` + `eu_18` |
| Pair 3 | `eu_13` + `eu_21` |

Shared-MPPT pairs occasionally exhibit **saturation**: their combined power output is
constrained below what the available irradiance would allow. The constraint has **no
known fixed power threshold** — the effective MPPT limit drifts with operating
conditions (irradiance, temperature, combined loading) — and the events follow no
obvious temporal pattern.

**Objective.** Learn the characteristic saturation pattern from the historical data
itself, and build an automatic detector that flags affected timestamps across the whole
dataset — while keeping true saturation strictly separated from technical faults
(inverter failures, outages, communication issues, sensor artifacts).

**Explicit non-goal:** no predefined power threshold is applied anywhere; every limit in
the method is estimated from the data and adapts over time.

---

## 2. Data Overview

| Property | Value |
|---|---|
| Rows | 51,005 (5-minute steps) |
| Span | 2025-05-01 06:15 → 2026-07-25 21:10 (≈ 15 months) |
| Days with data | 328 |
| Full 5-min grid for the span | 129,780 rows (i.e. night hours mostly absent) |
| Acquisition gaps > 1 h | 331 |
| Major gaps | ≈ Aug–Sep 2025 and Mar–Apr 2026 (each ≈ 61 days) |

First-pass findings that shaped the design:

1. **`eu_7` is on a different scale** (max ≈ 98 W vs ≈ 9–10 kW for all others) and only
   moderately correlates with the rest (r ≈ 0.63). It is treated as an odd/reference
   unit and **excluded** from the irradiance proxy and from pair logic.
2. **Negative values occur only in the six shared-MPPT units**
   (`eu_8`: 83, `eu_16`: 51, `eu_10`: 12, `eu_18`: 9, `eu_13`: 7, `eu_21`: 26 rows).
   Within a pair they are equal-and-opposite (e.g. `eu_8 = −15 W`, `eu_16 = +15 W`), so
   the **pair sum is unaffected** — a sensor cross-talk artifact, not physics, and
   definitely not saturation.
3. Several units have long offline blocks (`eu_2`, `eu_5/6/13/14/21/22` — identical
   counts point to a common inverter block, `eu_8/16`, `eu_24`). These are faults,
   handled by screening (§3), and they require per-day activity masking.
4. No duplicated timestamps, no NaNs, no stale/frozen values found.

---

## 3. Data-Quality Screening

Implemented in notebook §2. Four per-timestamp flags plus a per-day activity mask.
**No timestamp carrying a fault flag is ever eligible for a saturation label.**

| Flag | Rule | Interpretation |
|---|---|---|
| `site_outage` | > 80 % of reliable units < 50 W during 09:00–15:00 | plant outage / acquisition down (486 rows) |
| `unit_off[eu]` | unit ≤ 1 W while `ref` > 0.4, sustained ≥ 12 samples (1 h) | inverter/MPPT offline (58,430 rows total) |
| `stale[eu]` | identical non-zero value ≥ 12 samples while `ref` varies | communication freeze (**0 found**) |
| `negative[eu]` | value < 0 | sensor cross-talk artifact (188 rows, shared units only) |

**Activity mask** (per unit, per day): a unit-day is *active* if
`daily energy > 20 % × p95(daily energy of that unit)`. Saturation is only evaluated on
days where both units of a pair are active.

Fault episodes are condensed into `data_quality_events.csv`
(columns: `type, unit, start, end, duration_min`; 1,321 events) so bad-data periods can
be audited or repaired independently of the saturation analysis.

---

## 4. Reference Signal (irradiance proxy)

The method needs a per-timestamp measure of *available* solar resource that is
independent of the shared-MPPT units.

```
ref(t) = median over the 11 reliable units of  eu_c(t) / p99.9(eu_c)
RELIABLE = eu_1, eu_3, eu_4, eu_9, eu_11, eu_12, eu_15, eu_17, eu_19, eu_20, eu_23
```

Selection criterion for `RELIABLE`: independent-MPPT units that are essentially never
offline at midday (2–12 midday-zero rows each vs thousands for the unreliable ones).
The median makes `ref` robust to any single unit failing.

**Validation (notebook §3):**
- correlation with every well-behaved unit: 0.982–0.997;
- normalized gain of independent units vs `ref` is **flat at 1.0 across all irradiance
  bins** — i.e. independent units show *no* droop. Any gain droop in the shared pairs
  is therefore pair-specific (MPPT limiting), not site-wide physics such as temperature.

---

## 5. The Saturation Signature

The MPPT constrains the **pair sum** (the quantity it actually sees), so all pair
analysis uses `s(t) = eu_a(t) + eu_b(t)`.

### 5.1 Adaptive ceiling `cap(t)`

For each pair: per-day 99th percentile of `s` over active rows with `ref > 0.3`,
smoothed by a centered 31-day rolling median (`min_periods = 5`). Result: a slowly
varying ceiling of ≈ 10.2–13.5 kW depending on pair and season — confirming that **no
single fixed threshold exists**.

### 5.2 Gain and the droop diagnostic

```
gain(t) = s(t) / ( cap(t) · ref(t) )
g0      = median gain over active rows with ref ∈ [0.35, 0.60]   (unsaturated regime)
```

| Pair | g0 | Behavior at ref → 1.0 |
|---|---|---|
| eu_8+eu_16 | 1.45 | gain droops to ≈ 0.78 |
| eu_10+eu_18 | 1.41 | gain droops to ≈ 0.73 |
| eu_13+eu_21 | 1.41 | gain droops to ≈ 0.81 |

Gain is flat ≈ `g0` up to `ref ≈ 0.6`, then rolls off progressively — a **soft knee**,
the fingerprint of a current-limited shared MPPT. Independent units stay flat (§4), so
the droop is the learned, discriminating pattern.

### 5.3 Expected-output model

```
expected(t) = g0 · cap(t) · ref(t)
deficit(t)  = 1 − s(t) / expected(t)
```

`deficit` = fractional shortfall vs the unsaturated expectation; `deficit ≈ 0` means
healthy, `deficit = 0.3` means 30 % constrained. The model is *adaptive* in two places
(`cap(t)` rolling; `g0` from mid-irradiance behavior), so seasonal drift, soiling and
slow degradation do not produce false saturation calls.

---

## 6. Detection Algorithm

Per pair, per timestamp — a row is flagged when **all** of the following hold:

```
1. ref(t)  ≥ REF_ON            = 0.70   (test only at high irradiance)
2. deficit(t) > DEFICIT        = 0.15 moderate / 0.30 severe
3. s(t) > NEAR_CAP · cap(t)    = 0.60   (near the ceiling → not a fault/shading)
4. both units active (activity mask, §3)
5. no DQ flags on eu_a, eu_b, and no site_outage
6. persistence: condition held for ≥ PERSIST consecutive samples
                = 3 (15 min) moderate / 6 (30 min) severe
```

All thresholds live in a single parameter dict in notebook §5:

```python
P = dict(REF_ON=0.70, DEF_MOD=0.15, DEF_SEV=0.30,
         NEAR_CAP=0.60, PERSIST_MOD=3, PERSIST_SEV=6)
```

Rationale for the two "anti-fault" guards: condition 3 rejects pair-off events
(`s ≈ 0`) and heavy local shading (small `s`); conditions 4–5 remove every timestamp
already explained by a technical fault. Persistence (6) removes single-sample cloud-edge
noise.

Two severity tiers are emitted: **moderate** (deficit > 15 %, ≥ 15 min) and **severe**
(deficit > 30 %, ≥ 30 min); severe is always a subset of moderate.

---

## 7. Validation

### 7.1 Control experiment (specificity)

> **The interpretation below is falsified** (see `SATURATION_REVIEW.md` §7.2 and the
> fleet-wide null in §3.12). The arithmetic is right; the reading of the 122 rows is not.
> Those flags are **baseline bias**, not real constraint events: the published expectation
> fabricates a median 803 kWh/pair of phantom loss across 117 independent pairs that cannot
> saturate, against 92 kWh/pair for vat-v1. Note also that only `eu_1+eu_3` crosses
> inverters — `eu_4+eu_12` and `eu_15+eu_23` are same-inverter pairs and carry a
> co-movement common mode worth roughly 2× tail flags, so they are magnitude nulls rather
> than flag-level nulls. Left in place as the record of the published claim.

The identical detector was run on three **independent pseudo-pairs**
(`eu_1+eu_3`, `eu_4+eu_12`, `eu_15+eu_23`), which cannot saturate from MPPT sharing:

| Group | Flagged rows (moderate) |
|---|---|
| Control pairs (3, total) | 122 |
| Shared pairs (3, total) | 17,519 |

**144× separation.** Inspection of the rare control flags (e.g. `eu_15+eu_23` on hot
June days) shows they land on genuine few-percent midday sags — consistent with mild
inverter AC-limit clipping, i.e. small *real* constraint events rather than detector
noise. The detector answers "constrained vs peers"; on shared MPPTs that state is
chronic and severe, elsewhere it is rare and mild.

### 7.2 Visual audit

Per pair, the notebook plots the two most saturated days, one marginal and one clean
high-irradiance day: flags land exactly on flat-top clipping plateaus and rolloff
shoulders (incl. subtle cold-day winter clipping, e.g. 2026-02-22), and clean days get
no flags. A scatter of flags in the (ref, deficit) plane shows the flags concentrated
precisely in the high-irradiance / high-deficit corner, as designed.

### 7.3 Sensitivity to the deficit threshold

Flagged **hours** per pair vs threshold (moderate persistence):

| deficit > | eu_8+eu_16 | eu_10+eu_18 | eu_13+eu_21 |
|---|---|---|---|
| 0.10 | 557 | 714 | 508 |
| 0.15 | 473 | 593 | 394 |
| 0.20 | 342 | 410 | 253 |
| 0.30 | 61 | 58 | 34 |
| 0.40 | 6 | 2 | 3 |

Monotonic, smooth, no cliff → the method does not hinge on a finely tuned cutoff.

### 7.4 Results summary (moderate tier)

| Pair | Flagged rows | Hours | Days affected |
|---|---|---|---|
| eu_8+eu_16 | 5,680 | 473 | 118 |
| eu_10+eu_18 | 7,111 | 593 | 149 |
| eu_13+eu_21 | 4,728 | 394 | 106 |

Temporal pattern: saturation concentrates in May–Jul (strong sun) with a smaller peak
on clear cold February days; zero during acquisition gaps and during pair-off fault
blocks (correctly masked). Severe events: 368 / 334 / 243 rows respectively.

---

## 8. Output Files

### 8.1 `saturation_flags.csv` — 51,005 × 12, aligned 1:1 with the source timestamps

| Column | Type | Meaning |
|---|---|---|
| `time` | timestamp | row key (join key to the source CSV) |
| `ref` | float | irradiance proxy, 0 … ≈ 1.05 |
| `dq_site_outage` | 0/1 | site outage flag (saturation logic suspended) |
| `<pair>_deficit` | float | continuous severity: `1 − s/expected` per pair |
| `<pair>_sat_moderate` | 0/1 | moderate saturation flag |
| `<pair>_sat_severe` | 0/1 | severe saturation flag (⊂ moderate) |

`<pair>` ∈ `eu_8_eu_16`, `eu_10_eu_18`, `eu_13_eu_21`.

Caveats: `deficit` is empty near the two long gaps (rolling ceiling unavailable) and
`-inf` at `ref = 0` (division artifact; flags can never fire there — filter
`ref ≥ 0.7` before analyzing deficits). A `0` flag means "no saturation detected",
which includes "not testable" (night, gaps, faults) — use `ref`, `dq_site_outage` and
the events log to distinguish.

> **Both caveats are historical for the canonical file.** The canonical
> `saturation_flags.csv` is generated by vat-v1, which cannot produce `-inf`, and its
> `deficit` column is masked to `NaN` outside the decision domain
> (`active ∧ ref ≥ 0.70 ∧ no DQ`), so the "filter before analyzing" step is no longer
> required — an unfiltered `.mean()` is the in-domain mean. This paragraph still
> describes the published column, which is what the rest of this document specifies.
> Canonical schema is 51,005 × 21, not × 12: vat-v1 adds `null_d`,
> `excess_vs_controls` and `sat_conservative` per pair.

### 8.2 `data_quality_events.csv` — 1,321 events

`type` ∈ {`site_outage`, `unit_off`, `negative_value`}, `unit`, `start`, `end`,
`duration_min`. Long `unit_off` blocks document the major fault/repair periods of the
plant (e.g. the common `eu_5/6/13/14/21/22` block, `eu_8/16` mid-2025 and early 2026,
`eu_24` summer 2026, `eu_2` most of the period).

---

## 9. Reproduction

1. Open `all_data/saturation_detection.ipynb`.
2. Select kernel **`solar-venv`** (`all_data/venv`; pandas 2.3 / numpy 2.3 /
   matplotlib 3.10 / scikit-learn 1.8 — though no ML model is used).
3. Run top to bottom; the notebook reads `combined_may2025_jul2026_area1.csv` with a
   relative path, so its working directory must be `all_data/` (default when run in
   place) and rewrites both output CSVs.

Exploration scratch used during development (one-off scripts and review PNGs) is under
`all_data/sat_work/` and can be deleted safely.

---

## 10. Limitations and Tuning Guidance

- **Interpretation of tiers.** The moderate tier captures the *chronic* high-irradiance
  rolloff (structural MPPT undersizing — present on most strong-sun days); the severe
  tier isolates episodic hard-clipping events. Choose per use case.
- **Local shading** of one pair member is indistinguishable from mild saturation in a
  single timestamp; the persistence test and the near-ceiling guard suppress it, but
  brief partial-shade episodes may still flag. If a shade-free reference is available
  later, subtract its expectation.
- **Winter low-sun period** (Oct–Jan) is mostly untestable because `ref < 0.7`; this is
  a property of the climate, not a detector failure.
- **Tuning.** Adjust `P` in §5 of the notebook; or skip re-running entirely and
  threshold the exported `<pair>_deficit` columns directly (add your own persistence
  filter). §7.3's sensitivity table shows the impact of the deficit cutoff.
- **Extensions.** `g0` is currently a global-per-pair scalar; making it rolling
  (31-day median of mid-ref gain) would track long-term soiling even more tightly.
  `eu_7` was excluded as an unreliable irradiance proxy; if a proper pyranometer signal
  becomes available, it can replace `ref` without any other change to the method.
