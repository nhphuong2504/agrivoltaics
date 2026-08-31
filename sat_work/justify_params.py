import pandas as pd, numpy as np, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

CSV = r'c:\Users\nhphuong\Desktop\Solar\all_data\combined_may2025_jul2026_area1.csv'
df = pd.read_csv(CSV, parse_dates=['time']).set_index('time').sort_index()
eu = [f'eu_{i}' for i in range(1, 25)]
PAIRS = [('eu_8', 'eu_16'), ('eu_10', 'eu_18'), ('eu_13', 'eu_21')]
reliable = ['eu_1', 'eu_3', 'eu_4', 'eu_9', 'eu_11', 'eu_12', 'eu_15', 'eu_17', 'eu_19', 'eu_20', 'eu_23']
ref = df[reliable].div(df[reliable].quantile(0.999)).median(axis=1)
daily = df[eu].resample('1D').sum()
dailyN = daily.div(daily.quantile(0.95).replace(0, np.nan))
dayidx = df.index.normalize()
active = dailyN.loc[dayidx, eu].reset_index(drop=True).set_index(df.index) > 0.2

def run_lengths(cond):
    grp = cond.ne(cond.shift()).cumsum()
    return cond.groupby(grp).transform('sum')

def model(a, b):
    s = df[a] + df[b]
    act = active[a] & active[b]
    ok = act & (ref > 0.3)
    cap_day = s[ok].groupby(dayidx[ok]).quantile(0.99)
    cap_t = cap_day.rolling(31, center=True, min_periods=5).median().reindex(dayidx).to_numpy()
    gain = s / (cap_t * ref.replace(0, np.nan))
    g0 = gain[act & (ref >= 0.35) & (ref <= 0.6)].median()
    deficit = 1 - s / (g0 * cap_t * ref)
    return s, act, cap_t, deficit

print('=== [REF_ON] noise of deficit vs ref bin — CONTROL pairs (unsaturated) ===')
print('std/p95 of |deficit| per ref bin; low ref => noisy, unstable division')
for a, b in [('eu_1', 'eu_3'), ('eu_15', 'eu_23')]:
    s, act, cap_t, deficit = model(a, b)
    bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.05]
    labels = ['0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9+']
    for lo, hi, lab in zip(bins[:-1], bins[1:], labels):
        sel = act & (ref >= lo) & (ref < hi)
        d = deficit[sel].replace([np.inf, -np.inf], np.nan).dropna()
        print(f'{a}+{b} ref {lab:7s} n={len(d):5d}  std={d.std():.3f}  p95={d.abs().quantile(.95):.3f}  p99={d.abs().quantile(.99):.3f}')
    print()

print('=== [DEF_MOD] healthy deficit tail (controls, ref>=0.7) vs saturated deficits ===')
ctrl_d = []
for a, b in [('eu_1', 'eu_3'), ('eu_4', 'eu_12'), ('eu_15', 'eu_23')]:
    s, act, cap_t, deficit = model(a, b)
    d = deficit[act & (ref >= 0.7)].replace([np.inf, -np.inf], np.nan).dropna()
    ctrl_d.append(d)
ctrl_d = pd.concat(ctrl_d)
print('controls ref>=0.7: p50=%.3f p90=%.3f p95=%.3f p99=%.3f  frac>0.15 = %.4f' % (
    ctrl_d.median(), ctrl_d.quantile(.9), ctrl_d.quantile(.95), ctrl_d.quantile(.99), (ctrl_d > 0.15).mean()))
for a, b in PAIRS:
    s, act, cap_t, deficit = model(a, b)
    d = deficit[act & (ref >= 0.7)].replace([np.inf, -np.inf], np.nan).dropna()
    print(f'{a}+{b} ref>=0.7: p50={d.median():.3f} p90={d.quantile(.9):.3f}  frac>0.15={(d > 0.15).mean():.3f}')

print()
print('=== [NEAR_CAP] s/cap during saturated rows vs during fault/shade rows ===')
for a, b in PAIRS:
    s, act, cap_t, deficit = model(a, b)
    ratio = s / cap_t
    hi = act & (ref >= 0.7)
    sat_rows = hi & (deficit > 0.15)
    print(f'{a}+{b}: s/cap on deficit>0.15 rows  p5={ratio[sat_rows].quantile(.05):.2f} p25={ratio[sat_rows].quantile(.25):.2f} p50={ratio[sat_rows].quantile(.5):.2f}')
    # fault-like rows: ref high but pair near zero (excluded by near-cap)
    off = hi & (s <= 1)
    print(f'   pair ~0 W while ref>=0.7: {int(off.sum())} rows (deficit there = 1.0, would false-flag without NEAR_CAP)')

print()
print('=== [PERSIST] run-length distribution of raw moderate condition ===')
for a, b in PAIRS:
    s, act, cap_t, deficit = model(a, b)
    raw = act & (ref >= 0.7) & (deficit > 0.15) & (s > 0.6 * cap_t)
    grp = raw.ne(raw.shift()).cumsum()
    runs = raw.groupby(grp).sum()
    runs = runs[runs > 0]
    print(f'{a}+{b}: raw events total={len(runs):4d} | len 1-2 samples: {int((runs <= 2).sum())} ({100*(runs<=2).mean():.0f}%) | len>=3: {int((runs>=3).sum())}')
