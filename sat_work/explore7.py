import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = r'c:\Users\nhphuong\Desktop\Solar\all_data\combined_may2025_jul2026_area1.csv'
OUT = r'c:\Users\nhphuong\Desktop\Solar\all_data\sat_work'

df = pd.read_csv(CSV, parse_dates=['time']).set_index('time').sort_index()
eu = [f'eu_{i}' for i in range(1, 25)]
reliable = ['eu_1', 'eu_3', 'eu_4', 'eu_9', 'eu_11', 'eu_12', 'eu_15', 'eu_17', 'eu_19', 'eu_20', 'eu_23']

norm = df[reliable].div(df[reliable].quantile(0.999))
ref = norm.median(axis=1)

# activity masks (daily energy > 20% of own p95)
daily = df[eu].resample('1D').sum()
dailyN = daily.div(daily.quantile(0.95).replace(0, np.nan))

pairs = [('eu_8', 'eu_16'), ('eu_10', 'eu_18'), ('eu_13', 'eu_21')]

# rolling adaptive estimates: use day-centered windows
dayidx = df.index.normalize()
udays = pd.Series(dailyN.index)

def rolling_stat(series, mask, win_days, q):
    """rolling quantile of series over +-win_days, computed on masked rows only"""
    ds = series.groupby(dayidx).apply(lambda x: x[mask.loc[x.index].values] if False else x)
    # simpler: build daily series of the target quantity then rolling-quantile over days
    return None

# per-day quantities we need:
#  cap_day  : p99 of pair sum on active rows with ref>0.3
#  g0_day   : median gain s/(cap*ref) on active rows with ref in [0.35,0.6]
# We compute daily then take rolling median over 31 days (min_periods 5).
rows = []
for a, b in pairs:
    s = df[a] + df[b]
    actA = dailyN[a].loc[dayidx].values > 0.2
    actB = dailyN[b].loc[dayidx].values > 0.2
    act = actA & actB
    tmp = pd.DataFrame({'s': s, 'ref': ref, 'act': act}, index=df.index)
    tmp['day'] = dayidx
    for day, g in tmp.groupby('day'):
        ga = g[g['act']]
        cap = ga.loc[ga['ref'] > 0.3, 's'].quantile(0.99) if (ga['ref'] > 0.3).sum() > 20 else np.nan
        mid = ga[(ga['ref'] >= 0.35) & (ga['ref'] <= 0.6)]
        rows.append({'pair': f'{a}+{b}', 'day': day, 'cap_day': cap,
                     'n_act': len(ga), 'mid_gain_raw': (mid['s'] / mid['ref']).median() if len(mid) > 10 else np.nan})
capdf = pd.DataFrame(rows).pivot(index='day', columns='pair')

cap_roll = capdf['cap_day'].rolling(31, center=True, min_periods=5).median()
print('=== rolling cap (W) per pair: sample months ===')
print(cap_roll.resample('1MS').median().round(0).to_string())

# gain using rolling cap
fig, axes = plt.subplots(3, 1, figsize=(14, 10))
for j, (a, b) in enumerate(pairs):
    p = f'{a}+{b}'
    s = df[a] + df[b]
    cap_t = cap_roll[p].loc[dayidx].values
    g = s / (cap_t * ref.replace(0, np.nan))
    actA = dailyN[a].loc[dayidx].values > 0.2
    actB = dailyN[b].loc[dayidx].values > 0.2
    act = actA & actB
    m = act & (ref > 0.3)
    bins = np.linspace(0.3, 1.02, 25)
    idx = np.digitize(ref[m], bins).clip(1, len(bins) - 1)
    bm = g[m].groupby(idx).median().reindex(range(1, len(bins)))
    b25 = g[m].groupby(idx).quantile(0.25).reindex(range(1, len(bins)))
    b75 = g[m].groupby(idx).quantile(0.75).reindex(range(1, len(bins)))
    centers = 0.5 * (bins[1:] + bins[:-1])
    ax = axes[j]
    ax.plot(centers, bm, 'b.-', label='median gain')
    ax.fill_between(centers, b25, b75, alpha=0.25)
    g0 = g[m & (ref >= 0.35) & (ref <= 0.6)].median()
    ax.axhline(g0, color='g', ls='--', label=f'g0={g0:.2f}')
    ax.axhline(g0 * 0.85, color='r', ls=':', label='g0*0.85 (flag line)')
    ax.set_title(f'{p}: gain = s/(cap_rolling*ref) vs ref'); ax.legend(); ax.set_ylim(0.4, 1.8)
plt.tight_layout(); plt.savefig(OUT + r'\gain_all_pairs.png', dpi=85)
print('saved gain_all_pairs.png')
