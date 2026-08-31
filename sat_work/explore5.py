import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = r'c:\Users\nhphuong\Desktop\Solar\all_data\combined_may2025_jul2026_area1.csv'
OUT = r'c:\Users\nhphuong\Desktop\Solar\all_data\sat_work'

df = pd.read_csv(CSV, parse_dates=['time']).set_index('time').sort_index()
eu = [f'eu_{i}' for i in range(1, 25)]
shared = ['eu_8', 'eu_16', 'eu_10', 'eu_18', 'eu_13', 'eu_21']
reliable = ['eu_1', 'eu_3', 'eu_4', 'eu_9', 'eu_11', 'eu_12', 'eu_15', 'eu_17', 'eu_19', 'eu_20', 'eu_23']

norm = df[reliable].div(df[reliable].quantile(0.999))
ref = norm.median(axis=1)
pairA, pairB = 'eu_8', 'eu_16'
s = df[pairA] + df[pairB]

# active-day mask for the pair
daily = df[eu].resample('1D').sum()
dailyN = daily.div(daily.quantile(0.95).replace(0, np.nan))
act = dailyN[pairA].loc[df.index.normalize()].values > 0.2

m = act & (ref > 0.3)
cap = s[m].quantile(0.995)
resid = s / cap - ref          # 0 if tracking, negative if capped
print(f'cap(p99.5 of sum at act,ref>0.3) = {cap:.0f}')

hi = m & (ref > 0.75)
print('\n=== resid (s/cap - ref) at ref>0.75: percentiles ===')
print(resid[hi].describe(percentiles=[.01, .05, .1, .25, .5]).round(3).to_string())

# candidate: resid < -0.15 at ref>0.75  => how many, distribution over days
sat = hi & (resid < -0.15)
print('\nsat candidate count:', sat.sum(), ' days affected:', sat[sat].index.normalize().nunique())
per_day = sat.groupby(sat.index.date).sum()
print('per-day counts describe:'); print(per_day.describe().round(1).to_string())

# visualize a few candidate days with heaviest saturation
top_days = per_day.sort_values(ascending=False).head(6).index
fig, axes = plt.subplots(len(top_days), 1, figsize=(15, 2.6 * len(top_days)), sharex=False)
for i, d in enumerate(top_days):
    dd = df.loc[str(d)]
    rr = ref.loc[str(d)]
    ss = dd[pairA] + dd[pairB]
    ax = axes[i]
    ax.plot(dd.index, ss, 'b', lw=1.4, label=f'{pairA}+{pairB}')
    ax.plot(dd.index, rr * cap, 'orange', lw=1.2, label='ref*cap')
    ax.axhline(cap, color='r', ls=':', lw=1)
    satmask = sat.loc[str(d)]
    if satmask.any():
        ax.scatter(dd.index[satmask.reindex(dd.index, fill_value=False)],
                   ss[satmask.reindex(dd.index, fill_value=False)], c='r', s=14, zorder=5, label='sat flag')
    ax.set_title(str(d)); ax.legend(fontsize=7); ax.set_ylim(-200, cap * 1.15)
plt.tight_layout(); plt.savefig(OUT + r'\sat_days.png', dpi=85)
print('saved sat_days.png, top days:', [str(d) for d in top_days])

# how does resid distribution look across ref bins - the saturation "knee"
bins = np.linspace(0.3, 1.02, 37)
idx = np.digitize(ref[m], bins).clip(1, len(bins) - 1)
ratio = (s / ref.replace(0, np.nan))[m]
bm = ratio.groupby(idx).median().reindex(range(1, len(bins)))
b10 = ratio.groupby(idx).quantile(0.10).reindex(range(1, len(bins)))
b90 = ratio.groupby(idx).quantile(0.90).reindex(range(1, len(bins)))
centers = 0.5 * (bins[1:] + bins[:-1])
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(centers, bm, 'b.-', label='median sum/ref')
ax.plot(centers, b10, 'g.-', label='p10')
ax.plot(centers, b90, 'r.-', label='p90')
ax.set_xlabel('ref'); ax.set_ylabel('pair_sum / ref'); ax.legend()
ax.set_title('gain vs irradiance proxy: droop at high ref = saturation')
plt.tight_layout(); plt.savefig(OUT + r'\gain_droop.png', dpi=85)
print('saved gain_droop.png')
