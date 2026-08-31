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

# daily energy per EU (sum * 5min), normalized per column -> activity heatmap
daily = df[eu].resample('1D').sum()
dailyN = daily.div(daily.quantile(0.95).replace(0, np.nan))

fig, ax = plt.subplots(figsize=(19, 6))
im = ax.imshow(dailyN.T, aspect='auto', cmap='viridis', vmin=0, vmax=1,
               extent=[0, len(dailyN), 24.5, 0.5])
ax.set_yticks(range(1, 25)); ax.set_yticklabels(eu)
xt = np.arange(0, len(dailyN), 14)
ax.set_xticks(xt); ax.set_xticklabels(dailyN.index[xt].strftime('%y-%m-%d'), rotation=90, fontsize=7)
for s in shared:
    ax.axhline(eu.index(s) + 0.5, color='r', lw=0.4, alpha=0.5)
ax.set_title('daily energy per EU (col-normalized) - red lines bracket shared-MPPT units')
plt.colorbar(im, ax=ax, shrink=0.7)
plt.tight_layout(); plt.savefig(OUT + r'\activity_heatmap.png', dpi=85)
print('saved activity_heatmap.png')

# reliable reference
norm = df[reliable].div(df[reliable].quantile(0.999))
ref = norm.median(axis=1)

pairs = [('eu_8', 'eu_16'), ('eu_10', 'eu_18'), ('eu_13', 'eu_21')]
# ceiling vs ref with reliable ref, zoom
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
for j, (a, b) in enumerate(pairs):
    s = df[a] + df[b]
    act = dailyN[a].loc[df.index.normalize()] > 0.2  # pair-unit active days only
    m = (ref > 0.3) & act.values
    ax = axes[j]
    ax.scatter(ref[m], s[m], s=1.5, alpha=0.07)
    bins = np.linspace(0.3, 1.02, 37)
    idx = np.digitize(ref[m], bins).clip(1, len(bins) - 1)
    bm = s[m].groupby(idx).median().reindex(range(1, len(bins)))
    b95 = s[m].groupby(idx).quantile(0.95).reindex(range(1, len(bins)))
    centers = 0.5 * (bins[1:] + bins[:-1])
    ax.plot(centers, bm.values, 'r.-', lw=2, label='median')
    ax.plot(centers, b95.values, 'g.-', lw=1.3, label='p95')
    # linear expectation from unsaturated region (ref<0.7): slope via median ratio
    low = m & (ref < 0.7) & (ref > 0.4)
    k = (s[low] / ref[low]).median()
    ax.plot([0.3, 1.0], [0.3 * k, 1.0 * k], 'k--', label=f'linear exp (k={k:.0f})')
    ax.set_title(f'{a}+{b}'); ax.set_xlabel('ref'); ax.legend()
plt.tight_layout(); plt.savefig(OUT + r'\ceiling2.png', dpi=85)
print('saved ceiling2.png')
