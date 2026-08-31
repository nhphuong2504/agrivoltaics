import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = r'c:\Users\nhphuong\Desktop\Solar\all_data\combined_may2025_jul2026_area1.csv'
OUT = r'c:\Users\nhphuong\Desktop\Solar\all_data\sat_work'

df = pd.read_csv(CSV, parse_dates=['time']).set_index('time').sort_index()
eu = [f'eu_{i}' for i in range(1, 25)]
shared = ['eu_8', 'eu_16', 'eu_10', 'eu_18', 'eu_13', 'eu_21']
indep = [c for c in eu if c not in shared + ['eu_7']]

norm = df[indep].div(df[indep].quantile(0.999))
ref = norm.median(axis=1)

pairs = [('eu_8', 'eu_16'), ('eu_10', 'eu_18'), ('eu_13', 'eu_21')]

# 1) pair sum (raw, W) vs ref with LOWESS-ish binned median to see ceiling
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
for j, (a, b) in enumerate(pairs):
    s = df[a] + df[b]
    m = ref > 0.3
    ax = axes[j]
    ax.scatter(ref[m], s[m], s=1, alpha=0.05)
    # binned median + p95
    bins = np.linspace(0.3, 1.05, 40)
    idx = np.digitize(ref[m], bins).clip(1, len(bins) - 1)
    bm = s[m].groupby(idx).median().reindex(range(1, len(bins)))
    bp = s[m].groupby(idx).quantile(0.95).reindex(range(1, len(bins)))
    centers = 0.5 * (bins[1:] + bins[:-1])
    ax.plot(centers, bm.values, 'r.-', lw=2, label='binned median')
    ax.plot(centers, bp.values, 'g.-', lw=1.5, label='binned p95')
    ax.set_title(f'{a}+{b}'); ax.set_xlabel('ref'); ax.set_ylabel('pair sum [W]')
    ax.legend()
plt.tight_layout(); plt.savefig(OUT + r'\ceiling_bins.png', dpi=85)
print('saved ceiling_bins.png')

# 2) two clear days time series: pair sum vs scaled reference
days = ['2025-06-15', '2025-06-20', '2026-06-10', '2026-07-01']
fig, axes = plt.subplots(len(days), 3, figsize=(19, 3.2 * len(days)), sharex=False)
for i, day in enumerate(days):
    dd = df.loc[day]
    rr = ref.loc[day]
    if len(dd) == 0:
        continue
    for j, (a, b) in enumerate(pairs):
        ax = axes[i, j]
        s = dd[a] + dd[b]
        ax.plot(dd.index, s, 'b', lw=1.2, label=f'{a}+{b}')
        # scaled ref to pair "capacity": use pair p99.9
        cap = (df[a] + df[b]).quantile(0.999)
        ax.plot(dd.index, rr * cap, 'orange', lw=1, alpha=0.8, label='ref*cap')
        ax.plot(dd.index, dd[a], 'g--', lw=0.6, alpha=0.7, label=a)
        ax.plot(dd.index, dd[b], 'm--', lw=0.6, alpha=0.7, label=b)
        ax.set_title(f'{day} {a}+{b}')
        if i == 0 and j == 0: ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(OUT + r'\days.png', dpi=85)
print('saved days.png')
