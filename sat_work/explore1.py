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

# normalized reference: each independent unit scaled by its own 99.9th pct, then median
norm = df[indep].div(df[indep].quantile(0.999))
ref = norm.median(axis=1)

pairs = [('eu_8', 'eu_16'), ('eu_10', 'eu_18'), ('eu_13', 'eu_21')]
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
for j, (a, b) in enumerate(pairs):
    s = df[a] + df[b]
    ax = axes[0, j]
    m = ref > 0.05
    ax.scatter(ref[m], s[m], s=1, alpha=0.08)
    ax.set_title(f'{a}+{b} raw sum vs ref')
    ax.set_xlabel('ref (norm irradiance proxy)'); ax.set_ylabel('pair sum [W]')

    ax2 = axes[1, j]
    cap_est = df[a].quantile(0.999) + df[b].quantile(0.999)
    sn = s / cap_est
    ax2.scatter(ref[m], sn[m], s=1, alpha=0.08)
    ax2.axhline(1, color='r', ls='--')
    ax2.set_title(f'{a}+{b} sum / cap_est={cap_est:.0f} vs ref')
    ax2.set_xlabel('ref')
plt.tight_layout()
plt.savefig(OUT + r'\scatter_pairs.png', dpi=80)
print('saved scatter_pairs.png')

# what do the strongest independent units look like on the same axes (sanity check)
fig, ax = plt.subplots(figsize=(8, 5))
m = ref > 0.05
ax.scatter(ref[m], (df['eu_1'][m] / df['eu_1'].quantile(0.999)), s=1, alpha=0.08, label='eu_1 norm')
ax.scatter(ref[m], (df['eu_15'][m] / df['eu_15'].quantile(0.999)), s=1, alpha=0.08, label='eu_15 norm')
ax.legend(); ax.set_xlabel('ref'); ax.set_title('independent units normalized vs ref')
plt.tight_layout(); plt.savefig(OUT + r'\scatter_indep.png', dpi=80)
print('saved scatter_indep.png')
