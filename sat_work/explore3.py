import pandas as pd, numpy as np

CSV = r'c:\Users\nhphuong\Desktop\Solar\all_data\combined_may2025_jul2026_area1.csv'

df = pd.read_csv(CSV, parse_dates=['time']).set_index('time').sort_index()
eu = [f'eu_{i}' for i in range(1, 25)]
shared = ['eu_8', 'eu_16', 'eu_10', 'eu_18', 'eu_13', 'eu_21']
indep = [c for c in eu if c not in shared + ['eu_7']]

norm = df[indep].div(df[indep].quantile(0.999))
ref = norm.median(axis=1)

pairs = [('eu_8', 'eu_16'), ('eu_10', 'eu_18'), ('eu_13', 'eu_21')]

print('=== per-pair ceiling stats (raw pair sum at ref>0.9) ===')
for a, b in pairs:
    s = df[a] + df[b]
    hi = ref > 0.9
    print(f'{a}+{b}: p50={s[hi].median():.0f} p90={s[hi].quantile(.9):.0f} '
          f'p99={s[hi].quantile(.99):.0f} max={s.max():.0f}')

print()
print('=== negative values context (eu_8) ===')
neg = df[df['eu_8'] < 0].head(15)
print(neg[['eu_8', 'eu_16']].assign(ref=ref.loc[neg.index].round(2)).to_string())

print()
print('=== simultaneous all-zero (site outage) check ===')
allzero = (df[indep].sum(axis=1) < 100)
dayref = ref > 0.3
print('rows where ALL independent ~0 while ref-history says day: ', (allzero).sum())

# zero patterns: which EUs zero at midday (ref high) -> suspicious faults
print()
print('=== midday (ref>0.6) zero counts per EU ===')
mid = ref > 0.6
zc = (df[eu][mid] <= 0).sum()
print(zc[zc > 0].sort_values(ascending=False).to_string())

# staleness: repeated identical values
print()
print('=== staleness: longest run of identical consecutive values per EU ===')
for c in eu:
    v = df[c]
    ch = v.ne(v.shift())
    runs = (~ch).groupby(ch.cumsum()).sum()
    mx = runs.max()
    if mx >= 6:
        print(c, 'max stale run (5-min steps):', int(mx))
