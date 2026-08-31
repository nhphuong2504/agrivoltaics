import pandas as pd, numpy as np

CSV = r'c:\Users\nhphuong\Desktop\Solar\all_data\combined_may2025_jul2026_area1.csv'
df = pd.read_csv(CSV, parse_dates=['time']).set_index('time').sort_index()
eu = [f'eu_{i}' for i in range(1, 25)]
shared = ['eu_8', 'eu_16', 'eu_10', 'eu_18', 'eu_13', 'eu_21']
reliable = ['eu_1', 'eu_3', 'eu_4', 'eu_9', 'eu_11', 'eu_12', 'eu_15', 'eu_17', 'eu_19', 'eu_20', 'eu_23']

norm = df[reliable].div(df[reliable].quantile(0.999))
ref = norm.median(axis=1)

print('=== per-column correlation with ref (active-ish rows, ref>0.3) ===')
m = ref > 0.3
cors = {}
for c in eu:
    cors[c] = df[c][m].corr(ref[m])
cs = pd.Series(cors).sort_values()
print(cs.round(3).to_string())

print()
print('=== ratio-to-ref droop test for INDEPENDENT units (sanity) ===')
bins = np.linspace(0.3, 1.02, 13)
centers = 0.5 * (bins[1:] + bins[:-1])
for c in ['eu_1', 'eu_15', 'eu_2', 'eu_24']:
    ratio = (df[c] / df[c].quantile(0.999)) / ref.replace(0, np.nan)
    idx = np.digitize(ref[m], bins).clip(1, len(bins) - 1)
    bm = ratio[m].groupby(idx).median().reindex(range(1, len(bins)))
    print(c, np.round(bm.values, 2))

print()
print('=== droop for shared units and pair sums ===')
pairs = [('eu_8', 'eu_16'), ('eu_10', 'eu_18'), ('eu_13', 'eu_21')]
for a, b in pairs:
    s = df[a] + df[b]
    cap = s[(ref > 0.3)].quantile(0.995)
    ratio = (s / cap) / ref.replace(0, np.nan)
    idx = np.digitize(ref[m], bins).clip(1, len(bins) - 1)
    bm = ratio[m].groupby(idx).median().reindex(range(1, len(bins)))
    print(f'{a}+{b} sum gain by ref bin:', np.round(bm.values, 2))

print()
print('=== eu_7 as irradiance: corr with ref, head ===')
print('corr eu_7 vs ref (ref>0.3):', round(df['eu_7'][m].corr(ref[m]), 3))
print(df['eu_7'].describe().round(2).to_string())
