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
daily = df[eu].resample('1D').sum()
dailyN = daily.div(daily.quantile(0.95).replace(0, np.nan))
dayidx = df.index.normalize()
active = dailyN.loc[dayidx, eu].reset_index(drop=True).set_index(df.index) > 0.2

def run_lengths(cond):
    grp = cond.ne(cond.shift()).cumsum()
    return cond.groupby(grp).transform('sum')

def rolling_cap(s, act):
    ok = act & (ref > 0.3)
    per_day = s[ok].groupby(dayidx[ok]).quantile(0.99)
    return per_day.rolling(31, center=True, min_periods=5).median()

a, b = 'eu_15', 'eu_23'
s = df[a] + df[b]
act = active[a] & active[b]
cap_t = rolling_cap(s, act).reindex(dayidx).to_numpy()
gain = s / (cap_t * ref.replace(0, np.nan))
g0 = gain[act & (ref >= 0.35) & (ref <= 0.6)].median()
expected = g0 * cap_t * ref
deficit = 1 - s / expected
raw = (ref >= 0.7) & (deficit > 0.15) & (s > 0.6 * cap_t) & act
flag = raw & (run_lengths(raw) >= 3)
fl = flag[flag]
print('control eu_15+eu_23 flags:', len(fl))
print('by day:'); print(fl.groupby(fl.index.date).size().to_string())

# visualize the worst control day
per = fl.groupby(fl.index.date).size().sort_values(ascending=False)
d = per.index[0]
fig, ax = plt.subplots(figsize=(13, 4))
dd = slice(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=1))
ax.plot(s.loc[dd].index, s.loc[dd], 'b', lw=1.3, label='eu_15+eu_23')
ax.plot(s.loc[dd].index, (g0 * pd.Series(cap_t, index=df.index) * ref).loc[dd], 'orange', label='expected')
ax.plot(s.loc[dd].index, (df['eu_1'] + df['eu_3']).loc[dd], 'g--', lw=0.9, label='eu_1+eu_3 (peers)')
f = flag.loc[dd]
ax.scatter(s.loc[dd].index[f], s.loc[dd][f], c='r', s=14, zorder=5)
ax.legend(); ax.set_title(f'worst control day {d}'); plt.tight_layout()
plt.savefig(OUT + r'\control_worst.png', dpi=85)
print('saved control_worst.png')
