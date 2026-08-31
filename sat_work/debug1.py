import pandas as pd, numpy as np

CSV = r'c:\Users\nhphuong\Desktop\Solar\all_data\combined_may2025_jul2026_area1.csv'
df = pd.read_csv(CSV, parse_dates=['time']).set_index('time').sort_index()
eu = [f'eu_{i}' for i in range(1, 25)]
reliable = ['eu_1', 'eu_3', 'eu_4', 'eu_9', 'eu_11', 'eu_12', 'eu_15', 'eu_17', 'eu_19', 'eu_20', 'eu_23']

norm = df[reliable].div(df[reliable].quantile(0.999))
ref = norm.median(axis=1)
daily = df[eu].resample('1D').sum()
dailyN = daily.div(daily.quantile(0.95).replace(0, np.nan))
dayidx = df.index.normalize()

a, b = 'eu_8', 'eu_16'
s = df[a] + df[b]
d0 = '2026-06-18'
print('dailyN eu_8:', round(dailyN[a].loc[d0], 3), ' eu_16:', round(dailyN[b].loc[d0], 3))

sel = df.loc[d0].between_time('11:00', '14:00')
rr = ref.loc[sel.index]
ss = s.loc[sel.index]
print('ref midday:', rr.min().round(2), rr.max().round(2))
print('s midday min/max:', ss.min().round(0), ss.max().round(0))

# unit_off for eu_8 / eu_16 that day
for c in [a, b]:
    cond = (df[c] <= 1) & (ref > 0.4)
    grp = cond.ne(cond.shift()).cumsum()
    runlen = cond.groupby(grp).transform('sum')
    off = cond & (runlen >= 12)
    print(c, 'unit_off rows on', d0, ':', int(off.loc[d0].sum()))

# check activity
actA = dailyN[a].loc[dayidx].values > 0.2
actB = dailyN[b].loc[dayidx].values > 0.2
act = actA & actB
print('act on', d0, ':', bool(pd.Series(act, index=df.index).loc[d0].all()))
