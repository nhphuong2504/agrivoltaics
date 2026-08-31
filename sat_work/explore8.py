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
ref_disp = ref.rolling(3, center=True, min_periods=1).median()  # light smoothing for display

daily = df[eu].resample('1D').sum()
dailyN = daily.div(daily.quantile(0.95).replace(0, np.nan))
dayidx = df.index.normalize()

pairs = [('eu_8', 'eu_16'), ('eu_10', 'eu_18'), ('eu_13', 'eu_21')]

# ---------- data-quality flags ----------
# site outage: >80% of reliable units ~0 while sun should be up (use hour 9..15 and previous/next day activity)
site_zero = (df[reliable] < 50).mean(axis=1) > 0.8
midday = (df.index.hour >= 9) & (df.index.hour <= 15)
site_outage = site_zero & midday

# unit offline: unit ~0 while ref>0.4, sustained >= 1h (12 steps)
unit_off = pd.DataFrame(False, index=df.index, columns=eu)
for c in eu:
    cond = (df[c] <= 1) & (ref > 0.4)
    grp = cond.ne(cond.shift()).cumsum()
    runlen = cond.groupby(grp).transform('sum')
    unit_off[c] = cond & (runlen >= 12)

# stale: identical values >= 12 consecutive steps while ref varies
stale = pd.DataFrame(False, index=df.index, columns=eu)
refvar = ref.diff().abs().rolling(6).sum() > 0.02
for c in eu:
    same = df[c].eq(df[c].shift()) & df[c].gt(0)  # exclude zero runs (handled by offline)
    grp = same.ne(same.shift()).cumsum()
    runlen = same.groupby(grp).transform('sum')
    stale[c] = same & (runlen >= 12) & refvar

neg = df[eu] < 0

print('site outage rows:', int(site_outage.sum()))
print('unit_off rows total:', int(unit_off.sum().sum()))
print('stale rows total:', int(stale.sum().sum()))
print('negative rows total:', int(neg.sum().sum()))

# ---------- saturation detector ----------
results = {}
for a, b in pairs:
    p = f'{a}_{b}'
    s = df[a] + df[b]
    actA = dailyN[a].loc[dayidx].values > 0.2
    actB = dailyN[b].loc[dayidx].values > 0.2
    act = pd.Series(actA & actB, index=df.index)

    # daily cap -> rolling 31d median
    tmp = pd.DataFrame({'s': s.values, 'ref': ref.values, 'act': act.values}, index=df.index)
    cap_day = tmp[tmp['act'] & (tmp['ref'] > 0.3)].groupby(dayidx[tmp['act'] & (tmp['ref'] > 0.3)])['s'].quantile(0.99)
    cap_roll = cap_day.rolling(31, center=True, min_periods=5).median()
    cap_t = cap_roll.reindex(dayidx).values

    gain = s / (cap_t * ref.replace(0, np.nan))
    g0 = gain[act & (ref >= 0.35) & (ref <= 0.6)].median()

    expect = g0 * cap_t * ref
    deficit = 1 - s / expect

    raw = (ref >= 0.7) & (deficit > 0.15) & (s > 0.6 * cap_t) & act
    # exclude rows where either unit flagged off/stale/neg
    bad = unit_off[a] | unit_off[b] | stale[a] | stale[b] | neg[a] | neg[b] | site_outage
    raw = raw & ~bad
    # persistence >= 3 samples
    grp = raw.ne(raw.shift()).cumsum()
    runlen = raw.groupby(grp).transform('sum')
    flag = raw & (runlen >= 3)

    results[p] = dict(s=s, cap_t=cap_t, gain=gain, g0=g0, flag=flag, act=act)
    fdays = flag[flag].index.normalize().nunique()
    print(f'{p}: g0={g0:.3f}  flagged rows={int(flag.sum())}  days={fdays}')

# ---------- validation figure: random flagged & clear days for eu_8+16 ----------
p = 'eu_8_eu_16'
r = results[p]
flag_days = pd.Series(r['flag'][r['flag']].index.date).unique()[:0]
perday = r['flag'].groupby(dayidx).sum()
heavy = perday[perday >= 12].sort_values(ascending=False)
print('\ndays with >=1h saturation (eu_8_16):', len(heavy))
show = list(heavy.index[:4]) + list(perday[perday == 0].index[-2:])
fig, axes = plt.subplots(len(show), 1, figsize=(15, 2.5 * len(show)))
for i, d in enumerate(show):
    dd = slice(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=1))
    ax = axes[i]
    ss = r['s'].loc[dd]; rr = ref.loc[dd]
    ax.plot(ss.index, ss, 'b', lw=1.3, label='pair sum')
    ax.plot(ss.index, (r['g0'] * pd.Series(r['cap_t'], index=df.index).loc[dd] * rr), 'orange', lw=1.2, label='expected (g0*cap*ref)')
    ax.plot(ss.index, pd.Series(r['cap_t'], index=df.index).loc[dd], 'r:', lw=1, label='rolling cap')
    fl = r['flag'].loc[dd]
    if fl.any():
        ax.scatter(ss.index[fl], ss[fl], c='r', s=16, zorder=5, label='SAT flag')
    ax.set_title(f'{pd.Timestamp(d).date()}  (flags={int(fl.sum())})', fontsize=9)
    ax.legend(fontsize=7, loc='upper left'); ax.set_ylim(bottom=-100)
plt.tight_layout(); plt.savefig(OUT + r'\validate_eu8_16.png', dpi=85)
print('saved validate_eu8_16.png')
