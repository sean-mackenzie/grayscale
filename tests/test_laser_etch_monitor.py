import os
from os.path import join

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from graycart.utils import process

sciblue = '#0C5DA5'
scigreen = '#00B945'
scired = '#FF2C00'
sciorange = '#FF9500'
scipurple = '#845B97'
sciblack = '#474747'
scigray = '#9e9e9e'
sci_color_list = [sciblue, scigreen, scired, sciorange, scipurple, sciblack, scigray]

plt.style.use(['science', 'ieee', 'std-colors'])  # , 'std-colors'
fig, ax = plt.subplots()
size_x_inches, size_y_inches = fig.get_size_inches()
plt.close(fig)

# ---

"""
LaserEtch Monitoring for smOOth.V2

base_dir = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer17/DSEiii-LaserMon'
fn = 'step{}_smOOth.V2-60s'

# known values (measured elsewhere)
etch_rate_step4 = 0.552  # nm/min
etch_rate_step5 = 0.547
etch_time = 60

# ---

px = 't'
pr = 'ref'

# ---


lmons = {}
for step, er in zip([4, 5], [etch_rate_step4, etch_rate_step5]):

    file = [f for f in os.listdir(join(base_dir, fn.format(step))) if f.endswith('.csv')]
    df = pd.read_csv(join(base_dir, fn.format(step), file[0]), names=[px, pr])

    ref = df.ref.to_numpy()
    height = (ref.max() - ref.min()) * 0.8
    min_width = None
    distance = None
    prominence = 0.95
    rel_height = 0.95

    peaks, peak_properties = find_peaks(ref, height=height, width=min_width, distance=distance, prominence=prominence,
                                        rel_height=rel_height)

    if len(peaks) > 4:
        peaks = peaks[:4]

    period_idx = np.mean(np.diff(peaks))
    qperiod_idx = int(period_idx)
    idx_i = int(peaks[0] - qperiod_idx)
    idx_f = int(peaks[-1] + qperiod_idx)

    df = df.iloc[idx_i:idx_f]
    df['t'] = df['t'] - df.iloc[peaks[0] - idx_i].t

    sampling_rate = (df.t.max() - df.t.min()) / len(df)
    sampling_var = np.std(df.t.diff())
    period = period_idx * sampling_rate

    lmon = {'df': df,
            'peaks': peaks,
            'idx_i': idx_i,
            'idx_f': idx_f,
            'period_idx': period_idx,
            'period': period,
            'sampling_rate': sampling_rate,
            'sampling_std': sampling_var,
            'etch_time': etch_time,
            'etch_rate': er,
            }
    lmons.update({step: lmon})

# ---

# evaluate collection
t_is = []
periods = []
etch_rates = []
etch_depths = []
etch_depth_per_wavelengths = []

for step, lmon in lmons.items():
    t_is.append(lmon['df']['t'].min())
    periods.append(lmon['period'])
    etch_rates.append(lmon['etch_rate'])
    ed = lmon['etch_rate'] * 1000 * lmon['etch_time'] / 60
    etch_depths.append(ed)
    etch_depth_per_wavelengths.append(ed / (60 / lmon['period']))

t_i_min = np.min(t_is)
period_m, period_std = np.mean(periods), np.std(periods)
er_m, er_std = np.mean(etch_rates), np.std(etch_rates)
erpp_m, erpp_std = np.mean(etch_depth_per_wavelengths), np.std(etch_depth_per_wavelengths)
print(etch_depth_per_wavelengths)
erpp_std = 4

# ---

# plot

fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches))

for step, lmon in lmons.items():
    df = lmon['df'].copy()

    ax.plot(df.t - t_i_min, df.ref, '-o', ms=0.75,
            label='{}: {} s, {}'.format(step,
                                        np.round(lmon['period'], 1),
                                        int(lmon['etch_rate'] * 1000),
                                        )
            )

    lbl = None
    for k, pk in enumerate(peaks):
        if k == len(peaks) - 1:
            lbl = r'$\lambda=$' + ' {} '.format(np.round(period_idx * sampling_rate, 1)) + r'$s$'
        ax.axvline(df.iloc[pk - idx_i].t, linewidth=0.5, linestyle='--', color='b', alpha=0.25, label=lbl)

ax.set_ylabel('Ref. ' + r'$(\%)$')
ax.set_xlabel(r'$t \: (s)$')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
          title='Step: ' + r'$T$' + '(s), E.R. (nm/min)',
          )
ax.set_title(r'$\lambda =$' + ' {} '.format(int(erpp_m)) + r'$\pm$' + '{}'.format(erpp_std) + ' nm (SPR220-7)')

plt.suptitle('DSEiii recipe: smOOth.V2')
plt.tight_layout()
plt.savefig(join(base_dir, fn.format('4-5_') + '.png'))
plt.show()
plt.close()
"""

# ----


base_dir = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer13/DSEiii-LaserMon'
fn = 'step{}_SF6O2V6-{}min'

# known values (measured elsewhere)
steps = [7, 8]
etch_rates = [0.221, 0.211]  # nm/min
etch_times = [20, 6]
num_peaks = [15, 6]

# ---

px = 't'
pr = 'ref'

# ---


lmons = {}
for step, er, et, npk in zip(steps, etch_rates, etch_times, num_peaks):

    # file = [f for f in os.listdir(join(base_dir, fn.format(step))) if f.endswith('.csv')]
    # df = pd.read_csv(join(base_dir, fn.format(step), file), names=[px, pr])

    fp = join(base_dir, fn.format(step, et) + '.csv')
    df = pd.read_csv(fp, names=[px, pr])

    ref = df.ref.to_numpy()
    height = (ref.max() - ref.min()) * 0.8
    min_width = None
    distance = None
    prominence = 0.95
    rel_height = 0.95

    peaks, peak_properties = find_peaks(ref, height=height, width=min_width, distance=distance, prominence=prominence,
                                        rel_height=rel_height)

    if len(peaks) > npk:
        peaks = peaks[:npk]

    period_idx = np.mean(np.diff(peaks))
    qperiod_idx = int(period_idx / 4)
    idx_i = int(peaks[0] - qperiod_idx)
    idx_f = int(peaks[-1] + qperiod_idx)

    """fig, ax = plt.subplots()
    ax.plot(df.t, df.ref)
    sampling_rate = 1
    lbl = None
    for k, pk in enumerate(peaks):
        ax.axvline(df.iloc[pk - idx_i].t, linewidth=0.5, linestyle='--', color='b', alpha=0.25, label=lbl)
    plt.show()"""


    df = df.iloc[idx_i:idx_f]
    df['t'] = df['t'] - df.iloc[peaks[0] - idx_i].t

    sampling_rate = (df.t.max() - df.t.min()) / len(df)
    sampling_var = np.std(df.t.diff())
    period = period_idx * sampling_rate

    lmon = {'df': df,
            'peaks': peaks,
            'idx_i': idx_i,
            'idx_f': idx_f,
            'period_idx': period_idx,
            'period': period,
            'sampling_rate': sampling_rate,
            'sampling_std': sampling_var,
            'etch_time': et * 60,
            'etch_rate': er * 1000,
            }
    lmons.update({step: lmon})

# ---

# evaluate collection
t_is = []
periods = []
etch_rates = []
etch_depths = []
etch_depth_per_wavelengths = []

for step, lmon in lmons.items():
    t_is.append(lmon['df']['t'].min())
    periods.append(lmon['period'])
    etch_rates.append(lmon['etch_rate'])
    ed = lmon['etch_rate'] * lmon['etch_time'] / 60
    etch_depths.append(ed)
    etch_depth_per_wavelengths.append(ed / (lmon['etch_time'] / lmon['period']))

t_i_min = np.min(t_is)
period_m, period_std = np.mean(periods), np.std(periods)
er_m, er_std = np.mean(etch_rates), np.std(etch_rates)
erpp_m, erpp_std = np.mean(etch_depth_per_wavelengths), np.std(etch_depth_per_wavelengths)
print(etch_depth_per_wavelengths)
erpp_std = 6

# ---

# plot

fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches))

for step, lmon in lmons.items():
    df = lmon['df'].copy()

    ax.plot(df.t - t_i_min, df.ref, '-o', ms=0.75,
            label='{}: {} s, {}'.format(step,
                                        np.round(lmon['period'], 1),
                                        int(lmon['etch_rate']),
                                        )
            )

    """lbl = None
    for k, pk in enumerate(peaks):
        if k == len(peaks) - 1:
            lbl = r'$\lambda=$' + ' {} '.format(np.round(period_idx * sampling_rate, 1)) + r'$s$'
        ax.axvline(df.iloc[pk - idx_i].t, linewidth=0.5, linestyle='--', color='b', alpha=0.25, label=lbl)"""

ax.set_ylabel('Ref. ' + r'$(\%)$')
ax.set_xlabel(r'$t \: (s)$')
ax.legend(loc='upper left', bbox_to_anchor=(1, 1),
          title='Step: ' + r'$T$' + '(s), E.R. (nm/min)',
          )
ax.set_title(r'$\lambda =$' + ' {} '.format(int(erpp_m)) + r'$\pm$' + '{}'.format(erpp_std) + ' nm (SPR220-7)')

plt.suptitle('DSEiii recipe: ' + r'$SF_{6} + O_{2}$' + '.V6')
plt.tight_layout()
plt.savefig(join(base_dir, fn.format('7-8_', 'mixed') + '.png'))
plt.show()
plt.close()