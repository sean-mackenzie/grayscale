from os.path import join, isdir
from os import makedirs
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

from graycart.GraycartFeature import ProcessFeature
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

# ------------------------------------------------------------------------------------------------------------------
# Plot manually entered data

plot_etch_rates = True
if plot_etch_rates:

    recipe_SF6_O2_V6 = {'Pressure': 10,
                        'SF6': 50,
                        'O2': 50,
                        'HF Bias': 10,
                        'ICP': 800,
                        }

    SF6_flows = np.array([50, 55, 60, 65, 70])
    O2_flows = np.ones_like(SF6_flows) * 50
    Si_etch_rates = np.array([1.875, 2.36, 2.52, 2.64, 2.7])
    PR_etch_rates = np.array([0.222, 0.21, 0.2075, 0.2025, 0.2])

    # ---

    x = SF6_flows / O2_flows
    y1a = Si_etch_rates
    y1b = PR_etch_rates
    y2 = Si_etch_rates / PR_etch_rates

    # -

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

    ax1.plot(x, y1a, '-o', color=sciblue, label='Si')
    ax1.plot(x, y1b, '-o', color=scired, label='PR')
    ax1.set_ylabel(r'$E.R. \: (\mu m/min)$')
    ax1.legend(loc='center right')

    ax2.plot(x, y2, '-s', color=sciblack, label='Si:PR Selectivity')
    ax2.set_xlabel(r'$SF_{6}/O_{2} ratio$')
    ax2.set_ylabel('Selectivity')

    plt.suptitle('Si Chip on PR-coated Si Wafer')
    plt.tight_layout()
    plt.show()

# ---

plot_etch_rates_laser_monitor = False
if plot_etch_rates_laser_monitor:

    read_dir = '/Users/mackenzie/Desktop/Zipper/Fabrication/etch-characterization/SF6+O2-etch-char/DSEiii-LaserMon'
    data_dirs = ['s1', 's2', 's3', 's4']

    px = 't'
    pr = 'ref'
    px_min = 15
    px_max = 300

    fig, ax = plt.subplots(figsize=(size_x_inches * 1.25, size_y_inches * 0.75))
    for ddir in data_dirs:

        fp = join(read_dir, ddir, 'data' + '.csv')
        df = pd.read_csv(fp, names=[px, pr])

        # filter
        df = df[(df[px] > px_min) & (df[px] < px_max)]

        # find peaks
        ref = df[pr].to_numpy()
        height = (ref.max() - ref.min()) * 0.8
        min_width = None
        distance = None
        prominence = 0.95
        rel_height = 0.95

        peaks, peak_properties = find_peaks(ref, height=height, width=min_width, distance=distance,
                                            prominence=prominence,
                                            rel_height=rel_height)

        t_pks = []
        for k in range(len(peaks) - 1):
            t_pks.append(df.iloc[peaks[k + 1]].t - df.iloc[peaks[k]].t)
        period_pks = np.mean(t_pks)

        # plot each peak for each laser mon
        """
        figg, axx = plt.subplots()
        sampling_rate = 1
        lbl = None
        axx.plot(df[px], df[pr], '-o', ms=1)
        for k, pk in enumerate(peaks):
            axx.axvline(df.iloc[pk].t, linewidth=0.5, linestyle='--', color='r', alpha=0.25, label=lbl)
        axx.set_title(ddir)
        figg.show()
        """


        df[px] = df[px] - df.iloc[peaks[0]][px]

        ax.plot(df[px], df[pr], label='{}: {} s'.format(ddir, np.round(period_pks, 2)))

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()