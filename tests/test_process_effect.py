from os.path import join
import pandas as pd
import matplotlib.pyplot as plt

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

base_dir = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer18/results'
d_fn = 'w18_merged_process_profiles.xlsx'
dff = pd.read_excel(join(base_dir, d_fn))

"""
steps = [4, 5]
fid = 0

for step in steps:
    df1 = dff[(dff['step'] == step - 1) & (dff['fid'] == fid)]
    df2 = dff[(dff['step'] == step) & (dff['fid'] == fid)]

    x_new, y1, y2 = process.uniformize_x_dataframes(dfs=[df1, df2], xy_cols=['r', 'z'], num_points=None, sampling_rate=10)
    dy = y1 - y2

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 2]})

    ax1.plot(df1.r, df1.z, '.', ms=0.5, alpha=0.5)
    ax1.plot(df2.r, df2.z, '.', ms=0.5, alpha=0.5)
    ax1.plot(x_new, y1, linewidth=0.5, color='b', label=step-1)
    ax1.plot(x_new, y2, linewidth=0.5, color='darkgreen', label=step)
    ax1.legend(title='Step')

    ax2.plot(x_new, dy, label=r'$\Delta_{ij}$')
    ax2.grid(alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.show()
"""

# filter
step = 7
df = dff[dff['step'] == step]

# plot
fids = df.fid.unique()
z_offset = -2.55
dx = 4200
z_flip = 1


fig, ax = plt.subplots(figsize=(size_x_inches * 3.5 / 2, size_y_inches * 0.4 / 2 * 2))

for i, fid in enumerate(fids):
    dff = df[df['fid'] == fid]
    dff['z'] = dff['z'].where(dff['z'] < z_offset, z_offset)

    if z_flip == -1:
        # ax.plot((dff.r + i * 5000 - 5000) * 1e-3, (dff.z - z_offset) * z_flip, color='k')
        ax.fill_between((dff.r + i * dx - dx) * 1e-3, (dff.z - z_offset) * z_flip, 75,
                        color='silver')
    else:
        # ax.plot((dff.r + i * dx - dx) * 1e-3, (dff.z - z_offset) * z_flip, color='k')
        ax.fill_between((dff.r + i * dx - dx) * 1e-3, -75, (dff.z - z_offset) * z_flip, color='silver')

# fill in plot
# ax.plot([-2.6, -2.4], [0, 0], color='k')
# ax.plot([2.6, 2.4], [0, 0], color='k')

ax.set_xlim([-(dx + dx / 2) / 1000, (dx + dx / 2) / 1000])
ax.set_xticks([-dx / 1000, 0, dx / 1000], labels=[-4, 0, 4])
ax.set_xlabel(r'$x \: (mm)$')


if z_flip == -1:
    ax.set_ylim([-10, 75])
    ax.set_yticks([0, 50])
else:
    ax.set_ylim([-75, 10])
    ax.set_yticks([-50, 0])

ax.set_ylabel(r'$z \: (\mu m)$')


plt.tight_layout()
plt.savefig(join(base_dir, 'figs/array_profile.png'))
# plt.show()
j = 1