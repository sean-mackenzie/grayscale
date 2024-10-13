from os.path import join

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---

base_dir = '/Users/mackenzie/Desktop/Zipper/Fabrication'
fn = 'w{}_merged_dose-depths.xlsx'

# ---

px = 'exposure_dose'
pr = 'exposure_r'
py = 'focus'
pz = 'z'
step = 3

# ---

dfs = []
for wid in np.arange(10, 17):
    df = pd.read_excel(join(base_dir, 'Wafer{}/results'.format(wid), fn.format(wid)))
    df = df[df['step'] == step]
    df['wid'] = wid
    dfs.append(df)

df = pd.concat(dfs)
# df.to_excel(join(base_dir, 'w10-16_merged-dose-depths.xlsx'), index=False)

# ---

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

"""fig, ax = plt.subplots()
sc = ax.scatter(df[px], df[pz], c=df[py], s=1, alpha=0.25)
ax.set_xlabel('Exposure Dose ' + r'$(mJ)$')
ax.set_ylabel('Exposure Depth ' + r'$(\mu m)$')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(sc, cax=cax, label='Focus', ticks=[-25, 0, 25])

plt.tight_layout()
plt.savefig(join(base_dir, 'w10-16_dose-depth_cb-focus.png'))
plt.show()
plt.close()

# ---

fig, ax = plt.subplots()
sc = ax.scatter(df[px], df[pz], c=df[pr], s=1, alpha=0.25)
ax.set_xlabel('Exposure Dose ' + r'$(mJ)$')
ax.set_ylabel('Exposure Depth ' + r'$(\mu m)$')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(sc, cax=cax, label=r'$r \: (\mu m)$')

plt.tight_layout()
plt.savefig(join(base_dir, 'w10-16_dose-depth_cb-r.png'))
plt.show()
plt.close()

# ---

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

sc = ax.scatter(df[px], df[pr], df[pz], c=df[pz], s=1, alpha=0.25)

ax.set_xlabel('Exposure Dose ' + r'$(mJ)$')
ax.set_ylabel(r'$r \: (\mu m)$')
ax.set_zlabel('Exposure Depth ' + r'$(\mu m)$')

plt.tight_layout()
plt.savefig(join(base_dir, 'w10-16_dose-r-depth_cb-z.png'))
plt.show()
plt.close()

# ---

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

sc = ax.scatter(df[px], df[py], df[pz], c=df[pz], s=1, alpha=0.25)

ax.set_xlabel('Exposure Dose ' + r'$(mJ)$')
ax.set_ylabel(r'$Focus \: (a.u.)$')
ax.set_zlabel('Exposure Depth ' + r'$(\mu m)$')

plt.tight_layout()
plt.savefig(join(base_dir, 'w10-16_dose-focus-depth_cb-z.png'))
plt.show()
plt.close()


# ---

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# ax.view_init(elev=0, azim=0)

sc = ax.scatter(df[px], df[py], df[pz], c=df[pz], s=1, alpha=0.25)

ax.set_xlabel('Exposure Dose ' + r'$(mJ)$')
ax.set_ylabel(r'$Focus \: (a.u.)$')
ax.set_zlabel('Exposure Depth ' + r'$(\mu m)$')

plt.tight_layout()
plt.savefig(join(base_dir, 'w10-16_dose-focus-depth_cb-z_view-yz.png'))
plt.show()
plt.close()

# ---

fig, ax = plt.subplots()
sc = ax.scatter(df[pr], df[pz], c=df[px], s=1, alpha=0.25)
ax.set_xlabel(r'$r \: (\mu m)$')
ax.set_ylabel('Exposure Depth ' + r'$(\mu m)$')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(sc, cax=cax, label='Dose (mJ)')

plt.tight_layout()
plt.savefig(join(base_dir, 'w10-16_r-depth_cb-dose.png'))
plt.show()
plt.close()

# ---

fig, ax = plt.subplots(figsize=(size_x_inches / 1.8, size_y_inches))
sc = ax.scatter(df[pr], df[px], c=df[pz].abs(), s=1, alpha=0.2)
ax.set_xlabel(r'$r \: (\mu m)$')
ax.set_ylabel('Exposure Dose ' + r'$(mJ)$')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(sc, cax=cax, label=r'$z \: (\mu m)$')

plt.tight_layout()
plt.savefig(join(base_dir, 'w10-16_r-dose_cb-depth_resized.png'))
plt.show()
plt.close()"""

df = df[df['did'] == 0]

fig, ax = plt.subplots(figsize=(size_x_inches / 1.25, size_y_inches))
i = 1

for wid in df.wid.unique():
    for fid in df.fid.unique():
        dff = df[(df['wid'] == wid) & (df['fid'] == fid)]
        sc = ax.scatter(dff[pr], dff[pz] / dff[pz].min(), c=dff[px], s=0.25, alpha=0.15)

ax.set_xlabel(r'$r \: (\mu m)$')
ax.set_ylabel(r'$z/z_{o}$')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(sc, cax=cax, label='Dose ' + r'$(mJ)$')

plt.tight_layout()
plt.savefig(join(base_dir, 'w10-16_r-normz_cb-dose_n=63.png'))
plt.show()
plt.close()