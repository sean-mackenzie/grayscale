import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from graycart.utils.process import bin_by_column

base_dir = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer28/KLATencor-P7/'

fp0 = 'step8_Strip/a1.txt'
fp1 = 'step8_Strip/3d/a1_rot45.txt'
fp2 = 'step8_Strip/3d/a1_rot90.txt'

dfs = []
# fig, ax = plt.subplots()
ax = plt.figure().add_subplot(projection='3d')

for fp, deg in zip([fp0, fp1, fp2], [0, 45, 90]):
    df = pd.read_csv(base_dir + fp, names=['x', 'z'])

    df['xx'] = (df['x'] - df['x'].mean()) * np.cos(np.deg2rad(deg))
    df['yy'] = (df['x'] - df['x'].mean()) * np.sin(np.deg2rad(deg))

    # ax.plot(df.xx, df.yy, df.z, 'o', ms=1, label=fp)

    dfs.append(df)

dfs = pd.concat(dfs)

# filter
dfs = dfs[np.abs(dfs['x'] - dfs['x'].mean()) < 2350]

x = dfs['xx'].to_numpy()
y = dfs['yy'].to_numpy()
z = dfs['z'].to_numpy() * 2.5e-2

# Create a custom triangulation.
triang = tri.Triangulation(x, y)
# ax.tricontourf(triang, z, cmap=plt.cm.CMRmap)


X = np.arange(x.min(), x.max(), 5)
Y = np.arange(y.min(), y.max(), 5)
X, Y = np.meshgrid(X, Y)

tri_interp_less_good = False
if tri_interp_less_good:
    cubic_triang = tri.CubicTriInterpolator(triang, z, kind='min_E')  # 'geom'
    linear_triang = tri.LinearTriInterpolator(triang, z)

    # Z = cubic_triang(X, Y)
    Z = linear_triang(X, Y)

    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    surf = ax.plot_surface(X, Y, Z, cmap='RdBu', alpha=0.85, antialiased=False)

tri_interp_more_good = True
if tri_interp_more_good:
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(z) * 1.25))
    X, Y = X.flatten(), Y.flatten()
    tri = tri.Triangulation(X, Y)
    ax.plot_trisurf(triang, z, cmap=plt.cm.Spectral)


# bin to average
# dfs = bin_by_column(dfs, column_to_bin='x', number_of_bins=int(len(dfs) / 3), round_to_decimal=2)
# dfg = dfs.groupby('bin').mean()
# ax.plot(dfg.x, dfg.z, 'o', ms=1, label='Bin')
# ax.legend(bbox_to_anchor=(1, 1))

ax.view_init(35, 120)  # height, rotation
# ax.set_zlim(top=15000, bottom=-35000)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])
# plt.tight_layout()
plt.show()

j = 1