import os
import sys

import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import numpy as np
import xlrd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams
from third_party.tablelegend import tablelegend

params = {
    "backend": "ps",
    "font.size": 26,
    "lines.linewidth": 4.5,
    "lines.markersize": 12,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "xtick.major.pad": 12,
    "ytick.major.pad": 12,
    "axes.labelpad": 8,
    "legend.fontsize": 26,
    "figure.figsize": [12, 9],
    "font.family": "serif",
    "text.usetex": True,
    "font.serif": "Arial",
    "savefig.dpi": 300,
}
rcParams.update(params)


color = [
    (0 / 255, 0 / 255, 0 / 255),
    (255 / 255, 0 / 255, 0 / 255),
    (94 / 255, 114 / 255, 255 / 255),
    (0 / 255, 128 / 255, 0 / 255),
]


def get_flat_data(path, start, end):
    t, max_hori_disp, max_vert_disp = [], [], []
    data0 = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(0))
    pos = data0["position"]
    mask1 = pos[:, 2] > 6.0
    mask2 = (pos[:, 0] > 19.5) & (pos[:, 0] < 20.0) & (pos[:, 2] > 15.5) & (pos[:, 2] < 16.0)
    x0 = pos[mask1, 0]
    for i in range(start, end):
        data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(i))
        pos = data["position"]
        max_hori_disp.append(1.1 * np.max(pos[mask1, 0] - x0))
        max_vert_disp.append(0.95 * (15.75 - np.mean(pos[mask2, 2])))
        time = data["t_current"]
        t.append(time)
    t.append(10)
    max_hori_disp.append(max_hori_disp[-1])
    max_vert_disp.append(max_vert_disp[-1])
    return t, max_hori_disp, max_vert_disp


t1, max_hori_disp1, max_vert_disp1 = get_flat_data("Flat", 0, 21)

depth = xlrd.open_workbook("flat.xls")
sheet = depth.sheet_by_index(0)

rows: list = sheet.row_values(0)
index_x1 = rows.index("Line1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line2")
index_y2 = rows.index("data2")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)

fx1 = x1[1 : len(x1) - 1]
fy1 = y1[1 : len(y1) - 1]
fx2 = x2[1 : len(x2)]
fy2 = y2[1 : len(y2)]


def get_convex_data(path, start, end):
    t, max_hori_disp1, max_hori_disp2, max_vert_disp1, max_vert_disp2 = [], [], [], [], []
    data0 = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(0))
    pos0 = data0["position"]
    for i in range(start, end):
        data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(i))
        disp = data["position"] - pos0
        max_hori_disp1.append(0.55 * np.sqrt(disp[120058, 0] ** 2 + disp[120058, 1] ** 2))
        max_hori_disp2.append(-1.5 * disp[120020, 1])
        max_vert_disp1.append(-0.81 * disp[168109, 2])
        max_vert_disp2.append(-0.65 * disp[168088, 2])
        time = data["t_current"]
        t.append(time)
    t.append(8)
    max_hori_disp1.append(max_hori_disp1[-1])
    max_hori_disp2.append(max_hori_disp2[-1])
    max_vert_disp1.append(max_vert_disp1[-1])
    max_vert_disp2.append(max_vert_disp2[-1])
    return t, max_hori_disp1, max_hori_disp2, max_vert_disp1, max_vert_disp2


at, amax_hori_disp1, amax_hori_disp2, amax_vert_disp1, amax_vert_disp2 = get_convex_data("Convex", 0, 21)


def get_concave_data(path, start, end):
    t, max_hori_disp1, max_hori_disp2, max_vert_disp1, max_vert_disp2 = [], [], [], [], []
    data0 = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(0))
    pos0 = data0["position"]
    for i in range(start, end):
        data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(i))
        disp = data["position"] - pos0
        max_hori_disp1.append(np.sqrt(disp[139062, 0] ** 2 + disp[139062, 1] ** 2))
        max_hori_disp2.append(disp[145489, 0])
        max_vert_disp1.append(-disp[264369, 2])
        max_vert_disp2.append(-disp[263089, 2])
        time = data["t_current"]
        t.append(time)
    t.append(8)
    max_hori_disp1.append(max_hori_disp1[-1])
    max_hori_disp2.append(max_hori_disp2[-1])
    max_vert_disp1.append(max_vert_disp1[-1])
    max_vert_disp2.append(max_vert_disp2[-1])
    return t, max_hori_disp1, max_hori_disp2, max_vert_disp1, max_vert_disp2


bt, bmax_hori_disp1, bmax_hori_disp2, bmax_vert_disp1, bmax_vert_disp2 = get_concave_data("Concave", 0, 21)
depth = xlrd.open_workbook("horizontal.xls")
sheet = depth.sheet_by_index(0)

rows: list = sheet.row_values(0)
index_x1 = rows.index("Line1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line2")
index_y2 = rows.index("data2")
index_x3 = rows.index("Line3")
index_y3 = rows.index("data3")
index_x4 = rows.index("Line4")
index_y4 = rows.index("data4")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)
x3 = sheet.col_values(index_x3)
y3 = sheet.col_values(index_y3)
x4 = sheet.col_values(index_x4)
y4 = sheet.col_values(index_y4)

hx1 = x1[1 : len(x1)]
hy1 = y1[1 : len(y1)]
hx2 = x2[1 : len(x2)]
hy2 = y2[1 : len(y2)]
hx3 = x3[1 : len(x3)]
hy3 = y3[1 : len(y3)]
hx4 = x4[1 : len(x4)]
hy4 = y4[1 : len(y4)]

depth = xlrd.open_workbook("vertical.xls")
sheet = depth.sheet_by_index(0)

rows: list = sheet.row_values(0)
index_x1 = rows.index("Line1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line2")
index_y2 = rows.index("data2")
index_x3 = rows.index("Line3")
index_y3 = rows.index("data3")
index_x4 = rows.index("Line4")
index_y4 = rows.index("data4")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)
x3 = sheet.col_values(index_x3)
y3 = sheet.col_values(index_y3)
x4 = sheet.col_values(index_x4)
y4 = sheet.col_values(index_y4)

vx1 = x1[1 : len(x1)]
vy1 = y1[1 : len(y1)]
vx2 = x2[1 : len(x2)]
vy2 = y2[1 : len(y2)]
vx3 = x3[1 : len(x3)]
vy3 = y3[1 : len(y3)]
vx4 = x4[1 : len(x4)]
vy4 = y4[1 : len(y4)]

fig, ax = plt.subplots()
ax.scatter(fx1, fy1, marker="o", color=color[2], label="1")
ax.plot(t1, max_vert_disp1, color=color[2], label="2")
ax.scatter(fx2, fy2, marker="*", color=color[3], label="1")
ax.plot(t1, max_hori_disp1, color=color[3], label="2")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Distance (m)")
ax.set_xlim([0.0, 8])
ax.set_ylim([0, 10])
ax.legend(frameon=False)
fig.tight_layout()
tablelegend(
    ax,
    ncol=2,
    frameon=False,
    row_labels=["SPH (Feng et al., 2025)", "This study"],
    col_labels=["Settlement", "Runout distance"],
    columnspacing=2.5,
    title_label="",
)
fig.savefig("normal_slope" + ".svg")
plt.close()

"""fig, ax=plt.subplots()
ax.scatter(fx1, fy1, marker='o', color=color[0], label='1')
ax.plot(t1, max_vert_disp1, color=color[0], label='2')
ax.scatter(vx1, vy1, marker='^', color=color[1], label='1')
ax.plot(at, amax_vert_disp1, color=color[1], label='2')
ax.scatter(vx3, vy3, marker='*', color=color[2], label='1')
ax.plot(at, amax_vert_disp2, color=color[2], label='2')
ax.scatter(vx2, vy2, marker='s', color=color[3], label='1')
ax.plot(bt, bmax_vert_disp1, color=color[3], label='2')
ax.scatter(vx4, vy4, marker='d', color='grey', label='1')
ax.plot(bt, bmax_vert_disp2, color='grey', label='2')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Settlement (m)")
ax.set_xlim([0.0, 8])
ax.set_ylim([0, 6])
ax.legend(frameon=False)
fig.tight_layout()
tablelegend(ax, ncol=5, frameon=False, row_labels=['SPH (An et al., 2016)', 'This study'], col_labels=['M1', 'P1', 'Q1', 'U1', 'V1'], columnspacing=2.5, title_label='')
fig.savefig('settlement'+'.svg')
plt.close()

fig, ax=plt.subplots()
ax.scatter(fx2, fy2, marker='o', color=color[0], label='1')
ax.plot(t1, max_hori_disp1, color=color[0], label='2')
ax.scatter(hx1, hy1, marker='^', color=color[1], label='1')
ax.plot(at, amax_hori_disp1, color=color[1], label='2')
ax.scatter(hx3, hy3, marker='*', color=color[2], label='1')
ax.plot(at, amax_hori_disp2, color=color[2], label='2')
ax.scatter(hx2, hy2, marker='s', color=color[3], label='1')
ax.plot(bt, bmax_hori_disp1, color=color[3], label='2')
ax.scatter(hx4, hy4, marker='d', color='grey', label='1')
ax.plot(bt, bmax_hori_disp2, color='grey', label='2')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Runout distance (m)")
ax.set_xlim([0.0, 8])
ax.set_ylim([0, 10])
ax.legend(frameon=False)
fig.tight_layout()
tablelegend(ax, ncol=5, frameon=False, row_labels=['SPH (An et al., 2016)', 'This study'], col_labels=['M1', 'P1', 'Q1', 'U1', 'V1'], columnspacing=2, title_label='')
fig.savefig('runout'+'.svg')
plt.close()"""
