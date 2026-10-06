import os
import sys

import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import numpy as np
import xlrd
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


def get_data(path, start, end):
    t, disp = [], []
    for i in range(start, end):
        data = np.load(path + "LSDEMRigid{0:06d}.npz".format(i))
        disp.append(data["mass_center"][0][0] - 3.5)
        time = data["t_current"]
        t.append(time)
    return t, disp


t, disp = get_data("", 0, 50)

depth = xlrd.open_workbook("data.xls")
sheet = depth.sheet_by_index(0)

rows: list = sheet.row_values(0)
index_x1 = rows.index("Line1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line2")
index_y2 = rows.index("data2")
index_x3 = rows.index("Line3")
index_y3 = rows.index("data3")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)
x3 = sheet.col_values(index_x3)
y3 = sheet.col_values(index_y3)

x1 = x1[1 : len(x1)]
y1 = y1[1 : len(y1)]
x2 = x2[1 : len(x2) - 6]
y2 = y2[1 : len(y2) - 6]
x3 = x3[1 : len(x3) - 8]
y3 = y3[1 : len(y3) - 8]
fig, ax = plt.subplots()

ax.scatter(x2, y2, marker="o", c="none", edgecolors=color[1], label="Experiment (Canelas et al., 2016)")
ax.plot(x1, y1, color=color[0], label="SPH result (Canelas et al., 2016)")
ax.plot(x3, y3, color=color[2], label="SPH result (Sun et al., 2023)")
ax.plot(t, disp, marker="^", color=color[3], label="This study")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Particle position at $x$-axis (m)")
# ax.set_xlim([0.05, 0.55])
# ax.set_ylim([-10, 100])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("displacement" + ".pdf")
plt.close()
