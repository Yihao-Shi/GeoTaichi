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


def SetToRotate(q):
    qw, qx, qy, qz = q[3], q[0], q[1], q[2]
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


def get_data(path, start, end):
    t, rot = [], []
    for i in range(start, end):
        data = np.load(path + "/LSDEMRigid{0:06d}.npz".format(i))
        dirs = SetToRotate(data["quanternion"][1]).T @ np.array([0, 0, 1])
        dirs = dirs / np.linalg.norm(dirs)
        theta = 90 - np.arccos(np.dot(dirs, np.array([0, 0, 1]))) / math.pi * 180
        rot.append(theta)
        time = data["t_current"]
        t.append(time)
    return t, rot


t, rot = get_data("OutputData/particles", 0, 11)

depth = xlrd.open_workbook("data.xls")
sheet = depth.sheet_by_index(0)

rows: list = sheet.row_values(0)
index_x1 = rows.index("Line #1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line #2")
index_y2 = rows.index("data2")
index_x3 = rows.index("Line #3")
index_y3 = rows.index("data3")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)
x3 = sheet.col_values(index_x3)
y3 = sheet.col_values(index_y3)

x1 = x1[1 : len(x1)]
y1 = y1[1 : len(y1)]
x2 = x2[1 : len(x2)]
y2 = y2[1 : len(y2)]
x3 = x3[1 : len(x3)]
y3 = y3[1 : len(y3)]
fig, ax = plt.subplots()


ax.plot(x3, y3, marker="o", color=color[1], label="Experimental (Liu et al., 2018)", clip_on=False)
ax.plot(x1, y1, marker="*", color=color[0], label="Numerical (Liu et al., 2018)", clip_on=False)
ax.plot(x2, y2, marker="^", color=color[2], label="Numerical (Jiang et al., 2020)", clip_on=False)
ax.plot(t, rot, marker="p", color=color[3], label="This study", clip_on=False)

ax.set_xlabel("Time (s)")
ax.set_ylabel("Orientation ($^\circ$)")
ax.set_xlim([0.0, 0.5])
ax.set_ylim([-10, 100])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("rotation" + ".pdf")
plt.close()
