#!/usr/bin/env python
import numpy as np
import math, xlrd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

params = {
    "backend": "ps",
    "font.size": 26,
    "lines.linewidth": 4.5,
    "lines.markersize": 10,
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

depth = xlrd.open_workbook("experiment.xls")
sheet = depth.sheet_by_index(0)

rows: list = sheet.row_values(0)
index_x1 = rows.index("Line 1")
index_y1 = rows.index("data1")
index_x2 = rows.index("Line 2")
index_y2 = rows.index("data2")
index_x3 = rows.index("Line 3")
index_y3 = rows.index("data3")

x1 = sheet.col_values(index_x1)
y1 = sheet.col_values(index_y1)
x2 = sheet.col_values(index_x2)
y2 = sheet.col_values(index_y2)
x3 = sheet.col_values(index_x3)
y3 = sheet.col_values(index_y3)

x1 = x1[1 : len(x1) - 9]
y1 = y1[1 : len(y1) - 9]
x2 = x2[1 : len(x2) - 0]
y2 = y2[1 : len(y2) - 0]
x3 = x3[1 : len(x3) - 1]
y3 = y3[1 : len(y3) - 1]

start_num = 1
end_num = 41


def macro(path):
    tau = []
    epsilon = []
    shearf0 = 0.0

    for printNum in range(start_num, end_num):
        data = np.load(path + "/DEMWall{0:06d}.npz".format(printNum), allow_pickle=True)
        top_force = -(data["contact_force"][0][2] + data["contact_force"][1][2])
        left_force = -(data["contact_force"][12][0] + data["contact_force"][13][0])
        right_force = data["contact_force"][16][0] + data["contact_force"][17][0]
        up_position = (
            data["point1"][2][2]
            + data["point2"][2][2]
            + data["point3"][2][2]
            + data["point1"][3][2]
            + data["point2"][3][2]
            + data["point3"][3][2]
        ) / 6.0
        left_position = (
            data["point1"][12][0]
            + data["point2"][12][0]
            + data["point3"][12][0]
            + data["point1"][13][0]
            + data["point2"][13][0]
            + data["point3"][13][0]
        ) / 6.0

        if printNum == 1:
            shearf0 = (left_force - right_force) / ((up_position - 0.1225) * 0.3) / 1000

        tau.append((left_force - right_force) / ((up_position - 0.1225) * 0.3) / 1000 - shearf0)
        epsilon.append((left_position - 0.3) / 0.145 * 100)
    return tau, epsilon


tau1, epsilon1 = macro("50kPa/shear/walls")
tau2, epsilon2 = macro("100kPa/shear/walls")
tau3, epsilon3 = macro("200kPa/shear/walls")

plt.plot(epsilon1, tau1, linestyle="-", color=color[0], label="50kPa")
plt.scatter(x1, y1, color=color[0])
plt.plot(epsilon2, tau2, linestyle="--", color=color[1], label="100kPa")
plt.scatter(x2, y2, color=color[1])
plt.plot(epsilon3, tau3, linestyle="-.", color=color[2], label="200kPa")
plt.scatter(x3, y3, color=color[2])
plt.xlim([0, 20])
plt.ylim([0.0, 250])
plt.xlabel("Shear strain, $\\gamma$ (\%)")
plt.ylabel("Shear stress, $\\tau$")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("stress_strain.svg")
plt.close()
