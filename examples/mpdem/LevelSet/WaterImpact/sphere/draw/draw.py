import numpy as np
import xlrd
import matplotlib.pyplot as plt
from matplotlib import rcParams

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


def get_data0(path, start, end):
    t, disp = [], []
    for i in range(start, end):
        data = np.load(path + "DEMParticle{0:06d}.npz".format(i))
        disp.append(0.1327 - data["position"][0][2])
        time = data["t_current"]
        t.append(time)
    return t, disp


def get_data(path, start, end):
    t, disp = [], []
    for i in range(start, end):
        data = np.load(path + "LSDEMRigid{0:06d}.npz".format(i))
        disp.append(0.1327 - data["mass_center"][0][2])
        time = data["t_current"]
        t.append(time)
    return t, disp


t1, disp1 = get_data("10d/particles/", 0, 60)
t2, disp2 = get_data("5d/particles/", 0, 60)
t3, disp3 = get_data("3d/particles/", 0, 60)
t4, disp4 = get_data("12d/particles/", 0, 60)
t4, disp4 = get_data0("OutputData/particles/", 0, 60)

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
x2 = x2[1 : len(x2) - 11]
y2 = y2[1 : len(y2) - 11]
x3 = x3[1 : len(x3) - 7]
y3 = y3[1 : len(y3) - 7]

fig, ax = plt.subplots()
ax.plot(x1, y1, color=color[0], label="Analytical solution (Aristoff et al., 2010)")
ax.scatter(x2, y2, marker="o", c="none", edgecolors=color[1], label="Experiment (Aristoff et al., 2010)")
ax.plot(x3, y3, linestyle="--", color=color[2], label="SPH-DEM (Zhou et al., 2010)")
ax.plot(t4, disp4, linestyle="-.", color=color[3], label="MPM-DEM")
ax.plot(t4, disp4, linestyle=":", color="purple", label="MPM-LSDEM")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Vertical displacement (cm)")
ax.set_xlim([0.0, 0.06])
ax.set_ylim([0.0, 0.08])
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("displacement" + ".svg")
plt.close()

fig, ax = plt.subplots()
ax.plot(x1, y1, color=color[0], label="Analytical solution (Aristoff et al., 2010)")
ax.scatter(x2, y2, marker="o", c="none", edgecolors=color[1], label="Experiment (Aristoff et al., 2010)")
ax.plot(t4, disp4, color=color[3], label="d / $\\Delta$ = 12")
ax.plot(t1, disp1, color="purple", label="d / $\\Delta$ = 10")
ax.plot(t2, disp2, color="grey", label="d / $\\Delta$ = 5")
ax.plot(t3, disp3, color="blue", label="d / $\\Delta$ = 3")

ax.set_xlabel("Time (s)")
ax.set_ylabel("Vertical displacement (cm)")
ax.set_xlim([0.0, 0.06])
ax.set_ylim([0.0, 0.08])
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig("resolution" + ".svg")
plt.close()
