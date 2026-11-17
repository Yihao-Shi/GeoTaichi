import numpy as np
import math
import xlrd
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

dt = 1e-4
time = 0.3


def read_file1(path):
    data = []
    with open(path, "r") as f:
        for ann in f.readlines():
            data.append(time / dt / float(ann.strip("\n")))
            # print(float(ann))
    return data


depth = xlrd.open_workbook("CoSim.xls")
sheet = depth.sheet_by_index(0)

rows = sheet.row_values(0)
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

xaxis = [35828, 128000, 432000, 1024000]
data1 = [95.42359352111816, 340.94364404678345, 1689.6237671375275, 5867.148678064346]
data1 = [
    data1[0] / (0.4 / (1e-5) / 1000),
    data1[1] / (0.4 / (7.26e-6) / 1000),
    data1[2] / (0.4 / (3.95e-6) / 1000),
    data1[3] / (0.4 / (2.56e-6) / 1000),
]

plt.plot(xaxis, data1, marker="o", linestyle="-", color=color[0], label="GeoTaichi (GeForce RTX 3070)")
plt.plot(x1, y1, marker="^", linestyle="--", color=color[1], label="CoSim (Tesla V100)")
plt.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
plt.xlim([30000, 1500000])
plt.ylim([1, 1000])
plt.xscale("log")
plt.yscale("log")
plt.text(35000, 10, "Average speedup: 4.93")
plt.annotate(
    "",
    xy=(128000, 32.9),
    xytext=(128000, 6.3),
    arrowprops=dict(facecolor="blue", edgecolor="blue", arrowstyle="<-", linewidth=2),
)
plt.annotate(
    "",
    xy=(432000, 89.6),
    xytext=(432000, 17.6),
    arrowprops=dict(facecolor="blue", edgecolor="blue", arrowstyle="<-", linewidth=2),
)
plt.annotate(
    "",
    xy=(1024000, 212.5),
    xytext=(1024000, 39.5),
    arrowprops=dict(facecolor="blue", edgecolor="blue", arrowstyle="<-", linewidth=2),
)
plt.xlabel("Number of material points")
plt.ylabel("Run time (s/$10^3$ steps)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("efficiency.eps")
plt.close()
