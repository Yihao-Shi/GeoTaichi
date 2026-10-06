import numpy as np
import math
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

T = []
LT = []
for printNum in range(23):
    data = np.load("MPMParticle{0:06d}.npz".format(printNum))
    pos_x = data["position"][:, 0] - 0.05
    x = np.max(pos_x)
    LT.append(x / 0.6)
    t = printNum * 0.02
    T.append(t * math.sqrt(9.8 / 0.6))

x1, y1 = [], []
x2, y2 = [], []
x3, y3 = [], []
with open("Line1.txt") as file:
    data = file.readlines()
    for i in range(len(data)):
        x1.append(eval(data[i][0:21]))
        y1.append(eval(data[i][25:46]))

with open("Line2.txt") as file:
    data = file.readlines()
    for i in range(len(data)):
        x2.append(eval(data[i][0:21]))
        y2.append(eval(data[i][25:46]))

with open("Line3.txt") as file:
    data = file.readlines()
    for i in range(len(data)):
        x3.append(eval(data[i][0:21]))
        y3.append(eval(data[i][25:46]))

plt.plot(x2, y2, color=color[0], linestyle="-", label="Experiment (Lobovsky $et\ al.$, 2014)")
plt.plot(x3, y3, color=color[2], linestyle="-.", label="SPH (Zhang $et\ al.$, 2017)")
plt.plot(x1, y1, color=color[1], linestyle="--", label="MPM (Sun $et\ al.$, 2018)")
plt.plot(T, LT, color=color[3], linestyle=":", label="MPM (GeoTaichi)")
plt.xlim([0, 1.6])
plt.ylim([1, 2.8])
plt.xlabel("Normalized time")
plt.ylabel("Normalized position of water front")
plt.tight_layout()
plt.legend(loc="best")
plt.legend(frameon=False)
plt.savefig("fig.eps")
