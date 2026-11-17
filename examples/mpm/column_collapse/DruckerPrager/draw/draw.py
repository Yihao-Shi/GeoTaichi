import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import numpy as np
import math
import matplotlib
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

printNum = 50
length = 0.6
dbin = 100


def get_data(path):
    data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(printNum), allow_pickle=True)
    x = data["position"][:, 0] - 0.005
    y = data["position"][:, 2] - 0.005
    epeff = data["state_vars"].item()["epstrain"]

    bins = np.linspace(0, length, dbin)
    info1 = np.zeros(dbin)
    info2 = np.zeros(dbin)
    for i in range(len(x)):
        loc = math.floor(x[i] / (length / dbin))
        if y[i] > info2[loc]:
            info2[loc] = y[i]
        if epeff[i] < 1e-1:
            if y[i] > info1[loc]:
                info1[loc] = y[i]

    bins1 = bins[info1 > 1e-15]
    info1 = info1[info1 > 1e-15]
    bins2 = bins[info2 > 1e-15]
    info2 = info2[info2 > 1e-15]
    return bins1, info1, bins2, info2


bins_f1, info_f1, bins_s1, info_s1 = get_data("standard")
bins_f2, info_f2, bins_s2, info_s2 = get_data("bbar")
# bins_f3, info_f3, bins_s3, info_s3 = get_data("fbar")


X1 = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18]
Y1 = [0.1, 0.1, 0.0902954, 0.0797468, 0.064557, 0.0535865, 0.0367089, 0.0257384, 0.0109705, 0]
X2 = [
    0,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.12,
    0.14,
    0.16,
    0.18,
    0.2,
    0.22,
    0.24,
    0.26,
    0.28,
    0.3,
    0.32,
    0.34,
    0.36,
    0.38,
    0.4,
    0.42,
    0.44,
    0.46,
    0.48,
    0.5,
]
Y2 = [
    0.1,
    0.1,
    0.0966245,
    0.0898734,
    0.0839662,
    0.0772152,
    0.0721519,
    0.0670886,
    0.0603376,
    0.0535865,
    0.0472574,
    0.0421941,
    0.035443,
    0.0303797,
    0.0265823,
    0.0227848,
    0.0194093,
    0.0151899,
    0.0109705,
    0.00886076,
    0.00421941,
    0.00295359,
    0.0021097,
    0.00126582,
    0.000421941,
    0,
]
fig, ax = plt.subplots()
ax.scatter(X1, Y1, marker="o", color=color[0], label="Failure surface from experiment")
ax.scatter(X2, Y2, marker="s", color="grey", label="Free surface from experiment")
ax.plot(bins_f1, info_f1, linestyle="--", color=color[1], label="Standard GIMP")
ax.plot(bins_s1, info_s1, linestyle="--", color=color[1])
ax.plot(bins_f2, info_f2, linestyle="-.", color=color[2], label="GIMP with B-bar stabilization")
ax.plot(bins_s2, info_s2, linestyle="-.", color=color[2])
# ax.plot(bins_f3, info_f3, linestyle=':', color=color[3], label='F-bar GIMP')
# ax.plot(bins_s3, info_s3, linestyle=':', color=color[3])
ax.set_xlim([0, 0.6])
ax.set_ylim([0, 0.2])
ax.set_xlabel("$x$ (m)")
ax.set_ylabel("$y$ (m)")
ax.legend(frameon=False, loc="best")
fig.tight_layout()
fig.savefig("fig.eps")
plt.close()

t = [0, 0.18, 0.283, 0.387, 0.551, 1.0]
x = [0.2, 0.25, 0.307, 0.368, 0.444, 0.525]
start = 0
end = 61


def get_runout(path):
    ts = []
    xs = []
    for printNum in range(start, end):
        data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(printNum), allow_pickle=True)
        xpos = data["position"][:, 0] - 0.005
        xs.append(np.max(xpos))
        ts.append(data["t_current"] / 0.6)
    return xs, ts


xss, tss = get_runout("standard")
xsb, tsb = get_runout("bbar")
fig, ax = plt.subplots()
ax.scatter(t, x, marker="s", color="black", label="Experimental data")
ax.plot(tss, xss, linestyle="--", color=color[1], label="Standard GIMP")
ax.plot(tsb, xsb, linestyle="-.", color=color[2], label="GIMP with B-bar stabilization")
ax.set_xlim([0, 1.0])
ax.set_ylim([0.2, 0.55])
ax.set_xlabel("Normalized time, $T$")
ax.set_ylabel("Runout distance (m)")
ax.legend(frameon=False, loc="best")
fig.tight_layout()
fig.savefig("runout.eps")
plt.close()
