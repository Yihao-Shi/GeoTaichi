#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

params = {
    "backend": "ps",
    "font.size": 26,
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "xtick.major.pad": 12,
    "ytick.major.pad": 12,
    "axes.labelpad": 8,
    "legend.fontsize": 26,
    "figure.figsize": [12, 9],
    "font.family": "serif",
    "text.usetex": False,
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

particle = np.load("DEM_Compression/particles/DEMParticle{0:06d}.npz".format(0), allow_pickle=True)
radius = particle["radius"]
vol = np.sum(4.0 / 3.0 * math.pi * radius**3)


data = np.loadtxt("time_series.txt")
time = data[:, 0]
ztop = data[:, 1]
vmax = data[:, 2]
totalf = data[:, 3]

total_v = (ztop - 3.35e-6) * 3.481956e-8
fraction = (vol) / (total_v)
pressure = totalf / 3.481956e-8 / 1000000

plt.plot(time, pressure, linestyle="-", marker="o", color=color[0], label="Wall stress")
plt.xlabel("Time, $T$ (s)")
plt.ylabel("$\\sigma$ / $\\sigma_{target}$")
plt.legend(frameon=False)
plt.savefig("stress_strain.png")
plt.close()

plt.plot(time, fraction, linestyle="-", color=color[0], label="case 1")
plt.xlabel("Time, $T$ (s)")
plt.ylabel("$\\phi$")
plt.legend(frameon=False)
plt.savefig("fraction.png")
plt.close()
