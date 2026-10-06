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

rho0 = [700.0, 2200.0]
mu0 = [0.3, 0.5]
h0 = [0.05, 0.1, 0.2]


def get_data(path):
    x = []
    y = []
    for rho in rho0:
        for mu in mu0:
            for h in h0:
                try:
                    data = np.load(path + f"Rho_{rho}_Mu_{mu}_H_{h}_0/particles/LSDEMRigid{1:06d}.npz")
                except:
                    continue
                zpos = data["mass_center"][0][2]
                d = 0.06 - (zpos - 0.0125)
                normalized_x = (rho / 1510) ** (0.5) * (2.0 * 0.0125) ** (2.0 / 3.0) * (h + d) ** (1.0 / 3.0)
                x.append(normalized_x)
                y.append(d)
    print(x, y)
    return x, y


numx, numy = get_data("")
anax = np.linspace(0.0, 0.08, 200)
anay0 = 0.14 / mu0[0] * anax
anay1 = 0.14 / mu0[1] * anax


fig, ax = plt.subplots()
ax.plot(anax, anay0, linestyle="-.", color=color[0])
ax.plot(anax, anay1, linestyle="-.", color=color[0])
ax.scatter(numx, numy, marker="o", color=color[2], label="GeoTaichi")
ax.set_xlabel("$(\\frac{\\rho_{sphere}}{\\rho_{granular}})^{1/2}D_{sphere}^{2/3}H_{drop}^{1/3}$")
ax.set_ylabel("d (m)")
# ax.set_xlim([0.2, 0.4])
# ax.set_ylim([0, 100])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("rotation" + ".eps")
plt.close()
