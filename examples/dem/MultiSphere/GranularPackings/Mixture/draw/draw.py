#!/usr/bin/env python
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial.transform import Rotation as R

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

start_num = 0
end_num = 100
kn = kt = 2e5


def macro(path):
    ek = []
    ep = []
    total = []
    t = []

    for printNum in range(start_num, end_num):
        data = np.load(path + "particles/DEMClump{0:06d}.npz".format(printNum), allow_pickle=True)

        mass = data["mass"]
        v = data["velocity"]
        w = data["omega"]
        q = data["quanternion"]
        r = R.from_quat(q)

        wl = w.copy()

        v = data["velocity"]
        w = data["omega"]
        inertia = 1.0 / data["inverse_inertia"]
        kinetic = np.sum(0.5 * mass * (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2])) + np.sum(
            0.5
            * (
                inertia[:, 0] * wl[:, 0] * wl[:, 0]
                + inertia[:, 1] * wl[:, 1] * wl[:, 1]
                + inertia[:, 2] * wl[:, 2] * wl[:, 2]
            )
        )

        potential = np.sum(mass * 9.8 * (data["centerOfMass"][:, 2]))

        ek.append(kinetic / 1e6)
        ep.append(potential / 1e6)
        total.append(kinetic / 1e6 + potential / 1e6)
        t.append(data["t_current"])
    return ek, ep, total, t


ek, ep, total, t = macro("")
plt.plot(t, total, color=color[0], label="Total mechanical energy")
plt.plot(t, ek, color=color[1], label="Kinetic energy")
plt.plot(t, ep, color=color[2], label="Potential energy")
# plt.xlim([0, 3.0])
# plt.ylim([0., 2.0])
plt.xlabel("Time (s)")
plt.ylabel("Energy of particles [$10^3$ KJ]")
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig("energy.pdf")
