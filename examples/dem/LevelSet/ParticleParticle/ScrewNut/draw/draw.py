import numpy as np
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


Iz = 1081.66
m = 105.59088124485913

ana = []
num = []
time = []


def get_data(path, start, end):
    for i in range(start, end):
        data = np.load(path + "LSDEMRigid{0:06d}.npz".format(i))
        pomega = data["omega"][0][1]
        t = data["t_current"]
        omega_z = (2.0 * np.pi * m * 10 * 1.3 * t) / (4.0 * np.pi * np.pi * Iz + m * 1.3 * 1.3)

        ana.append(omega_z)
        num.append(pomega)
        time.append(t)


get_data("particles/", 0, 60)
plt.plot(time, ana, color=color[0], label="Analytical solution")
plt.scatter(time, num, edgecolors=color[3], facecolors="none", linewidths=2, label="This study")
plt.xlabel("Time (s)")
plt.ylabel("Rotational velocity (rad/s)")
plt.xlim([0.0, 14])
plt.ylim([0.0, 3])
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("spin" + ".svg")
plt.close()
