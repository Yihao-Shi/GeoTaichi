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


def LaceyIndex(path):
    t, mixing = [], []
    bin_size = 0.02
    for i in range(0, 51):
        data = np.load(path + "LSDEMRigid{0:06d}.npz".format(i))
        positions = data["mass_center"]
        print(positions.shape)
        group_ids = data["groupID"]
        assert positions.shape[0] == group_ids.shape[0]

        mins = np.array([0, 0, 0])
        maxs = np.array([0.2, 0.2, 0.2])
        num_bins = np.ceil((maxs - mins) / bin_size).astype(int)

        bin_indices = ((positions - mins) / bin_size).astype(int)
        bin_keys = [tuple(idx) for idx in bin_indices]

        from collections import defaultdict

        bin_total = defaultdict(int)
        bin_group0 = defaultdict(int)

        for key, gid in zip(bin_keys, group_ids):
            bin_total[key] += 1
            if gid == 0:
                bin_group0[key] += 1

        x_list = []
        for key in bin_total:
            if bin_total[key] > 0:
                frac = bin_group0[key] / bin_total[key]
                x_list.append(frac)

        x = np.array(x_list)
        x_bar = 0.5
        var_mixed = np.mean((x - x_bar) ** 2)
        var_unmixed = x_bar * (1 - x_bar)

        LMI = 1 - var_mixed / var_unmixed
        mixing.append(LMI)
        t.append(data["t_current"] / 3.0)
    return t, mixing


t1, mixing1 = LaceyIndex("20rpm/particles/")

t3 = [0.0, 0.497, 1.01, 1.51, 1.99, 2.52, 3.0, 3.49, 4.01, 4.5, 5.0]
mixing3 = [0, 0.173, 0.398, 0.542, 0.787, 0.92, 0.92, 0.96, 0.98, 0.972, 0.96]
t4 = [
    0.00e00,
    2.02e-01,
    2.48e-01,
    4.04e-01,
    6.83e-01,
    7.76e-01,
    9.78e-01,
    1.09e00,
    1.30e00,
    1.48e00,
    1.63e00,
    1.75e00,
    2.05e00,
    2.31e00,
    2.44e00,
    2.50e00,
    2.62e00,
    2.80e00,
    3.00e00,
    3.09e00,
    3.39e00,
    3.68e00,
    3.79e00,
    3.91e00,
    4.15e00,
    4.33e00,
    4.38e00,
    4.53e00,
    4.70e00,
    4.97e00,
]
mixing4 = [
    0.00e00,
    1.08e-01,
    1.20e-01,
    1.33e-01,
    2.29e-01,
    2.53e-01,
    3.33e-01,
    4.06e-01,
    4.90e-01,
    5.62e-01,
    6.67e-01,
    6.67e-01,
    7.99e-01,
    8.80e-01,
    8.88e-01,
    8.92e-01,
    9.04e-01,
    9.28e-01,
    9.28e-01,
    9.44e-01,
    9.32e-01,
    9.72e-01,
    9.60e-01,
    9.80e-01,
    9.44e-01,
    9.68e-01,
    9.60e-01,
    9.92e-01,
    9.60e-01,
    9.56e-01,
]
plt.scatter(t3, mixing3, color=color[0], label="Experiment")
plt.plot(t4, mixing4, color=color[2], label="Super-ellipsoid (Ma $et\ al$., 2017)")
plt.plot(t1, mixing1, color=color[3], label="This study")
plt.xlabel("Revolutions")
plt.ylabel("Mixing Index")
plt.xlim([0, 5])
plt.ylim([0, 1])
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig("index.pdf")
plt.close()

x1 = [
    1.85e-02,
    2.63e-02,
    3.41e-02,
    5.07e-02,
    6.24e-02,
    8.63e-02,
    9.95e-02,
    1.13e-01,
    1.29e-01,
    1.46e-01,
    1.56e-01,
    1.68e-01,
    1.82e-01,
    1.91e-01,
]
y1 = [
    4.64e-02,
    4.74e-02,
    5.13e-02,
    5.38e-02,
    5.82e-02,
    7.18e-02,
    8.61e-02,
    1.00e-01,
    1.15e-01,
    1.33e-01,
    1.39e-01,
    1.45e-01,
    1.52e-01,
    1.50e-01,
]
x2 = [
    2.98e-02,
    4.49e-02,
    5.78e-02,
    7.06e-02,
    8.80e-02,
    1.06e-01,
    1.20e-01,
    1.33e-01,
    1.50e-01,
    1.70e-01,
    1.82e-01,
]
y2 = [
    4.33e-02,
    4.71e-02,
    5.10e-02,
    5.86e-02,
    6.92e-02,
    8.59e-02,
    1.05e-01,
    1.19e-01,
    1.34e-01,
    1.48e-01,
    1.48e-01,
]
x3 = [
    3.36e-02,
    5.25e-02,
    6.54e-02,
    9.03e-02,
    1.13e-01,
    1.35e-01,
    1.55e-01,
    1.75e-01,
    1.82e-01,
]
y3 = [
    3.65e-02,
    3.80e-02,
    4.71e-02,
    6.92e-02,
    9.66e-02,
    1.21e-01,
    1.40e-01,
    1.50e-01,
    1.51e-01,
]
plt.plot(x2, y2, color=color[0], linestyle="--", label="Experiment (Ma $et\ al$., 2017)")
plt.plot(x3, y3, color=color[2], label="Super-ellipsoid (Ma $et\ al$., 2017)")
plt.plot(x1, y1, color=color[3], label="This study")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
# plt.xlim([0,5])
# plt.ylim([0,1])
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig("profile.pdf")
