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


def get_data(path, end=1):
    data0 = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(0), allow_pickle=True)
    pos0 = data0["position"]
    data = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(end), allow_pickle=True)
    epdstrain = data["state_vars"].item()["epdstrain"][pos0[:, 2] > 6]
    disp = np.linalg.norm(data["position"] - pos0, axis=1)[pos0[:, 2] > 6]
    volume = 0.5**3 * np.sum((disp > 0.15))
    return volume


def anisotropy():
    volume = []
    get_data("Flat")
    volume.append(get_data("Flat332_0_r1.0_a0.0", end=20))
    volume.append(get_data("Flat332_0_r1.0_a30.0", end=20))
    volume.append(get_data("Flat332_0_r1.0_a45.0", end=20))
    volume.append(get_data("Flat332_0_r1.0_a60.0", end=20))
    volume.append(get_data("Flat332_0_r1.0_a90.0", end=20))
    volume.append(get_data("Flat332_0_r1.0_a120.0", end=20))
    volume.append(get_data("Flat332_0_r1.0_a135.0", end=20))
    volume.append(get_data("Flat332_0_r1.0_a150.0", end=20))
    volume.append(get_data("Flat332_0_r1.0_a0.0", end=20))

    x = [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0]
    plt.plot(x, volume, color=color[3], marker="^", markersize=20)
    plt.xlabel("Angle ($\\circ$)")
    plt.ylabel("Sliding volume (m$^3$)")
    # plt.xlim([0.0, 180])
    # plt.ylim([3800, 4400])
    # plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("volume_anisotropy" + ".svg")
    plt.close()


def shape0():
    volume1, volume2, volume3 = [], [], []
    volume4, volume5, volume6 = [], [], []
    volume7, volume8, volume9 = [], [], []

    volume1.append(get_data("Flat108_0_r1.0", end=20))
    volume1.append(get_data("Flat108_0_r1.0_rand1"))
    volume1.append(get_data("Flat108_0_r1.0_rand2"))
    volume1.append(get_data("Flat108_0_r1.0_rand3"))
    volume1.append(get_data("Flat108_0_r1.0_rand4"))
    volume1.append(get_data("Flat108_0_r1.0_rand5"))

    volume2.append(get_data("Flat108_1_r1.0", end=20))
    volume2.append(get_data("Flat108_1_r1.0_rand1"))
    volume2.append(get_data("Flat108_1_r1.0_rand2"))
    volume2.append(get_data("Flat108_1_r1.0_rand3"))
    volume2.append(get_data("Flat108_1_r1.0_rand4"))
    volume2.append(get_data("Flat108_1_r1.0_rand5"))

    volume3.append(get_data("Flat108_2_r1.0", end=20))
    volume3.append(get_data("Flat108_2_r1.0_rand1"))
    volume3.append(get_data("Flat108_2_r1.0_rand2"))
    volume3.append(get_data("Flat108_2_r1.0_rand3"))
    volume3.append(get_data("Flat108_2_r1.0_rand4"))
    volume3.append(get_data("Flat108_2_r1.0_rand5"))

    volume4.append(get_data("Flat216_0_r1.0", end=20))
    volume4.append(get_data("Flat216_0_r1.0_rand1"))
    volume4.append(get_data("Flat216_0_r1.0_rand2"))
    volume4.append(get_data("Flat216_0_r1.0_rand3"))
    volume4.append(get_data("Flat216_0_r1.0_rand4"))
    volume4.append(get_data("Flat216_0_r1.0_rand5"))

    volume5.append(get_data("Flat216_1_r1.0", end=20))
    volume5.append(get_data("Flat216_1_r1.0_rand1"))
    volume5.append(get_data("Flat216_1_r1.0_rand2"))
    volume5.append(get_data("Flat216_1_r1.0_rand3"))
    volume5.append(get_data("Flat216_1_r1.0_rand4"))
    volume5.append(get_data("Flat216_1_r1.0_rand5"))

    volume6.append(get_data("Flat216_2_r1.0", end=20))
    volume6.append(get_data("Flat216_2_r1.0_rand1"))
    volume6.append(get_data("Flat216_2_r1.0_rand2"))
    volume6.append(get_data("Flat216_2_r1.0_rand3"))
    volume6.append(get_data("Flat216_2_r1.0_rand4"))
    volume6.append(get_data("Flat216_2_r1.0_rand5"))

    volume7.append(get_data("Flat332_0_r1.0", end=20))
    volume7.append(get_data("Flat332_0_r1.0_rand1"))
    volume7.append(get_data("Flat332_0_r1.0_rand2"))
    volume7.append(get_data("Flat332_0_r1.0_rand3"))
    volume7.append(get_data("Flat332_0_r1.0_rand4"))
    volume7.append(get_data("Flat332_0_r1.0_rand5"))

    volume8.append(get_data("Flat332_1_r1.0", end=20))
    volume8.append(get_data("Flat332_1_r1.0_rand1"))
    volume8.append(get_data("Flat332_1_r1.0_rand2"))
    volume8.append(get_data("Flat332_1_r1.0_rand3"))
    volume8.append(get_data("Flat332_1_r1.0_rand4"))
    volume8.append(get_data("Flat332_1_r1.0_rand5"))

    volume9.append(get_data("Flat332_2_r1.0", end=20))
    volume9.append(get_data("Flat332_2_r1.0_rand1"))
    volume9.append(get_data("Flat332_2_r1.0_rand2"))
    volume9.append(get_data("Flat332_2_r1.0_rand3"))
    volume9.append(get_data("Flat332_2_r1.0_rand4"))
    volume9.append(get_data("Flat332_2_r1.0_rand5"))

    x = [0.47, 0.78, 1.0]

    F1 = [np.mean(volume1), np.mean(volume2), np.mean(volume3)]
    F2 = [np.mean(volume4), np.mean(volume5), np.mean(volume6)]
    F3 = [np.mean(volume7), np.mean(volume8), np.mean(volume9)]

    E1 = [np.std(volume1, ddof=1), np.std(volume2, ddof=1), np.std(volume3, ddof=1)]
    E2 = [np.std(volume4, ddof=1), np.std(volume5, ddof=1), np.std(volume6, ddof=1)]
    E3 = [np.std(volume7, ddof=1), np.std(volume8, ddof=1), np.std(volume9, ddof=1)]

    plt.errorbar(x, F1, yerr=E1, color=color[0], fmt="o-", capsize=10, capthick=4, label="AR1-F1")
    plt.errorbar(x, F2, yerr=E2, color=color[1], fmt="o-", capsize=10, capthick=4, label="AR1-F2")
    plt.errorbar(x, F3, yerr=E3, color=color[2], fmt="o-", capsize=10, capthick=4, label="AR1-F3")

    plt.xlabel("Aspect ratio")
    plt.ylabel("Sliding volume (m$^3$)")
    # plt.xlim([0.0, 8])
    # plt.ylim([0, 10])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("volume_shape0" + ".svg")
    plt.close()


def shape1():
    volume1, volume2, volume3 = [], [], []
    volume4, volume5, volume6 = [], [], []
    volume7, volume8, volume9 = [], [], []
    volume1.append(get_data("Concave180_0_r1.0", end=20))
    volume1.append(get_data("Concave180_0_r1.0_rand1"))
    volume1.append(get_data("Concave180_0_r1.0_rand2"))
    volume1.append(get_data("Concave180_0_r1.0_rand3"))
    volume1.append(get_data("Concave180_0_r1.0_rand4"))
    volume1.append(get_data("Concave180_0_r1.0_rand5"))

    volume2.append(get_data("Concave180_1_r1.0", end=20))
    volume2.append(get_data("Concave180_1_r1.0_rand1"))
    volume2.append(get_data("Concave180_1_r1.0_rand2"))
    volume2.append(get_data("Concave180_1_r1.0_rand3"))
    volume2.append(get_data("Concave180_1_r1.0_rand4"))
    volume2.append(get_data("Concave180_1_r1.0_rand5"))

    volume3.append(get_data("Concave180_2_r1.0", end=20))
    volume3.append(get_data("Concave180_2_r1.0_rand1"))
    volume3.append(get_data("Concave180_2_r1.0_rand2"))
    volume3.append(get_data("Concave180_2_r1.0_rand3"))
    volume3.append(get_data("Concave180_2_r1.0_rand4"))
    volume3.append(get_data("Concave180_2_r1.0_rand5"))

    volume4.append(get_data("Concave360_0_r1.0", end=20))
    volume4.append(get_data("Concave360_0_r1.0_rand1"))
    volume4.append(get_data("Concave360_0_r1.0_rand2"))
    volume4.append(get_data("Concave360_0_r1.0_rand3"))
    volume4.append(get_data("Concave360_0_r1.0_rand4"))
    volume4.append(get_data("Concave360_0_r1.0_rand5"))

    volume5.append(get_data("Concave360_1_r1.0", end=20))
    volume5.append(get_data("Concave360_1_r1.0_rand1"))
    volume5.append(get_data("Concave360_1_r1.0_rand2"))
    volume5.append(get_data("Concave360_1_r1.0_rand3"))
    volume5.append(get_data("Concave360_1_r1.0_rand4"))
    volume5.append(get_data("Concave360_1_r1.0_rand5"))

    volume6.append(get_data("Concave360_2_r1.0", end=20))
    volume6.append(get_data("Concave360_2_r1.0_rand1"))
    volume6.append(get_data("Concave360_2_r1.0_rand2"))
    volume6.append(get_data("Concave360_2_r1.0_rand3"))
    volume6.append(get_data("Concave360_2_r1.0_rand4"))
    volume6.append(get_data("Concave360_2_r1.0_rand5"))

    volume7.append(get_data("Concave480_0_r1.0", end=20))
    volume7.append(get_data("Concave480_0_r1.0_rand1"))
    volume7.append(get_data("Concave480_0_r1.0_rand2"))
    volume7.append(get_data("Concave480_0_r1.0_rand3"))
    volume7.append(get_data("Concave480_0_r1.0_rand4"))
    volume7.append(get_data("Concave480_0_r1.0_rand5"))

    volume8.append(get_data("Concave480_1_r1.0", end=20))
    volume8.append(get_data("Concave480_1_r1.0_rand1"))
    volume8.append(get_data("Concave480_1_r1.0_rand2"))
    volume8.append(get_data("Concave480_1_r1.0_rand3"))
    volume8.append(get_data("Concave480_1_r1.0_rand4"))
    volume8.append(get_data("Concave480_1_r1.0_rand5"))

    volume9.append(get_data("Concave480_2_r1.0", end=20))
    volume9.append(get_data("Concave480_2_r1.0_rand1"))
    volume9.append(get_data("Concave480_2_r1.0_rand2"))
    volume9.append(get_data("Concave480_2_r1.0_rand3"))
    volume9.append(get_data("Concave480_2_r1.0_rand4"))
    volume9.append(get_data("Concave480_2_r1.0_rand5"))

    x = [0.47, 0.78, 1.0]
    F1 = [np.mean(volume1), np.mean(volume2), np.mean(volume3)]
    F2 = [np.mean(volume4), np.mean(volume5), np.mean(volume6)]
    F3 = [np.mean(volume7), np.mean(volume8), np.mean(volume9)]

    E1 = [np.std(volume1, ddof=1), np.std(volume2, ddof=1), np.std(volume3, ddof=1)]
    E2 = [np.std(volume4, ddof=1), np.std(volume5, ddof=1), np.std(volume6, ddof=1)]
    E3 = [np.std(volume7, ddof=1), np.std(volume8, ddof=1), np.std(volume9, ddof=1)]

    plt.errorbar(x, F1, yerr=E1, color=color[0], fmt="o-", capsize=10, capthick=4, label="AR1-F1")
    plt.errorbar(x, F2, yerr=E2, color=color[1], fmt="o-", capsize=10, capthick=4, label="AR1-F2")
    plt.errorbar(x, F3, yerr=E3, color=color[2], fmt="o-", capsize=10, capthick=4, label="AR1-F3")
    plt.xlabel("Aspect ratio")
    plt.ylabel("Sliding volume (m$^3$)")
    # plt.xlim([0.0, 8])
    # plt.ylim([0, 10])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("volume_shape1" + ".svg")
    plt.close()


def shape2():
    volume1, volume2, volume3 = [], [], []
    volume4, volume5, volume6 = [], [], []
    volume7, volume8, volume9 = [], [], []

    # Convex70
    volume1.append(get_data("Convex70_0_r1.0", end=20))
    volume1.append(get_data("Convex70_0_r1.0_rand1"))
    volume1.append(get_data("Convex70_0_r1.0_rand2"))
    volume1.append(get_data("Convex70_0_r1.0_rand3"))
    volume1.append(get_data("Convex70_0_r1.0_rand4"))
    volume1.append(get_data("Convex70_0_r1.0_rand5"))

    volume2.append(get_data("Convex70_1_r1.0", end=20))
    volume2.append(get_data("Convex70_1_r1.0_rand1"))
    volume2.append(get_data("Convex70_1_r1.0_rand2"))
    volume2.append(get_data("Convex70_1_r1.0_rand3"))
    volume2.append(get_data("Convex70_1_r1.0_rand4"))
    volume2.append(get_data("Convex70_1_r1.0_rand5"))

    volume3.append(get_data("Convex70_2_r1.0", end=20))
    volume3.append(get_data("Convex70_2_r1.0_rand1"))
    volume3.append(get_data("Convex70_2_r1.0_rand2"))
    volume3.append(get_data("Convex70_2_r1.0_rand3"))
    volume3.append(get_data("Convex70_2_r1.0_rand4"))
    volume3.append(get_data("Convex70_2_r1.0_rand5"))

    # Convex140
    volume4.append(get_data("Convex140_0_r1.0", end=20))
    volume4.append(get_data("Convex140_0_r1.0_rand1"))
    volume4.append(get_data("Convex140_0_r1.0_rand2"))
    volume4.append(get_data("Convex140_0_r1.0_rand3"))
    volume4.append(get_data("Convex140_0_r1.0_rand4"))
    volume4.append(get_data("Convex140_0_r1.0_rand5"))

    volume5.append(get_data("Convex140_1_r1.0", end=20))
    volume5.append(get_data("Convex140_1_r1.0_rand1"))
    volume5.append(get_data("Convex140_1_r1.0_rand2"))
    volume5.append(get_data("Convex140_1_r1.0_rand3"))
    volume5.append(get_data("Convex140_1_r1.0_rand4"))
    volume5.append(get_data("Convex140_1_r1.0_rand5"))

    volume6.append(get_data("Convex140_2_r1.0", end=20))
    volume6.append(get_data("Convex140_2_r1.0_rand1"))
    volume6.append(get_data("Convex140_2_r1.0_rand2"))
    volume6.append(get_data("Convex140_2_r1.0_rand3"))
    volume6.append(get_data("Convex140_2_r1.0_rand4"))
    volume6.append(get_data("Convex140_2_r1.0_rand5"))

    # Convex210
    volume7.append(get_data("Convex210_0_r1.0", end=20))
    volume7.append(get_data("Convex210_0_r1.0_rand1"))
    volume7.append(get_data("Convex210_0_r1.0_rand2"))
    volume7.append(get_data("Convex210_0_r1.0_rand3"))
    volume7.append(get_data("Convex210_0_r1.0_rand4"))
    volume7.append(get_data("Convex210_0_r1.0_rand5"))

    volume8.append(get_data("Convex210_1_r1.0", end=20))
    volume8.append(get_data("Convex210_1_r1.0_rand1"))
    volume8.append(get_data("Convex210_1_r1.0_rand2"))
    volume8.append(get_data("Convex210_1_r1.0_rand3"))
    volume8.append(get_data("Convex210_1_r1.0_rand4"))
    volume8.append(get_data("Convex210_1_r1.0_rand5"))

    volume9.append(get_data("Convex210_2_r1.0", end=20))
    volume9.append(get_data("Convex210_2_r1.0_rand1"))
    volume9.append(get_data("Convex210_2_r1.0_rand2"))
    volume9.append(get_data("Convex210_2_r1.0_rand3"))
    volume9.append(get_data("Convex210_2_r1.0_rand4"))
    volume9.append(get_data("Convex210_2_r1.0_rand5"))

    # 统计
    x = [0.47, 0.78, 1.0]

    F1 = [np.mean(volume1), np.mean(volume2), np.mean(volume3)]
    F2 = [np.mean(volume4), np.mean(volume5), np.mean(volume6)]
    F3 = [np.mean(volume7), np.mean(volume8), np.mean(volume9)]

    E1 = [np.std(volume1, ddof=1), np.std(volume2, ddof=1), np.std(volume3, ddof=1)]
    E2 = [np.std(volume4, ddof=1), np.std(volume5, ddof=1), np.std(volume6, ddof=1)]
    E3 = [np.std(volume7, ddof=1), np.std(volume8, ddof=1), np.std(volume9, ddof=1)]

    # 画图
    plt.errorbar(x, F1, yerr=E1, color=color[0], fmt="o-", capsize=10, capthick=4, label="AR1-F1")
    plt.errorbar(x, F2, yerr=E2, color=color[1], fmt="o-", capsize=10, capthick=4, label="AR1-F2")
    plt.errorbar(x, F3, yerr=E3, color=color[2], fmt="o-", capsize=10, capthick=4, label="AR1-F3")

    plt.xlabel("Aspect ratio")
    plt.ylabel("Sliding volume (m$^3$)")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("volume_shape2" + ".svg")
    plt.close()


anisotropy()
shape0()
shape1()
shape2()
