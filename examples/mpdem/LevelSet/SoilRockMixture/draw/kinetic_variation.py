import os
import sys

import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.spatial.transform import Rotation as R

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


def get_data(path, start=0, end=20):
    t, kinetic = [], []
    for i in range(start, end):
        mdata = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(i), allow_pickle=True)
        ddata = np.load(path + "/particles/LSDEMRigid{0:06d}.npz".format(i), allow_pickle=True)

        t.append(mdata["t_current"])
        mass = ddata["mass"]
        v = ddata["velocity"]
        w = ddata["omega"]
        q = ddata["quanternion"]
        r = R.from_quat(q)
        wl = w.copy()
        for j in range(w.shape[0]):
            wl[j] = r.as_matrix()[j, ...] @ w[j]
        inertia = 1.0 / ddata["inverse_inertia"]

        ke_mpm = 0.5 * 0.5**3 * 1800.0 * np.sum(np.linalg.norm(mdata["velocity"], axis=1) ** 2)
        ke_dem_t = np.sum(0.5 * mass * (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2]))
        ke_dem_r = np.sum(
            0.5
            * (
                inertia[:, 0] * wl[:, 0] * wl[:, 0]
                + inertia[:, 1] * wl[:, 1] * wl[:, 1]
                + inertia[:, 2] * wl[:, 2] * wl[:, 2]
            )
        )

        kinetic.append((ke_mpm + ke_dem_t + ke_dem_r) / 1e6)
    return np.array(t), np.array(kinetic)


def get_dem_data(path, start=0, end=20):
    t, kinetic = [], []
    for i in range(start, end):
        ddata = np.load(path + "/particles/LSDEMRigid{0:06d}.npz".format(i), allow_pickle=True)

        t.append(ddata["t_current"])
        mass = ddata["mass"]
        v = ddata["velocity"]
        w = ddata["omega"]
        q = ddata["quanternion"]
        r = R.from_quat(q)
        wl = w.copy()
        for j in range(w.shape[0]):
            wl[j] = r.as_matrix()[j, ...] @ w[j]
        inertia = 1.0 / ddata["inverse_inertia"]

        ke_dem_t = np.sum(0.5 * mass * (v[:, 0] * v[:, 0] + v[:, 1] * v[:, 1] + v[:, 2] * v[:, 2]))
        ke_dem_r = np.sum(
            0.5
            * (
                inertia[:, 0] * wl[:, 0] * wl[:, 0]
                + inertia[:, 1] * wl[:, 1] * wl[:, 1]
                + inertia[:, 2] * wl[:, 2] * wl[:, 2]
            )
        )

        kinetic.append((ke_dem_t + ke_dem_r) / 1e6)
    return np.array(t), np.array(kinetic)


def get_mpm_data(path, start=0, end=20):
    t, kinetic = [], []
    for i in range(start, end):
        mdata = np.load(path + "/particles/MPMParticle{0:06d}.npz".format(i), allow_pickle=True)

        t.append(mdata["t_current"])
        ke_mpm = 0.5 * 0.5**3 * 1800.0 * np.sum(np.linalg.norm(mdata["velocity"], axis=1) ** 2)
        kinetic.append((ke_mpm) / 1e6)
    return np.array(t), np.array(kinetic)


def get_case_paths(base_name):
    """
    例如:
    Flat332_0_r1.0
    Flat332_0_r1.0_rand1
    ...
    Flat332_0_r1.0_rand5
    """
    paths = []
    for i in range(1, 6):
        paths.append(base_name + "_rand{}".format(i))
    return paths


def get_mean_std_data(base_name, func, start=0, end=20):
    """
    对同一组6次随机结果，返回:
    t_mean: 时间
    y_mean: 动能均值
    y_std : 动能标准差
    """
    paths = get_case_paths(base_name)

    all_t = []
    all_y = []

    for path in paths:
        t, y = func(path, start=start, end=end)
        all_t.append(t)
        all_y.append(y)

    all_t = np.array(all_t)  # shape = (6, nt)
    all_y = np.array(all_y)  # shape = (6, nt)

    # 默认假设 6 次模拟的时间步一致
    t_mean = np.mean(all_t, axis=0)
    y_mean = np.mean(all_y, axis=0)
    y_std = np.std(all_y, axis=0, ddof=1)

    return t_mean, y_mean, y_std


def plot_with_shade(base_name, c, label, func=get_data, start=0, end=20):
    t, y_mean, y_std = get_mean_std_data(base_name, func=func, start=start, end=end)

    plt.plot(t, y_mean, color=c, label=label)
    plt.fill_between(t, y_mean - y_std, y_mean + y_std, color=c, alpha=0.25)


def shape0():
    # -------- geometry shape --------
    plot_with_shade("Flat332_0_r1.0", color[0], "G1-AR1-F3")
    plot_with_shade("Flat332_1_r1.0", color[1], "G1-AR2-F3")
    plot_with_shade("Flat332_2_r1.0", color[2], "G1-AR3-F3")

    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_shape0.svg")
    plt.close()

    # -------- number effect --------
    plot_with_shade("Flat108_0_r1.0", color[2], "G1-AR1-F1")
    plot_with_shade("Flat216_0_r1.0", color[1], "G1-AR1-F2")
    plot_with_shade("Flat332_0_r1.0", color[0], "G1-AR1-F3")

    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_number0.svg")
    plt.close()


def shape1():
    # -------- geometry shape --------
    plot_with_shade("Concave480_0_r1.0", color[0], "G2-AR1-F3")
    plot_with_shade("Concave480_1_r1.0", color[1], "G2-AR2-F3")
    plot_with_shade("Concave480_2_r1.0", color[2], "G2-AR3-F3")

    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_shape1.svg")
    plt.close()

    # -------- number effect --------
    plot_with_shade("Concave180_0_r1.0", color[2], "G2-AR1-F1")
    plot_with_shade("Concave360_0_r1.0", color[1], "G2-AR1-F2")
    plot_with_shade("Concave480_0_r1.0", color[0], "G2-AR1-F3")

    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_number1.svg")
    plt.close()


def shape2():
    # -------- geometry shape --------
    plot_with_shade("Convex210_0_r1.0", color[0], "G3-AR1-F3")
    plot_with_shade("Convex210_1_r1.0", color[1], "G3-AR2-F3")
    plot_with_shade("Convex210_2_r1.0", color[2], "G3-AR3-F3")

    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_shape2.svg")
    plt.close()

    # -------- number effect --------
    plot_with_shade("Convex70_0_r1.0", color[2], "G3-AR1-F1")
    plot_with_shade("Convex140_0_r1.0", color[1], "G3-AR1-F2")
    plot_with_shade("Convex210_0_r1.0", color[0], "G3-AR1-F3")

    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_number2.svg")
    plt.close()


def geometry_shape():
    plot_with_shade("Flat332_0_r1.0", color[0], "G1-AR1-F3")
    plot_with_shade("Concave480_0_r1.0", color[1], "G2-AR1-F3")
    plot_with_shade("Convex210_0_r1.0", color[2], "G3-AR1-F3")

    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_geometry_shape_AR0F0.svg")
    plt.close()

    plot_with_shade("Flat216_0_r1.0", color[0], "G1-AR1-F2")
    plot_with_shade("Concave360_0_r1.0", color[1], "G2-AR1-F2")
    plot_with_shade("Convex140_0_r1.0", color[2], "G3-AR1-F2")

    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_geometry_shape_AR0F1.svg")
    plt.close()

    plot_with_shade("Flat108_0_r1.0", color[0], "G1-AR1-F1")
    plot_with_shade("Concave180_0_r1.0", color[1], "G2-AR1-F1")
    plot_with_shade("Convex70_0_r1.0", color[2], "G3-AR1-F1")

    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_geometry_shape_AR0F2.svg")
    plt.close()


# shape0()
# shape1()
# shape2()
# geometry_shape()


def split_energy():
    plot_with_shade("Flat332_0_r1.0", color[0], "Soil phase1", func=get_mpm_data)
    plot_with_shade("Flat332_0_r1.0", color[0], "Rock phase1", func=get_dem_data)
    plot_with_shade("Flat332_1_r1.0", color[1], "Soil phase2", func=get_mpm_data)
    plot_with_shade("Flat332_1_r1.0", color[1], "Rock phase2", func=get_dem_data)
    plot_with_shade("Flat332_2_r1.0", color[2], "Soil phase3", func=get_mpm_data)
    plot_with_shade("Flat332_2_r1.0", color[2], "Rock phase3", func=get_dem_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_split0.svg")
    plt.close()

    plot_with_shade("Concave480_0_r1.0", color[0], "Soil phase1", func=get_mpm_data)
    plot_with_shade("Concave480_0_r1.0", color[0], "Rock phase1", func=get_dem_data)
    plot_with_shade("Concave480_1_r1.0", color[1], "Soil phase2", func=get_mpm_data)
    plot_with_shade("Concave480_1_r1.0", color[1], "Rock phase2", func=get_dem_data)
    plot_with_shade("Concave480_2_r1.0", color[2], "Soil phase3", func=get_mpm_data)
    plot_with_shade("Concave480_2_r1.0", color[2], "Rock phase3", func=get_dem_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_split1.svg")
    plt.close()

    plot_with_shade("Convex210_0_r1.0", color[0], "Soil phase1", func=get_mpm_data)
    plot_with_shade("Convex210_0_r1.0", color[0], "Rock phase1", func=get_dem_data)
    plot_with_shade("Convex210_1_r1.0", color[1], "Soil phase2", func=get_mpm_data)
    plot_with_shade("Convex210_1_r1.0", color[1], "Rock phase2", func=get_dem_data)
    plot_with_shade("Convex210_2_r1.0", color[2], "Soil phase3", func=get_mpm_data)
    plot_with_shade("Convex210_2_r1.0", color[2], "Rock phase3", func=get_dem_data)
    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (10$^6$ J)")
    plt.xlim([0.0, 6])
    plt.ylim([0, 6])
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("kinetic_split2.svg")
    plt.close()


split_energy()
