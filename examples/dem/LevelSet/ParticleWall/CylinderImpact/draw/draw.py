import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rcParams

params = {
    "backend": "ps",
    "font.size": 36,
    "lines.linewidth": 4.5,
    "lines.markersize": 10,
    "xtick.labelsize": 32,
    "ytick.labelsize": 32,
    "xtick.major.pad": 12,
    "ytick.major.pad": 12,
    "axes.labelpad": 8,
    "legend.fontsize": 32,
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

epsilon = 1.0
r0 = math.sqrt(2) / 50.0
fai = math.pi / 4.0
vel1 = -0.1
omega1 = 0.0
theta, thetaa, v, w, va, wa = [], [], [], [], [], []
bins = 50


def analytical():
    m = 0.13304891256771037  # math.pi * 0.02 * 0.02 * 0.04 * 2650
    I = 3.10580090e-05
    for i in range(bins + 1):
        theta = math.pi / 2.0 / bins * i
        omega3 = (
            m
            * vel1
            * (1 + epsilon)
            * r0
            * math.cos(fai + theta)
            / (I + m * r0 * r0 * math.cos(fai + theta) * math.cos(fai + theta))
        )
        vel3 = omega3 * r0 * math.cos(fai + theta) - epsilon * vel1
        va.append(vel3 / vel1)
        wa.append(omega3 * r0 / vel1)
        thetaa.append(theta)


def getdata(path, theta0):
    data1 = np.load(path + "particles/LSDEMRigid{0:06d}.npz".format(1))
    m = data1["mass"][0]
    i = 1.0 / data1["inverse_inertia"][0]
    vel2 = data1["velocity"][0][2]
    omega2 = data1["omega"][0][1]

    v.append(vel2 / vel1)
    w.append(omega2 * r0 / vel1)
    theta.append(theta0)


analytical()
getdata("5degrees/", 5 / 180 * math.pi)
getdata("10degrees/", 10 / 180 * math.pi)
getdata("15degrees/", 15 / 180 * math.pi)
getdata("20degrees/", 20 / 180 * math.pi)
getdata("25degrees/", 25 / 180 * math.pi)
getdata("30degrees/", 30 / 180 * math.pi)
getdata("35degrees/", 35 / 180 * math.pi)
getdata("40degrees/", 40 / 180 * math.pi)
getdata("45degrees/", 45 / 180 * math.pi)
getdata("50degrees/", 50 / 180 * math.pi)
getdata("55degrees/", 55 / 180 * math.pi)
getdata("60degrees/", 60 / 180 * math.pi)
getdata("65degrees/", 65 / 180 * math.pi)
getdata("70degrees/", 70 / 180 * math.pi)
getdata("75degrees/", 75 / 180 * math.pi)
getdata("80degrees/", 80 / 180 * math.pi)
getdata("85degrees/", 85 / 180 * math.pi)

plt.plot(thetaa, va, color=color[0], label="Analytical")
plt.scatter(theta, v, marker="o", color=color[1], s=185, edgecolors=color[1], label="This study")
# plt.xlim([0, 0.02])
# plt.ylim([0., 1.6])
plt.xlabel("$\\upsilon_0$")
plt.ylabel("$v_z^+/v_z^-$")
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig("vel.svg")
plt.close()

plt.plot(thetaa, wa, color=color[0], label="Analytical")
plt.scatter(theta, w, marker="o", color=color[1], s=185, edgecolors=color[1], label="This study")
# plt.xlim([0, 0.02])
# plt.ylim([0., 1.6])
plt.xlabel("$\\upsilon_0$")
plt.ylabel("$\omega_y^+r_0/v_z^-$")
plt.tight_layout()
plt.legend(frameon=False)
plt.savefig("omega.svg")
plt.close()
