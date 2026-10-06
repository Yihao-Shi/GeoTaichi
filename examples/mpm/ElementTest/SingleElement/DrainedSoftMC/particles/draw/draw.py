#!/usr/bin/env python
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib import rcParams

from third_party.tablelegend import tablelegend

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

pid = 4
c = 250
cp = 250
fai = 0 / 180 * math.pi
start_num = 0
end_num = 101


def MeanStress(stress):
    return (stress[2] + stress[1] + stress[0]) / 3.0


def equistress(stress):
    return math.sqrt(
        3.0
        * (
            (
                (stress[0] - stress[1]) * (stress[0] - stress[1])
                + (stress[1] - stress[2]) * (stress[1] - stress[2])
                + (stress[0] - stress[2]) * (stress[0] - stress[2])
            )
            / 6.0
            + stress[3] * stress[3]
            + stress[4] * stress[4]
            + stress[5] * stress[5]
        )
    )


def ComputeInvariantJ2(stress):
    J2 = (
        (
            (stress[0] - stress[1]) * (stress[0] - stress[1])
            + (stress[1] - stress[2]) * (stress[1] - stress[2])
            + (stress[0] - stress[2]) * (stress[0] - stress[2])
        )
        / 6.0
        + stress[3] * stress[3]
        + stress[4] * stress[4]
        + stress[5] * stress[5]
    )
    return J2


def DeviatoricStress(stress):
    sigma = MeanStress(stress)

    deviatoric_stress = stress
    for i in range(3):
        deviatoric_stress[i] -= sigma
    return deviatoric_stress


def ComputeInvariantJ3(stress):
    deviatoric_stress = DeviatoricStress(stress)
    J3 = (
        deviatoric_stress[0] * deviatoric_stress[1] * deviatoric_stress[2]
        - deviatoric_stress[2] * deviatoric_stress[3] * deviatoric_stress[3]
        + 2 * deviatoric_stress[3] * deviatoric_stress[4] * deviatoric_stress[5]
        - deviatoric_stress[0] * deviatoric_stress[4] * deviatoric_stress[4]
        - deviatoric_stress[1] * deviatoric_stress[5] * deviatoric_stress[5]
    )
    return J3


def ComputeLodeAngle(stress):
    J2 = ComputeInvariantJ2(stress)
    J3 = ComputeInvariantJ3(stress)

    load_angle = 0.0
    if abs(J2) > 1e-12:
        load_angle = (3.0 * math.sqrt(3.0) / 2.0) * (J3 / (J2**1.5))
    load_angle = max(-1.0, min(1.0, load_angle))
    return 1.0 / 3.0 * math.acos(load_angle)


def get_result(path):
    p = []
    q = []
    Rq = []
    ratio = []
    time = []
    for printNum in range(start_num, end_num):
        data = np.load(path + "MPMParticle{0:06d}.npz".format(printNum), allow_pickle=True)

        cos_fai = math.cos(fai)
        tan_fai = math.tan(fai)
        lode = ComputeLodeAngle(data["stress"][pid])
        Rmc = (
            math.sin(lode + math.pi / 3.0) / (math.sqrt(3.0) * cos_fai) + math.cos(lode + math.pi / 3.0) * tan_fai / 3.0
        )
        p.append(-MeanStress(data["stress"][pid]) / 1000)
        Rq.append(Rmc * equistress(data["stress"][pid]) / 1000)
        q.append(equistress(data["stress"][pid]) / 1000)
        ratio.append(equistress(data["stress"][pid]) / -MeanStress(data["stress"][pid]))
        time.append(0.5 - data["position"][4][2] + data["position"][0][2])
    return p, q, Rq, ratio, time


p1, q1, Rq1, ratio1, time1 = get_result("")

t0 = np.linspace(0, 0.05, 200)
q0 = (
    cp / 1000 * (1 + math.sin(fai)) / (1 - math.sin(fai))
    + 2 * c / 1000 * math.cos(fai) / (1 - math.sin(fai))
    - cp / 1000
)
fig, ax = plt.subplots()
ax.plot(time1, q1, markerfacecolor="none", markersize=12, color=color[0], label="$\epsilon^r_p$=0.05, $c^r$=2.5kPa")
ax.set_xlabel("Axial strain, $\epsilon_a$ (\%)")
ax.set_ylabel("Equivalent stress, $q$ (kpa)")
ax.set_xlim([0, 0.01])
# ax.set_ylim([0,5])
ax.legend(loc="best")
fig.tight_layout()
fig.savefig("qtcurve.svg")
plt.close()

print((q[-1] - q[0]) / (p[-1] - p[0]))
