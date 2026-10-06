#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style


p = []
q = []
Rq = []
ratio = []
time = []
pid = 4
c = 2500
fai = 30 / 180 * math.pi


def SphericalTensor(stress):
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


def ComputeStressInvariantJ2(stress):
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


def DeviatoricTensor(stress):
    sigma = SphericalTensor(stress)

    deviatoric_stress = stress
    for i in range(3):
        deviatoric_stress[i] -= sigma
    return deviatoric_stress


def ComputeStressInvariantJ3(stress):
    deviatoric_stress = DeviatoricTensor(stress)
    J3 = (
        deviatoric_stress[0] * deviatoric_stress[1] * deviatoric_stress[2]
        - deviatoric_stress[2] * deviatoric_stress[3] * deviatoric_stress[3]
        + 2 * deviatoric_stress[3] * deviatoric_stress[4] * deviatoric_stress[5]
        - deviatoric_stress[0] * deviatoric_stress[4] * deviatoric_stress[4]
        - deviatoric_stress[1] * deviatoric_stress[5] * deviatoric_stress[5]
    )
    return J3


def ComputeLodeAngle(stress):
    J2 = ComputeStressInvariantJ2(stress)
    J3 = ComputeStressInvariantJ3(stress)

    load_angle = 0.0
    if abs(J2) > 1e-12:
        load_angle = (3.0 * math.sqrt(3.0) / 2.0) * (J3 / (J2**1.5))
    load_angle = max(-1.0, min(1.0, load_angle))
    return 1.0 / 3.0 * math.acos(load_angle)


start_num = 0
end_num = 120
for printNum in range(start_num, end_num):
    data = np.load("MPMParticle{0:06d}.npz".format(printNum), allow_pickle=True)

    cos_fai = math.cos(fai)
    tan_fai = math.tan(fai)
    lode = ComputeLodeAngle(data["stress"][pid])
    Rmc = math.sin(lode + math.pi / 3.0) / (math.sqrt(3.0) * cos_fai) + math.cos(lode + math.pi / 3.0) * tan_fai / 3.0
    p.append(-SphericalTensor(data["stress"][pid]))
    Rq.append(Rmc * equistress(data["stress"][pid]))
    q.append(equistress(data["stress"][pid]))
    ratio.append(equistress(data["stress"][pid]) / -SphericalTensor(data["stress"][pid]))
    time.append(0.5 - data["position"][4][2] + data["position"][0][2])

p0 = np.linspace(0, 1000000, 200)
q0 = c + math.tan(fai) * p0

fig = plt.figure(figsize=(18, 6))
fig.suptitle("drained Test (MohrCoulomb)", size=18)

ax1 = plt.subplot(1, 2, 1)
ax1.scatter(p, Rq, label="ti-MPM")
ax1.scatter(p0, q0, label="analytical")
ax1.set_xlabel("mean stress")
ax1.set_ylabel("equivalent stress")
ax1.legend(loc="best")

ax2 = plt.subplot(1, 2, 2)
ax2.plot(time, q)
ax2.set_xlabel("axial strain")
ax2.set_ylabel("equivalent stress")


print((q[-1] - q[0]) / (p[-1] - p[0]))

plt.show()
