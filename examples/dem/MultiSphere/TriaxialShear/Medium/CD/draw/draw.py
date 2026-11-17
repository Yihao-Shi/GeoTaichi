#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

"""
data1 = np.load('contacts/DEMContactPP{0:06d}.npz'.format(2), allow_pickle=True)
data2 = np.load('contacts/DEMContactPW{0:06d}.npz'.format(2), allow_pickle=True)
end1=data1["end1"]
end2=data1["end2"]
val=data1["oldTangentialOverlap"]

for i in range(end1.shape[0]):
    print(end1[i], end2[i], val[i])
"""

start_num = 0
end_num = 83


def macro(path):
    q = []
    p = []
    qp = []
    evol = []
    epsilon = []
    vol0 = 0.0
    leng0 = 0.0

    for printNum in range(start_num, end_num):
        data = np.load(path + "/DEMWall{0:06d}.npz".format(printNum), allow_pickle=True)

        down_force = -(data["contact_force"][0][2] + data["contact_force"][1][2])
        up_force = data["contact_force"][2][2] + data["contact_force"][3][2]
        left_force = -(data["contact_force"][4][0] + data["contact_force"][5][0])
        right_force = data["contact_force"][6][0] + data["contact_force"][7][0]
        front_force = -(data["contact_force"][8][1] + data["contact_force"][9][1])
        back_force = data["contact_force"][10][1] + data["contact_force"][11][1]

        down_position = (
            data["point1"][0][2]
            + data["point2"][0][2]
            + data["point3"][0][2]
            + data["point1"][1][2]
            + data["point2"][1][2]
            + data["point3"][1][2]
        ) / 6.0
        up_position = (
            data["point1"][2][2]
            + data["point2"][2][2]
            + data["point3"][2][2]
            + data["point1"][3][2]
            + data["point2"][3][2]
            + data["point3"][3][2]
        ) / 6.0
        left_position = (
            data["point1"][4][0]
            + data["point2"][4][0]
            + data["point3"][4][0]
            + data["point1"][5][0]
            + data["point2"][5][0]
            + data["point3"][5][0]
        ) / 6.0
        right_position = (
            data["point1"][6][0]
            + data["point2"][6][0]
            + data["point3"][6][0]
            + data["point1"][7][0]
            + data["point2"][7][0]
            + data["point3"][7][0]
        ) / 6.0
        front_position = (
            data["point1"][8][1]
            + data["point2"][8][1]
            + data["point3"][8][1]
            + data["point1"][9][1]
            + data["point2"][9][1]
            + data["point3"][9][1]
        ) / 6.0
        back_position = (
            data["point1"][10][1]
            + data["point2"][10][1]
            + data["point3"][10][1]
            + data["point1"][11][1]
            + data["point2"][11][1]
            + data["point3"][11][1]
        ) / 6.0

        down_pressure = down_force / (right_position - left_position) / (back_position - front_position)
        up_pressure = up_force / (right_position - left_position) / (back_position - front_position)
        left_pressure = left_force / (up_position - down_position) / (back_position - front_position)
        right_pressure = right_force / (up_position - down_position) / (back_position - front_position)
        front_pressure = front_force / (up_position - down_position) / (right_position - left_position)
        back_pressure = back_force / (up_position - down_position) / (right_position - left_position)

        wexx = 0.5 * (left_pressure + right_pressure)
        weyy = 0.5 * (front_pressure + back_pressure)
        wezz = 0.5 * (down_pressure + up_pressure)

        if printNum == start_num:
            vol0 = (up_position - down_position) * (right_position - left_position) * (back_position - front_position)
            leng0 = up_position - down_position
        leng = up_position - down_position

        p.append(1.0 / 3.0 * (wexx + weyy + wezz))
        q.append(wezz - 0.5 * (weyy + wexx))
        # q.append(0.5*math.sqrt((wexx-weyy)**2+(weyy-wezz)**2+(wexx-wezz)**2))
        # if (1./3.*(wexx+weyy+wezz))==0: wexx=weyy=wezz=1
        qp.append(
            (0.5 * math.sqrt((wexx - weyy) ** 2 + (weyy - wezz) ** 2 + (wexx - wezz) ** 2))
            / (1.0 / 3.0 * (wexx + weyy + wezz))
        )
        epsilon.append(math.log(leng0 / leng))
        evol.append(
            math.log(
                vol0
                / ((up_position - down_position) * (right_position - left_position) * (back_position - front_position))
            )
        )
        print(wexx, weyy, wezz)
    return p, q, qp, epsilon, evol


pm, qm, qpm, epsilonm, evolm = macro("walls")
plt.plot(epsilonm, qpm)
plt.xlim([0, 0.5])
plt.show()
plt.plot(epsilonm, evolm)
plt.xlim([0, 0.5])
plt.ylim([-0.08, 0.05])
plt.gca().invert_yaxis()
plt.show()
