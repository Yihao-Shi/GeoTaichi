#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

params = {
    "backend": "ps",
    "font.size": 26,
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "xtick.major.pad": 12,
    "ytick.major.pad": 12,
    "axes.labelpad": 8,
    "legend.fontsize": 26,
    "figure.figsize": [12, 9],
    "font.family": "serif",
    "text.usetex": False,
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


start_num = 1
end_num = 90

love_stress = []
p = []
q = []
qp = []
epsilon = []
density = []

particle = np.load("particles/DEMParticle{0:06d}.npz".format(start_num), allow_pickle=True)
radius = particle["radius"]
vol = np.sum(7158 * 4.0 / 3.0 * math.pi * radius**3)

data = np.load("walls/DEMWall{0:06d}.npz".format(0), allow_pickle=True)
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
vol0 = (up_position - down_position) * (right_position - left_position) * (back_position - front_position)
leng0 = up_position - down_position
leng = up_position - down_position
wexx = 200000
weyy = 200000
wezz = 200000
p.append(1.0 / 3.0 * (wexx + weyy + wezz))
q.append(wezz - 0.5 * (weyy + wexx))
qp.append((wezz - 0.5 * (weyy + wexx)) / (1.0 / 3.0 * (wexx + weyy + wezz)))
epsilon.append(math.log(leng0 / leng))
height = up_position - down_position
length = right_position - left_position
width = back_position - front_position
total_v = height * length * width
density.append(vol / (total_v))

for printNum in range(start_num, end_num):
    if printNum > start_num:
        data = np.load("walls/DEMWall{0:06d}.npz".format(printNum), allow_pickle=True)
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

        height = up_position - down_position
        length = right_position - left_position
        width = back_position - front_position
        total_v = height * length * width
        density.append(vol / (total_v))

        data = np.load("contacts/DEMContactPW{0:06d}.npz".format(printNum), allow_pickle=True)
        end2 = data["end2"][np.linalg.norm(data["normal_force"], axis=1) > 0.0]
        fn = data["normal_force"][np.linalg.norm(data["normal_force"], axis=1) > 0.0]
        ft = data["tangential_force"][np.linalg.norm(data["normal_force"], axis=1) > 0.0]

        down_force = (
            np.sum(fn[end2 == 0], axis=0)
            + np.sum(fn[end2 == 1], axis=0)
            + np.sum(ft[end2 == 0], axis=0)
            + np.sum(ft[end2 == 1], axis=0)
        )[2]
        up_force = -(
            np.sum(fn[end2 == 2], axis=0)
            + np.sum(fn[end2 == 3], axis=0)
            + np.sum(ft[end2 == 2], axis=0)
            + np.sum(ft[end2 == 3], axis=0)
        )[2]
        left_force = (
            np.sum(fn[end2 == 4], axis=0)
            + np.sum(fn[end2 == 5], axis=0)
            + np.sum(ft[end2 == 4], axis=0)
            + np.sum(ft[end2 == 5], axis=0)
        )[0]
        right_force = -(
            np.sum(fn[end2 == 6], axis=0)
            + np.sum(fn[end2 == 7], axis=0)
            + np.sum(ft[end2 == 6], axis=0)
            + np.sum(ft[end2 == 7], axis=0)
        )[0]
        front_force = (
            np.sum(fn[end2 == 8], axis=0)
            + np.sum(fn[end2 == 9], axis=0)
            + np.sum(ft[end2 == 8], axis=0)
            + np.sum(ft[end2 == 9], axis=0)
        )[1]
        back_force = -(
            np.sum(fn[end2 == 10], axis=0)
            + np.sum(fn[end2 == 11], axis=0)
            + np.sum(ft[end2 == 10], axis=0)
            + np.sum(ft[end2 == 11], axis=0)
        )[1]

        down_pressure = down_force / (right_position - left_position) / (back_position - front_position)
        up_pressure = up_force / (right_position - left_position) / (back_position - front_position)
        left_pressure = left_force / (up_position - down_position) / (back_position - front_position)
        right_pressure = right_force / (up_position - down_position) / (back_position - front_position)
        front_pressure = front_force / (up_position - down_position) / (right_position - left_position)
        back_pressure = back_force / (up_position - down_position) / (right_position - left_position)

        wexx = 0.5 * (left_pressure + right_pressure)
        weyy = 0.5 * (front_pressure + back_pressure)
        wezz = 0.5 * (down_pressure + up_pressure)

        p.append(1.0 / 3.0 * (wexx + weyy + wezz))
        q.append(wezz - 0.5 * (weyy + wexx))
        qp.append((wezz - 0.5 * (weyy + wexx)) / (1.0 / 3.0 * (wexx + weyy + wezz)))
        epsilon.append(math.log(leng0 / height))

    particle = np.load("particles/DEMParticle{0:06d}.npz".format(0), allow_pickle=True)
    pos = particle["position"]

    data = np.load("contacts/DEMContactPP{0:06d}.npz".format(printNum), allow_pickle=True)
    contact_num = data["contact_num"][-1]
    end1 = data["end1"][:contact_num]
    end2 = data["end2"][:contact_num]
    normal_force = data["normal_force"][:contact_num]
    tangential_force = data["tangential_force"][:contact_num]
    forces = normal_force + tangential_force

    branch = pos[end1] - pos[end2]
    sigma = np.einsum("ci,cj->ij", forces, branch)
    sigma /= total_v
    sigma = 0.5 * (sigma + sigma.T)
    s = sigma - np.eye(3) * np.trace(sigma) / 3.0
    R = np.linalg.norm(s) * np.sqrt(1.5)
    love_stress.append(R)


plt.plot(epsilon, q, linestyle="-", color=color[0], label="Wall stress")
plt.plot(epsilon, love_stress, linestyle="--", color=color[1], label="Particle stress")
plt.xlabel("Time, $T$ (s)")
plt.ylabel("$\\sigma$ / $\\sigma_{target}$")
plt.legend(frameon=False)
plt.savefig("stress_strain.png")
plt.close()

plt.plot(epsilon, density, linestyle="-", color=color[0], label="case 1")
plt.xlabel("Time, $T$ (s)")
plt.ylabel("$\\rho$")
plt.legend(frameon=False)
plt.savefig("density.png")
plt.close()
