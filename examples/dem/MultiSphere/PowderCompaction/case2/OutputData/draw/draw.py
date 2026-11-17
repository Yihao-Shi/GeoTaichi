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


pressure = []
love_stress = []
time = []
density = []

particle = np.load("particles/DEMParticle{0:06d}.npz".format(0), allow_pickle=True)
radius = particle["radius"]
vol = np.sum(7158 * 4.0 / 3.0 * math.pi * radius**3)

Lx = 0.00036
Ly = 0.00036


start_num = 10
end_num = 21
for printNum in range(start_num, end_num):

    data = np.load("walls/DEMWall{0:06d}.npz".format(printNum), allow_pickle=True)
    zcoord = data["point1"][2][2]

    wall_planes = {
        0: (0, 0.0, np.array([0.0, 0.0, 1.0])),
        1: (0, 0.0, np.array([0.0, 0.0, 1.0])),
        2: (0, zcoord, np.array([0.0, 0.0, -1.0])),
        3: (0, zcoord, np.array([0.0, 0.0, -1.0])),
    }
    wall_ids = np.array(list(wall_planes.keys()), dtype=int)
    wall_normals = np.array([wall_planes[k][2] for k in wall_ids])  # (Nw, 3)
    wall_d = np.array([wall_planes[k][1] for k in wall_ids])  # (Nw,)

    particle = np.load("particles/DEMParticle{0:06d}.npz".format(0), allow_pickle=True)
    pos = particle["position"]

    total_v = zcoord * Lx * Ly
    density.append((vol) / (total_v))
    down_force = abs(data["contact_force"][2][2] + data["contact_force"][3][2])
    pressure.append(down_force / 1.296e-7)
    # print(down_force,np.sum(wall_normal_force[:,2]))
    time.append(data["t_current"])

    data = np.load("contacts/DEMContactPW{0:06d}.npz".format(printNum), allow_pickle=True)
    contact_num = data["contact_num"][-1]
    particle_id = data["end1"][:contact_num]
    wall_id = data["end2"][:contact_num]
    mask = (wall_id == 2) | (wall_id == 3)
    wall_normal_force = data["normal_force"][:contact_num]
    wall_tangential_force = data["tangential_force"][:contact_num]
    wall_forces = wall_normal_force + wall_tangential_force

    data = np.load("contacts/DEMContactPP{0:06d}.npz".format(printNum), allow_pickle=True)
    contact_num = data["contact_num"][-1]
    end1 = data["end1"][:contact_num]
    end2 = data["end2"][:contact_num]
    normal_force = data["normal_force"][:contact_num]
    tangential_force = data["tangential_force"][:contact_num]
    forces = normal_force + tangential_force

    branch = pos[end1] - pos[end2]
    branch[:, 0] -= Lx * np.round(branch[:, 0] / Lx)
    branch[:, 1] -= Ly * np.round(branch[:, 1] / Ly)
    sigma_pp = np.einsum("ci,cj->ij", forces, branch)

    ppos = pos[particle_id]
    pradius = particle["radius"][particle_id]
    wnorm = wall_normals[wall_id]
    d = wall_d[wall_id]
    Delta = np.einsum("ij,ij->i", ppos, wnorm) + d
    pw_branch = -Delta[:, None] * wnorm
    sigma_pw = np.einsum("ci,cj->ij", wall_forces, pw_branch)

    sigma = (sigma_pp) / total_v
    sigma = 0.5 * (sigma + sigma.T)
    love_stress.append(sigma[2, 2])


plt.plot(time, pressure, linestyle="-", marker="o", color=color[0], label="Wall stress")
plt.plot(time, love_stress, linestyle="--", marker="x", color=color[1], label="Particle stress")
plt.plot(time, np.repeat(245000, len(time)), color=color[2], label="Target stress")
plt.xlabel("Time, $T$ (s)")
plt.ylabel("$\\sigma$ / $\\sigma_{target}$")
plt.ylim([0, 500000])
plt.legend(frameon=False)
plt.savefig("stress_strain.png")
plt.close()

plt.plot(time, density, linestyle="-", color=color[0], label="case 1")
plt.xlabel("Time, $T$ (s)")
plt.ylabel("$\\rho$")
plt.legend(frameon=False)
plt.savefig("density.png")
plt.close()
