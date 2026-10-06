#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

bottom_pressure = []
top_pressure = []
vol0 = 0.0
targs = 200000


time = []

start_num = 0
end_num = 21
for printNum in range(start_num, end_num):
    data = np.load("walls/DEMWall{0:06d}.npz".format(printNum), allow_pickle=True)

    bottom_force = -(data["contact_force"][0][2] + data["contact_force"][1][2])

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

    vol0 = (up_position - down_position) * 0.09

    bottom_pressure.append(bottom_force / 0.09)

    time.append(data["t_current"])


plt.scatter(time, bottom_pressure)
# plt.ylim([100000, 300000])
plt.show()

data = np.load("particles/DEMParticle{0:06d}.npz".format(end_num - 1), allow_pickle=True)
particle_vol = 4.0 / 3.0 * math.pi * (np.power(data["radius"], 3).sum())
print(f"Void Ratio: {(vol0-particle_vol)/particle_vol}")
print(f"Porosity: {(vol0-particle_vol)/vol0}")
