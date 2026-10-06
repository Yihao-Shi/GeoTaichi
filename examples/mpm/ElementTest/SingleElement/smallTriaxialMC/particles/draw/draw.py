#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style


p = []
q = []
szz = []
time = []
p0 = 1000
c = 3000
fai = 30 * math.pi / 180
pid = 1
start_num = 10
end_num = 260


def meanstress(stress):
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


szz0 = 0.0
for printNum in range(start_num, end_num):

    data = np.load("MPMParticle{0:06d}.npz".format(printNum), allow_pickle=True)
    p.append(-meanstress(data["stress"][pid]))
    q.append(equistress(data["stress"][pid]))

    szz.append(-(data["stress"][pid][2] - szz0))
    time.append((data["t_current"] - 0.1) * 0.01)


fig = plt.figure(figsize=(12, 6))
fig.suptitle("drained Test (Modified Cam Clay)", size=18)

ax1 = plt.subplot(1, 2, 1)
ax1.plot(p, q)
ax1.set_xlabel("axial strain")
ax1.set_ylabel("equivalent stress")


ax2 = plt.subplot(1, 2, 2)
ax2.plot(time, q)
ax2.set_xlim([0, 0.04])

plt.show()
