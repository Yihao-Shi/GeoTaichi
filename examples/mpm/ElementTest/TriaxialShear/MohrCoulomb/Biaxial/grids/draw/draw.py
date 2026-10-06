import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import style

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

p0 = 100000
fai = 30.5 / 180 * math.pi
c = 8500
start = 10
end = 61


def calculate(path):
    q = []
    p = []
    q_p = []
    epslion = []
    time = []
    vertical_stress0 = 0.0
    for printNum in range(start, end):
        data = np.load(path + "MPMGrid{0:06d}.npz".format(printNum))

        contact_force = data["contact_force"]
        vertical_stress = np.sum(contact_force[:, 1], 0)[2] / 1.0

        q.append(vertical_stress - p0)
        p.append((vertical_stress + 2 * p0) / 3.0)
        q_p.append((vertical_stress - p0) / ((vertical_stress + 2 * p0) / 3.0))
        epslion.append((data["t_current"] - 0.1) * 0.01)
        time.append(data["t_current"])
    return p, q, q_p, epslion, time


p, q, q_p, epslion, time = calculate("")
q0 = (
    0 * np.array(epslion)
    + p0 * (1 + math.sin(fai)) / (1 - math.sin(fai))
    + 2 * c * math.cos(fai) / (1 - math.sin(fai))
    - p0
)
plt.plot(epslion, q, label="stress ratio")
plt.plot(epslion, q0, label="stress ratio")
plt.xlabel("Axial strain")
plt.ylabel("stress ratio")
plt.legend(loc="best")
plt.show()
