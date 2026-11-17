import numpy as np
import math
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

start = 30
end = 50


def cond1(path, mu):
    time0 = 0
    t, vel, forcex, forcez, theory = [], [], [], [], []
    for i in range(start, end):
        data = np.load(path + "DEMParticle{0:06d}.npz".format(i))
        if i == start:
            time0 = data["t_current"]
        vel.append(math.sqrt(data["velocity"][0][0] ** 2 + data["velocity"][0][1] ** 2 + data["velocity"][0][2] ** 2))
        time = data["t_current"] - time0
        forcex.append(math.sqrt(data["contact_force"][0][0] ** 2) / 1e5)
        forcez.append(math.sqrt(data["contact_force"][0][2] ** 2) / 1e5)
        theory.append(9.8 * time * math.sqrt(2.0) / 2.0 * (1 - mu))
        t.append(time)
    return t, vel, forcex, forcez, theory


def cond2(path):
    time0 = 0
    t, vel, forcex, forcez, theory = [], [], [], [], []
    for i in range(start, end):
        data = np.load(path + "DEMParticle{0:06d}.npz".format(i))
        if i == start:
            time0 = data["t_current"]
        vel.append(math.sqrt(data["velocity"][0][0] ** 2 + data["velocity"][0][1] ** 2 + data["velocity"][0][2] ** 2))
        forcex.append(math.sqrt(data["contact_force"][0][0] ** 2) / 1e5)
        forcez.append(math.sqrt(data["contact_force"][0][2] ** 2) / 1e5)
        time = data["t_current"] - time0
        theory.append(5.0 / 7.0 * 9.8 * time * math.sqrt(2.0) / 2.0)
        t.append(time)
    return t, vel, forcex, forcez, theory


t1, vel1, forcex1, forcez1, theory1 = cond1("iP2SContact/mu=0.1/particles/", 0.1)
t2, vel2, forcex2, forcez2, theory2 = cond1("iP2SContact/mu=0.2/particles/", 0.2)
# t3, vel3, forcex3, forcez3, theory3 = cond1("iP2SContact/mu=0.3/particles/",0.3)
# t4, vel4, forcex4, forcez4, theory4 = cond2("iP2SContact/mu=0.4/particles/")
t5, vel5, forcex5, forcez5, theory5 = cond2("iP2SContact/mu=0.5/particles/")
t6, vel6, forcex6, forcez6, theory6 = cond2("P2PContact/mu=0.1/particles/")

plt.scatter(t1, vel1, marker="^", color=color[1], label="$\mu$=0.1")
plt.scatter(t2, vel2, marker="h", color=color[2], label="$\mu$=0.2")
plt.scatter(t5, vel5, marker="*", color=color[3], label="$\mu$=0.5")
plt.plot(t1, theory1, color=color[0], label="Analytical solution")
plt.plot(t2, theory2, color=color[0])
plt.plot(t5, theory5, color=color[0])
plt.xlabel("Time $(s)$")
plt.ylabel("Velocity $(m/s)$")
plt.xlim([0, 2.0])
plt.ylim([0, 12])
plt.legend(frameon=False)
plt.savefig("figure1.svg")

plt.clf()
plt.plot(t1[1:end], forcez1[1:end], linestyle="-.", color=color[0], label="Normal force (analytical solution)")
plt.plot(t1[1:end], forcex1[1:end], linestyle="-.", color=color[1], label="tangential force (analytical solution)")
plt.plot(t6[1:end], forcez6[1:end], marker="^", color=color[2], label="Normal force (this study)")
plt.plot(t6[1:end], forcex6[1:end], marker="^", color=color[3], label="tangential force (this study)")
plt.xlabel("Time $(s)$")
plt.ylabel("Contact force $(10^5N)$")
plt.xlim([0, 2.0])
plt.ylim([0, 4])
plt.legend(frameon=False)
plt.savefig("figure2.svg")
