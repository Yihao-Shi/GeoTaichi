import numpy as np
import math
import matplotlib.pyplot as plt

t, vel, force, theory = [], [], [], []
for i in range(30, 50):
    data = np.load("DEMParticle{0:06d}.npz".format(i))
    vel.append(math.sqrt(data["velocity"][0][0] ** 2 + data["velocity"][0][1] ** 2 + data["velocity"][0][2] ** 2))
    time = data["t_current"] - 3
    force.append(math.sqrt(data["contact_force"][0][2] ** 2))
    theory.append(9.8 * time * math.sqrt(2.0) / 2.0 * (1 - 0.2))
    t.append(time)


plt.scatter(t, vel, marker="^", color="orange", label="Simulation: DEMPM_Taichi")
plt.plot(t, theory, color="black", label="Theory")
plt.xlabel("$t$ (s)")
plt.ylabel("$v$ (m/s)")
plt.legend(loc="best")
plt.show()


plt.scatter(t, force, marker="^", color="orange", label="Simulation: DEMPM_Taichi")
plt.plot(t, np.repeat(315068.88, len(t)))
plt.xlabel("$t$ (s)")
plt.ylabel("$v$ (m/s)")
plt.legend(loc="best")
plt.ylim([0, 800000])
plt.show()
