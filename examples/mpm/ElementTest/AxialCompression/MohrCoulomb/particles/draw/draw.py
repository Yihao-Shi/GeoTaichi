import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

height = 0.05
data = np.load("MPMParticle000000.npz")
posz0 = data["position"][200][2]
print(data["position"][1000][2])

sigmazz = []
sigmaxz = []
epslion = []
time = []
pid = 1852
for printNum in range(0, 99):
    data = np.load("MPMParticle{0:06d}.npz".format(printNum))
    sigmaxz.append((-data["stress"][pid][2] + data["stress"][pid][1]) / 2.0)
    sigmazz.append(-data["stress"][pid][2])
    posz = data["position"][200][2]
    epslion.append(abs(posz - posz0) / height)
    time.append(data["t_current"])


fig = plt.figure(figsize=(18, 6))
fig.suptitle("Triaxial Undrained Tests", fontsize=24)
ax1 = plt.subplot(1, 2, 1)
ax1.plot(epslion, sigmaxz)
ax1.set_xlabel("$\epsilon_a$")
ax1.set_ylabel("$\tau_x$")

ax2 = plt.subplot(1, 2, 2)
ax2.plot(epslion, sigmazz)
ax2.set_xlabel("$\epsilon_a$")
ax2.set_ylabel("$\sigma_z$")
plt.show()
