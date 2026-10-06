import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

height = 0.05
data = np.load("MPMParticle000000.npz")
posz0 = data["position"][200][2]

q = []
p = []
epslion = []
time = []
pid = 1000
for printNum in range(0, 50):
    data = np.load("MPMParticle{0:06d}.npz".format(printNum))
    stress = data["stress"][pid]
    p.append((stress[0] + stress[1] + stress[2]) / 3.0)
    q.append(
        math.sqrt(
            3.0
            * (
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
    posz = data["position"][200][2]
    epslion.append(abs(posz - posz0) / height)
    time.append(data["t_current"])

plt.plot(epslion, q)
plt.xlabel("$\epsilon_a$")
plt.ylabel("q")
plt.show()
