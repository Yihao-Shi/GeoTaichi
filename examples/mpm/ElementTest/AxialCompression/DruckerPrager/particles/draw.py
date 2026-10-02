import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

height = 0.05
data = np.load('MPMParticle000000.npz')
posz0 = data["position"][200][2]
print(data["position"][1000][2])

sigmazz = []
sigmaxz = []
epslion = []
time = []
pid = 1000
for printNum in range(0, 50):
    data = np.load('MPMParticle{0:06d}.npz'.format(printNum))
    sigmaxz.append(data['stress'][pid][4])
    sigmazz.append(-data['stress'][pid][2])
    posz = data["position"][200][2]
    epslion.append(abs(posz-posz0)/height)
    time.append(data['t_current'])
    
plt.plot(epslion, sigmazz)
plt.xlabel("$\epsilon_a$")
plt.ylabel("$\sigma_z$")
plt.show()
