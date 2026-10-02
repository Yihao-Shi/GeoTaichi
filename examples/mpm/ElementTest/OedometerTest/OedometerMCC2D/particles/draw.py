#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style



sig1 = []
sig3 = []
epsilon = []
pid = 10000

start_num=0
end_num=130
for printNum in range(start_num, end_num):
    data = np.load('MPMParticle{0:06d}.npz'.format(printNum), allow_pickle=True)
    
    sig = -data['stress'][pid]
    stress = np.array([
    [sig[0], sig[3], sig[5]],
    [sig[3], sig[1], sig[4]],
    [sig[5], sig[4], sig[2]]
    ])
    eigvals, eigvecs = np.linalg.eigh(stress)
    sigma1 = np.max(eigvals)
    sigma3 = np.min(eigvals)
    sig1.append(sigma1)
    sig3.append(sigma3)
    if printNum <=100:
        epsilon.append(data['t_current']*0.005/0.25)
    else:
        epsilon.append(0.02-(data['t_current']-1.0)*0.005/0.25)

fig = plt.figure(figsize=(15,6))
fig.suptitle('drained Test (Modified Cam Clay)', size=18)

ax1=plt.subplot(1,1,1)    
ax1.scatter(sig1, epsilon, label='stress path')
ax1.set_xlabel("stress")
ax1.set_ylabel("strain")
ax1.legend(loc='best')
ax1.set_xscale('log')
ax1.invert_yaxis()

plt.show()

