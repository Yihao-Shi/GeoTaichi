import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

p0 = 2000
pc0 = 2000
m_theta = 1.02
start = 0
end = 75


def calculate(path):
    q = []
    epsilon = []
    zpos0 = 0.0
    for printNum in range(start, end):
        data = np.load(path + "MPMGrid{0:06d}.npz".format(printNum))
        """datap = np.load(path+'../particles/MPMParticle{0:06d}.npz'.format(printNum))
        position=datap['position']
        bodyID=datap['bodyID']
        coord=np.where(position[bodyID==0,2]>(np.max(position[bodyID==0,2])-0.025))
        xmin=np.min(position[coord,0])-0.025
        xmax=np.max(position[coord,0])+0.025
        ymin=np.min(position[coord,1])-0.025
        ymax=np.max(position[coord,1])+0.025
        area=(ymax-ymin)*(xmax-xmin)
        if printNum==start:
            zpos0=np.mean(position[bodyID==1, 2])
        zpos=np.mean(position[bodyID==1, 2])"""

        contact_force = data["contact_force"]
        vertical_stress = np.sum(contact_force[:, 1], 0)[1] / 0.5

        q.append(vertical_stress)
        if printNum <= 50:
            epsilon.append(data["t_current"] * 0.01 / 0.25)
        else:
            epsilon.append(0.02 - (data["t_current"] - 0.5) * 0.01 / 0.25)
    return q, epsilon


q, epsilon = calculate("")
print(q)

fig = plt.figure(figsize=(12, 6))
fig.suptitle("drained Test (Modified Cam Clay)", size=18)

ax1 = plt.subplot(1, 2, 1)
ax1.scatter(q, epsilon, label="stress path")
ax1.set_xlabel("mean stress")
ax1.set_ylabel("equivalent stress")
ax1.legend(loc="best")
ax1.set_xscale("log")
ax1.invert_yaxis()
# ax1.set_xlim([0, 160000])
# ax1.set_ylim([0, 160000])

"""ax2=plt.subplot(1,2,2)    
ax2.scatter(epslion, q, label='stress ratio')
ax2.set_xlabel("Axial strain")
ax2.set_ylabel("stress ratio")
ax2.legend(loc='best')"""
plt.show()
