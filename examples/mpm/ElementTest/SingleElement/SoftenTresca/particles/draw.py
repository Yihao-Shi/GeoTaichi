#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style


p = []
q = []
ratio = []
time = []
pid = 4


def meanstress(stress):
    return (stress[2]+stress[1]+stress[0])/3.
    
def equistress(stress):
    return math.sqrt(3.*(((stress[0] - stress[1]) * (stress[0] - stress[1]) \
                          + (stress[1] - stress[2]) * (stress[1] - stress[2]) \
                          + (stress[0] - stress[2]) * (stress[0] - stress[2])) / 6. \
                          + stress[3] * stress[3] + stress[4] * stress[4] + stress[5] * stress[5]))

start_num=0
end_num=51
for printNum in range(start_num, end_num):
    data = np.load('MPMParticle{0:06d}.npz'.format(printNum), allow_pickle=True)
    
    p.append(-meanstress(data['stress'][pid]))
    q.append(equistress(data['stress'][pid]))
    ratio.append(equistress(data['stress'][pid])/-meanstress(data['stress'][pid]))
    time.append(data['t_current']*0.01)
    print(data['stress'][pid])
    

fig = plt.figure(figsize=(18,6))
fig.suptitle('drained Test (MohrCoulomb)', size=18)

ax1=plt.subplot(1,2,1)    
ax1.scatter(p, q, label='ti-MPM')
ax1.set_xlabel("mean stress")
ax1.set_ylabel("equivalent stress")
ax1.legend(loc='best')

ax2=plt.subplot(1,2,2)    
ax2.plot(time, q)
ax2.set_xlabel("axial strain")
ax2.set_ylabel("equivalent stress")


print((q[-1]-q[0])/(p[-1]-p[0]))

plt.show()

