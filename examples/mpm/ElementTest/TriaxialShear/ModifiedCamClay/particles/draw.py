#!/usr/bin/env python
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style


p = []
q = []
pc = []
f = []
ratio = []
time = []
vel0 = []
vel1 = []
pid = 3280
    
m_theta = 1.5

def meanstress(stress):
    return (stress[2]+stress[1]+stress[0])/3.
    
def equistress(stress):
    return math.sqrt(3.*(((stress[0] - stress[1]) * (stress[0] - stress[1]) \
                          + (stress[1] - stress[2]) * (stress[1] - stress[2]) \
                          + (stress[0] - stress[2]) * (stress[0] - stress[2])) / 6. \
                          + stress[3] * stress[3] + stress[4] * stress[4] + stress[5] * stress[5]))

start_num=10
end_num=61
for printNum in range(start_num, end_num):
    data = np.load('MPMParticle{0:06d}.npz'.format(printNum), allow_pickle=True)
    
    vel0.append(data['velocity'][0][0])
    vel1.append(data['velocity'][4][0])
    p.append(-meanstress(data['stress'][pid]))
    q.append(equistress(data['stress'][pid]))
    pc.append(data['state_vars'].item()['pc'][pid])
    f.append(m_theta**2*(meanstress(data['stress'][pid])**2+(data['state_vars'].item()['pc'][pid]*meanstress(data['stress'][pid])))+equistress(data['stress'][pid])**2)
    ratio.append(equistress(data['stress'][pid])/-meanstress(data['stress'][pid]))
    time.append((data['t_current']-0.1)*0.01)
print(q)
    
    

yield_p = np.linspace(0, 2*pc[0], 200)    
yield_p0 = np.linspace(0, pc[0], 200)
yield_q0 = np.sqrt(m_theta**2*yield_p0*(pc[0]-yield_p0))

yield_p1 = np.linspace(0, pc[end_num-start_num-1], 200)
yield_q1 = np.sqrt(m_theta**2*yield_p1*(pc[end_num-start_num-1]-yield_p1))

CSL = m_theta*yield_p


fig = plt.figure(figsize=(15,6))
fig.suptitle('drained Test (Modified Cam Clay)', size=18)

ax1=plt.subplot(1,2,1)    
ax1.scatter(p, q, label='ti-MPM')
ax1.plot(yield_p0, yield_q0, label='initial yield surface')
ax1.plot(yield_p1, yield_q1, label='final yield surface')
ax1.plot(yield_p, CSL, label='critical state line')
ax1.set_xlabel("mean stress")
ax1.set_ylabel("equivalent stress")
ax1.legend(loc='best')

x=np.linspace(0, 0.3, 10)
y=np.zeros(10)+m_theta
ax2=plt.subplot(1,2,2)    
ax2.plot(time, ratio)
#ax2.plot(x, y)
#ax2.set_xlim([0,0.1])
#ax2.set_ylim([0,2.0])
ax2.set_xlabel("axial strain")
ax2.set_ylabel("equivalent stress")


print((q[-1]-q[0])/(p[-1]-p[0]))

plt.show()

