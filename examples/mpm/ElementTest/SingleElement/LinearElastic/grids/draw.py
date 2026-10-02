import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

pc0=100000
m_theta=1.5
start=10
end=101

def calculate(path):
    q = []
    p = []
    epslion = []
    time = []
    for printNum in range(start, end):
        data = np.load(path+'MPMGrid{0:06d}.npz'.format(printNum))
        datap = np.load(path+'../particles/MPMParticle{0:06d}.npz'.format(printNum))
        
        position=datap['position']
        bodyID=datap['bodyID']

        coord=np.where(position[bodyID==0,2]>(np.max(position[bodyID==0,2])-0.025))
        xmin=np.min(position[coord,0])-0.25
        xmax=np.max(position[coord,0])+0.25
        ymin=np.min(position[coord,1])-0.25
        ymax=np.max(position[coord,1])+0.25
        area=(ymax-ymin)*(xmax-xmin)
        
        vertical_stress = 0.
        contact_force = data['contact_force']
        for i in range(contact_force.shape[0]):
            vertical_stress += contact_force[i, 1][2]
                
        vertical_stress = vertical_stress / area
        if printNum==10: p0=vertical_stress
        print(area)
        
        q.append(vertical_stress-p0)
        p.append(1./3.*(vertical_stress+2*p0))
        epslion.append((data['t_current']-0.1)*0.01)
        time.append(data['t_current'])
    return p, q, epslion, time
        
p, q, epslion, time = calculate('')
print((q[-1]-q[0])/(p[-1]-p[0]))
yield_p0 = np.linspace(0, pc0, 200)
yield_q0 = np.sqrt(m_theta**2*yield_p0*(pc0-yield_p0))


CSL = m_theta*yield_p0


fig = plt.figure(figsize=(12,6))
fig.suptitle('drained Test (Modified Cam Clay)', size=18)

ax1=plt.subplot(1,2,1)    
ax1.scatter(p, q, label='stress path')
ax1.plot(yield_p0, yield_q0, label='initial yield surface')
ax1.plot(yield_p0, CSL, label='critical state line')
ax1.set_xlabel("mean stress")
ax1.set_ylabel("equivalent stress")
ax1.legend(loc='best')
#ax1.set_xlim([0, 160000])
#ax1.set_ylim([0, 160000])

ax2=plt.subplot(1,2,2)    
ax2.scatter(epslion, q, label='stress ratio')
ax2.set_xlabel("Axial strain")
ax2.set_ylabel("stress ratio")
ax2.legend(loc='best')
plt.show()
