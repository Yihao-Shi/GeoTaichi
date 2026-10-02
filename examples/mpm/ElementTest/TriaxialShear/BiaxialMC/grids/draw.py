import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

p0=100000
fai=30./180*math.pi
c=100.
start=10
end=60
def calculate(path):
    q = []
    p = []
    q_p=[]
    epslion = []
    time = []
    vertical_stress0 = 0.
    for printNum in range(start, end):
        data = np.load(path+'MPMGrid{0:06d}.npz'.format(printNum))
        
        contact_force = data['contact_force']
        vertical_stress = np.sum(contact_force[:, 1], 0)[-1] / 1.
       
        q.append(vertical_stress-p0)
        p.append((vertical_stress+2*p0)/3.)
        q_p.append((vertical_stress-p0)/((vertical_stress+2*p0)/3.))
        epslion.append((data['t_current']-0.1)*0.01)
        time.append(data['t_current'])
    return p, q, q_p, epslion, time
        
p, q, q_p, epslion, time = calculate('')
q0 = 0*np.array(epslion)+p0*(1+math.sin(fai))/(1-math.sin(fai))+2*c*math.cos(fai)/(1-math.sin(fai))-p0
plt.plot(epslion, q, label='stress ratio')
plt.plot(epslion, q0, label='stress ratio')
plt.xlabel("Axial strain")
plt.ylabel("stress ratio")
plt.legend(loc='best')
plt.show()
