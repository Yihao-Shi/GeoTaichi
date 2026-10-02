import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style

p0=200000
pc0=200000
m_theta=1.5
start=0
end=50
pid = 0
e0 = 1.042331204152

def calculate(path):
    e = []
    p = []
    zpos0=0.
    for printNum in range(start, end):
        data = np.load(path+'MPMParticle{0:06d}.npz'.format(printNum), allow_pickle=True)
        void_ratio = data['state_vars'].item()["void_ratio"][pid]
        stress = data['stress'][pid]
        
        e.append(void_ratio)
        p.append(math.log(-(stress[0]+stress[1]+stress[2])/3.))
    return e, p
        
e, p = calculate('')
yield_p0 = np.linspace(0, 2.5*pc0, 200)
yield_q0 = np.sqrt(m_theta**2*yield_p0*(pc0-yield_p0))

CSL = m_theta*yield_p0

fig = plt.figure(figsize=(12,6))
fig.suptitle('drained Test (Modified Cam Clay)', size=18)

ax1=plt.subplot(1,1,1)    
ax1.scatter(p, e, label='stress path')
#ax1.plot(yield_p0, yield_q0, label='initial yield surface')
#ax1.plot(yield_p0, CSL, label='critical state line')
ax1.set_xlabel("ln(p)")
ax1.set_ylabel("e")
ax1.legend(loc='best')
#ax1.set_xlim([0, 160000])
#ax1.set_ylim([0, 160000])
plt.show()
