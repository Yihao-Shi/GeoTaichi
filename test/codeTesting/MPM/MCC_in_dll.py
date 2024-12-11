import matplotlib.pyplot as plt
import numpy
from ctypes import cdll

dllpath = '././third_party/A3DModifiedCamClay.dll'
mcc = cdll.LoadLibrary(dllpath)


possion = 0.25
lambda_ = 0.12
kappa = 0.023
m_theta = 1.02
void_ratio0 = 1.7
pc0 = 392
p0 = 98
strain_rate = 0.0005
strain = 0.08
nsub = int(strain / strain_rate)



yield_p = numpy.linspace(0, pc0, 200)
yield_q = numpy.sqrt(m_theta**2*yield_p*(pc0-yield_p))
CSL = m_theta*yield_p

fig = plt.figure(figsize=(18,9))
fig.suptitle('Undrained Test (Modified Cam Clay)', size=24)

ax1=plt.subplot(1,2,1)
ax1.plot(yield_p, yield_q,'k-',lw=2,label='Initial yield surface')
ax1.plot(yield_p, CSL,'r-',lw=3,label='Critical state line')
ax1.plot(series[:,1],series[:,2],'b--')
ax1.set_xlabel("$p$ (Pa)", size=18)
ax1.set_ylabel("$q$ (Pa)", size=18)
ax1.annotate('Stress Path', xy=(0.5, 1), xytext=(0, 12), xycoords='axes fraction', textcoords='offset points', size=22, ha='center', va='center')

ax2=plt.subplot(1,2,2)
ax2.plot(series[:,0],series[:,2],'b--')
ax2.set_xlabel("$\epsilon$", size=18)
ax2.set_ylabel("$q$ (Pa)", size=18)
ax2.annotate('q-t', xy=(0.5, 1), xytext=(0, 12), xycoords='axes fraction', textcoords='offset points', size=22, ha='center', va='center')

plt.show()

