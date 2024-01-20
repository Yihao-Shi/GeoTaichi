import taichi as ti
ti.init(default_fp=ti.f64)
import matplotlib.pyplot as plt
import sys, os, numpy
sys.path.append('/home/eleven/work/GeoTaichi')


possion = 0.25
lambda_ = 0.12
kappa = 0.023
m_theta = 1.02
void_ratio0 = 1.7
pc0 = 392000
p0 = 306000
strain_rate = 0.0005
strain = 0.08
nsub = int(strain / strain_rate)
res = ti.field(float, shape=(nsub+1, 5))

from src.mpm.BaseStruct import ParticleCloud
particle = ParticleCloud.field(shape=1)
particle[0].stress = [-p0, -p0, -p0, 0., 0., 0.]
from src.mpm.materials.infinitesimal_strain.ModifiedCamClay import ModifiedCamClay
material = ModifiedCamClay(1, 1)
material.model_initialization(materials=[{
                                         "MaterialID": 0,
                                         "Density":1530,
                                         "PoissionRatio": possion,
                                         "StressRatio": m_theta,
                                         "lambda": lambda_,
                                         "kappa": kappa,
                                         "void_ratio_ref":void_ratio0,
                                         "ConsolidationPressure":pc0
                                        }])
material.state_vars_initialize(0, 1, particle)
dt = ti.field(float, ())
dt[None] = 1
from src.utils.MaterialKernel import MeanStress, EquivalentStress
@ti.kernel
def TraxialTest_undrained():
    p = MeanStress(-particle.stress[0])
    q = EquivalentStress(-particle.stress[0])
    res[0, 1] = p
    res[0, 2] = q
    res[0, 3] = pc0
    dep = strain_rate
    pc = pc0
    de = ti.Matrix([[-dep, 0., 0.], [0., 0.5*dep, 0.], [0., 0., 0.5*dep]])
    ti.loop_config(serialize=True)
    for i in range(nsub):
        material.matProps[0].ComputeStress(np=0,velocity_gradient=de,stateVars=material.stateVars,particle=particle,dt=dt)
        p = MeanStress(-particle.stress[0])
        q = EquivalentStress(-particle.stress[0])
        res[i+1, 0] = dep * (i + 1)
        res[i+1, 1] = p
        res[i+1, 2] = q
        res[i+1, 3] = pc


TraxialTest_undrained()

series = res.to_numpy()

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

