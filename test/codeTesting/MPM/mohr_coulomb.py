import taichi as ti
ti.init(default_fp=ti.f64)
import matplotlib.pyplot as plt


strain_rate = 0.0002
strain = 0.008
nsub = int(strain / strain_rate)
res = ti.field(float, shape=(nsub+1, 3))


from src.mpm.BaseStruct import ParticleCloud
particle = ParticleCloud.field(shape=1)

from src.mpm.materials.infinitesimal_strain.MohrCoulomb import MohrCoulomb
material = MohrCoulomb(1, 1)
material.model_initialization(materials=[{
                                         "MaterialID":                   0,
                                         "Density":                      1000.,
                                         "YoungModulus":                 1e7,
                                         "PoissionRatio":                0.3,
                                         "Softening":                    True,
                                         "Cohesion":                     2000,
                                         "Friction":                     30.,
                                         "Dilation":                     15.,
                                         "ResidualCohesion":             1000.,
                                         "ResidualFriction":             0.,
                                         "ResidualDilation":             0.,
                                         "PlasticDevStrain":             1e-15,
                                         "ResidualPlasticDevStrain":     0.001,
                                         "Tensile":                      0.
                                     }])
material.state_vars_initialize(0, 1, particle)
dt = ti.field(float, ())
dt[None] = 1
from src.utils.MaterialKernel import *
de = ti.Vector([0., 0., 0., 0., 0., 0.])
dw = ti.Vector([0., 0., 0.])

@ti.kernel
def test():
    bulk_modulus = material.matProps[0].bulk
    shear_modulus = material.matProps[0].shear
    particle[0].stress = [-5000., -6000., -7000., -1000., -2000., -3000.]

    stress = particle[0].stress
    sigrot = Sigrot(stress, dw)
    dstress = ElasticTensorMultiplyVector(de, shear_modulus, bulk_modulus)
    trial_stress = stress + dstress + sigrot
    epsilon, sqrt2J2, lode = material.matProps[0].ComputeStressInvariant(trial_stress)
    print(epsilon, sqrt2J2, lode)
    
    material.stateVars[0].epstrain = 0.00009110433579
    material.stateVars[0].fai = 0.4758966569262
    material.stateVars[0].psi = 0.2379483284631
    material.stateVars[0].c = 1908.89566421
    print(material.stateVars[0].c)
    yield_state, yield_tension_trial, yield_shear_trial = material.matProps[0].YieldState(lode, sqrt2J2, epsilon, material.stateVars[0])
    print('\n')
    print(yield_state, yield_tension_trial, yield_shear_trial)
    
    df_dsigma_trial, dp_dsigma_trial, dp_dq_trial, softening_trial = material.matProps[0].Compute_DfDp(yield_state, sqrt2J2, lode, trial_stress, material.stateVars[0])
    print('\n')
    print(df_dsigma_trial, dp_dsigma_trial)
    
    de = ti.Vector([0.0001, 0., 0., 0.001, 0.002, 0.003])
    stress = particle[0].stress
    sigrot = Sigrot(stress, dw)
    dstress = ElasticTensorMultiplyVector(de, shear_modulus, bulk_modulus)
    trial_stress = stress + dstress + sigrot
    epsilon, sqrt2J2, lode = material.matProps[0].ComputeStressInvariant(trial_stress)
    print('\n')
    print(epsilon, sqrt2J2, lode, material.stateVars[0].fai, material.stateVars[0].psi, material.stateVars[0].c, material.stateVars[0].epstrain)
    
    
    yield_state, yield_tension_trial, yield_shear_trial = material.matProps[0].YieldState(lode, sqrt2J2, epsilon, material.stateVars[0])
    print('\n')
    print(yield_state, yield_tension_trial, yield_shear_trial)
    
    df_dsigma_trial, dp_dsigma_trial, dp_dq_trial, softening_trial = material.matProps[0].Compute_DfDp(yield_state, sqrt2J2, lode, trial_stress, material.stateVars[0])
    print('\n')
    print(df_dsigma_trial, dp_dsigma_trial)
    
    material.stateVars[0].epstrain = 0.00009110433579
    material.stateVars[0].fai = 0.4758966569262
    material.stateVars[0].psi = 0.2379483284631
    material.stateVars[0].c = 1908.89566421
    material.matProps[0].ComputeStress(np=0,strain_rate=de,vort=dw,stateVars=material.stateVars,particle=particle,dt=dt)
    print('\n')
    print(particle[0].stress, material.stateVars[0].epstrain)
    
    particle[0].stress = [-5000., -6000., -7000., -1000., -2000., -3000.]
    material.stateVars[0].epstrain = 0.
    material.stateVars[0].fai = 0.52359878
    material.stateVars[0].psi = 0.26179939
    material.stateVars[0].c = 2000.
    stress = particle[0].stress
    epsilon, sqrt2J2, lode = material.matProps[0].ComputeStressInvariant(stress)
    print('\n')
    print(epsilon, sqrt2J2, lode, material.stateVars[0].fai, material.stateVars[0].psi, material.stateVars[0].c, material.stateVars[0].epstrain)
    
    material.matProps[0].ComputeStress(np=0,strain_rate=de,vort=dw,stateVars=material.stateVars,particle=particle,dt=dt)
    print('\n')
    print(particle[0].stress, material.stateVars[0].epstrain)
    
    
    particle[0].stress = [-5000., -6000., -6000., -100., -200., -300.]
    stress = particle[0].stress
    material.stateVars[0].epstrain = 0.00244948974278
    material.stateVars[0].fai = 0.
    material.stateVars[0].psi = 0.
    material.stateVars[0].c = 1000.
    epsilon, sqrt2J2, lode = material.matProps[0].ComputeStressInvariant(stress)
    print('\n')
    print(epsilon, sqrt2J2, lode, material.stateVars[0].fai, material.stateVars[0].psi, material.stateVars[0].c)
    
    yield_state, yield_tension_trial, yield_shear_trial = material.matProps[0].YieldState(lode, sqrt2J2, epsilon, material.stateVars[0])
    print('\n')
    print(yield_state, yield_tension_trial, yield_shear_trial)
    
    df_dsigma_trial, dp_dsigma_trial, dp_dq_trial, softening_trial = material.matProps[0].Compute_DfDp(yield_state, sqrt2J2, lode, stress, material.stateVars[0])
    print('\n')
    print(df_dsigma_trial, dp_dsigma_trial)
    
    de = ti.Vector([0.0001, 0., 0., 0.001, 0.002, 0.003])
    stress = particle[0].stress
    sigrot = Sigrot(stress, dw)
    dstress = ElasticTensorMultiplyVector(de, shear_modulus, bulk_modulus)
    trial_stress = stress + dstress + sigrot
    epsilon, sqrt2J2, lode = material.matProps[0].ComputeStressInvariant(trial_stress)
    print('\n')
    print(epsilon, sqrt2J2, lode, material.stateVars[0].fai, material.stateVars[0].psi, material.stateVars[0].c, material.stateVars[0].epstrain)
    
    yield_state, yield_tension_trial, yield_shear_trial = material.matProps[0].YieldState(lode, sqrt2J2, epsilon, material.stateVars[0])
    print('\n')
    print(yield_state, yield_tension_trial, yield_shear_trial)
    
    df_dsigma_trial, dp_dsigma_trial, dp_dq_trial, softening_trial = material.matProps[0].Compute_DfDp(yield_state, sqrt2J2, lode, trial_stress, material.stateVars[0])
    print('\n')
    print(df_dsigma_trial, dp_dsigma_trial)
    
    material.matProps[0].ComputeStress(np=0,strain_rate=de,vort=dw,stateVars=material.stateVars,particle=particle,dt=dt)
    print('\n')
    print(particle[0].stress, material.stateVars[0].epstrain)
    
    '''particle[0].stress = [-5000., -6000., -7000., -1000., 0., 0.]
    de = ti.Vector([0.001, 0., 0., 0., 0., 0.])
    stress = particle[0].stress
    sigrot = Sigrot(stress, dw)
    dstress = ElasticTensorMultiplyVector(de, shear_modulus, bulk_modulus)
    trial_stress = stress + dstress + sigrot
    
    epsilon, sqrt2J2, lode = material.matProps[0].ComputeStressInvariant(trial_stress)
    print(epsilon, sqrt2J2, lode)
    yield_state, yield_tension_trial, yield_shear_trial = material.matProps[0].YieldState(lode, sqrt2J2, epsilon, variables)
    print(yield_state, yield_tension_trial, yield_shear_trial)
    df_dsigma_trial, dp_dsigma_trial, dp_dq_trial, softening_trial = material.matProps[0].Compute_DfDp(yield_state, sqrt2J2, lode, trial_stress, variables)
    print(df_dsigma_trial, dp_dsigma_trial)
    
    particle[0].stress = [-5000., -6000., -7000., -1000., 0., 0.]
    material.matProps[0].ComputeStress(np=0,strain_rate=de,vort=dw,stateVars=material.stateVars,particle=particle,dt=dt)
    print(particle[0].stress)'''
test()

'''
@ti.kernel
def TraxialTest_undrained():
    particle[0].stress = [-1000., -1000., -1000., 0., 0., 0.]
    p = MeanStress(particle[0].stress)
    q = particle[0].stress[2]
    res[0, 1] = p
    res[0, 2] = q
    dep = strain_rate
    de = ti.Vector([-dep, 0.5*dep, 0.5*dep, 0., 0., 0.])
    ti.loop_config(serialize=True)
    for i in range(nsub):
        material.matProps[0].ComputeStress(np=0,strain_rate=de,vort=ti.Vector([0., 0., 0.]),stateVars=material.stateVars,particle=particle,dt=dt)
        p = MeanStress(particle[0].stress)
        q = particle[0].stress[0]
        res[i+1, 0] = dep * (i + 1)
        res[i+1, 1] = p
        res[i+1, 2] = q


TraxialTest_undrained()

series = res.to_numpy()


fig = plt.figure(figsize=(18,9))
fig.suptitle('Undrained Test (Mohr Coulomb)', size=24)

ax1=plt.subplot(1,2,1)
ax1.plot(series[:,1],series[:,2],'b--')
ax1.set_xlim([-2000,0])
ax1.set_xlabel("$p$ (Pa)", size=18)
ax1.set_ylabel("$q$ (Pa)", size=18)
ax1.annotate('Stress Path', xy=(0.5, 1), xytext=(0, 12), xycoords='axes fraction', textcoords='offset points', size=22, ha='center', va='center')

ax2=plt.subplot(1,2,2)
ax2.plot(series[:,0],series[:,2],'b--')
ax2.set_xlabel("$\epsilon$", size=18)
ax2.set_ylabel("$q$ (Pa)", size=18)
ax2.annotate('q-t', xy=(0.5, 1), xytext=(0, 12), xycoords='axes fraction', textcoords='offset points', size=22, ha='center', va='center')

plt.show()
'''
