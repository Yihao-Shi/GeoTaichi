import taichi as ti
ti.init(default_fp=ti.f64, cpu_max_num_threads=1)
import matplotlib.pyplot as plt


strain_rate = 0.0002
strain = 0.008
nsub = int(strain / strain_rate)
res = ti.field(float, shape=(nsub+1, 3))


from src.mpm.BaseStruct import ParticleCloud
particle = ParticleCloud.field(shape=984)

from src.mpm.materials.infinitesimal_strain.ModifiedCamClay import ModifiedCamClay
material = ModifiedCamClay(2, 984)
material.model_initialization(materials=[{
                                         "MaterialID": 1,
                                         "Density":1530,
                                         "PossionRatio": 0.3,
                                         "StressRatio": 1.2,
                                         "lambda": 0.12,
                                         "kappa": 0.02,
                                         "void_ratio_ref":1.0,
                                         "pressure_ref": 392,
                                         "ConsolidationPressure":396
                                        }])
particle[983].materialID=1
particle[983].stress = [-392., -392., -392., -0., -0., -0.]
material.state_vars_initialize(983, 984, particle)
dt = ti.field(float, ())
dt[None] = 1
from src.utils.MaterialKernel import *
de = ti.Matrix([[0.005, 0., 0.], [0., -0.0100000, 0.], [0., 0., 0.]])

@ti.kernel
def test1(de: ti.types.matrix(3, 3, float), matProps: ti.template(), stateVars: ti.template(), particle: ti.template()): 
    particle[983].stress = matProps[1].ComputeStress2D(np=983,previous_stress=particle[983].stress,velocity_gradient=de,stateVars=stateVars,dt=dt)
    mtheta = matProps[1].m_theta
    p0 = stateVars[983].pc
    stress = -particle[983].stress
    p = MeanStress(stress)
    q = EquivalentStress(stress)
    F = mtheta*mtheta*(p**2-p0*p)+q**2
    print('\n')
    print(particle[983].stress, material.stateVars[983].epstrain, EquivalentStress(particle[983].stress), F)
    
test1(de, material.matProps, material.stateVars, particle)

'''from Library.MPMLib3D.materials.ModifiedCamClay_copy import ModifiedCamClay as ModifiedCamClay1
material1 = ModifiedCamClay1(1, 1)
material1.model_initialization(materials=[{
                                         "MaterialID": 0,
                                         "Density":1530,
                                         "PossionRatio": 0.25,
                                         "StressRatio": 1.02,
                                         "lambda": 0.12,
                                         "kappa": 0.023,
                                         "void_ratio_ref":1.7,
                                         "ConsolidationPressure":396
                                        }])
particle1 = ParticleCloud.field(shape=1)
particle1[983].stress = [-392., -392., -392., -0., -0., -0.]
material1.state_vars_initialize(983, 984, particle1)
dt1 = ti.field(float, ())
dt1[None] = 1

dw = ti.Vector([0., 0., 0.])

@ti.kernel
def test2(de: ti.types.vector(6, float), matProps: ti.template(), stateVars: ti.template(), particle: ti.template()):
    material1.matProps[0].ComputeStress(np=983,strain_rate=de,vort_rate=dw,stateVars=material1.stateVars,particle=particle1,dt=dt)
    mtheta = matProps[0].m_theta
    p0 = stateVars[983].pc
    stress = particle[983].stress
    p = -MeanStress(stress)
    q = EquivalentStress(stress)
    F = mtheta*mtheta*(p**2-p0*p)+q**2
    print('\n')
    print(particle1[983].stress, material1.stateVars[0].epstrain, F)
    
test2(de, material.matProps, material.stateVars, particle)'''

'''material.model_initialization(materials={
                               "MaterialID":                    0,
                               "Density":                       1530,
                               "PossionRatio":                  0.3,
                               "StressRatio":                   1.3,
                               "lambda":                        0.161,
                               "kappa":                         0.062,
                               "void_ratio_ref":                1.76,
                               "ConsolidationPressure":         75002.294469681146
                 })
particle[983].stress = [-65017.363492487129, -65017.363492487129, -85042.757069157669, 0.000000000000, 0.000000000000, 0.000000000000]
material.state_vars_initialize(983, 984, particle)
de = ti.Vector([0.000000000000, 0.000000000000, -0.000000050000, 0.000000000000, 0.000000000000, 0.000000000000])
dw = ti.Vector([0., 0., 0.])
test1(de, material.matProps, material.stateVars, particle)'''

