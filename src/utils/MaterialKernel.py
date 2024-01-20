import taichi as ti

from src.utils.constants import EYE, ZEROMAT3x3, ZEROVEC6f, Threshold
from src.utils.ScalarFunction import clamp
from src.utils.TypeDefination import vec6f, vec3f


# ================================================================================ #
# Voigt Notation:                                                                  #
#                 notation1: tensor11,                                             #
#                 notation2: tensor22,                                             # 
#                 notation3: tensor33,                                             #
#                 notation4: tensor12 = tensor21,                                  #
#                 notation5: tensor23 = tensor32,                                  #
#                 notation6: tensor13 = tensor31;                                  #
# ================================================================================ #
# ========================== Constitutive Model Utility ========================== #
@ti.func
def calculate_strain_increment2D(velocity_gradient, dt):
    return calculate_strain_rate2D(velocity_gradient) * dt[None]

@ti.func
def calculate_strain_increment(velocity_gradient, dt):
    return calculate_strain_rate(velocity_gradient) * dt[None]

@ti.func
def calculate_vorticity_increment2D(velocity_gradient, dt):
    return 0.5 * calculate_vorticity_rate2D(velocity_gradient) * dt[None]

@ti.func
def calculate_vorticity_increment(velocity_gradient, dt):
    return 0.5 * calculate_vorticity_rate(velocity_gradient) * dt[None]

@ti.func
def calculate_strain_rate2D(velocity_gradient):
    return vec6f(velocity_gradient[0, 0],
                 velocity_gradient[1, 1],
                 0.,
                 velocity_gradient[0, 1] + velocity_gradient[1, 0], 0., 0.)

@ti.func
def calculate_strain_rate(velocity_gradient):
    return vec6f(velocity_gradient[0, 0],
                 velocity_gradient[1, 1],
                 velocity_gradient[2, 2],
                 velocity_gradient[0, 1] + velocity_gradient[1, 0],
                 velocity_gradient[1, 2] + velocity_gradient[2, 1],
                 velocity_gradient[0, 2] + velocity_gradient[2, 0])

@ti.func
def calculate_vorticity_rate2D(velocity_gradient):
    return vec3f(velocity_gradient[1, 0] - velocity_gradient[0, 1], 0., 0.)

@ti.func
def calculate_vorticity_rate(velocity_gradient):
    return vec3f(velocity_gradient[1, 0] - velocity_gradient[0, 1],
                 velocity_gradient[2, 1] - velocity_gradient[1, 2],
                 velocity_gradient[2, 0] - velocity_gradient[0, 2])

@ti.kernel
def VisualizeStressByVonMises(particleNum: ti.types.ndarray(), particle: ti.template(), stateVars: ti.template()):
    for np in range(particleNum[0]):
        stateVars[np].estress = VonMisesStress(particle[np].stress)

@ti.kernel
def VisualizeStressByMean(particleNum: ti.types.ndarray(), particle: ti.template(), stateVars: ti.template()):
    for np in range(particleNum[0]):
        stateVars[np].estress = MeanStress(particle[np].stress)

@ti.func
def AssembleStress(sigma, deviatoric_stress):
    return deviatoric_stress + sigma * EYE

@ti.func
def Sigrot(stress, dw):
    sigrot = ZEROVEC6f
    sigrot[0] = 2. * (dw[0] * stress[3] + dw[2] * stress[5])
    sigrot[1] = 2. * (-dw[0] * stress[3] + dw[1] * stress[4])
    sigrot[2] = -2. * (dw[2] * stress[5] + dw[1] * stress[4])
    sigrot[3] = dw[1] * stress[5] + dw[2] * stress[4] + dw[0] * (stress[1] - stress[0])
    sigrot[4] = -dw[0] * stress[5] - dw[2] * stress[3] + dw[1] * (stress[2] - stress[1])
    sigrot[5] = dw[0] * stress[4] - dw[1] * stress[3] + dw[2] * (stress[2] - stress[0])
    return sigrot

@ti.func
def MeanStress(stress):
    sigma = (stress[0] + stress[1] + stress[2]) / 3.
    return sigma

@ti.func
def DeviatoricStress(stress):
    sigma = MeanStress(stress)

    deviatoric_stress = stress
    for i in ti.static(range(3)):
        deviatoric_stress[i] -= sigma
    return deviatoric_stress

@ti.func
def VonMisesStress(stress):
    return ti.sqrt(0.5 * ((stress[0] - stress[1]) * (stress[0] - stress[1]) \
                        + (stress[1] - stress[2]) * (stress[1] - stress[2]) \
                        + (stress[0] - stress[2]) * (stress[0] - stress[2])))

@ti.func
def VonMisesPKStress(stress):
    return ti.sqrt(0.5 * ((stress[0, 0] - stress[1, 1]) * (stress[0, 0] - stress[1, 1]) \
                        + (stress[1, 1] - stress[2, 2]) * (stress[1, 1] - stress[2, 2]) \
                        + (stress[0, 0] - stress[2, 2]) * (stress[0, 0] - stress[2, 2])))

@ti.func
def ComputeInvariantJ2(stress):
    J2 = ((stress[0] - stress[1]) * (stress[0] - stress[1]) \
        + (stress[1] - stress[2]) * (stress[1] - stress[2]) \
        + (stress[0] - stress[2]) * (stress[0] - stress[2])) / 6. \
        + stress[3] * stress[3] + stress[4] * stress[4] + stress[5] * stress[5]
    return J2

@ti.func
def ComputeInvariantJ3(stress):
    deviatoric_stress = DeviatoricStress(stress)
    J3 = deviatoric_stress[0] * deviatoric_stress[1] * deviatoric_stress[2] \
        - deviatoric_stress[2] * deviatoric_stress[3] * deviatoric_stress[3] \
        + 2 * deviatoric_stress[3] * deviatoric_stress[4] * deviatoric_stress[5] \
        - deviatoric_stress[0] * deviatoric_stress[4] * deviatoric_stress[4] \
        - deviatoric_stress[1] * deviatoric_stress[5] * deviatoric_stress[5] 
    return J3

@ti.func
def EquivalentStress(stress):
    J2 = ComputeInvariantJ2(stress)
    seqv = ti.sqrt(3 * J2)
    return seqv

@ti.func
def ComputeLodeAngle(stress):
    J2 = ComputeInvariantJ2(stress)
    J3 = ComputeInvariantJ3(stress)

    load_angle = 0.
    if ti.abs(J2) > Threshold:
        load_angle = (3. * ti.sqrt(3.)/2.) * (J3 / (J2 ** 1.5))
    load_angle = clamp(-1.0, 1.0, load_angle)
    return 1./3. * ti.acos(load_angle)

@ti.func
def DpDsigma():
    dp_dsigma = ZEROVEC6f
    dp_dsigma[0] = 1./3.
    dp_dsigma[1] = 1./3.
    dp_dsigma[2] = 1./3.
    return dp_dsigma

@ti.func
def DqDsigma(stress):
    seqv = EquivalentStress(stress)
    deviatoric_stress = DeviatoricStress(stress)
    dq_dsigma = ZEROVEC6f
    if ti.abs(seqv) > Threshold:
        dq_dsigma[0] = 3. / (2. * seqv) * deviatoric_stress[0]
        dq_dsigma[1] = 3. / (2. * seqv) * deviatoric_stress[1]
        dq_dsigma[2] = 3. / (2. * seqv) * deviatoric_stress[2]
        dq_dsigma[3] = 3. / seqv * deviatoric_stress[3]
        dq_dsigma[4] = 3. / seqv * deviatoric_stress[4]
        dq_dsigma[5] = 3. / seqv * deviatoric_stress[5]
    return dq_dsigma

@ti.func
def Dj2Dsigma(stress):
    dj2_dsigma = DeviatoricStress(stress)
    dj2_dsigma[3] *= 2.
    dj2_dsigma[4] *= 2.
    dj2_dsigma[5] *= 2.
    return dj2_dsigma

@ti.func
def Dj3Dsigma(stress):
    J2 = ComputeInvariantJ2(stress)
    deviatoric_stress = DeviatoricStress(stress)

    dev1 = ti.Matrix.zero(float, 3)
    dev1[0] = deviatoric_stress[0]
    dev1[1] = deviatoric_stress[3]
    dev1[2] = deviatoric_stress[5]

    dev2 = ti.Matrix.zero(float, 3)
    dev2[0] = deviatoric_stress[3]
    dev2[1] = deviatoric_stress[1]
    dev2[2] = deviatoric_stress[4]

    dev3 = ti.Matrix.zero(float, 3)
    dev3[0] = deviatoric_stress[5]
    dev3[1] = deviatoric_stress[4]
    dev3[2] = deviatoric_stress[2]

    dj3_dsigma = ZEROVEC6f
    dj3_dsigma[0] = dev1.dot(dev1) - 2./3. * J2
    dj3_dsigma[1] = dev2.dot(dev2) - 2./3. * J2
    dj3_dsigma[2] = dev3.dot(dev3) - 2./3. * J2
    dj3_dsigma[3] = 2. * dev1.dot(dev2) 
    dj3_dsigma[4] = 2. * dev2.dot(dev3)
    dj3_dsigma[5] = 2. * dev1.dot(dev3)

    return dj3_dsigma

@ti.func
def DlodeDsigma(stress):
    J2 = ComputeInvariantJ2(stress)
    J3 = ComputeInvariantJ3(stress)
    dj2_dsigma = Dj2Dsigma(stress)
    dj3_dsigma = Dj3Dsigma(stress)

    dr_dj2 = -9./4.*ti.sqrt(3.) * J3
    dr_dj3 = 3./2.*ti.sqrt(3.)

    dtheta_dr = -1./3.

    if ti.abs(J2) > Threshold:
        r = J3 / 2. * (J2 / 3.) ** (-1.5)
        dr_dj2 *= J2 ** (-2.5)
        dr_dj3 *= J2 ** (-1.5)
        factor = ti.max(ti.abs(1 - r * r), Threshold)
        dtheta_dr = -1. / (3. * ti.sqrt(factor))

    return (dtheta_dr * (dr_dj2 * dj2_dsigma + dr_dj3 * dj3_dsigma))

@ti.func
def ElasticTensorMultiplyVector(vector, bulk_modulus, shear_modulus):
    a = bulk_modulus + (4./3.) * shear_modulus
    b = bulk_modulus - (2./3.) * shear_modulus
    return vec6f([a * vector[0] + b * (vector[1] + vector[2]),
                  a * vector[1] + b * (vector[0] + vector[2]),
                  a * vector[2] + b * (vector[0] + vector[1]),
                  shear_modulus * vector[3], shear_modulus * vector[4], shear_modulus * vector[5]])

@ti.func
def ComputePlasticDeviatoricStrain(plastic_strain):
    pdstrain = ti.sqrt(2./9. * ((plastic_strain[0] - plastic_strain[1]) * (plastic_strain[0] - plastic_strain[1]) \
                + (plastic_strain[1] - plastic_strain[2]) * (plastic_strain[1] - plastic_strain[2]) \
                + (plastic_strain[0] - plastic_strain[2]) * (plastic_strain[0] - plastic_strain[2])) \
                + 1./3. * plastic_strain[3] * plastic_strain[3] + plastic_strain[4] * plastic_strain[4] + plastic_strain[5] * plastic_strain[5])
    return pdstrain

@ti.func
def ThermodynamicPressure(np, matID, matProps, particle, gamma=1):
    bulk_modulus = matProps[matID].modulus
    dvolumertic = particle[np].J

    pressure = -bulk_modulus * (dvolumertic ** gamma - 1)
    return pressure

@ti.func
def VigotVec2Tensor(vector):
    matrix_tensor = ZEROMAT3x3
    matrix_tensor[0, 0] = vector[0]
    matrix_tensor[0, 1] = vector[3]
    matrix_tensor[0, 2] = vector[5]
    matrix_tensor[1, 0] = vector[3]
    matrix_tensor[1, 1] = vector[1]
    matrix_tensor[1, 2] = vector[4]
    matrix_tensor[2, 0] = vector[5]
    matrix_tensor[2, 1] = vector[4]
    matrix_tensor[2, 2] = vector[2]
    return matrix_tensor

@ti.func
def Tensor2VigotVec(matrix_tensor):
    vector = ZEROVEC6f
    vector[0] = matrix_tensor[0, 0]
    vector[1] = matrix_tensor[1, 1]
    vector[2] = matrix_tensor[2, 2]
    vector[3] = matrix_tensor[1, 0]
    vector[4] = matrix_tensor[1, 2]
    vector[5] = matrix_tensor[2, 0]
    return vector

@ti.kernel
def find_max_sound_speed_(matProps: ti.template()) -> float:
    max_sound_speed = 0.
    for matID in range(matProps.shape[0]):
        if matID == 0: continue
        sound_speed = matProps[matID]._get_sound_speed()
        # TODO: utilize reduce max to accelerate
        ti.atomic_max(max_sound_speed, sound_speed)
    return max_sound_speed

@ti.kernel
def kernel_initial_state_variables(to_beg: int, to_end: int, particle: ti.template(), stateVars: ti.template(), matProps: ti.template()):
    for np in range(to_beg, to_end):
        stateVars[np]._initialize_vars(np, particle, matProps)

@ti.func
def ComputeElasticStiffnessTensor(np, bulk_modulus, shear_modulus, stiffness):
    a1 = bulk_modulus + (4./3.) * shear_modulus
    a2 = bulk_modulus - (2./3.) * shear_modulus
    stiffness[np][0, 0] = stiffness[np][1, 1] = stiffness[np][2, 2] = a1
    stiffness[np][0, 1] = stiffness[np][0, 2] = stiffness[np][1, 2] = a2
    stiffness[np][1, 0] = stiffness[np][2, 0] = stiffness[np][2, 1] = a2
    stiffness[np][3, 3] = stiffness[np][4, 4] = stiffness[np][5, 5] = shear_modulus

@ti.kernel
def compute_stiffness_matrix(stiffness_matrix: ti.template(), particleNum: int, particle: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        stress = particle[np].stress
        matProps[materialID].compute_stiffness_tensor(np, stress, stiffness_matrix, stateVars)

@ti.kernel
def compute_elastic_stiffness_matrix(stiffness_matrix: ti.template(), particleNum: int, particle: ti.template(), matProps: ti.template(), stateVars: ti.template()):
    for np in range(particleNum):
        materialID = int(particle[np].materialID)
        stress = particle[np].stress
        matProps[materialID].compute_elastic_tensor(np, stress, stiffness_matrix, stateVars)