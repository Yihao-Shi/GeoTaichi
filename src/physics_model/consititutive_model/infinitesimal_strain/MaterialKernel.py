import taichi as ti

from src.physics_model.consititutive_model.MaterialKernel import calculate_vorticity_increment2D, calculate_vorticity_increment, \
                                                                 calculate_vorticity_rate2D, calculate_vorticity_rate
from src.utils.constants import EYE, ZEROMAT3x3, ZEROVEC6f, ZEROMAT6x6, Threshold
from src.utils.ScalarFunction import clamp
from src.utils.TypeDefination import vec6f

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
def calculate_strain_rate2D(velocity_gradient):
    return vec6f(velocity_gradient[0, 0],
                 velocity_gradient[1, 1],
                 0.,
                 0.5 * (velocity_gradient[0, 1] + velocity_gradient[1, 0]), 0., 0.)

@ti.func
def calculate_strain_rate(velocity_gradient):
    return vec6f(velocity_gradient[0, 0],
                 velocity_gradient[1, 1],
                 velocity_gradient[2, 2],
                 0.5 * (velocity_gradient[0, 1] + velocity_gradient[1, 0]),
                 0.5 * (velocity_gradient[1, 2] + velocity_gradient[2, 1]),
                 0.5 * (velocity_gradient[0, 2] + velocity_gradient[2, 0]))

@ti.kernel
def VisualizeStressByVonMises(particleNum: ti.types.ndarray(), particle: ti.template(), stateVars: ti.template()):
    for np in range(particleNum[0]):
        stateVars[np].estress = VonMisesStress(particle[np].stress)

@ti.kernel
def VisualizeStressByMean(particleNum: ti.types.ndarray(), particle: ti.template(), stateVars: ti.template()):
    for np in range(particleNum[0]):
        stateVars[np].estress = SphericalTensor(particle[np].stress)

@ti.func
def AssembleStress(sigma, deviatoric_stress):
    return deviatoric_stress + sigma * EYE

@ti.func
def Sigrot(stress, dw):
    sigrot = ZEROVEC6f
    sigrot[0] = 2. * (-dw[2] * stress[5] + dw[0] * stress[3])
    sigrot[1] = 2. * (-dw[0] * stress[3] + dw[1] * stress[4])
    sigrot[2] = 2. * (-dw[1] * stress[4] + dw[2] * stress[5])
    sigrot[3] = -dw[2] * stress[4] + dw[1] * stress[5] + dw[0] * (stress[1] - stress[0])
    sigrot[4] = -dw[0] * stress[5] + dw[2] * stress[3] + dw[1] * (stress[2] - stress[1])
    sigrot[5] = -dw[1] * stress[3] + dw[0] * stress[4] + dw[2] * (stress[0] - stress[2])
    return sigrot

@ti.func
def SphericalTensor(tensor):
    return (tensor[0] + tensor[1] + tensor[2]) / 3.

@ti.func
def DeviatoricTensor(tensor):
    sigma = SphericalTensor(tensor)
    return vec6f(tensor[0] - sigma, tensor[1] - sigma, tensor[2] - sigma, tensor[3], tensor[4], tensor[5])

@ti.func
def VonMisesStress(stress):
    return ti.sqrt(0.5 * ((stress[0] - stress[1]) * (stress[0] - stress[1]) \
                        + (stress[1] - stress[2]) * (stress[1] - stress[2]) \
                        + (stress[0] - stress[2]) * (stress[0] - stress[2])))

@ti.func
def ComputeDeviatoricStressTensor(stress):
    n = ZEROVEC6f
    p = -SphericalTensor(stress)
    q = EquivalentDeviatoricStress(stress)
    dev_stress = vec6f(stress[0], stress[1], stress[2], stress[3], stress[4], stress[5])
    for i in range(3): dev_stress[i] += p
    if q > Threshold: n = dev_stress / q
    return n

@ti.func
def EquivalentDeviatoricStress(stress):
    J2 = ComputeStressInvariantJ2(stress)
    return ti.sqrt(3 * J2)

@ti.func
def ComputeStressInvariantI1(stress):
    I1 = stress[0] + stress[1] + stress[2]
    return I1

@ti.func
def ComputeStressInvariantI2(stress):
    I2 = stress[0] * stress[1] + stress[1] * stress[2] + stress[0] * stress[2] \
         - stress[3] * stress[3] - stress[4] * stress[4] - stress[5] * stress[5]
    return I2

@ti.func
def ComputeStressInvariantI3(stress):
    I3 = stress[0] * stress[1] * stress[2] + 2. * stress[3] * stress[4] * stress[5] \
         - stress[0] * stress[4] * stress[4] - stress[1] * stress[5] * stress[5] - stress[2] * stress[3] * stress[3]
    return I3

@ti.func
def ComputeStressInvariantJ1(stress):
    J1 = 0.
    return J1

@ti.func
def ComputeStressInvariantJ2(stress):
    J2 = ((stress[0] - stress[1]) * (stress[0] - stress[1]) \
        + (stress[1] - stress[2]) * (stress[1] - stress[2]) \
        + (stress[0] - stress[2]) * (stress[0] - stress[2])) / 6. \
        + stress[3] * stress[3] + stress[4] * stress[4] + stress[5] * stress[5]
    return J2

@ti.func
def ComputeStressInvariantJ3(stress):
    deviatoric_stress = DeviatoricTensor(stress)
    J3 = deviatoric_stress[0] * deviatoric_stress[1] * deviatoric_stress[2] \
        + 2 * deviatoric_stress[3] * deviatoric_stress[4] * deviatoric_stress[5] \
        - deviatoric_stress[2] * deviatoric_stress[3] * deviatoric_stress[3] \
        - deviatoric_stress[0] * deviatoric_stress[4] * deviatoric_stress[4] \
        - deviatoric_stress[1] * deviatoric_stress[5] * deviatoric_stress[5] 
    return J3

@ti.func
def ComputeLodeAngle(stress):
    J2 = ComputeStressInvariantJ2(stress)
    J3 = ComputeStressInvariantJ3(stress)
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
    seqv = EquivalentDeviatoricStress(stress)
    iseqv = 1./seqv if ti.abs(seqv) > Threshold else 0.
    deviatoric_stress = DeviatoricTensor(stress)
    factor1 = 1.5 * iseqv
    # TODO: need to multiply 2?
    factor2 = factor1
    return vec6f(factor1 * deviatoric_stress[0], factor1 * deviatoric_stress[1], factor1 * deviatoric_stress[2],
                 factor2 * deviatoric_stress[3], factor2 * deviatoric_stress[4], factor2 * deviatoric_stress[5])

@ti.func
def Dj2Dsigma(stress):
    dj2_dsigma = DeviatoricTensor(stress)
    # TODO: need to multiply 2?
    #dj2_dsigma[3] *= 2.
    #dj2_dsigma[4] *= 2.
    #dj2_dsigma[5] *= 2.
    return dj2_dsigma

@ti.func
def Dj3Dsigma(stress):
    J2 = ComputeStressInvariantJ2(stress)
    deviatoric_stress = DeviatoricTensor(stress)

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
    # TODO: need to multiply 2?
    dj3_dsigma[3] = dev1.dot(dev2) #2. * dev1.dot(dev2) 
    dj3_dsigma[4] = dev2.dot(dev3) #2. * dev2.dot(dev3)
    dj3_dsigma[5] = dev1.dot(dev3) #2. * dev1.dot(dev3)
    return dj3_dsigma

@ti.func
def DlodeDsigma(stress):
    J2 = ComputeStressInvariantJ2(stress)
    J3 = ComputeStressInvariantJ3(stress)
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
def EquivalentStrain(strain):
    return ti.sqrt(2./3. * ComputeStrainInvariantI2(strain))

@ti.func
def EquivalentDeviatoricStrain(strain):
    J2 = ComputeStrainInvariantJ2(strain)
    return ti.sqrt(2./3. * J2)

@ti.func
def ComputeStrainInvariantI1(strain):
    I1 = strain[0] + strain[1] + strain[2]
    return I1

@ti.func
def ComputeStrainInvariantI2(strain):
    I2 = strain[0] * strain[1] + strain[1] * strain[2] + strain[0] * strain[2] \
         - (strain[3] * strain[3] - strain[4] * strain[4] - strain[5] * strain[5])
    return I2

@ti.func
def ComputeStrainInvariantI3(strain):
    I3 = strain[0] * strain[1] * strain[2] + (2. * strain[3] * strain[4] * strain[5] \
         - strain[0] * strain[4] * strain[4] - strain[1] * strain[5] * strain[5] - strain[2] * strain[3] * strain[3])
    return I3

@ti.func
def ComputeStrainInvariantJ1(strain):
    return 0.

@ti.func
def ComputeStrainInvariantJ2(strain):
    J2 = ((strain[0] - strain[1]) * (strain[0] - strain[1]) \
        + (strain[1] - strain[2]) * (strain[1] - strain[2]) \
        + (strain[0] - strain[2]) * (strain[0] - strain[2])) / 6. \
        + (strain[3] * strain[3] + strain[4] * strain[4] + strain[5] * strain[5])
    return J2

@ti.func
def ComputeStrainInvariantJ3(strain):
    deviatoric_strain = DeviatoricTensor(strain)
    J3 = deviatoric_strain[0] * deviatoric_strain[1] * deviatoric_strain[2] \
        + (2. * deviatoric_strain[3] * deviatoric_strain[4] * deviatoric_strain[5] \
        - deviatoric_strain[0] * deviatoric_strain[4] * deviatoric_strain[4] \
        - deviatoric_strain[1] * deviatoric_strain[5] * deviatoric_strain[5] \
        - deviatoric_strain[2] * deviatoric_strain[3] * deviatoric_strain[3])
    return J3

@ti.func
def DeqepsilonvDepsilon():
    return vec6f(1., 1., 1., 0., 0., 0.)

@ti.func
def DeqepsilonqDepsilon(strain):
    seqv = EquivalentDeviatoricStrain(strain)
    iseqv = 1./seqv if ti.abs(seqv) > Threshold else 0.
    deviatoric_strain = DeviatoricTensor(strain)
    factor = 2. / 3. * iseqv
    return factor * deviatoric_strain

@ti.func
def DeqepsilonDepsilon(strain):
    dpstrain = EquivalentStrain(strain)
    deqepsilonq_depsilon = 2./3. * strain / dpstrain
    return deqepsilonq_depsilon 

@ti.func
def ElasticTensorMultiplyVector(vector, bulk_modulus, shear_modulus):
    a = bulk_modulus + (4./3.) * shear_modulus
    b = bulk_modulus - (2./3.) * shear_modulus
    c = 2. * shear_modulus
    return vec6f([a * vector[0] + b * (vector[1] + vector[2]),
                  a * vector[1] + b * (vector[0] + vector[2]),
                  a * vector[2] + b * (vector[0] + vector[1]),
                  c * vector[3], c * vector[4], c * vector[5]])

@ti.func
def AssembleMeanDeviaStress(dstress, mean):
    stress = vec6f(dstress[0], dstress[1], dstress[2], dstress[3], dstress[4], dstress[5])
    for i in ti.static(range(3)):
        stress[i] += mean
    return stress

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

@ti.func
def ComputeElasticStiffnessTensor(bulk_modulus, shear_modulus):
    smatrix = ZEROMAT6x6
    a1 = bulk_modulus + (4./3.) * shear_modulus
    a2 = bulk_modulus - (2./3.) * shear_modulus
    smatrix[0, 0] = smatrix[1, 1] = smatrix[2, 2] = a1
    smatrix[0, 1] = smatrix[0, 2] = smatrix[1, 2] = a2
    smatrix[1, 0] = smatrix[2, 0] = smatrix[2, 1] = a2
    smatrix[3, 3] = shear_modulus
    smatrix[4, 4] = shear_modulus
    smatrix[5, 5] = shear_modulus
    return smatrix

if __name__ == '__main__':
    ti.init(default_fp=ti.f64)
    import sys
    sys.path.append('/home/eleven/work/GeoTaichi')
    sigma = ti.Vector([20516., 321546., 2130., 21543., 202165., 84651.])
    strain = ti.Vector([0.512, 0.355, 20.21, 310.3, 215.2, 6.32])

    @ti.kernel
    def test(vector: ti.types.vector(6, float), value: ti.template(), gradient: ti.template()):
        delta = 1e-5*vector.norm()
        fdm = ti.Vector.zero(float, 6)
        analytical = gradient(sigma)
        for i in range(6):
            vector0 = vector
            vector0[i] += delta
            vector1 = vector
            vector1[i] -= delta
            fdm[i] = 0.5 * (value(vector0) - value(vector1)) / delta
        print("Finite Difference = ", fdm, "Analytical results = ", analytical, 'q = ', value(vector))

    @ti.func
    def mean(mat):
        return 1./3.*(mat[0,0]+mat[1,1]+mat[2,2])
    @ti.func
    def deviatoric(mat):
        return mat-mean(mat)*ti.Matrix.identity(float,3)
    @ti.func
    def double_dot(mat):
        result=0.
        for i in range(3):
            for j in range(3):
                result+=mat[i, j]**2
        return result
    @ti.func
    def q(mat):
        J2 = 0.5*double_dot(deviatoric(mat))
        return ti.sqrt(3.*J2)
    @ti.func
    def dqdsigma(mat):
        return 1.5/q(mat)*deviatoric(mat)
    @ti.func
    def norm(mat):
        return ti.sqrt(mat[0,0]+mat[0,1]+mat[0,2]+mat[1,0]+mat[1,1]+mat[1,2]+mat[2,0]+mat[2,1]+mat[2,2])

    @ti.kernel
    def test_tensor(matrix: ti.types.matrix(3, 3, float)):
        delta = 1e-5*norm(matrix)
        fdm = ti.Matrix.zero(float, 3, 3)
        analytical = dqdsigma(matrix)
        for i in range(3):
            for j in range(3):
                matrix0 = matrix
                matrix0[i,j] += delta
                matrix1 = matrix
                matrix1[i,j] -= delta
                fdm[i,j] = 0.5 * (q(matrix0) - q(matrix1)) / delta
        print("Finite Difference = ", fdm, "Analytical results = ", analytical, 'q = ', q(matrix))

    test(sigma, EquivalentDeviatoricStress, DqDsigma)
    test_tensor(ti.Matrix([[20516.,21543.,84651.],[21543.,321546.,202165.],[84651.,202165.,2130.]]))

