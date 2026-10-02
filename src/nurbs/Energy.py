import numpy as np

class SurfaceEnergy:
    def __init__(self, dimension=2):
        if dimension == 2:
            self.dPsi_div_dF = self.dPsi_div_dF_2x2
            self.d2Psi_div_dF2 = self.d2Psi_div_dF2_2x2
            self.d2Psi_div_dF2_spd = self.d2Psi_div_dF2_spd_2x2
        elif dimension == 2:
            self.dPsi_div_dF = self.dPsi_div_dF_3x2
            self.d2Psi_div_dF2 = self.d2Psi_div_dF2_3x2
            self.d2Psi_div_dF2_spd = self.d2Psi_div_dF2_spd_3x2

    def Psi(self, deformation_gradient):
        raise NotImplementedError
    
    def dPsi_div_dF_2x2(self, deformation_gradient):
        raise NotImplementedError

    def d2Psi_div_dF2_2x2(self, deformation_gradient):
        raise NotImplementedError
    
    def d2Psi_div_dF2_spd_2x2(self, deformation_gradient):
        raise NotImplementedError
    
    def dPsi_div_dF_3x2(self, deformation_gradient):
        raise NotImplementedError

    def d2Psi_div_dF2_3x2(self, deformation_gradient):
        raise NotImplementedError
    
    def d2Psi_div_dF2_spd_3x2(self, deformation_gradient):
        raise NotImplementedError
    
    def dPsi_div_dx(self, deformation_gradient, dF_dx):  
        '''
        applying chain-rule, ∂Ψ/∂x ((nx3)x1) = ∂Ψ/∂F * ∂F/∂x
        '''
        return self.dPsi_div_dF(deformation_gradient) @ dF_dx

    def d2Psi_div_dx2(self, deformation_gradient, dF_dx):  
        '''
        applying chain-rule, ∂²Ψ/∂x² ((nx3)x(nx3)) = (∂F/∂x)^T * ∂²Ψ/∂F² * ∂F/∂x (note that ∂²F/∂x² = 0)
        '''
        return dF_dx.transpose() @ self.d2Psi_div_dF2(deformation_gradient) @ dF_dx
    
    def I1(self, sigma):
        return np.sum(sigma)
    
    def I2(self, sigma):
        return np.dot(sigma, sigma)
    
    def I3(self, sigma):
        return np.prod(sigma)


class ASAP(SurfaceEnergy):
    def __init__(self, dimension=2):
        super().__init__(dimension)

    def Psi(self, deformation_gradient):
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        return self.I2(sigma) - 2. * self.I1(sigma) + 2
    
    def dPsi_div_dF_2x2(self, deformation_gradient):
        '''
        return a 4x1 vector (dPsi_div_dF), index: [∂Ψ/∂F₀₀, ∂Ψ/∂F₁₀, ∂Ψ/∂F₀₁, ∂Ψ/∂F₁₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        rotation_matrix = U @ VT
        return 2 * (deformation_gradient - rotation_matrix).flatten(order='F')
    
    def dPsi_div_dF_3x2(self, deformation_gradient):
        '''
        return a 6x1 vector (dPsi_div_dF), index: [∂Ψ/∂F₀₀, ∂Ψ/∂F₁₀, ∂Ψ/∂F₂₀, ∂Ψ/∂F₀₁, ∂Ψ/∂F₁₁, ∂Ψ/∂F₂₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        rotation_matrix = (U @ np.array([[VT[0, 0], VT[0, 1]], [VT[1, 0], VT[1, 1]], [0., 0.]]))
        return 2 * (deformation_gradient - rotation_matrix).flatten(order='F')

    def d2Psi_div_dF2_2x2(self, deformation_gradient):
        '''
        return a 4x4 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        dPsi_dI1 = -2.
        dPsi_dI2 = 1.
        d2I1_dF2 = self.d2I1_div_dF2_2x2(U, sigma, VT)
        d2I2_dF2 = 2. * np.eye(4, 4)
        return dPsi_dI1 * d2I1_dF2 + dPsi_dI2 * d2I2_dF2

    def d2Psi_div_dF2_3x2(self, deformation_gradient):
        '''
        return a 6x6 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₂₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁, ∂(∂Ψ/∂F)/∂F₂₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        dPsi_dI1 = -2.
        dPsi_dI2 = 1.
        d2I1_dF2 = self.d2I1_div_dF2_3x2(U, sigma, VT)
        d2I2_dF2 = 2. * np.eye(6, 6)
        return dPsi_dI1 * d2I1_dF2 + dPsi_dI2 * d2I2_dF2

    def d2Psi_div_dF2_spd_2x2(self, deformation_gradient):
        '''
        return a 6x6 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₂₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁, ∂(∂Ψ/∂F)/∂F₂₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        Hq = np.zeros((6, 6))
        e1 = np.sqrt(0.5) * (U @ np.array([[0., 1.], [1., 0.]]) @ VT).flatten(order='F')
        Hq += 2. * np.outer(e1, e1)
        
        e2 = np.sqrt(0.5) * (U @ np.array([[0., -1.], [1., 0.]]) @ VT).flatten(order='F')
        Hq += max(0.0, 2. - 4.0 / (sigma[0] + sigma[1])) * np.outer(e2, e2)
        
        e3 = (U @ np.array([[1., 0.], [0., 0.]]) @ VT).flatten(order='F')
        Hq += 2. * np.outer(e3, e3)

        e4 = (U @ np.array([[0., 0.], [0., 1.]]) @ VT).flatten(order='F')
        Hq += 2. * np.outer(e4, e4)
        return Hq

    def d2Psi_div_dF2_spd_3x2(self, deformation_gradient):
        '''
        return a 6x6 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₂₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁, ∂(∂Ψ/∂F)/∂F₂₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        Hq = np.zeros((6, 6))
        e1 = np.sqrt(0.5) * (U @ np.array([[0., 1.], [1., 0.], [0., 0.]]) @ VT).flatten(order='F')
        Hq += 2. * np.outer(e1, e1)
        
        e2 = np.sqrt(0.5) * (U @ np.array([[0., -1.], [1., 0.], [0., 0.]]) @ VT).flatten(order='F')
        Hq += max(0.0, 2. - 4.0 / (sigma[0] + sigma[1])) * np.outer(e2, e2)
        
        e3 = (U @ np.array([[1., 0.], [0., 0.], [0., 0.]]) @ VT).flatten(order='F')
        Hq += 2. * np.outer(e3, e3)

        e4 = (U @ np.array([[0., 0.], [0., 1.], [0., 0.]]) @ VT).flatten(order='F')
        Hq += 2. * np.outer(e4, e4)
        return Hq

    def d2I1_div_dF2_2x2(self, U, sigma, VT):
        T = np.sqrt(0.5) * (U @ np.array([[0., -1.], [1., 0.]]) @ VT).flatten(order='F')
        return 2. / (sigma[0] + sigma[1]) * np.outer(T, T)

    def d2I1_div_dF2_3x2(self, U, sigma, VT):
        T = np.sqrt(0.5) * (U @ np.array([[0., -1.], [1., 0.], [0., 0.]]) @ VT).flatten(order='F')
        return 2. / (sigma[0] + sigma[1]) * np.outer(T, T)


class MIPS(SurfaceEnergy):
    def __init__(self, dimension=2):
        super().__init__(dimension)

    def Psi(self, deformation_gradient):
        U, sigma, V = np.linalg.svd(deformation_gradient)
        return self.I2(sigma) / self.I3(sigma)
    
    def dPsi_div_dF_2x2(self, deformation_gradient):
        '''
        return a 4x1 vector (dPsi_div_dF), index: [∂Ψ/∂F₀₀, ∂Ψ/∂F₁₀, ∂Ψ/∂F₀₁, ∂Ψ/∂F₁₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.I2(sigma)
        I3 = self.I3(sigma)
        inv_I3 = 1. / I3
        dPsi_dI2 = inv_I3
        dPsi_dI3 = -I2 * inv_I3 * inv_I3
        dI2_dF = 2. * deformation_gradient
        dI3_dF = U @ np.array([[sigma[1], 0.], [0., sigma[0]]]) @ VT
        return (dPsi_dI2 * dI2_dF + dPsi_dI3 * dI3_dF).flatten(order='F')
    
    def dPsi_div_dF_3x2(self, deformation_gradient):
        '''
        return a 6x1 vector (dPsi_div_dF), index: [∂Ψ/∂F₀₀, ∂Ψ/∂F₁₀, ∂Ψ/∂F₂₀, ∂Ψ/∂F₀₁, ∂Ψ/∂F₁₁, ∂Ψ/∂F₂₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.I2(sigma)
        I3 = self.I3(sigma)
        inv_I3 = 1. / I3
        dPsi_dI2 = inv_I3
        dPsi_dI3 = -I2 * inv_I3 * inv_I3
        dI2_dF = 2. * deformation_gradient
        dI3_dF = U @ np.array([[sigma[1], 0.], [0., sigma[0]], [0., 0.]]) @ VT
        return (dPsi_dI2 * dI2_dF + dPsi_dI3 * dI3_dF).flatten(order='F')

    def d2Psi_div_dF2_2x2(self, deformation_gradient):
        '''
        return a 4x4 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.I2(sigma)
        I3 = self.I3(sigma)
        dPsi_dI2 = 1. / I3
        dPsi_dI3 = -I2 / (I3 * I3)
        d2Psi_dI22 = 0.
        d2Psi_dI32 = 2 * I2 / (I3 * I3 * I3)
        d2Psi_dI2I3 = -1. / (I3 * I3)
        dI2_dF = (2. * deformation_gradient).flatten(order='F')
        dI3_dF = (U @ np.array([[sigma[1], 0.], [0., sigma[0]]]) @ VT).flatten(order='F')
        d2I2_dF2 = 2. * np.eye(4, 4)
        d2I3_dF2 = self.d2I3_div_dF2_2x2(U, sigma, VT)
        return dPsi_dI3 * d2I3_dF2 + dPsi_dI2 * d2I2_dF2 + d2Psi_dI22 * np.outer(dI2_dF, dI2_dF) + d2Psi_dI2I3 * (np.outer(dI2_dF, dI3_dF) + np.outer(dI3_dF, dI2_dF)) + d2Psi_dI32 * np.outer(dI3_dF, dI3_dF)

    def d2Psi_div_dF2_3x2(self, deformation_gradient):
        '''
        return a 6x6 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₂₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁, ∂(∂Ψ/∂F)/∂F₂₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.I2(sigma)
        I3 = self.I3(sigma)
        dPsi_dI2 = 1. / I3
        dPsi_dI3 = -I2 / (I3 * I3)
        d2Psi_dI22 = 0.
        d2Psi_dI32 = 2 * I2 / (I3 * I3 * I3)
        d2Psi_dI2I3 = -1. / (I3 * I3)
        dI2_dF = (2. * deformation_gradient).flatten(order='F')
        dI3_dF = (U @ np.array([[sigma[1], 0.], [0., sigma[0]], [0., 0.]]) @ VT).flatten(order='F')
        d2I2_dF2 = 2. * np.eye(6, 6)
        d2I3_dF2 = self.d2I3_div_dF2_3x2(U, sigma, VT)
        return dPsi_dI3 * d2I3_dF2 + dPsi_dI2 * d2I2_dF2 + d2Psi_dI22 * np.outer(dI2_dF, dI2_dF) + d2Psi_dI2I3 * (np.outer(dI2_dF, dI3_dF) + np.outer(dI3_dF, dI2_dF)) + d2Psi_dI32 * np.outer(dI3_dF, dI3_dF)
        
    def d2Psi_div_dF2_spd_2x2(self, deformation_gradient):
        '''
        return a 4x4 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.I2(sigma)
        I3 = self.I3(sigma)

        Hq = np.zeros((4, 4))
        e1 = np.sqrt(0.5) * (U @ np.array([[0., 1.], [1., 0.]]) @ VT).flatten(order='F')
        Hq += 2. * np.outer(e1, e1)
        
        e2 = np.sqrt(0.5) * (U @ np.array([[0., -1.], [1., 0.]]) @ VT).flatten(order='F')
        Hq += max(0.0, 2. - 4.0 / (sigma[0] + sigma[1])) * np.outer(e2, e2)
        
        alpha = np.sqrt(I2 * I2 - 3. * I3 * I3)
        beta = I2 / (sigma[1] * sigma[1] - sigma[0] * sigma[0] + alpha)
        gamma = np.sqrt(1. + beta * beta)
        d1 = (U @ np.array([[1., 0.], [0., 0.]]) @ VT).flatten(order='F')
        d2 = (U @ np.array([[0., 0.], [0., 1.]]) @ VT).flatten(order='F')
        e3 = 1. / gamma * (beta * d1 + d2)
        e4 = 1. / gamma * (d1 - beta * d2)
        Hq += max(0.0, I2 * (I2 - alpha) - 2. * I3 * I3) * np.outer(e3, e3)
        Hq += max(0.0, I2 * (I2 + alpha) - 2. * I3 * I3) * np.outer(e4, e4)
        return Hq
        
    def d2Psi_div_dF2_spd_3x2(self, deformation_gradient):
        '''
        return a 6x6 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₂₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁, ∂(∂Ψ/∂F)/∂F₂₁]
        '''
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.I2(sigma)
        I3 = self.I3(sigma)

        Hq = np.zeros((6, 6))
        e1 = np.sqrt(0.5) * (U @ np.array([[0., 1.], [1., 0.], [0., 0.]]) @ VT).flatten(order='F')
        Hq += 2. * np.outer(e1, e1)
        
        e2 = np.sqrt(0.5) * (U @ np.array([[0., -1.], [1., 0.], [0., 0.]]) @ VT).flatten(order='F')
        Hq += max(0.0, 2. - 4.0 / (sigma[0] + sigma[1])) * np.outer(e2, e2)
        
        alpha = np.sqrt(I2 * I2 - 3. * I3 * I3)
        beta = I2 / (sigma[1] * sigma[1] - sigma[0] * sigma[0] + alpha)
        gamma = np.sqrt(1. + beta * beta)
        d1 = (U @ np.array([[1., 0.], [0., 0.], [0., 0.]]) @ VT).flatten(order='F')
        d2 = (U @ np.array([[0., 0.], [0., 1.], [0., 0.]]) @ VT).flatten(order='F')
        e3 = 1. / gamma * (beta * d1 + d2)
        e4 = 1. / gamma * (d1 - beta * d2)
        Hq += max(0.0, I2 * (I2 - alpha) - 2. * I3 * I3) * np.outer(e3, e3)
        Hq += max(0.0, I2 * (I2 + alpha) - 2. * I3 * I3) * np.outer(e4, e4)
        return Hq
    
    def d2I3_div_dF2_2x2(self, U, sigma, VT):
        L = np.sqrt(0.5) * (U @ np.array([[0., 1.], [1., 0.]]) @ VT).flatten(order='F')
        P = np.sqrt(0.5) * (U @ np.array([[1., 0.], [0., -1.]]) @ VT).flatten(order='F')
        T = np.sqrt(0.5) * (U @ np.array([[0., -1.], [1., 0.]]) @ VT).flatten(order='F')
        R = np.sqrt(0.5) * (U @ np.array([[VT[0, 0], VT[0, 1]], [VT[1, 0], VT[1, 1]]])).flatten(order='F')
        return np.outer(R, R) + np.outer(T, T) - np.outer(P, P) - np.outer(L, L)
    
    def d2I3_div_dF2_3x2(self, U, sigma, VT):
        L = np.sqrt(0.5) * (U @ np.array([[0., 1.], [1., 0.], [0., 0.]]) @ VT).flatten(order='F')
        P = np.sqrt(0.5) * (U @ np.array([[1., 0.], [0., -1.], [0., 0.]]) @ VT).flatten(order='F')
        T = np.sqrt(0.5) * (U @ np.array([[0., -1.], [1., 0.], [0., 0.]]) @ VT).flatten(order='F')
        R = np.sqrt(0.5) * (U @ np.array([[VT[0, 0], VT[0, 1]], [VT[1, 0], VT[1, 1]], [0., 0.]])).flatten(order='F')
        return np.outer(R, R) + np.outer(T, T) - np.outer(P, P) - np.outer(L, L)