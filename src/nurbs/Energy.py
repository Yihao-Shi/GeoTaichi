import numpy as np


class SurfaceEnergy:
    def __init__(self, dimension=2):
        if dimension == 2:
            self.d_psi_div_d_f = self.d_psi_div_d_f_2x2
            self.d2Psi_div_dF2 = self.d2_psi_div_d_f2_2x2
            self.d2Psi_div_dF2_spd = self.d2_psi_div_d_f2_spd_2x2
        elif dimension == 2:
            self.d_psi_div_d_f = self.d_psi_div_d_f_3x2
            self.d2Psi_div_dF2 = self.d2_psi_div_d_f2_3x2
            self.d2Psi_div_dF2_spd = self.d2_psi_div_d_f2_spd_3x2

    def psi(self, deformation_gradient):
        raise NotImplementedError

    def d_psi_div_d_f_2x2(self, deformation_gradient):
        raise NotImplementedError

    def d2_psi_div_d_f2_2x2(self, deformation_gradient):
        raise NotImplementedError

    def d2_psi_div_d_f2_spd_2x2(self, deformation_gradient):
        raise NotImplementedError

    def d_psi_div_d_f_3x2(self, deformation_gradient):
        raise NotImplementedError

    def d2_psi_div_d_f2_3x2(self, deformation_gradient):
        raise NotImplementedError

    def d2_psi_div_d_f2_spd_3x2(self, deformation_gradient):
        raise NotImplementedError

    def d_psi_div_dx(self, deformation_gradient, dF_dx):
        """
        applying chain-rule, ∂Ψ/∂x ((nx3)x1) = ∂Ψ/∂F * ∂F/∂x
        """
        return self.d_psi_div_d_f(deformation_gradient) @ dF_dx

    def d2_psi_div_dx2(self, deformation_gradient, dF_dx):
        """
        applying chain-rule, ∂²Ψ/∂x² ((nx3)x(nx3)) = (∂F/∂x)^T * ∂²Ψ/∂F² * ∂F/∂x (note that ∂²F/∂x² = 0)
        """
        return dF_dx.transpose() @ self.d2Psi_div_dF2(deformation_gradient) @ dF_dx

    def invariant_i1(self, sigma):
        return np.sum(sigma)

    def invariant_i2(self, sigma):
        return np.dot(sigma, sigma)

    def invariant_i3(self, sigma):
        return np.prod(sigma)
