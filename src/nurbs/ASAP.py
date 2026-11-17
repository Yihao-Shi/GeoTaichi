from src.nurbs.Energy import SurfaceEnergy

import numpy as np


class ASAP(SurfaceEnergy):

    def psi(self, deformation_gradient):
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        return self.invariant_i2(sigma) - 2.0 * self.invariant_i1(sigma) + 2

    def d_psi_div_d_f_2x2(self, deformation_gradient):
        """
        return a 4x1 vector (dPsi_div_dF), index: [∂Ψ/∂F₀₀, ∂Ψ/∂F₁₀, ∂Ψ/∂F₀₁, ∂Ψ/∂F₁₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        rotation_matrix = U @ VT
        return 2 * (deformation_gradient - rotation_matrix).flatten(order="F")

    def d_psi_div_d_f_3x2(self, deformation_gradient):
        """
        return a 6x1 vector (dPsi_div_dF), index: [∂Ψ/∂F₀₀, ∂Ψ/∂F₁₀, ∂Ψ/∂F₂₀, ∂Ψ/∂F₀₁, ∂Ψ/∂F₁₁, ∂Ψ/∂F₂₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        rotation_matrix = U @ np.array([[VT[0, 0], VT[0, 1]], [VT[1, 0], VT[1, 1]], [0.0, 0.0]])
        return 2 * (deformation_gradient - rotation_matrix).flatten(order="F")

    def d2_psi_div_d_f2_2x2(self, deformation_gradient):
        """
        return a 4x4 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        dPsi_dI1 = -2.0
        dPsi_dI2 = 1.0
        d2I1_dF2 = self.d2_i1_div_d_f2_2x2(U, sigma, VT)
        d2I2_dF2 = 2.0 * np.eye(4, 4)
        return dPsi_dI1 * d2I1_dF2 + dPsi_dI2 * d2I2_dF2

    def d2_psi_div_d_f2_3x2(self, deformation_gradient):
        """
        return a 6x6 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₂₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁, ∂(∂Ψ/∂F)/∂F₂₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        dPsi_dI1 = -2.0
        dPsi_dI2 = 1.0
        d2I1_dF2 = self.d2_i1_div_d_f2_3x2(U, sigma, VT)
        d2I2_dF2 = 2.0 * np.eye(6, 6)
        return dPsi_dI1 * d2I1_dF2 + dPsi_dI2 * d2I2_dF2

    def d2_psi_div_d_f2_spd_2x2(self, deformation_gradient):
        """
        return a 6x6 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₂₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁, ∂(∂Ψ/∂F)/∂F₂₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        Hq = np.zeros((6, 6))
        e1 = np.sqrt(0.5) * (U @ np.array([[0.0, 1.0], [1.0, 0.0]]) @ VT).flatten(order="F")
        Hq += 2.0 * np.outer(e1, e1)

        e2 = np.sqrt(0.5) * (U @ np.array([[0.0, -1.0], [1.0, 0.0]]) @ VT).flatten(order="F")
        Hq += max(0.0, 2.0 - 4.0 / (sigma[0] + sigma[1])) * np.outer(e2, e2)

        e3 = (U @ np.array([[1.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        Hq += 2.0 * np.outer(e3, e3)

        e4 = (U @ np.array([[0.0, 0.0], [0.0, 1.0]]) @ VT).flatten(order="F")
        Hq += 2.0 * np.outer(e4, e4)
        return Hq

    def d2_psi_div_d_f2_spd_3x2(self, deformation_gradient):
        """
        return a 6x6 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₂₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁, ∂(∂Ψ/∂F)/∂F₂₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        Hq = np.zeros((6, 6))
        e1 = np.sqrt(0.5) * (U @ np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        Hq += 2.0 * np.outer(e1, e1)

        e2 = np.sqrt(0.5) * (U @ np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        Hq += max(0.0, 2.0 - 4.0 / (sigma[0] + sigma[1])) * np.outer(e2, e2)

        e3 = (U @ np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        Hq += 2.0 * np.outer(e3, e3)

        e4 = (U @ np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        Hq += 2.0 * np.outer(e4, e4)
        return Hq

    def d2_i1_div_d_f2_2x2(self, U, sigma, VT):
        T = np.sqrt(0.5) * (U @ np.array([[0.0, -1.0], [1.0, 0.0]]) @ VT).flatten(order="F")
        return 2.0 / (sigma[0] + sigma[1]) * np.outer(T, T)

    def d2_i1_div_d_f2_3x2(self, U, sigma, VT):
        T = np.sqrt(0.5) * (U @ np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        return 2.0 / (sigma[0] + sigma[1]) * np.outer(T, T)
