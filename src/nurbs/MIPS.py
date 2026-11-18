from src.nurbs.Energy import SurfaceEnergy

import numpy as np


class MIPS(SurfaceEnergy):

    def psi(self, deformation_gradient):
        U, sigma, V = np.linalg.svd(deformation_gradient)
        return self.invariant_i2(sigma) / self.invariant_i3(sigma)

    def d_psi_div_d_f_2x2(self, deformation_gradient):
        """
        return a 4x1 vector (dPsi_div_dF), index: [∂Ψ/∂F₀₀, ∂Ψ/∂F₁₀, ∂Ψ/∂F₀₁, ∂Ψ/∂F₁₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.invariant_i2(sigma)
        I3 = self.invariant_i3(sigma)
        inv_I3 = 1.0 / I3
        dPsi_dI2 = inv_I3
        dPsi_dI3 = -I2 * inv_I3 * inv_I3
        dI2_dF = 2.0 * deformation_gradient
        dI3_dF = U @ np.array([[sigma[1], 0.0], [0.0, sigma[0]]]) @ VT
        return (dPsi_dI2 * dI2_dF + dPsi_dI3 * dI3_dF).flatten(order="F")

    def d_psi_div_d_f_3x2(self, deformation_gradient):
        """
        return a 6x1 vector (dPsi_div_dF), index: [∂Ψ/∂F₀₀, ∂Ψ/∂F₁₀, ∂Ψ/∂F₂₀, ∂Ψ/∂F₀₁, ∂Ψ/∂F₁₁, ∂Ψ/∂F₂₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.invariant_i2(sigma)
        I3 = self.invariant_i3(sigma)
        inv_I3 = 1.0 / I3
        dPsi_dI2 = inv_I3
        dPsi_dI3 = -I2 * inv_I3 * inv_I3
        dI2_dF = 2.0 * deformation_gradient
        dI3_dF = U @ np.array([[sigma[1], 0.0], [0.0, sigma[0]], [0.0, 0.0]]) @ VT
        return (dPsi_dI2 * dI2_dF + dPsi_dI3 * dI3_dF).flatten(order="F")

    def d2_psi_div_d_f2_2x2(self, deformation_gradient):
        """
        return a 4x4 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.invariant_i2(sigma)
        I3 = self.invariant_i3(sigma)
        dPsi_dI2 = 1.0 / I3
        dPsi_dI3 = -I2 / (I3 * I3)
        d2Psi_dI22 = 0.0
        d2Psi_dI32 = 2 * I2 / (I3 * I3 * I3)
        d2Psi_dI2I3 = -1.0 / (I3 * I3)
        dI2_dF = (2.0 * deformation_gradient).flatten(order="F")
        dI3_dF = (U @ np.array([[sigma[1], 0.0], [0.0, sigma[0]]]) @ VT).flatten(order="F")
        d2I2_dF2 = 2.0 * np.eye(4, 4)
        d2I3_dF2 = self.d2_i3_div_d_f2_2x2(U, sigma, VT)
        return (
            dPsi_dI3 * d2I3_dF2
            + dPsi_dI2 * d2I2_dF2
            + d2Psi_dI22 * np.outer(dI2_dF, dI2_dF)
            + d2Psi_dI2I3 * (np.outer(dI2_dF, dI3_dF) + np.outer(dI3_dF, dI2_dF))
            + d2Psi_dI32 * np.outer(dI3_dF, dI3_dF)
        )

    def d2_psi_div_d_f2_3x2(self, deformation_gradient):
        """
        return a 6x6 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₂₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁, ∂(∂Ψ/∂F)/∂F₂₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.invariant_i2(sigma)
        I3 = self.invariant_i3(sigma)
        dPsi_dI2 = 1.0 / I3
        dPsi_dI3 = -I2 / (I3 * I3)
        d2Psi_dI22 = 0.0
        d2Psi_dI32 = 2 * I2 / (I3 * I3 * I3)
        d2Psi_dI2I3 = -1.0 / (I3 * I3)
        dI2_dF = (2.0 * deformation_gradient).flatten(order="F")
        dI3_dF = (U @ np.array([[sigma[1], 0.0], [0.0, sigma[0]], [0.0, 0.0]]) @ VT).flatten(order="F")
        d2I2_dF2 = 2.0 * np.eye(6, 6)
        d2I3_dF2 = self.d2_i3_div_d_f2_3x2(U, sigma, VT)
        return (
            dPsi_dI3 * d2I3_dF2
            + dPsi_dI2 * d2I2_dF2
            + d2Psi_dI22 * np.outer(dI2_dF, dI2_dF)
            + d2Psi_dI2I3 * (np.outer(dI2_dF, dI3_dF) + np.outer(dI3_dF, dI2_dF))
            + d2Psi_dI32 * np.outer(dI3_dF, dI3_dF)
        )

    def d2_psi_div_d_f2_spd_2x2(self, deformation_gradient):
        """
        return a 4x4 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.invariant_i2(sigma)
        I3 = self.invariant_i3(sigma)

        Hq = np.zeros((4, 4))
        e1 = np.sqrt(0.5) * (U @ np.array([[0.0, 1.0], [1.0, 0.0]]) @ VT).flatten(order="F")
        Hq += 2.0 * np.outer(e1, e1)

        e2 = np.sqrt(0.5) * (U @ np.array([[0.0, -1.0], [1.0, 0.0]]) @ VT).flatten(order="F")
        Hq += max(0.0, 2.0 - 4.0 / (sigma[0] + sigma[1])) * np.outer(e2, e2)

        alpha = np.sqrt(I2 * I2 - 3.0 * I3 * I3)
        beta = I2 / (sigma[1] * sigma[1] - sigma[0] * sigma[0] + alpha)
        gamma = np.sqrt(1.0 + beta * beta)
        d1 = (U @ np.array([[1.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        d2 = (U @ np.array([[0.0, 0.0], [0.0, 1.0]]) @ VT).flatten(order="F")
        e3 = 1.0 / gamma * (beta * d1 + d2)
        e4 = 1.0 / gamma * (d1 - beta * d2)
        Hq += max(0.0, I2 * (I2 - alpha) - 2.0 * I3 * I3) * np.outer(e3, e3)
        Hq += max(0.0, I2 * (I2 + alpha) - 2.0 * I3 * I3) * np.outer(e4, e4)
        return Hq

    def d2_psi_div_d_f2_spd_3x2(self, deformation_gradient):
        """
        return a 6x6 vector (d2Psi_div_dF2), index: [∂(∂Ψ/∂F)/∂F₀₀, ∂(∂Ψ/∂F)/∂F₁₀, ∂(∂Ψ/∂F)/∂F₂₀, ∂(∂Ψ/∂F)/∂F₀₁, ∂(∂Ψ/∂F)/∂F₁₁, ∂(∂Ψ/∂F)/∂F₂₁]
        """
        U, sigma, VT = np.linalg.svd(deformation_gradient)
        I2 = self.invariant_i2(sigma)
        I3 = self.invariant_i3(sigma)

        Hq = np.zeros((6, 6))
        e1 = np.sqrt(0.5) * (U @ np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        Hq += 2.0 * np.outer(e1, e1)

        e2 = np.sqrt(0.5) * (U @ np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        Hq += max(0.0, 2.0 - 4.0 / (sigma[0] + sigma[1])) * np.outer(e2, e2)

        alpha = np.sqrt(I2 * I2 - 3.0 * I3 * I3)
        beta = I2 / (sigma[1] * sigma[1] - sigma[0] * sigma[0] + alpha)
        gamma = np.sqrt(1.0 + beta * beta)
        d1 = (U @ np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        d2 = (U @ np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        e3 = 1.0 / gamma * (beta * d1 + d2)
        e4 = 1.0 / gamma * (d1 - beta * d2)
        Hq += max(0.0, I2 * (I2 - alpha) - 2.0 * I3 * I3) * np.outer(e3, e3)
        Hq += max(0.0, I2 * (I2 + alpha) - 2.0 * I3 * I3) * np.outer(e4, e4)
        return Hq

    def d2_i3_div_d_f2_2x2(self, U, sigma, VT):
        L = np.sqrt(0.5) * (U @ np.array([[0.0, 1.0], [1.0, 0.0]]) @ VT).flatten(order="F")
        P = np.sqrt(0.5) * (U @ np.array([[1.0, 0.0], [0.0, -1.0]]) @ VT).flatten(order="F")
        T = np.sqrt(0.5) * (U @ np.array([[0.0, -1.0], [1.0, 0.0]]) @ VT).flatten(order="F")
        R = np.sqrt(0.5) * (U @ np.array([[VT[0, 0], VT[0, 1]], [VT[1, 0], VT[1, 1]]])).flatten(order="F")
        return np.outer(R, R) + np.outer(T, T) - np.outer(P, P) - np.outer(L, L)

    def d2_i3_div_d_f2_3x2(self, U, sigma, VT):
        L = np.sqrt(0.5) * (U @ np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        P = np.sqrt(0.5) * (U @ np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        T = np.sqrt(0.5) * (U @ np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 0.0]]) @ VT).flatten(order="F")
        R = np.sqrt(0.5) * (U @ np.array([[VT[0, 0], VT[0, 1]], [VT[1, 0], VT[1, 1]], [0.0, 0.0]])).flatten(order="F")
        return np.outer(R, R) + np.outer(T, T) - np.outer(P, P) - np.outer(L, L)
