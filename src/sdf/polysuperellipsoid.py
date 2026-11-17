from src.sdf.SuperSurface import SuperSurface

import numpy as np


class polysuperellipsoid(SuperSurface):
    def __init__(self, xrad1, yrad1, zrad1, epsilon_e, epsilon_n, xrad2=None, yrad2=None, zrad2=None) -> None:
        super().__init__(xrad1, yrad1, zrad1, xrad2, yrad2, zrad2)
        self.epsilon_e = epsilon_e
        self.epsilon_n = epsilon_n
        self.physical_parameter.update(
            {
                "xrad1": self.xrad1,
                "yrad1": self.yrad1,
                "zrad1": self.zrad1,
                "epsilon_e": self.epsilon_e,
                "epsilon_n": self.epsilon_n,
                "xrad2": self.xrad2,
                "yrad2": self.yrad2,
                "zrad2": self.zrad2,
            }
        )

        if epsilon_e == epsilon_n == 0:
            raise ValueError("Please define non zero epsilons for superellipsoid shape.")
        self._reset = False

    def fx(self, p):
        p = self.transfer(p)
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        funcs = np.array(
            pow(
                pow(abs(x / np.where(x < 0.0, self.xrad1, self.xrad2)), 2.0 / self.epsilon_e)
                + pow(abs(y / np.where(y < 0.0, self.yrad1, self.yrad2)), 2.0 / self.epsilon_e),
                self.epsilon_e / self.epsilon_n,
            )
            + pow(abs(z / np.where(z < 0.0, self.zrad1, self.zrad2)), 2.0 / self.epsilon_n)
            - 1.0
        )
        return funcs.reshape(-1)

    def dfx(self, p):
        p = self.transfer(p)
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        xrad = np.where(x < 0.0, -1.0 / self.xrad1, 1.0 / self.xrad2)
        yrad = np.where(y < 0.0, -1.0 / self.yrad1, 1.0 / self.yrad2)
        zrad = np.where(z < 0.0, -1.0 / self.zrad1, 1.0 / self.zrad2)
        mult = (
            2.0
            / self.epsilon_n
            * pow(
                pow(abs(x * xrad), 2.0 / self.epsilon_e) + pow(abs(y * yrad), 2.0 / self.epsilon_e),
                self.epsilon_e / self.epsilon_n - 1.0,
            )
        )
        grad_f = np.array(
            [
                mult * pow(abs(x * xrad), 2.0 / self.epsilon_e - 1.0) * xrad,
                mult * pow(abs(y * yrad), 2.0 / self.epsilon_e - 1.0) * yrad,
                2.0 / self.epsilon_n * pow(abs(z * zrad), 2.0 / self.epsilon_n - 1) * zrad,
            ]
        ).T
        return (grad_f.T / np.linalg.norm(grad_f, axis=1)).T.reshape((-1, 3))

    def radial(self, theta, phi):
        para1 = np.sin(theta) * np.cos(phi)
        para2 = np.sin(theta) * np.sin(phi)
        para3 = np.cos(theta)

        coeff1 = np.where(para1 < 0.0, -1.0 / self.xrad1, 1.0 / self.xrad2)
        coeff2 = np.where(para2 < 0.0, -1.0 / self.yrad1, 1.0 / self.yrad2)
        coeff3 = np.where(para3 < 0.0, -1.0 / self.zrad1, 1.0 / self.zrad2)
        r_dist = 1.0 / (
            (np.abs(para1 * coeff1) ** (2.0 / self.epsilon_e) + np.abs(para2 * coeff2) ** (2.0 / self.epsilon_e))
            ** (self.epsilon_n / self.epsilon_e)
            + np.abs(para3 * coeff3) ** (2.0 / self.epsilon_n)
        )
        return r_dist ** (0.5 * self.epsilon_n)
