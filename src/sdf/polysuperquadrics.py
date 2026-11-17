from src.sdf.SuperSurface import SuperSurface
from src.utils.linalg import newton

import numpy as np


class polysuperquadrics(SuperSurface):
    def __init__(
        self, xrad1, yrad1, zrad1, epsilon_x, epsilon_y, epsilon_z, xrad2=None, yrad2=None, zrad2=None
    ) -> None:
        super().__init__(xrad1, yrad1, zrad1, xrad2, yrad2, zrad2)
        self.epsilon_x = epsilon_x
        self.epsilon_y = epsilon_y
        self.epsilon_z = epsilon_z
        self.physical_parameter.update(
            {
                "xrad1": self.xrad1,
                "yrad1": self.yrad1,
                "zrad1": self.zrad1,
                "epsilon_x": self.epsilon_x,
                "epsilon_y": self.epsilon_y,
                "epsilon_z": self.epsilon_z,
                "xrad2": self.xrad2,
                "yrad2": self.yrad2,
                "zrad2": self.zrad2,
            }
        )

        if epsilon_x == epsilon_y == epsilon_z == 0:
            raise ValueError("Please define non zero epsilons for polysuperquadrics shape.")
        self._reset = False

    def _estimate_bounding_box(self):
        self._lower_bound = -np.array([self.xrad1, self.yrad1, self.zrad1])
        self._higher_bound = np.array([self.xrad2, self.yrad2, self.zrad2])
        self._is_bound = True

    def fx(self, p):
        p = self.transfer(p)
        x = p[:, 0]
        y = p[:, 1]
        z = p[:, 2]

        funcs = np.array(
            pow(abs(x / np.where(x < 0.0, self.xrad1, self.xrad2)), 2.0 / self.epsilon_x)
            + pow(abs(y / np.where(y < 0.0, self.yrad1, self.yrad2)), 2.0 / self.epsilon_y)
            + pow(abs(z / np.where(z < 0.0, self.zrad1, self.zrad2)), 2.0 / self.epsilon_z)
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
        grad_f = np.array(
            [
                2 / self.epsilon_x * pow(abs(x * xrad), 2.0 / self.epsilon_x - 1.0) * xrad,
                2 / self.epsilon_y * pow(abs(y * yrad), 2.0 / self.epsilon_y - 1.0) * yrad,
                2 / self.epsilon_z * pow(abs(z * zrad), 2.0 / self.epsilon_z - 1.0) * zrad,
            ]
        ).T
        return (grad_f.T / np.linalg.norm(grad_f, axis=1)).T.reshape((-1, 3))

    def radial(self, theta, phi):
        para1 = np.sin(theta) * np.cos(phi)
        para2 = np.sin(theta) * np.sin(phi)
        para3 = np.cos(theta)

        coeff1 = pow(np.abs(para1 * np.where(para1 < 0.0, -1.0 / self.xrad1, 1.0 / self.xrad2)), 2.0 / self.epsilon_x)
        coeff2 = pow(np.abs(para2 * np.where(para2 < 0.0, -1.0 / self.yrad1, 1.0 / self.yrad2)), 2.0 / self.epsilon_y)
        coeff3 = pow(np.abs(para3 * np.where(para3 < 0.0, -1.0 / self.zrad1, 1.0 / self.zrad2)), 2.0 / self.epsilon_z)

        def f(r, a, b, c, ex, ey, ez):
            return a * pow(r, 2.0 / ex) + b * pow(r, 2.0 / ey) + c * pow(r, 2.0 / ez) - 1.0

        def df(r, a, b, c, ex, ey, ez):
            return (
                a * (2.0 / ex) * pow(r, 2.0 / ex - 1)
                + b * (2.0 / ey) * pow(r, 2.0 / ey - 1)
                + c * (2.0 / ez) * pow(r, 2.0 / ez - 1)
            )

        start_value = min(self.xrad1, self.xrad2, self.yrad1, self.yrad2, self.zrad1, self.zrad2)
        root = newton(
            func=f,
            x0=start_value,
            args=(coeff1, coeff2, coeff3, self.epsilon_x, self.epsilon_y, self.epsilon_z),
            fprime=df,
        )
        return root
