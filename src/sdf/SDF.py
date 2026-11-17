from src.sdf.BasicShape import BasicShape
from src.sdf.BasicShape import IL_THRESHOLD
from src.sdf.BasicShape import L_THRESHOLD

import numpy as np
from src.utils.linalg import generate_grid


class SDF(BasicShape):
    def __init__(self, ray=False):
        super().__init__(ray)
        self._smooth = 0.0
        self.is_simple_shape = 3

    @property
    def smooth(self):
        return self._smooth

    @smooth.setter
    def smooth(self, smooth):
        self._smooth = smooth

    def smooths(self, smooth):
        self._smooth = smooth
        return self

    def _distance(self, p):
        raise NotImplementedError

    def _normal(self, p):
        point = self.transfer(p)
        space = np.min(self.upper_bound - self.lower_bound)
        gapnx = self._distance(point + np.array([L_THRESHOLD * space, 0, 0])).reshape(-1)
        gapny = self._distance(point + np.array([0, L_THRESHOLD * space, 0])).reshape(-1)
        gapnz = self._distance(point + np.array([0, 0, L_THRESHOLD * space])).reshape(-1)
        gapnx_ = self._distance(point - np.array([L_THRESHOLD * space, 0, 0])).reshape(-1)
        gapny_ = self._distance(point - np.array([0, L_THRESHOLD * space, 0])).reshape(-1)
        gapnz_ = self._distance(point - np.array([0, 0, L_THRESHOLD * space])).reshape(-1)
        normals = 0.5 * IL_THRESHOLD * np.array([gapnx - gapnx_, gapny - gapny_, gapnz - gapnz_]).T / space
        return np.where(np.linalg.norm(normals, axis=1) > 0.0, (normals.T / np.linalg.norm(normals, axis=1)), 0.0).T

    def _estimate_bounding_box(self):
        # TODO: raise exception if bound estimation fails
        s = 31
        x0, y0, z0 = self._lower_bound
        x1, y1, z1 = self._higher_bound
        prev = None
        for _ in range(64):
            X, Y, Z, P = generate_grid(x0, y0, z0, x1, y1, z1, s)
            d = np.array([X[1] - X[0], Y[1] - Y[0], Z[1] - Z[0]])
            threshold = np.linalg.norm(d) / 2
            if threshold == prev:
                break
            prev = threshold
            volume = self(P).reshape((len(X), len(Y), len(Z)))
            where = np.argwhere(np.abs(volume) <= threshold)
            x1, y1, z1 = (x0, y0, z0) + where.max(axis=0) * d + d / 2
            x0, y0, z0 = (x0, y0, z0) + where.min(axis=0) * d - d / 2
        self._lower_bound = np.array([x0, y0, z0])
        self._higher_bound = np.array([x1, y1, z1])
        self._is_bound = True

    def radial(self, theta, phi):
        pass
