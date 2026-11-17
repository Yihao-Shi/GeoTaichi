from src.sdf.BasicShape import BasicShape
from src.sdf.BasicShape import DBL_EPSILON
from src.sdf.BasicShape import THRESHOLD

import numpy as np
import math
from src.sdf.FastMarchingMethod import FastMarchingMethod


class Surface(BasicShape):
    def __init__(self) -> None:
        super().__init__(ray=True)
        self.is_simple_shape = 2

    def fx(self, p):
        raise NotImplementedError

    def dfx(self, p):
        raise NotImplementedError

    def radial(self, theta, phi):
        raise NotImplementedError

    def _normal(self, p):
        return self.dfx(p)

    def side(self, p):
        p = self.transfer(p)
        field = self.fx(p)
        return np.select([field > THRESHOLD, field < -THRESHOLD], [math.inf, -math.inf], default=0.0)

    def _approximate_distance(self, p):
        # reference: Gabriel Taubin (1994). Distance approximations for rasterizing implicit curves. ACM Transactions on Graphics, 13(1), 3-42.
        # First order approximation
        fxyz = self.fx(p)
        grad_f = self.dfx(p)
        dnorm = np.select(
            [np.linalg.norm(grad_f) > (self.upper_bound - self.lower_bound) * DBL_EPSILON],
            [np.linalg.norm(grad_f)],
            default=1.0,
        )
        return_val = np.select([np.abs(fxyz) > DBL_EPSILON], [fxyz / dnorm], default=0.0)
        return return_val

    def fast_marching(self):
        nodal_coords = self.grid.node_coord.copy()
        step = self.grid.grid_space
        gnum = self.grid.gnum.copy()
        fmm = FastMarchingMethod(step, gnum, nodal_coords)
        fmm.phi_ini(self)
        fmm.phi()
        self.grid.generate_sdf(fmm.phiField.copy().reshape(-1))
        fmm.clear()
