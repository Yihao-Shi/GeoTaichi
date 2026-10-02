"""Scalar Armijo policy used by the device-resident implicit FEM solver."""

import math


class LineSearchError(RuntimeError):
    pass


class ArmijoLineSearch:
    def __init__(self, reduction=0.5, sufficient_decrease=1.0e-4, max_backtracks=20, minimum_step=1.0e-8):
        self.reduction = float(reduction)
        self.sufficient_decrease = float(sufficient_decrease)
        self.max_backtracks = int(max_backtracks)
        self.minimum_step = float(minimum_step)
        if not (0.0 < self.reduction < 1.0):
            raise ValueError("line-search reduction must lie strictly between zero and one")
        if not (0.0 < self.sufficient_decrease < 1.0):
            raise ValueError("Armijo sufficient_decrease must lie strictly between zero and one")
        if self.max_backtracks < 0 or self.minimum_step <= 0.0:
            raise ValueError("line-search iteration count/minimum step is invalid")

    def accepts(self, initial, value, step, slope, roundoff=0.0):
        armijo = initial + self.sufficient_decrease * step * slope
        return math.isfinite(value) and (value <= armijo or (-step * slope <= roundoff and value <= initial + roundoff))


__all__ = ["ArmijoLineSearch", "LineSearchError"]
