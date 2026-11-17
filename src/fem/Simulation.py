"""FEM simulation configuration."""

import math

from src.utils.TimeTicker import Timer


class FEMSimulation:
    def __init__(self):
        self.dimension = 3
        self.solver_type = "Explicit"
        self.is_axisymmetric = False
        self.axis_offset = 0.0
        self.kwargs = {}
        self.timer = Timer()

    def set_configuration(self, dimension=3, solver_type="Explicit", **kwargs):
        dimension = int(dimension)
        if dimension not in (2, 3):
            raise ValueError("FEM dimension must be 2 or 3")
        normalized = str(solver_type).strip().lower()
        if normalized not in ("explicit", "implicit"):
            raise ValueError("FEM solver_type must be 'Explicit' or 'Implicit'")
        self.dimension = dimension
        self.solver_type = normalized.capitalize()
        self.is_axisymmetric = bool(
            kwargs.get(
                "axisymmetric",
                kwargs.get("is_axisymmetric", kwargs.get("is_2DAxisy", False)),
            )
        )
        self.axis_offset = float(kwargs.get("axis_offset", 0.0))
        if self.is_axisymmetric and dimension != 2:
            raise ValueError("axisymmetric FEM requires dimension=2")
        if not math.isfinite(self.axis_offset):
            raise ValueError("FEM axis_offset must be finite")
        kwargs["axisymmetric"] = self.is_axisymmetric
        kwargs["axis_offset"] = self.axis_offset
        self.kwargs.update(kwargs)


__all__ = ["FEMSimulation"]
