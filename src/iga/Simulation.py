import math

from src.iga.config import set_dimension, get_dimension
from src.utils.TimeTicker import Timer


class IGASimulation:
    def __init__(self):
        self.dimension = get_dimension()
        self.solver_type = "Implicit"
        self.is_axisymmetric = False
        self.axis_offset = 0.0
        self.kwargs = {}
        self.timer = Timer()

    def set_configuration(self, dimension=3, solver_type="Implicit", **kwargs):
        set_dimension(dimension)
        self.dimension = int(dimension)
        self.solver_type = str(solver_type)
        self.is_axisymmetric = bool(
            kwargs.get(
                "axisymmetric",
                kwargs.get("is_axisymmetric", kwargs.get("is_2DAxisy", False)),
            )
        )
        self.axis_offset = float(kwargs.get("axis_offset", 0.0))
        if self.is_axisymmetric and self.dimension != 2:
            raise ValueError("axisymmetric IGA requires dimension=2")
        if not math.isfinite(self.axis_offset):
            raise ValueError("IGA axis_offset must be finite")
        self.kwargs.update(kwargs)
