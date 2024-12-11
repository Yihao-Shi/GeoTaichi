import taichi as ti

from src.utils.constants import ZEROVEC6f
from src.utils.TypeDefination import vec6f


@ti.dataclass
class HexahedronCell:
    active: ti.u8
    volume: float

    @ti.func
    def _reset(self):
        self.volume = 0.

    @ti.func
    def _update_cell_volume(self, volume):
        self.volume += volume


@ti.dataclass
class HexahedronGuassCell:
    stress: vec6f
    vol: float

    @ti.func
    def _reset(self):
        if self.vol > 0.:
            self.stress = ZEROVEC6f
            self.vol = 0.


@ti.data_oriented
class IncompressibleCell:
    def __init__(self) -> None:
        self.pressure = None
        self.type = None

    def set_ptr(self, solution, cell_type):
        self.pressure = solution
        self.type = cell_type