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
    def __init__(self, cnum, cellSum, ghost_cell=1) -> None:
        self.cnum = cnum
        self.cellSum = cellSum
        self.ghost_cell = ghost_cell
        self.pressure = None
        self.type = None
        self.surface_tension = None
        self.dofID = None

    def set_ptr(self, pressure=None, cell_type=None, surface_tension=None):
        if pressure is None:
            self.pressure = ti.field(dtype=float, shape=self.cnum, offset=0 * self.cnum - self.ghost_cell)
        else:
            self.pressure = pressure
        if cell_type is None:
            self.type = ti.field(dtype=ti.u8, shape=self.cnum, offset=0 * self.cnum - self.ghost_cell)
        else:
            self.type = cell_type
        if surface_tension is None:
            self.surface_tension = ti.field(dtype=float, shape=self.cnum, offset=0 * self.cnum - self.ghost_cell)
        else:
            self.surface_tension = surface_tension

@ti.data_oriented
class IncompressibleFace:
    def __init__(self, gnum, dimension=2):
        if dimension == 2:
            self.pressure_gradient_x = ti.field(dtype=float, shape=(gnum[0], gnum[1] - 1))
            self.pressure_gradient_y = ti.field(dtype=float, shape=(gnum[0] - 1, gnum[1]))
        elif dimension == 3:
            self.pressure_gradient_x = ti.field(dtype=float, shape=(gnum[0], gnum[1], gnum[2] - 1))
            self.pressure_gradient_y = ti.field(dtype=float, shape=(gnum[0], gnum[1] - 1, gnum[2]))
            self.pressure_gradient_z = ti.field(dtype=float, shape=(gnum[0] - 1, gnum[1], gnum[2]))
