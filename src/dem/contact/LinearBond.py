import taichi as ti
from src.dem.contact.Linear import ContactModelBase


class LinearBondModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.null_mode = False

    def calcu_critical_timestep(self):
        pass

    def add_surface_property(self, max_material_num, materialID1, materialID2, property):
        pass


@ti.dataclass
class LinearBondSurfaceProperty:
    pass
