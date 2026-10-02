import taichi as ti
from src.dem.contact.Linear import ContactModelBase


class LinearBondModel(ContactModelBase):
    def __init__(self, max_material_num) -> None:
        super().__init__()
        self.null_model = False

    def calcu_critical_timestep(self):
        pass
