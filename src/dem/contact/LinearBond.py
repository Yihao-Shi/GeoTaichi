import taichi as ti
from src.dem.contact.Linear import ContactModelBase


class LinearBondModel(ContactModelBase):
    def __init__(self, sims) -> None:
        super().__init__(sims)
        self.null_model = False
        self.model_type = 3

    def calcu_critical_timestep(self, scene):
        pass

    def add_surface_property(self, materialID1, materialID2, property):
        pass

