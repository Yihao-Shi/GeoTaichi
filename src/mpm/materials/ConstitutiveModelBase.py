import numpy as np

from src.consititutive_model.ConstitutiveModelBase import ConstitutiveBase
from src.utils.ObjectIO import DictIO


class ConstitutiveModelBase(ConstitutiveBase):
    def __init__(self) -> None:
        super().__init__()

    def add_material(self, max_material_num, material_type, contact_type, material_struct):
        if material_type == "TwoPhaseSingleLayer":
            material_struct.members.update({"solid_density": float, "fluid_density": float, "fluid_bulk": float, "porosity": float, "permeability": float})
        if contact_type == "DEMContact":
            material_struct.members.update({"kn": float, "kt": float, "friction": float})
        self.matProps = material_struct.field(shape=max_material_num)
        self.material_type = material_type

    def contact_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        if "Stiffness" in material:
            friction = DictIO.GetAlternative(material, "Friction", 0.)
            stiffness = DictIO.GetAlternative(material, "Stiffness", [1e6, 1e6])
            if len(list(stiffness)) != 2:
                raise RuntimeError("The dimension of Keyword:: /Stiffness/ should be 2")
            self.matProps[materialID].add_contact_parameter(friction, stiffness[0], stiffness[1])
    
        


