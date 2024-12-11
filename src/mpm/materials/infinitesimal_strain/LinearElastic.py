import numpy as np
import taichi as ti

from src.consititutive_model.infinitesimal_strain.LinearElastic import *
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO


class LinearElastic(ConstitutiveModelBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.is_elastic = True
        self.add_material(sims.max_material_num, sims.material_type, sims.contact_detection, LinearElasticModel)
        if sims.configuration == "ULMPM":
            self.stateVars = ULStateVariable.field(shape=sims.max_particle_num) 
        elif sims.configuration == "TLMPM":
            self.stateVars = TLStateVariable.field(shape=sims.max_particle_num) 

        if sims.solver_type == "Implicit":
            self.stiffness_matrix = ti.Matrix.field(6, 6, float, shape=sims.max_particle_num)

    def get_state_vars_dict(self, start_particle, end_particle):
        estress = np.ascontiguousarray(self.stateVars.estress.to_numpy()[start_particle:end_particle])
        return {'estress': estress}
    
    def reload_state_variables(self, state_vars):
        estress = state_vars.item()['estress']
        kernel_reload_state_variables(estress, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        self.check_materialID(materialID, self.matProps.shape[0])

        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        young = DictIO.GetEssential(material, 'YoungModulus')
        possion = DictIO.GetAlternative(material, 'PossionRatio', 0.3)
        if self.material_type == "TwoPhaseSingleLayer":
            solid_density = DictIO.GetAlternative(material, 'SolidDensity', 2650)
            fluid_density = DictIO.GetAlternative(material, 'FluidDensity', 1000)
            porosity = DictIO.GetEssential(material, 'Porosity')
            fluid_bulk = DictIO.GetEssential(material, 'FluidBulkModulus')
            permeability = DictIO.GetEssential(material, 'Permeability')
            self.matProps[materialID].add_coupling_material(porosity, solid_density, fluid_density, young, possion, fluid_bulk, permeability)
        else:
            density = DictIO.GetAlternative(material, 'Density', 2650)
            self.matProps[materialID].add_material(density, young, possion)
        self.contact_initialize(material)
        self.matProps[materialID].print_message(materialID)

    def get_lateral_coefficient(self, materialID):
        mu = self.matProps[materialID].possion
        return mu / (1. - mu)
    
    def compute_elasto_plastic_stiffness(self, particleNum, particle):
        pass
