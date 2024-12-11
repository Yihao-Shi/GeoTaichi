import numpy as np

from src.consititutive_model.strain_rate.Bingham import *
from src.mpm.materials.ConstitutiveModelBase import ConstitutiveModelBase
from src.mpm.Simulation import Simulation
from src.utils.ObjectIO import DictIO


class Bingham(ConstitutiveModelBase):
    def __init__(self, sims: Simulation):
        super().__init__()
        self.add_material(sims.max_material_num, sims.material_type, BinghamModel)
        self.stateVars = None
        if sims.solver_type == "Explicit":
            self.stateVars = ULStateVariable.field(shape=sims.max_particle_num) 

    def add_material(self, max_material_num, material_type, material_struct):
        self.matProps = material_struct.field(shape=max_material_num)
        self.material_type = material_type

    def get_state_vars_dict(self, start_particle, end_particle):
        pressure = np.ascontiguousarray(self.stateVars.pressure.to_numpy()[start_particle:end_particle])
        rho = np.ascontiguousarray(self.stateVars.rho.to_numpy()[start_particle:end_particle])
        return {'pressure': pressure, 'rho': rho}
    
    def reload_state_variables(self, state_vars):
        pressure = state_vars.item()['pressure']
        rho = state_vars.item()['rho']
        kernel_reload_state_variables(pressure, rho, self.stateVars)

    def model_initialize(self, material):
        materialID = DictIO.GetEssential(material, 'MaterialID')
        self.check_materialID(materialID, self.matProps.shape[0])
        
        if self.matProps[materialID].density > 0.:
            print("Previous Material Property will be overwritten!")
        density = DictIO.GetAlternative(material, 'Density', 1000)
        modulus = DictIO.GetEssential(material, 'Modulus')
        viscosity = DictIO.GetEssential(material, 'Viscosity')
        _yield = DictIO.GetEssential(material, 'YieldStress')
        critical_rate = DictIO.GetEssential(material, 'CriticalStrainRate')
        gamma = DictIO.GetAlternative(material, 'gamma', 7.)
        atmospheric_pressure = DictIO.GetAlternative(material, 'atmospheric_pressure', 0.)
        
        self.matProps[materialID].add_material(density, modulus, viscosity, _yield, critical_rate, gamma, atmospheric_pressure)
        self.contact_initialize(material)
        self.matProps[materialID].print_message(materialID)

    def get_lateral_coefficient(self, materialID):
        return 1.

