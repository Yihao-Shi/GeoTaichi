from src.consititutive_model.MaterialKernel import *


class ConstitutiveBase:
    def __init__(self) -> None:
        self.matProps = None
        self.stateVars = None
        self.stiffness_matrix = None
        self.is_elastic = False

    def check_materialID(self, materialID, max_material_num):
        if materialID <= 0: 
            raise RuntimeError(f"MaterialID {materialID} should be larger than 0")
        if materialID > max_material_num - 1:
            raise RuntimeError(f"Keyword:: /max_material_number/ should be set as {materialID + 1}")

    def model_initialization(self, materials):
        if type(materials) is dict:
            self.model_initialize(materials)
        elif type(materials) is list:
            for material in materials:
                self.model_initialize(material)

    def model_initialize(self, material):
        raise NotImplementedError
    
    def get_state_vars_dict(self, start_particle, end_particle):
        raise NotImplementedError
    
    def compute_stress(self):
        raise NotImplementedError
    
    def get_lateral_coefficient(self, materialID):
        raise NotImplementedError
    
    def reload_state_variables(self, state_vars):
        raise NotImplementedError
    
    def find_max_sound_speed(self):
        return find_max_sound_speed_(self.matProps)
    
    def state_vars_initialize(self, start_particle, end_particle, particle):
        kernel_initial_state_variables(start_particle, end_particle, particle, self.stateVars, self.matProps)

    def pre_compute_stiffness(self, particleNum, particle):
        compute_elastic_stiffness_matrix(self.stiffness_matrix, particleNum, particle, self.matProps, self.stateVars)
    
    def compute_elasto_plastic_stiffness(self, particleNum, particle):
        compute_stiffness_matrix(self.stiffness_matrix, particleNum, particle, self.matProps, self.stateVars)

