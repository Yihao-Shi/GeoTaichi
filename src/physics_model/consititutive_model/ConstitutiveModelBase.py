import numpy as np

from src.physics_model.consititutive_model.MaterialKernel import kernel_initial_state_variables, compute_stiffness_matrix
from src.utils.ListWrapper import ListWrapper
from src.utils.linalg import get_dataclass_to_dict


class ConstitutiveBase:
    def __init__(self) -> None:
        self.matProps = ListWrapper()
        self.stateDict = dict()
        self.stateVars = None
        self.stiffness_matrix = None

    def check_materialID(self, materialID):
        if materialID <= 0: 
            raise RuntimeError(f"MaterialID {materialID} should be larger than 0")
        if materialID < self.matProps.size():
            print(f"Material {materialID} Property will be overwritten!")
    
    def get_state_vars_dict(self, selected_vars=None, start_index=0, end_index=-1):
        state_vars = get_dataclass_to_dict(self.stateVars, selected_vars, start_index, end_index)
        if 'strain'  in state_vars:
            strain_tensor = state_vars['strain']
            N = strain_tensor.shape[0]
            trace = np.trace(strain_tensor, axis1=1, axis2=2)
            identity = np.eye(3)
            dev = strain_tensor - trace[:, None, None] / 3.0 * identity[None, :, :]
            j2 = 0.5 * np.sum(dev ** 2, axis=(1, 2))
            eq_strain = np.sqrt(4.0 / 3.0 * j2)
            state_vars['strain'] = eq_strain
        return state_vars
    
    def reload_state_variables(self, state_vars):
        target_length = self.stateVars.shape[0]
        for key, value in state_vars.items():
            npvalue = np.asarray(value)
            current_length = npvalue.shape[0]
            state_para = np.pad(npvalue, (0, target_length - current_length), mode='constant')
            getattr(self.stateVars, key).from_numpy(state_para)
    
    def find_max_sound_speed(self):
        max_sound_speed = 0.
        for matID in range(1, self.matProps.shape[0]):
            sound_speed = self.matProps[matID].max_sound_speed
            # TODO: utilize reduce max to accelerate
            max_sound_speed = np.maximum(max_sound_speed, sound_speed)
        return max_sound_speed
    
    def state_vars_initialize(self, materialID, start_particle, end_particle, particle):
        if self.stateDict and materialID > 0:
            kernel_initial_state_variables(start_particle, end_particle, particle, self.matProps[materialID], self.stateVars)
    
    def compute_elasto_plastic_stiffness(self, materialID, start_index, end_index, particle, material_mapping):
        compute_stiffness_matrix(self.stiffness_matrix, start_index, end_index, particle, material_mapping, self.matProps[materialID], self.stateVars)

