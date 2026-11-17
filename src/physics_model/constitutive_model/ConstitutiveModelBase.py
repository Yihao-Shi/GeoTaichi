from src.utils.FieldIO import get_dataclass_to_dict, runtime_float_numpy_dtype, validate_numpy_prefix
from src.utils.errors import SimulationStateError
import numpy as np

from src.physics_model.constitutive_model.MaterialKernel import (
    kernel_initial_state_variables,
    compute_stiffness_matrix,
)
from src.utils.ListWrapper import ListWrapper
from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import export_equivalent_strain


class ConstitutiveBase:
    def __init__(self) -> None:
        self.matProps = ListWrapper()
        self.stateDict = dict()
        self.stateVars = None
        self.stiffness_matrix = None

    def check_material_id(self, materialID):
        if materialID <= 0:
            raise SimulationStateError(f"MaterialID {materialID} should be larger than 0")
        if materialID < self.matProps.size():
            print(f"Material {materialID} Property will be overwritten!")

    def get_state_vars_dict(self, selected_vars=None, start_index=0, end_index=-1):
        if self.stateVars is None:
            return {}
        selected = list(selected_vars or self.stateVars.field_dict)
        state_vars = (
            get_dataclass_to_dict(
                self.stateVars, [name for name in selected if name != "strain"], start_index, end_index
            )
            if any(name != "strain" for name in selected)
            else {}
        )
        if "strain" in selected and "strain" in self.stateVars.field_dict:
            end = self.stateVars.shape[0] if end_index == -1 else end_index
            if not 0 <= start_index <= end <= self.stateVars.shape[0]:
                raise ValueError("Invalid constitutive output slice")
            strain = np.empty(end - start_index, dtype=runtime_float_numpy_dtype())
            export_equivalent_strain(start_index, self.stateVars.strain, strain)
            state_vars["strain"] = strain
        return state_vars

    def reload_state_variables(self, state_vars, particle_count=None, start_index=0):
        """Validate and upload complete raw history for a restart or appended batch."""
        for field, values in self.prepare_state_variables(state_vars, particle_count, start_index):
            field.from_numpy(values)

    def prepare_state_variables(self, state_vars, particle_count=None, start_index=0):
        """Prepare validated uploads without modifying live constitutive history."""
        if isinstance(state_vars, np.ndarray) and state_vars.shape == ():
            state_vars = state_vars.item()
        if not isinstance(state_vars, dict):
            raise TypeError("Constitutive restart history must be a dictionary")
        fields = self.stateVars.field_dict if self.stateVars is not None else {}
        if set(state_vars) != set(fields):
            raise ValueError(
                f"Constitutive restart history has missing fields {set(fields) - set(state_vars)} "
                f"or unknown fields {set(state_vars) - set(fields)}"
            )
        prepared = []
        for name, field in fields.items():
            values = np.asarray(state_vars[name])
            # Legacy visualization archives expanded symmetric Voigt tensors.
            if getattr(field, "n", None) == 6 and getattr(field, "m", None) == 1 and values.shape[1:] == (3, 3):
                values = values[:, [0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2]]
            values = validate_numpy_prefix(field, values)
            if particle_count is None:
                particle_count = values.shape[0]
            if values.shape[0] != particle_count or not 0 <= start_index <= field.shape[0] - particle_count:
                raise ValueError(f"Constitutive restart field {name} has an invalid particle count or offset")
            restored = (
                field.to_numpy() if start_index else np.zeros((field.shape[0], *values.shape[1:]), dtype=values.dtype)
            )
            restored[start_index : start_index + particle_count] = values
            prepared.append((field, restored))
        return prepared

    def find_max_sound_speed(self):
        max_sound_speed = 0.0
        for mat_id in range(1, self.matProps.shape[0]):
            sound_speed = self.matProps[mat_id].max_sound_speed
            # TODO: utilize reduce max to accelerate
            max_sound_speed = np.maximum(max_sound_speed, sound_speed)
        return max_sound_speed

    def state_vars_initialize(self, materialID, start_particle, end_particle, particle):
        if self.stateDict and materialID > 0:
            kernel_initial_state_variables(
                start_particle, end_particle, particle, self.matProps[materialID], self.stateVars
            )

    def compute_elasto_plastic_stiffness(self, materialID, start_index, end_index, particle, material_mapping):
        compute_stiffness_matrix(
            self.stiffness_matrix,
            start_index,
            end_index,
            particle,
            material_mapping,
            self.matProps[materialID],
            self.stateVars,
        )
