import taichi as ti
import numpy as np
import os
from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class MaterialModel:
    def __init__(self, material_type, configuration, solver_type):
        # TODO: Add essential material properties
        self.material = None
        self.density = 0.0
        self.max_sound_speed = 0.0
        self._initialize_vars = None
        self.random_field_file = None
        self.material_type = material_type
        self.configuration = configuration
        self.incompressible_fluid = solver_type == "SemiImplicit" or (
            material_type == "Fluid" and solver_type == "Implicit"
        )
        self.solver_type = {"Explicit": 0, "Implicit": 1}.get(solver_type)
        self.residual_frac = 1.0
        self.console_solver_name = None

    def print_console_header(self):
        owner = f"{self.console_solver_name} " if self.console_solver_name else ""
        print(f" {owner}Constitutive Model Information ".center(71, "-"))

    def members_update(self, **config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def add_material(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def validate_fluid_bulk_modulus(modulus, key):
        if not np.isfinite(modulus) or modulus <= 0.0:
            raise ValueError(f"{key} must be finite and positive for compressible fluid, got {modulus!r}")

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: Material Model")
        print("Material ID: ", materialID)
        print("Model density = ", self.density)

    def random_field_initialize(self, parameter):
        random_field_file = DictIO.get_alternative(parameter, "MaterialFile", "RandomField.txt")
        residual_frac = DictIO.get_alternative(parameter, "ResidualFraction", 1.0)
        self.random_field_file = random_field_file
        self.residual_frac = residual_frac
        if not os.access(random_field_file, os.F_OK):
            raise ValueError(f"File {random_field_file} does not exist!")

    def model_initialize(self, material):
        raise NotImplementedError

    def get_state_vars(self):
        raise NotImplementedError

    def define_state_vars(self):
        raise NotImplementedError

    def get_sound_speed(self):
        # TODO: Add proporiate equations of sound speed
        raise NotImplementedError

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, state_variables, dt):
        raise NotImplementedError

    @ti.func
    def update_particle_volume_2d(self, np, velocity_gradient, state_variables, dt):
        return self.update_particle_volume(np, velocity_gradient, state_variables, dt)

    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, state_variables, dt):
        raise NotImplementedError

    @ti.func
    def update_particle_volume_bbar_2d(self, np, velocity_gradient, state_variables, dt):
        return self.update_particle_volume_2d(np, velocity_gradient, state_variables, dt)

    @ti.func
    def compute_stress2d(
        self,
        np,  # particle id
        previous_stress,  # state variables
        velocity_gradient,  # velocity gradient
        state_variables,  # state variables
        dt,  # time step
    ):
        raise NotImplementedError

    @ti.func
    def compute_stress(
        self,
        np,  # particle id
        previous_stress,  # state variables
        velocity_gradient,  # velocity gradient
        state_variables,  # state variables
        dt,  # time step
    ):
        raise NotImplementedError
