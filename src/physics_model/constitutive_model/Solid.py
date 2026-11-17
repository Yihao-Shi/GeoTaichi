from src.physics_model.constitutive_model.MaterialModel import MaterialModel

import taichi as ti
import numpy as np
from src.physics_model.constitutive_model.infinitesimal_strain.MaterialKernel import *
import src.utils.GlobalVariable as GlobalVariable
from src.utils.MatrixFunction import matrix_form, matrix_form_2d
from src.utils.VectorFunction import voigt_form, voigt_form_2d
from src.utils.constants import DELTA, DELTA2D
from src.utils.FieldIO import field_to_numpy_slice
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class Solid(MaterialModel):
    supports_rate_dependent = False

    def __init__(self, material_type, configuration, solver_type):
        super().__init__(material_type, configuration, solver_type)
        self.young = 0.0
        self.poisson = 0.0
        self.shear = 0.0
        self.bulk = 0.0
        self.is_elastic = False
        self.core = None

    def add_material(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def validate_elastic_parameters(density, young, poisson):
        """Validate the scalar parameters used by isotropic elastic solids."""

        parameters = {
            "Density": density,
            "YoungModulus": young,
            "PoissonRatio": poisson,
        }
        for name, value in parameters.items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if density <= 0.0:
            raise ValueError(f"Density must be positive, got {density!r}")
        if young <= 0.0:
            raise ValueError(f"YoungModulus must be positive, got {young!r}")
        if not -1.0 < poisson < 0.5:
            raise ValueError("PoissonRatio must lie strictly between -1 and 0.5, " f"got {poisson!r}")

    @staticmethod
    def validate_nonnegative_parameters(**parameters):
        for name, value in parameters.items():
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")

    @staticmethod
    def validate_friction_angle(name, value):
        if not np.isfinite(value) or not 0.0 <= value < 90.0:
            raise ValueError(f"{name} must lie in [0, 90) degrees, got {value!r}")

    def initialize_coupling(self):
        if self.material_type == "TwoPhaseSingleLayer" or self.material_type == "TwoPhaseDoubleLayer":
            self.members_update(
                solid_density=0.0,
                fluid_density=0.0,
                fluid_bulk=None if self.incompressible_fluid else 0.0,
                porosity=0.0,
                permeability=0.0,
                fluid_viscosity=1.0e-3,
                grain_diameter=1.0e-3,
                fluid_unit_weight=9800.0,
                drag_model=0,
                maximum_porosity=1.0,
                cavitation_pressure=-1.0e30,
            )

    def validate_rate_dependent(self, material):
        enabled = DictIO.get_alternative(material, "RateDependent", False)
        if not isinstance(enabled, (bool, np.bool_)):
            raise ValueError("RateDependent must be a boolean")
        if enabled:
            if self.solver_type != 0:
                raise ValueError(
                    "mu(I) RateDependent requires an Explicit solver; implicit MPM does not "
                    "support RateDependent=True because a consistent rate-dependent tangent "
                    "has not been implemented"
                )
            if not self.supports_rate_dependent:
                raise ValueError(f"{type(self).__name__} does not implement mu(I) RateDependent")
        return bool(enabled)

    def add_coupling_material(self, material):
        enabled = self.validate_rate_dependent(material)
        if enabled and not getattr(self, "rate_dependent_function", False):
            raise ValueError(f"{type(self).__name__} does not implement mu(I) RateDependent")
        if self.material_type == "TwoPhaseSingleLayer" or self.material_type == "TwoPhaseDoubleLayer":
            solid_density = DictIO.get_alternative(material, "SolidDensity", 2650)
            fluid_density = DictIO.get_alternative(material, "FluidDensity", 1000)
            porosity = DictIO.get_essential(material, "Porosity")
            # Legacy inputs may retain this key; projection does not use an equation of state.
            fluid_bulk = None
            if not self.incompressible_fluid:
                fluid_bulk = DictIO.get_essential(material, "FluidBulkModulus")
                self.validate_fluid_bulk_modulus(fluid_bulk, "FluidBulkModulus")
            permeability = DictIO.get_essential(material, "Permeability")
            fluid_viscosity = DictIO.get_alternative(
                material, "FluidViscosity", DictIO.get_alternative(material, "Viscosity", 1.0e-3)
            )
            grain_diameter = DictIO.get_alternative(material, "GrainDiameter", max(permeability**0.5, 1.0e-6))
            fluid_unit_weight = DictIO.get_alternative(
                material, "FluidUnitWeight", fluid_density * DictIO.get_alternative(material, "Gravity", 9.8)
            )
            drag_model = DictIO.get_alternative(
                material, "DragModel", DictIO.get_alternative(material, "drag_model", "Ergun")
            )
            maximum_porosity = DictIO.get_alternative(material, "MaximumPorosity", 1.0)
            cavitation_pressure = DictIO.get_alternative(material, "CavitationPressure", -1.0e30)
            drag_models = {"Ergun": 0, "Darcy": 1, "Beetstra": 2}
            if not isinstance(drag_model, str) or drag_model not in drag_models:
                raise ValueError(f"DragModel must be one of {tuple(drag_models)}, got {drag_model!r}")
            drag_model = drag_models[drag_model]
            if not np.isfinite(porosity) or not 0.0 < porosity < 1.0:
                raise ValueError(f"Porosity must lie strictly between 0 and 1, got {porosity!r}")
            if not np.isfinite(maximum_porosity) or not porosity <= maximum_porosity <= 1.0:
                raise ValueError(
                    "MaximumPorosity must lie between the initial Porosity and 1, "
                    f"got {maximum_porosity!r} for Porosity {porosity!r}"
                )
            if not np.isfinite(cavitation_pressure):
                raise ValueError(f"CavitationPressure must be finite, got {cavitation_pressure!r}")

            self.density = fluid_density * porosity + solid_density * (1.0 - porosity)
            self.solid_density = solid_density
            self.fluid_density = fluid_density
            self.porosity = porosity
            self.fluid_bulk = fluid_bulk
            self.permeability = permeability
            self.fluid_viscosity = fluid_viscosity
            self.grain_diameter = grain_diameter
            self.fluid_unit_weight = fluid_unit_weight
            self.drag_model = int(drag_model)
            self.maximum_porosity = maximum_porosity
            self.cavitation_pressure = cavitation_pressure

    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        if GlobalVariable.RANDOMFIELD:
            particle_index = field_to_numpy_slice(materialID, start_index, end_index)
            active_end = int(np.max(particle_index)) + 1 if particle_index.size > 0 else 0
            shear = field_to_numpy_slice(stateVars.shear, 0, active_end)[particle_index]
            bulk = field_to_numpy_slice(stateVars.bulk, 0, active_end)[particle_index]
            poisson = 0.5 * (3.0 * bulk - 2.0 * shear) / (3.0 * bulk + shear)
            return poisson / (1.0 - poisson)
        else:
            if "k0" in self.material:
                return np.repeat(DictIO.get_essential(self.material, "k0"), end_index - start_index)
            else:
                poisson = self.poisson
                return np.repeat(poisson / (1.0 - poisson), end_index - start_index)

    def get_state_vars(self):
        raise NotImplementedError

    def compute_elasto_plastic_stiffness(self, particleNum, particle):
        raise NotImplementedError

    def get_sound_speed(self, density, young, poisson):
        return np.where(density > 0, np.sqrt(young * (1 - poisson) / (1 + poisson) / (1 - 2 * poisson) / density), 0)

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, state_variables, dt):
        raise NotImplementedError

    @ti.func
    def update_particle_volume_2d(self, np, velocity_gradient, state_variables, dt):
        raise NotImplementedError

    @ti.func  # two phase
    def update_particle_porosity(self, velocity_gradient, porosity, dt):
        return (
            1.0
            - (1.0 - porosity)
            / (ti.Matrix.identity(float, GlobalVariable.DIMENSION) + velocity_gradient * dt[None]).determinant()
        )

    @ti.func
    def update_particle_porosity_2d(self, np, velocity_gradient, state_variables, particle, dt):
        particle[np].porosity = (
            1.0 - (1.0 - particle[np].porosity) / (DELTA2D + velocity_gradient * dt[None]).determinant()
        )

    @ti.func
    def update_particle_porosity_2d_u_p(self, np, velocity_gradient, state_variables, particle, dt):
        particle[np].porosity = (
            1.0 - (1.0 - particle[np].porosity) / (DELTA2D + velocity_gradient * dt[None]).determinant()
        )

    @ti.func
    def update_particle_porosity_axisy(self, np, velocity_gradient, state_variables, particle, dt):
        particle[np].porosity = (
            1.0 - (1.0 - particle[np].porosity) / (DELTA + velocity_gradient * dt[None]).determinant()
        )

    @ti.func
    def compute_single_layer_effective_stress_2d(
        self, np, previous_stress, velocity_gradient, porosity, state_variables, dt
    ):
        updated_stress = previous_stress * 0.0
        if porosity <= self.maximum_porosity:
            updated_stress = self.compute_stress2d(np, previous_stress, velocity_gradient, state_variables, dt)
        return updated_stress

    @ti.func
    def compute_single_layer_effective_stress(
        self, np, previous_stress, velocity_gradient, porosity, state_variables, dt
    ):
        updated_stress = previous_stress * 0.0
        if porosity <= self.maximum_porosity:
            updated_stress = self.compute_stress(np, previous_stress, velocity_gradient, state_variables, dt)
        return updated_stress

    @ti.func  # two phase
    def compute_pressure(self, solid_velocity_gradient, fluid_velocity_gradient, porosity, dt):
        vs = (
            ti.Matrix.identity(float, GlobalVariable.DIMENSION) + solid_velocity_gradient * dt[None]
        ).determinant() - 1.0
        vf = (
            ti.Matrix.identity(float, GlobalVariable.DIMENSION) + fluid_velocity_gradient * dt[None]
        ).determinant() - 1.0
        return self.fluid_bulk / porosity * ((1.0 - porosity) * vs + porosity * vf)

    @ti.func
    def compute_pressure_2d(self, np, velocity_gradients, velocity_gradientf, state_variables, particle, dt):
        vs = (DELTA2D + velocity_gradients * dt[None]).determinant() - 1.0
        vf = (DELTA2D + velocity_gradientf * dt[None]).determinant() - 1.0
        particle[np].pressure -= (
            self.fluid_bulk / particle[np].porosity * ((1.0 - particle[np].porosity) * vs + particle[np].porosity * vf)
        )

    @ti.func  # two phase
    def update_particle_fluid_mass(self, pvolume, porosity):
        return pvolume * porosity * self.fluid_density

    @ti.func
    def update_particle_massf(self, np, state_variables, particle):
        # A single-point control volume follows the skeleton, not the water;
        # its pore-fluid mass changes with the current pore volume.
        particle[np].mf = particle[np].vol * particle[np].porosity * self.fluid_density
        particle[np].m = particle[np].ms + particle[np].mf

    @ti.func
    def get_density(self):
        return self.solid_density, self.fluid_density

    @ti.func
    def pk2_cauchy_stress(self, np, PKstress, state_variables):
        deformation_gradient = f3d(state_variables[np].deformation_gradient)
        inv_j = 1.0 / deformation_gradient.determinant()
        return voigt_form(PKstress @ deformation_gradient.transpose() * inv_j)

    @ti.func
    def pk2_cauchy_stress2d(self, np, PKstress, state_variables):
        deformation_gradient = state_variables[np].deformation_gradient
        inv_j = 1.0 / deformation_gradient.determinant()
        PKstress2D = ti.Matrix.zero(float, 2, 2)
        if ti.static(PKstress.n == 3):
            PKstress2D[0, 0] = PKstress[0, 0]
            PKstress2D[0, 1] = PKstress[0, 1]
            PKstress2D[1, 0] = PKstress[1, 0]
            PKstress2D[1, 1] = PKstress[1, 1]
        else:
            PKstress2D = PKstress
        cauchy_stress = voigt_form_2d(PKstress2D @ deformation_gradient.transpose() * inv_j)
        if ti.static(PKstress.n == 3):
            cauchy_stress[2] = PKstress[2, 2] * inv_j
        return cauchy_stress

    @ti.func
    def cauchy2_pk_stress(self, np, state_variables, stress):
        deformation_gradient = f3d(state_variables[np].deformation_gradient)
        j = deformation_gradient.determinant()
        return matrix_form(stress) @ deformation_gradient.inverse().transpose() * j

    @ti.func
    def cauchy2_pk_stress2d(self, np, state_variables, stress):
        deformation_gradient = state_variables[np].deformation_gradient
        j = deformation_gradient.determinant()
        return matrix_form_2d(stress) @ deformation_gradient.inverse().transpose() * j

    @ti.func
    def compute_pk_stress(self, np, previous_PKstress, velocity_gradient, state_variables, dt):
        raise NotImplementedError

    @ti.func
    def compute_stress2d(self, np, previous_stress, velocity_gradient, state_variables, dt):
        raise NotImplementedError

    @ti.func
    def compute_stress(self, np, previous_stress, velocity_gradient, state_variables, dt):
        raise NotImplementedError

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, state_variables):
        raise NotImplementedError

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, state_variables):
        raise NotImplementedError
