import taichi as ti
import numpy as np
import os

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
import src.utils.GlobalVariable as GlobalVariable
from src.utils.MatrixFunction import trace, matrix_form, matrix_form_2d
from src.utils.VectorFunction import voigt_form, voigt_form_2d, voigt_tensor_trace
from src.utils.constants import DELTA, DELTA2D
from src.utils.FieldIO import field_to_numpy_slice
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

    def print_message(self, materialID):
        self.print_console_header()
        print("Constitutive model: Material Model")
        print("Material ID: ", materialID)
        print("Model density = ", self.density)

    def random_field_initialize(self, parameter):
        random_field_file = DictIO.GetAlternative(parameter, "MaterialFile", "RandomField.txt")
        residual_frac = DictIO.GetAlternative(parameter, "ResidualFraction", 1.0)
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
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        raise NotImplementedError

    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        return self.update_particle_volume(np, velocity_gradient, stateVars, dt)

    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        raise NotImplementedError

    @ti.func
    def update_particle_volume_bbar_2D(self, np, velocity_gradient, stateVars, dt):
        return self.update_particle_volume_2D(np, velocity_gradient, stateVars, dt)

    @ti.func
    def ComputeStress2D(
        self,
        np,  # particle id
        previous_stress,  # state variables
        velocity_gradient,  # velocity gradient
        stateVars,  # state variables
        dt,  # time step
    ):
        raise NotImplementedError

    @ti.func
    def ComputeStress(
        self,
        np,  # particle id
        previous_stress,  # state variables
        velocity_gradient,  # velocity gradient
        stateVars,  # state variables
        dt,  # time step
    ):
        raise NotImplementedError


@ti.data_oriented
class Solid(MaterialModel):
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
                fluid_bulk=0.0,
                porosity=0.0,
                permeability=0.0,
                fluid_viscosity=1.0e-3,
                grain_diameter=1.0e-3,
                fluid_unit_weight=9800.0,
                drag_model=0,
                maximum_porosity=1.0,
                cavitation_pressure=-1.0e30,
            )

    def add_coupling_material(self, material):
        if self.material_type == "TwoPhaseSingleLayer" or self.material_type == "TwoPhaseDoubleLayer":
            solid_density = DictIO.GetAlternative(material, "SolidDensity", 2650)
            fluid_density = DictIO.GetAlternative(material, "FluidDensity", 1000)
            porosity = DictIO.GetEssential(material, "Porosity")
            fluid_bulk = DictIO.GetEssential(material, "FluidBulkModulus")
            permeability = DictIO.GetEssential(material, "Permeability")
            fluid_viscosity = DictIO.GetAlternative(
                material, "FluidViscosity", DictIO.GetAlternative(material, "Viscosity", 1.0e-3)
            )
            grain_diameter = DictIO.GetAlternative(material, "GrainDiameter", max(permeability**0.5, 1.0e-6))
            fluid_unit_weight = DictIO.GetAlternative(
                material, "FluidUnitWeight", fluid_density * DictIO.GetAlternative(material, "Gravity", 9.8)
            )
            drag_model = DictIO.GetAlternative(material, "DragModel", DictIO.GetAlternative(material, "drag_model", 0))
            maximum_porosity = DictIO.GetAlternative(material, "MaximumPorosity", 1.0)
            cavitation_pressure = DictIO.GetAlternative(material, "CavitationPressure", -1.0e30)
            if isinstance(drag_model, str):
                drag_model = {"ergun": 0, "darcy": 1, "permeability": 1, "beetstra": 2}.get(drag_model.lower(), 0)
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
                return np.repeat(DictIO.GetEssential(self.material, "k0"), end_index - start_index)
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
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        raise NotImplementedError

    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        raise NotImplementedError

    @ti.func  # two phase
    def update_particle_porosity(self, velocity_gradient, porosity, dt):
        return (
            1.0
            - (1.0 - porosity)
            / (ti.Matrix.identity(float, GlobalVariable.DIMENSION) + velocity_gradient * dt[None]).determinant()
        )

    @ti.func
    def update_particle_porosity_2D(self, np, velocity_gradient, stateVars, particle, dt):
        particle[np].porosity = (
            1.0 - (1.0 - particle[np].porosity) / (DELTA2D + velocity_gradient * dt[None]).determinant()
        )

    @ti.func
    def update_particle_porosity_2D_u_p(self, np, velocity_gradient, stateVars, particle, dt):
        particle[np].porosity = (
            1.0 - (1.0 - particle[np].porosity) / (DELTA2D + velocity_gradient * dt[None]).determinant()
        )

    @ti.func
    def update_particle_porosity_axisy(self, np, velocity_gradient, stateVars, particle, dt):
        particle[np].porosity = (
            1.0 - (1.0 - particle[np].porosity) / (DELTA + velocity_gradient * dt[None]).determinant()
        )

    @ti.func
    def compute_single_layer_effective_stress_2d(self, np, previous_stress, velocity_gradient, porosity, stateVars, dt):
        updated_stress = previous_stress * 0.0
        if porosity <= self.maximum_porosity:
            updated_stress = self.ComputeStress2D(np, previous_stress, velocity_gradient, stateVars, dt)
        return updated_stress

    @ti.func
    def compute_single_layer_effective_stress(self, np, previous_stress, velocity_gradient, porosity, stateVars, dt):
        updated_stress = previous_stress * 0.0
        if porosity <= self.maximum_porosity:
            updated_stress = self.ComputeStress(np, previous_stress, velocity_gradient, stateVars, dt)
        return updated_stress

    @ti.func  # two phase
    def ComputePressure(self, solid_velocity_gradient, fluid_velocity_gradient, porosity, dt):
        vs = (
            ti.Matrix.identity(float, GlobalVariable.DIMENSION) + solid_velocity_gradient * dt[None]
        ).determinant() - 1.0
        vf = (
            ti.Matrix.identity(float, GlobalVariable.DIMENSION) + fluid_velocity_gradient * dt[None]
        ).determinant() - 1.0
        return self.fluid_bulk / porosity * ((1.0 - porosity) * vs + porosity * vf)

    @ti.func
    def ComputePressure_2D(self, np, velocity_gradients, velocity_gradientf, stateVars, particle, dt):
        vs = (DELTA2D + velocity_gradients * dt[None]).determinant() - 1.0
        vf = (DELTA2D + velocity_gradientf * dt[None]).determinant() - 1.0
        particle[np].pressure -= (
            self.fluid_bulk / particle[np].porosity * ((1.0 - particle[np].porosity) * vs + particle[np].porosity * vf)
        )

    @ti.func  # two phase
    def update_particle_fluid_mass(self, pvolume, porosity):
        return pvolume * porosity * self.fluid_density

    @ti.func
    def update_particle_massf(self, np, stateVars, particle):
        # A single-point control volume follows the skeleton, not the water;
        # its pore-fluid mass changes with the current pore volume.
        particle[np].mf = particle[np].vol * particle[np].porosity * self.fluid_density
        particle[np].m = particle[np].ms + particle[np].mf

    @ti.func
    def get_density(self):
        return self.solid_density, self.fluid_density

    @ti.func
    def PK2CauchyStress(self, np, PKstress, stateVars):
        deformation_gradient = F3d(stateVars[np].deformation_gradient)
        inv_j = 1.0 / deformation_gradient.determinant()
        return voigt_form(PKstress @ deformation_gradient.transpose() * inv_j)

    @ti.func
    def PK2CauchyStress2D(self, np, PKstress, stateVars):
        deformation_gradient = stateVars[np].deformation_gradient
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
    def Cauchy2PKStress(self, np, stateVars, stress):
        deformation_gradient = F3d(stateVars[np].deformation_gradient)
        j = deformation_gradient.determinant()
        return matrix_form(stress) @ deformation_gradient.inverse().transpose() * j

    @ti.func
    def Cauchy2PKStress2D(self, np, stateVars, stress):
        deformation_gradient = stateVars[np].deformation_gradient
        j = deformation_gradient.determinant()
        return matrix_form_2d(stress) @ deformation_gradient.inverse().transpose() * j

    @ti.func
    def ComputePKStress(self, np, previous_PKstress, velocity_gradient, stateVars, dt):
        raise NotImplementedError

    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):
        raise NotImplementedError

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):
        raise NotImplementedError

    @ti.func
    def compute_elastic_tensor(self, np, current_stress, stateVars):
        raise NotImplementedError

    @ti.func
    def compute_stiffness_tensor(self, np, current_stress, stateVars):
        raise NotImplementedError


@ti.data_oriented
class Fluid(MaterialModel):
    def __init__(self, material_type, configuration, solver_type):
        super().__init__(material_type, configuration, solver_type)
        self.atmospheric_pressure = 0.0
        self.surface_tension = 0.0
        self.modulus = 0.0
        self.viscosity = 0.0
        self.element_length = 0.0
        self.cl = 0.0
        self.cq = 0.0
        self.gamma = 0.0
        if "TL" in self.configuration:
            raise RuntimeError("Fluid materials do not support total Lagrangian simulation")

    def add_material(self, *args, **kwargs):
        pass

    def initialize_coupling(self):
        pass

    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        return np.repeat(1.0, end_index - start_index)

    def get_state_vars(self):
        self._initialize_vars = self._initialize_vars_
        return self.define_state_vars()

    def get_sound_speed(self, density, modulus):
        return np.where(density > 0, np.sqrt(modulus / density), 0)

    @ti.func
    def _initialize_vars_(np, particle, stateVars):
        raise NotImplementedError

    @ti.func
    def _set_modulus(self, velocity):
        velocity = 1000 * velocity
        ti.atomic_max(self.modulus, self.density * velocity / self.gamma)

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        delta_jacobian = 1.0 + dt[None] * trace(velocity_gradient)
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian

    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        return self.update_particle_volume(np, velocity_gradient, stateVars, dt)

    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        delta_jacobian = 1.0 + dt[None] * voigt_tensor_trace(strain_rate)
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian

    @ti.func
    def thermodynamic_pressure(self, rho, volumertic_strain):
        pressure = -rho * self.modulus / self.density * volumertic_strain
        return pressure

    @ti.func
    def artifical_viscosity(self, np, volumetric_strain_rate, stateVars):
        # VonNeumann J. 1950, A method for the numerical calculation of hydrodynamic shocks. J. Appl. Phys.
        q = 0.0
        if volumetric_strain_rate < 0.0:
            q = (
                -stateVars[np].rho * self.cl * self.element_length * volumetric_strain_rate
                + stateVars[np].rho
                * self.cq
                * self.element_length
                * self.element_length
                * volumetric_strain_rate
                * volumetric_strain_rate
            )
        return q

    @ti.func
    def fluid_pressure(self, np, stateVars, strain_rate, dt):
        volumetric_strain_rate = voigt_tensor_trace(strain_rate)
        volumetric_strain_increment = volumetric_strain_rate * dt[None]
        pressure = ti.max(
            0.0, -stateVars[np].pressure + self.thermodynamic_pressure(stateVars[np].rho, volumetric_strain_increment)
        )
        artifical_pressure = self.artifical_viscosity(np, volumetric_strain_rate, stateVars)
        pressureAV = pressure + artifical_pressure
        stateVars[np].pressure = -pressure
        return pressureAV

    @ti.func
    def ComputeStress2D(self, np, previous_stress, velocity_gradient, stateVars, dt):
        strain_rate = calculate_strain_rate2D(velocity_gradient)
        return self.core(np, strain_rate, stateVars, dt)

    @ti.func
    def ComputeStress(self, np, previous_stress, velocity_gradient, stateVars, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        return self.core(np, strain_rate, stateVars, dt)

    @ti.func
    def ComputePressure2D(self, np, stateVars, velocity_gradient, dt):
        strain_rate = calculate_strain_rate2D(velocity_gradient)
        return self.fluid_pressure(np, stateVars, strain_rate, dt)

    @ti.func
    def ComputePressure(self, np, stateVars, velocity_gradient, dt):
        strain_rate = calculate_strain_rate(velocity_gradient)
        return self.fluid_pressure(np, stateVars, strain_rate, dt)

    @ti.func
    def ComputeShearStress2D(self, velocity_gradient):
        strain_rate = calculate_strain_rate2D(velocity_gradient)
        return self.shear_stress(strain_rate)

    @ti.func
    def ComputeShearStress(self, velocity_gradient):
        strain_rate = calculate_strain_rate(velocity_gradient)
        return self.shear_stress(strain_rate)

    @ti.func
    def shear_stress(self, strain_rate):
        raise NotImplementedError

    @ti.func
    def core(self, np, strain_rate, stateVars, dt):
        raise NotImplementedError
