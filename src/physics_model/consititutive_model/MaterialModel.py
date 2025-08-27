import taichi as ti
import numpy as np
import os

from src.physics_model.consititutive_model.infinitesimal_strain.MaterialKernel import *
import src.utils.GlobalVariable as GlobalVariable
from src.utils.MatrixFunction import trace, matrix_form
from src.utils.VectorFunction import voigt_form, voigt_tensor_trace
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class MaterialModel:
    def __init__(self, material_type, configuration, solver_type):
        # TODO: Add essential material properties
        self.density = 0.
        self.max_sound_speed = 0.
        self._initialize_vars = None
        self.random_field_file = None
        self.material_type = material_type
        self.configuration = configuration
        self.solver_type = {"Explicit": 0, "Implicit": 1}.get(solver_type)

    def members_update(self, **config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value) 

    def add_material(self, *args, **kwargs):
        raise NotImplementedError

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Material Model')
        print("Model ID: ", materialID)
        print("Model density = ",  self.density)

    def random_field_initialize(self, parameter):
        random_field_file = DictIO.GetAlternative(parameter, "MaterialFile", "RandomField.txt")
        self.random_field_file = random_field_file
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
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        raise NotImplementedError

    @ti.func
    def ComputeStress2D(self, 
                        np,                                                     # particle id 
                        previous_stress,                                        # state variables
                        velocity_gradient,                                      # velocity gradient
                        stateVars,                                              # state variables
                        dt                                                      # time step
                        ):  
        raise NotImplementedError

    @ti.func
    def ComputeStress(self,
                      np,                                                     # particle id 
                      previous_stress,                                        # state variables
                      velocity_gradient,                                      # velocity gradient
                      stateVars,                                              # state variables
                      dt                                                      # time step
                     ):                  
        raise NotImplementedError


@ti.data_oriented
class Solid(MaterialModel):
    def __init__(self, material_type, configuration, solver_type):
        super().__init__(material_type, configuration, solver_type)
        self.young = 0.
        self.poisson = 0.
        self.shear = 0.
        self.bulk = 0.
        self.is_elastic = False
        self.core = None

    def add_material(self, *args, **kwargs):
        raise NotImplementedError
    
    def initialize_coupling(self):
        if self.material_type == "TwoPhaseSingleLayer":
            self.members_update(solid_density=0., fluid_density=0., fluid_bulk=0., porosity=0., permeability=0.)

    def add_coupling_material(self, material):
        if self.material_type == "TwoPhaseSingleLayer":
            solid_density = DictIO.GetAlternative(material, 'SolidDensity', 2650)
            fluid_density = DictIO.GetAlternative(material, 'FluidDensity', 1000)
            porosity = DictIO.GetEssential(material, 'Porosity')
            fluid_bulk = DictIO.GetEssential(material, 'FluidBulkModulus')
            permeability = DictIO.GetEssential(material, 'Permeability')
        
            self.density = fluid_density * porosity + solid_density * (1.- porosity)
            self.solid_density = solid_density
            self.fluid_density = fluid_density
            self.porosity = porosity
            self.fluid_bulk = fluid_bulk
            self.permeability = permeability
    
    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        if GlobalVariable.RANDOMFIELD:
            particle_index = np.ascontiguousarray(materialID.to_numpy()[start_index:end_index])
            shear = np.ascontiguousarray(stateVars.shear.to_numpy()[particle_index])
            bulk = np.ascontiguousarray(stateVars.bulk.to_numpy()[particle_index])
            poisson = 0.5 * (3. * bulk - 2. * shear) / (3. * bulk + shear)
            return poisson / (1. - poisson)
        else:
            poisson = self.poisson
            return np.repeat(poisson / (1. - poisson), end_index - start_index)
    
    def get_state_vars(self):
        raise NotImplementedError
    
    def compute_elasto_plastic_stiffness(self, particleNum, particle):
        raise NotImplementedError

    def get_sound_speed(self, density, young, poisson):
        return np.where(density > 0, 0, np.sqrt(young * (1 - poisson) / (1 + poisson) / (1 - 2 * poisson) / density))
    
    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        raise NotImplementedError
    
    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        raise NotImplementedError
    
    @ti.func    # two phase
    def update_particle_porosity(self, velocity_gradient, porosity, dt):
        return 1.0 - (1.0 - porosity) / (ti.Matrix.identity(float, GlobalVariable.DIMENSION) + velocity_gradient * dt[None]).determinant()

    @ti.func    # two phase
    def ComputePressure(self, solid_velocity_gradient, fluid_velocity_gradient, porosity, dt):
        vs = ((ti.Matrix.identity(float, GlobalVariable.DIMENSION) + solid_velocity_gradient * dt[None]).determinant() - 1.0)
        vf = ((ti.Matrix.identity(float, GlobalVariable.DIMENSION) + fluid_velocity_gradient * dt[None]).determinant() - 1.0)
        return self.fluid_bulk / porosity * ((1.0 - porosity) * vs + porosity * vf)

    @ti.func    # two phase
    def update_particle_fluid_mass(self, pvolume, porosity):
        return pvolume * porosity * self.fluid_density
    
    @ti.func
    def PK2CauchyStress(self, np, PKstress, stateVars):
        deformation_gradient = stateVars[np].deformation_gradient
        inv_j = 1. / deformation_gradient.determinant()
        return voigt_form(PKstress @ deformation_gradient.transpose() * inv_j)

    @ti.func
    def Cauchy2PKStress(self, np, stateVars, stress):
        deformation_gradient = stateVars[np].deformation_gradient
        j = deformation_gradient.determinant()
        return matrix_form(stress) @ deformation_gradient.inverse().transpose() * j
    
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
        self.atmospheric_pressure = 0.
        self.modulus = 0.
        self.viscosity = 0.
        self.element_length = 0.
        self.cl = 0.
        self.cq = 0.
        self.gamma = 0.
        if 'TL' in self.configuration:
            raise RuntimeError("Fluid materials do not support total Lagrangian simulation")

    def add_material(self, *args, **kwargs):
        pass

    def initialize_coupling(self):
        pass

    def get_lateral_coefficient(self, start_index, end_index, materialID, stateVars):
        return np.repeat(1., end_index - start_index)
    
    def get_state_vars(self):
        if self.solver_type == 0:
            self._initialize_vars = self._initialize_vars_
        return self.define_state_vars()

    def get_sound_speed(self, density, modulus):
        return np.where(density > 0, 0, np.sqrt(modulus / density))

    @ti.func
    def _initialize_vars_(np, particle, stateVars):
        raise NotImplementedError

    @ti.func
    def _set_modulus(self, velocity):
        velocity = 1000 * velocity
        ti.atomic_max(self.modulus, self.density * velocity / self.gamma)

    @ti.func
    def update_particle_volume(self, np, velocity_gradient, stateVars, dt):
        delta_jacobian = 1. + dt[None] * trace(velocity_gradient)
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian
    
    @ti.func
    def update_particle_volume_2D(self, np, velocity_gradient, stateVars, dt):
        return self.update_particle_volume(np, velocity_gradient, stateVars, dt)
    
    @ti.func
    def update_particle_volume_bbar(self, np, strain_rate, stateVars, dt):
        delta_jacobian = 1. + dt[None] * voigt_tensor_trace(strain_rate)
        stateVars[np].rho /= delta_jacobian
        return delta_jacobian

    @ti.func
    def thermodynamic_pressure(self, rho, volumertic_strain):
        pressure = -rho * self.modulus / self.density * volumertic_strain
        return pressure
    
    @ti.func
    def artifical_viscosity(self, np, volumetric_strain_rate, stateVars):
        # VonNeumann J. 1950, A method for the numerical calculation of hydrodynamic shocks. J. Appl. Phys.
        q = 0.
        if volumetric_strain_rate < 0.:
            q = -stateVars[np].rho * self.cl * self.element_length * volumetric_strain_rate + \
                stateVars[np].rho * self.cq * self.element_length * self.element_length * volumetric_strain_rate * volumetric_strain_rate
        return q
    
    @ti.func
    def fluid_pressure(self, np, stateVars, strain_rate, dt):
        volumetric_strain_rate = voigt_tensor_trace(strain_rate) 
        volumetric_strain_increment = volumetric_strain_rate * dt[None]
        pressure = -stateVars[np].pressure + self.thermodynamic_pressure(stateVars[np].rho, volumetric_strain_increment)
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

