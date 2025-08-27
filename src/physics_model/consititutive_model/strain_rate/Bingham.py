import taichi as ti

from src.physics_model.consititutive_model.MaterialKernel import SphericalTensor
from src.physics_model.consititutive_model.MaterialModel import Fluid
from src.utils.constants import ZEROVEC6f, EYE, Threshold
from src.utils.ObjectIO import DictIO
from src.utils.VectorFunction import voigt_tensor_trace, voigt_tensor_dot


@ti.data_oriented
class BinghamModel(Fluid):
    def __init__(self, material_type="Fluid", configuration="UL", solver_type="Explicit"):
        super().__init__(material_type, configuration, solver_type)
        self._yield = 0.
        self.critical_rate = 0.

    def model_initialize(self, material):
        density = DictIO.GetAlternative(material, 'Density', 1000)
        modulus = DictIO.GetEssential(material, 'Modulus')
        viscosity = DictIO.GetEssential(material, 'Viscosity')
        _yield = DictIO.GetEssential(material, 'YieldStress')
        critical_rate = DictIO.GetEssential(material, 'CriticalStrainRate')
        gamma = DictIO.GetAlternative(material, 'gamma', 7.)
        atmospheric_pressure = DictIO.GetAlternative(material, 'atmospheric_pressure', 0.)
        self.add_material(density, modulus, viscosity, _yield, critical_rate, gamma, atmospheric_pressure)

    def add_material(self, density, modulus, viscosity, _yield, critical_rate, atmospheric_pressure, gamma=1.):
        self.density = density
        self.modulus = modulus
        self.viscosity = viscosity
        self._yield = _yield
        self.critical_rate = critical_rate
        self.atmospheric_pressure = atmospheric_pressure
        self.gamma = gamma
        self.max_sound_speed = self.get_sound_speed(self.density, self.modulus)

    def print_message(self, materialID):
        print(" Constitutive Model Information ".center(71, '-'))
        print('Constitutive model = Bingham Model')
        print("Model ID: ", materialID)
        print("Model density = ",  self.density)
        print('Bulk Modulus = ', self.modulus)
        print('Viscosity = ', self.viscosity)
        print('Yield Stress = ', self._yield)
        print('Critical Shear Rate = ', self.critical_rate, '\n')
    
    def define_state_vars(self):
        if self.solver_type == 0:
            return {'pressure': float, 'rho': float}
        elif self.solver_type == 1:
            return {}
    
    @ti.func
    def _initialize_vars_(self, np, particle, stateVars):
        stress = particle[np].stress
        stateVars[np].pressure = SphericalTensor(stress)
        stateVars[np].rho = self.density

    @ti.func
    def shear_stress(self, strain_rate):
        _yield = self._yield
        viscosity = self.viscosity
        critical_shear_rate = ti.min(1e-8, self.critical_rate)

        # Rate of shear = sqrt(2 * D_ij * D_ij)
        shear_rate = ti.sqrt(2. * voigt_tensor_dot(strain_rate, strain_rate))
        
        apparent_viscosity = 0.
        if shear_rate * shear_rate > critical_shear_rate * critical_shear_rate:
            apparent_viscosity = 2. * (_yield / shear_rate + viscosity)
        tau = apparent_viscosity * strain_rate

        trace_invariant2 = 0.5 * (tau[0] * tau[0] + tau[1] * tau[1] + tau[2] * tau[2])
        for d in ti.static(range(3)): tau[3 + d] *= 2
        if trace_invariant2 < _yield * _yield: tau = ZEROVEC6f
        return tau

    @ti.func
    def core(self, np, strain_rate, stateVars, dt):   
        tau = self.shear_stress(strain_rate)
        volumetric_strain_rate = voigt_tensor_trace(strain_rate)
        pressure = -stateVars[np].pressure + self.thermodynamic_pressure(stateVars[np].rho, volumetric_strain_rate * dt[None])
        updated_stress = -pressure * EYE + tau 
        stateVars[np].pressure = -pressure
        return updated_stress