from src.utils.MatrixFunction import _stress_increment_mat3
from src.utils.MatrixFunction import _stress_increment_voigt

import taichi as ti
from src.utils.TypeDefinition import real, vec3f, vec6f


@ti.dataclass
class LargeScaleParticle:
    particleID: int
    vol: real
    x: vec3f
    v: vec3f
    stress: vec6f

    @ti.func
    def _restart(
        self,
        particle_id,
        body_id,
        material_id,
        active,
        mass,
        position,
        velocity,
        volume,
        stress,
        velocity_gradient,
        fix_v,
    ):
        self.particleID = int(particle_id)
        self.x = float(position)
        self.v = float(velocity)
        self.vol = float(volume)
        self.stress = float(stress)

    @ti.func
    def _set_essential(self, particle_id, body_id, material_id, density, particle_volume, position, init_v, fix_v):
        self.particleID = particle_id
        self.m = float(particle_volume * density)
        self.x = float(position)
        self.v = float(init_v)
        self.vol = float(particle_volume)

    @ti.func
    def _add_gravity_field(self, gamma):
        if ti.static(self.stress.n == 6):
            self.stress += _stress_increment_voigt(gamma)
        elif ti.static(self.stress.n == 3):
            self.stress += _stress_increment_mat3(gamma)

    @ti.func
    def _update_stress(self, stress):
        if ti.static(self.stress.n == 6):
            self.stress += _stress_increment_voigt(stress)
        elif ti.static(self.stress.n == 3):
            self.stress += _stress_increment_mat3(stress)

    @ti.func
    def _get_mean_stress(self):
        pressure = 0.0
        if ti.static(self.stress.n == 6):
            pressure = 1.0 / 3.0 * (self.stress[0] + self.stress[1] + self.stress[2])
        elif ti.static(self.stress.n == 3):
            pressure = 1.0 / 3.0 * (self.stress[0, 0] + self.stress[1, 1] + self.stress[2, 2])
        return pressure


@ti.dataclass
class ParticleCPDI:
    r0: vec3f
    r1: vec3f
    r2: vec3f

    @ti.func
    def _set(self, r0, r1, r2):
        self.r0 = r0
        self.r1 = r1
        self.r2 = r2

    @ti.func
    def _update(self, deformation_gradient):
        self.r0 = deformation_gradient @ self.r0
        self.r1 = deformation_gradient @ self.r1
        self.r2 = deformation_gradient @ self.r2
