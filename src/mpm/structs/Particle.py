import taichi as ti
from src.utils.TypeDefinition import vec3f


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
