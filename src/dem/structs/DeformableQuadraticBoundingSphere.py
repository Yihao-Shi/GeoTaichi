import taichi as ti
import numpy as np
from src.utils.constants import ZEROVEC3F, ZEROMAT3X9
from src.utils.MatrixFunction import l_uinverse, eigenvalue
from src.utils.TypeDefinition import vec3f, mat3x3, vec9f


@ti.dataclass
class DeformableQuadraticBoundingSphere:
    active: ti.u8
    rad: float
    rad0: float
    mass_center0: vec3f
    x: vec3f
    x0: vec3f
    verletDisp: vec3f
    qmax: float
    mmax: float

    @ti.func
    def _restart(self, active, bounding_center, bounding_radius):
        self.active = ti.u8(active)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)

    @ti.func
    def _add_bounding_sphere(self, bounding_center, bounding_radius):
        self.active = ti.u8(1)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)

    @ti.func
    def _move(self, disp, true_disp):
        self.x += disp
        self.verletDisp += true_disp

    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.rad *= float(scale)
        self.x = float(scale * (self.x - centor_of_mass) + centor_of_mass)

    @ti.func
    def _translate(self, offset):
        self.x += offset

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3F

    @ti.func
    def _compute_auxiliary_parameter(self, start_index, end_index, particle):
        center = self.x
        for np in range(start_index, end_index):
            position = particle[np].x
            self.qmax = max(
                self.qmax,
                vec3f(
                    position[0] * position[0] - center[0] * center[0],
                    position[1] * position[1] - center[1] * center[1],
                    position[2] * position[2] - center[2] * center[2],
                ).norm(),
            )
            self.mmax = max(
                self.mmax,
                vec3f(
                    position[0] * position[1] - center[0] * center[1],
                    position[1] * position[2] - center[1] * center[2],
                    position[0] * position[2] - center[0] * center[2],
                ).norm(),
            )

    @ti.func
    def _get_multisphere_index1(self):
        return -1

    @ti.func
    def _get_multisphere_index2(self):
        return 1

    @ti.func
    def _get_position(self):
        return self.x

    @ti.func
    def _get_radius(self):
        return self.rad

    @ti.func
    def _get_verlet_displacement(self):
        return self.verletDisp

    @ti.func
    def _evolution(self, start_index, end_index, mass_center, particle):
        # Efficient updates of bounding sphere hierarchies for geometrically deformable models
        # Meshless Deformations based on shape matching
        apq = ZEROMAT3X9
        aqq = ti.Matrix.zero(float, 9, 9)
        mass_center = vec3f(0.0, 0.0, 0.0)
        for np in range(start_index, end_index):
            mass_center += particle[np].x
        mass_center /= end_index - start_index
        for np in range(start_index, end_index):
            pp = particle[np].x - self.mass_center
            qq0 = particle[np].x0 - self.mass_center0
            qq = vec9f(
                qq0[0],
                qq0[1],
                qq0[2],
                qq0[0] * qq0[0],
                qq0[1] * qq0[1],
                qq0[2] * qq0[2],
                qq0[0] * qq0[1],
                qq0[1] * qq0[2],
                qq0[2] * qq0[0],
            )
            apq += particle[np].m * pp.outer_product(qq)
            aqq += particle[np].m * qq.outer_product(qq)
        a_cap = apq @ l_uinverse(aqq)
        d = 0.0, 0.0, 0.0
        for np in range(start_index, end_index):
            pp = particle[np].x - mass_center
            qq0 = particle[np].x0 - self.mass_center0
            qq = vec9f(
                qq0[0],
                qq0[1],
                qq0[2],
                qq0[0] * qq0[0],
                qq0[1] * qq0[1],
                qq0[2] * qq0[2],
                qq0[0] * qq0[1],
                qq0[1] * qq0[2],
                qq0[2] * qq0[0],
            )
            d = max(d, (a_cap @ qq - pp).norm())
        a = mat3x3(
            [a_cap[0, 0], a_cap[0, 1], a_cap[0, 2]],
            [a_cap[1, 0], a_cap[1, 1], a_cap[1, 2]],
            [a_cap[2, 0], a_cap[2, 1], a_cap[2, 2]],
        )
        q = mat3x3(
            [a_cap[0, 3], a_cap[0, 4], a_cap[0, 5]],
            [a_cap[1, 3], a_cap[1, 4], a_cap[1, 5]],
            [a_cap[2, 3], a_cap[2, 4], a_cap[2, 5]],
        )
        m = mat3x3(
            [a_cap[0, 6], a_cap[0, 7], a_cap[0, 8]],
            [a_cap[1, 6], a_cap[1, 7], a_cap[1, 8]],
            [a_cap[2, 6], a_cap[2, 7], a_cap[2, 8]],
        )
        aval = eigenvalue(a.transpose() @ a)
        qval = eigenvalue(q.transpose() @ q)
        mval = eigenvalue(m.transpose() @ m)
        self.rad = (
            ti.sqrt(abs(aval[2])) * self.rad0
            + ti.sqrt(abs(qval[2])) * self.qmax
            + ti.sqrt(abs(mval[2])) * self.mmax
            + d
        )
        cc0 = self.x0 - self.mass_center0
        cc = vec9f(
            cc0[0],
            cc0[1],
            cc0[2],
            cc0[0] * cc0[0],
            cc0[1] * cc0[1],
            cc0[2] * cc0[2],
            cc0[0] * cc0[1],
            cc0[1] * cc0[2],
            cc0[2] * cc0[0],
        )
        self.x = a_cap @ cc + mass_center
