import taichi as ti
import numpy as np
from src.utils.constants import ZEROVEC3F
from src.utils.MatrixFunction import eigenvalue
from src.utils.TypeDefinition import vec3f, mat3x3


@ti.dataclass
class DeformableBoundingSphere:
    active: ti.u8
    rad: float
    rad0: float
    rad_verlet: float
    mass_center0: vec3f
    x: vec3f
    x0: vec3f
    verletDisp: vec3f

    @ti.func
    def _restart(self, active, bounding_center, bounding_radius):
        self.active = ti.u8(active)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)
        self.rad0 = self.rad
        self.rad_verlet = self.rad
        self.x0 = self.x
        self.mass_center0 = self.x
        self.verletDisp = ZEROVEC3F

    @ti.func
    def _add_bounding_sphere(self, bounding_center, bounding_radius):
        self.active = ti.u8(1)
        self.x = float(bounding_center)
        self.rad = float(bounding_radius)
        self.rad0 = self.rad
        self.rad_verlet = self.rad
        self.x0 = self.x
        self.mass_center0 = self.x
        self.verletDisp = ZEROVEC3F

    @ti.func
    def _move(self, disp, true_disp):
        self.x += disp
        self.verletDisp += true_disp

    @ti.func
    def _scale(self, scale, centor_of_mass):
        self.rad *= float(scale)
        self.rad0 *= float(scale)
        self.rad_verlet *= float(scale)
        self.x = float(scale * (self.x - centor_of_mass) + centor_of_mass)

    @ti.func
    def _translate(self, offset):
        self.x += offset

    @ti.func
    def _renew_verlet(self):
        self.verletDisp = ZEROVEC3F
        self.rad_verlet = self.rad

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
    def _set_deformed_shape(self, mass_center, rotate_matrix, shape_min, shape_max, shape_radius, sdf_padding):
        local_center = 0.5 * (shape_min + shape_max)
        local_extent = ti.max(shape_max - shape_min, ZEROVEC3F)
        box_radius = 0.5 * local_extent.norm()
        new_center = mass_center
        geometry_radius = ti.max(shape_radius, 0.0)
        if box_radius < geometry_radius:
            new_center = mass_center + rotate_matrix @ local_center
            geometry_radius = box_radius
        self.active = ti.u8(1)
        self.x = new_center
        self.rad = geometry_radius + ti.max(sdf_padding, 0.0)
        self.rad0 = self.rad
        self.rad_verlet = self.rad
        self.x0 = self.x
        self.mass_center0 = mass_center
        self.verletDisp = ZEROVEC3F

    @ti.func
    def _follow_deformed_shape(self, mass_center, rotate_matrix, shape_min, shape_max, shape_radius, sdf_padding):
        local_center = 0.5 * (shape_min + shape_max)
        local_extent = ti.max(shape_max - shape_min, ZEROVEC3F)
        box_radius = 0.5 * local_extent.norm()
        new_center = mass_center
        geometry_radius = ti.max(shape_radius, 0.0)
        if box_radius < geometry_radius:
            new_center = mass_center + rotate_matrix @ local_center
            geometry_radius = box_radius
        self.verletDisp += new_center - self.x
        self.x = new_center
        self.rad = geometry_radius + ti.max(sdf_padding, 0.0)

    @ti.func
    def _evolution(self, start_index, end_index, particle):
        # Efficient updates of bounding sphere hierarchies for geometrically deformable models
        # Meshless Deformations based on shape matching
        apq = mat3x3([0, 0, 0], [0, 0, 0], [0, 0, 0])
        aqq = mat3x3([0, 0, 0], [0, 0, 0], [0, 0, 0])
        mass_center = vec3f(0.0, 0.0, 0.0)
        for np in range(start_index, end_index):
            mass_center += particle[np].x
        mass_center /= end_index - start_index
        for np in range(start_index, end_index):
            pp = particle[np].x - mass_center
            qq = particle[np].x0 - self.mass_center0
            apq += particle[np].m * pp.outer_product(qq)
            aqq += particle[np].m * qq.outer_product(qq)
        a = apq @ aqq.inverse()
        d = 0.0
        for np in range(start_index, end_index):
            pp = particle[np].x - mass_center
            qq = particle[np].x0 - self.mass_center0
            d = max(d, (a @ qq - pp).norm())
        aTa = a.transpose() @ a
        val = eigenvalue(aTa)
        self.rad = ti.sqrt(abs(val[2])) * self.rad0 + d
        self.x = a @ (self.x0 - self.mass_center0) + mass_center
