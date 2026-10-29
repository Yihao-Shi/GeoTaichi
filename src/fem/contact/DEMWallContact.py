"""Device IPC contact between FEM surface nodes and existing DEM facet walls."""

import taichi as ti

from src.contact_detection.continuous_contact_detection import point_triangle_accd
from src.fem.contact.ContactAssembler import FEMContactAssembler, _ipc_barrier_terms
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_nd
from src.physics_model.contact_model.ipc.ContactDistance import point_triangle_distance_grad_hess
from src.physics_model.contact_model.ipc.IPC import (
    ipc_friction_f0,
    ipc_friction_f1_over_speed,
    ipc_friction_hessian_term,
)


@ti.data_oriented
class DEMWallContactAssembler(FEMContactAssembler):
    # shortcut: scan fixed triaxial wall sets; add broad-phase pruning if facet count grows.
    def __init__(self, mesh, material, contact, boundary):
        self.wall_count = 0
        super().__init__(mesh, material, contact)
        self.wall_boundary = boundary
        self.wall_count = boundary.wall_count
        shape = (max(1, self.wall_count), max(1, self.surface.vertices.size))
        self.wall_normal_force = ti.field(ti.f64, shape=shape)
        self.wall_friction_normal = ti.Vector.field(3, ti.f64, shape=shape)

    @ti.func
    def _wall_normal_terms(self, point, wall, vertex):
        distance2, gradient12, hessian12, unused_type = point_triangle_distance_grad_hess(
            point, self.wall_boundary.vertex1[wall], self.wall_boundary.vertex2[wall], self.wall_boundary.vertex3[wall]
        )
        distance = ti.sqrt(ti.max(distance2, 1.0e-30))
        derivative_distance = ti.Vector([gradient12[i] for i in ti.static(range(3))]) / (2.0 * distance)
        hessian_distance = ti.Matrix.zero(ti.f64, 3, 3)
        for i, j in ti.static(ti.ndrange(3, 3)):
            hessian_distance[i, j] = hessian12[i, j] / (2.0 * distance) - gradient12[i] * gradient12[j] / (
                4.0 * distance**3
            )
        value, first, second = 0.0, 0.0, 0.0
        measure = self.surface_vertex_measure[vertex]
        value, first, second = _ipc_barrier_terms(distance - self.dmin, self.dhat, self.kappa)
        measure *= self.dhat
        return (
            measure * value,
            measure * first * derivative_distance,
            measure * (second * derivative_distance.outer_product(derivative_distance) + first * hessian_distance),
            derivative_distance,
            ti.max(-measure * first, 0.0),
        )

    @ti.func
    def _wall_terms(self, point, node, wall, vertex):
        energy, gradient, hessian, normal, unused_load = self._wall_normal_terms(point, wall, vertex)
        coefficient = self.contact.friction_coefficient * self.wall_normal_force[wall, vertex]
        if coefficient > 0.0:
            normal = self.wall_friction_normal[wall, vertex]
            tangent = ti.Matrix.identity(ti.f64, 3) - normal.outer_product(normal)
            dt = self.friction_dt_field[None]
            velocity = tangent @ (point - self.friction_hat_position[node] - self.wall_boundary.offset[wall]) / dt
            speed = velocity.norm()
            profile = ipc_friction_f1_over_speed(speed, self.contact.epsv)
            energy += coefficient * ipc_friction_f0(speed, self.contact.epsv, dt)
            gradient += coefficient * profile * velocity
            hessian += coefficient / dt * profile * tangent
            if speed > 0.0:
                hessian += (
                    coefficient
                    / dt
                    * ipc_friction_hessian_term(speed, self.contact.epsv)
                    / speed
                    * velocity.outer_product(velocity)
                )
        return energy, gradient, hessian

    @ti.kernel
    def _freeze_wall_friction_device(self, positions: ti.template()):
        for wall, vertex in ti.ndrange(self.wall_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex]
            unused_energy, unused_gradient, unused_hessian, normal, load = self._wall_normal_terms(
                positions[node], wall, vertex
            )
            self.wall_normal_force[wall, vertex] = load
            self.wall_friction_normal[wall, vertex] = normal

    def begin_step_device(self, positions, dt):
        super().begin_step_device(positions, dt)

    def refresh_friction_device(self, positions):
        super().refresh_friction_device(positions)
        if self.activate_friction:
            self._freeze_wall_friction_device(positions)

    @ti.kernel
    def _assemble_walls_device(self, positions: ti.template(), force: ti.template()):
        for wall in range(self.wall_count):
            self.wall_boundary.fem_force[wall] = ti.Vector.zero(ti.f64, 3)
            self.wall_boundary.fem_stiffness[wall] = 0.0
        for wall, vertex in ti.ndrange(self.wall_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex]
            energy, gradient, hessian = self._wall_terms(positions[node], node, wall, vertex)
            ti.atomic_add(self.total_energy[None], energy)
            if energy != 0.0:
                ti.atomic_add(self.active_count[None], 1)
            for component in ti.static(range(3)):
                ti.atomic_add(force[node][component], gradient[component])
                ti.atomic_add(self.wall_boundary.fem_force[wall][component], gradient[component])
            normal = self.wall_boundary.normal[wall]
            ti.atomic_add(self.wall_boundary.fem_stiffness[wall], ti.max(0.0, normal.dot(hessian @ normal)))

    def assemble_device(self, positions, force, stiffness, *, need_stiffness=False):
        force, stiffness = super().assemble_device(positions, force, stiffness, need_stiffness=need_stiffness)
        self._assemble_walls_device(positions, force)
        self._last_energy = float(self.total_energy[None])
        self._active_contacts = int(self.active_count[None])
        return force, stiffness

    def stiffness_entry_count_device(self):
        return super().stiffness_entry_count_device() + 9 * self.wall_count * self.surface.vertices.size

    @ti.kernel
    def _scatter_walls_device(self, target: ti.template(), scalar_base: ti.i32, assemble_hash: ti.template()):
        for wall, vertex in ti.ndrange(self.wall_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex]
            unused_energy, unused_gradient, hessian = self._wall_terms(
                self.stiffness_positions[node], node, wall, vertex
            )
            if ti.static(self.contact.project_pd):
                hessian = psd_project_nd(hessian)
            for i, j in ti.static(ti.ndrange(3, 3)):
                if ti.static(assemble_hash):
                    ti.atomic_add(target.diag[node][3 * i + j], hessian[i, j])
                else:
                    entry = scalar_base + 9 * (wall * self.surface.vertices.size + vertex) + 3 * i + j
                    target.rows[entry], target.cols[entry], target.data[entry] = (
                        3 * node + i,
                        3 * node + j,
                        hessian[i, j],
                    )

    def scatter_stiffness_to_hash(self, matrix):
        super().scatter_stiffness_to_hash(matrix)
        self._scatter_walls_device(matrix, 0, True)

    def scatter_stiffness_to_coo(self, matrix, offset=0):
        super().scatter_stiffness_to_coo(matrix, offset)
        self._scatter_walls_device(matrix, int(offset) + super().stiffness_entry_count_device(), False)

    @ti.kernel
    def limit_wall_motion_device(self, positions: ti.template()):
        for wall, vertex in ti.ndrange(self.wall_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex]
            zero = ti.Vector.zero(ti.f64, 3)
            displacement = self.wall_boundary.offset[wall]
            alpha = point_triangle_accd(
                positions[node],
                self.wall_boundary.vertex1[wall],
                self.wall_boundary.vertex2[wall],
                self.wall_boundary.vertex3[wall],
                zero,
                displacement,
                displacement,
                displacement,
                1.0 - self.contact.ccd_safety,
                self.dmin,
                self.contact.ccd_max_iterations,
            )
            ti.atomic_min(self.wall_boundary.motion_alpha[None], alpha)

    @ti.kernel
    def _wall_ccd_device(self, positions: ti.template(), direction: ti.template()):
        for wall, vertex in ti.ndrange(self.wall_count, self.surface.vertices.size):
            node = self.surface_vertices[vertex]
            zero = ti.Vector.zero(ti.f64, 3)
            alpha = point_triangle_accd(
                positions[node],
                self.wall_boundary.vertex1[wall],
                self.wall_boundary.vertex2[wall],
                self.wall_boundary.vertex3[wall],
                direction[node],
                zero,
                zero,
                zero,
                1.0 - self.contact.ccd_safety,
                self.dmin,
                self.contact.ccd_max_iterations,
            )
            ti.atomic_min(self.ccd_alpha[None], alpha)

    def maximum_admissible_step_device(self, positions, direction):
        super().maximum_admissible_step_device(positions, direction)
        self._wall_ccd_device(positions, direction)
        return max(0.0, min(1.0, float(self.ccd_alpha[None])))
