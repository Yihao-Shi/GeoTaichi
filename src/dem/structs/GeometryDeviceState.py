import taichi as ti
import numpy as np


@ti.data_oriented
class GeometryDeviceState:
    """Persistent state for one deformable patch-wall rigid aggregate."""

    def __init__(
        self,
        *,
        mass,
        mass_center,
        orientation,
        velocity,
        angular_velocity,
        inertia_body,
        inertia_body_inv,
        rotate_center,
        external_force,
        external_torque,
        local_v1,
        local_v2,
        local_v3,
        local_norm,
    ):
        self.mass = ti.field(dtype=float, shape=())
        self.mass_center = ti.Vector.field(3, dtype=float, shape=())
        self.orientation = ti.Matrix.field(3, 3, dtype=float, shape=())
        self.velocity = ti.Vector.field(3, dtype=float, shape=())
        self.angular_velocity = ti.Vector.field(3, dtype=float, shape=())
        self.inertia_body = ti.Matrix.field(3, 3, dtype=float, shape=())
        self.inertia_body_inv = ti.Matrix.field(3, 3, dtype=float, shape=())
        self.rotate_center = ti.Vector.field(3, dtype=float, shape=())
        self.external_force = ti.Vector.field(3, dtype=float, shape=())
        self.external_torque = ti.Vector.field(3, dtype=float, shape=())
        self.total_force = ti.Vector.field(3, dtype=float, shape=())
        self.total_torque = ti.Vector.field(3, dtype=float, shape=())

        local_v1 = np.asarray(local_v1, dtype=float).reshape((-1, 3))
        local_v2 = np.asarray(local_v2, dtype=float).reshape((-1, 3))
        local_v3 = np.asarray(local_v3, dtype=float).reshape((-1, 3))
        local_norm = np.asarray(local_norm, dtype=float).reshape((-1, 3))
        self.face_count = int(local_v1.shape[0])
        capacity = max(self.face_count, 1)
        self.local_v1 = ti.Vector.field(3, dtype=float, shape=capacity)
        self.local_v2 = ti.Vector.field(3, dtype=float, shape=capacity)
        self.local_v3 = ti.Vector.field(3, dtype=float, shape=capacity)
        self.local_norm = ti.Vector.field(3, dtype=float, shape=capacity)

        self.mass[None] = float(mass)
        self.mass_center[None] = mass_center
        self.orientation[None] = orientation
        self.velocity[None] = velocity
        self.angular_velocity[None] = angular_velocity
        self.inertia_body[None] = inertia_body
        self.inertia_body_inv[None] = inertia_body_inv
        self.rotate_center[None] = rotate_center
        self.external_force[None] = external_force
        self.external_torque[None] = external_torque
        self.total_force[None] = external_force
        self.total_torque[None] = external_torque
        if self.face_count:
            self.local_v1.from_numpy(np.ascontiguousarray(local_v1))
            self.local_v2.from_numpy(np.ascontiguousarray(local_v2))
            self.local_v3.from_numpy(np.ascontiguousarray(local_v3))
            self.local_norm.from_numpy(np.ascontiguousarray(local_norm))

    @ti.kernel
    def _reset_contact_resultant(self):
        self.total_force[None] = self.external_force[None]
        self.total_torque[None] = self.external_torque[None]

    @ti.kernel
    def _accumulate_contact_resultant(self, start_index: int, end_index: int, wall: ti.template()):
        center = self.mass_center[None]
        for wall_id in range(start_index, end_index):
            contact_force = wall[wall_id].contact_force
            contact_torque = wall[wall_id].contact_torque
            arm = wall[wall_id]._get_center() - center
            torque = arm.cross(contact_force) + contact_torque
            for component in ti.static(range(3)):
                ti.atomic_add(
                    self.total_force[None][component],
                    contact_force[component],
                )
                ti.atomic_add(self.total_torque[None][component], torque[component])

    @ti.kernel
    def _integrate_rigid_state(self, dt: float):
        mass = self.mass[None]
        if mass > 0.0:
            velocity = self.velocity[None] + dt * self.total_force[None] / mass
            center = self.mass_center[None] + dt * velocity
            orientation = self.orientation[None]
            inverse_inertia_world = orientation @ self.inertia_body_inv[None] @ orientation.transpose()
            angular_velocity = self.angular_velocity[None] + dt * (inverse_inertia_world @ self.total_torque[None])

            magnitude = angular_velocity.norm()
            if magnitude > 1.0e-14:
                axis = angular_velocity / magnitude
                skew = ti.Matrix(
                    [
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0],
                    ]
                )
                identity = ti.Matrix.identity(float, 3)
                angle = magnitude * dt
                rotation = identity + ti.sin(angle) * skew + (1.0 - ti.cos(angle)) * (skew @ skew)
                orientation = rotation @ orientation

            # Match the previous polar/SVD re-orthonormalization without a
            # host-side ``numpy.linalg.svd`` round trip.
            left, _, right = ti.svd(orientation)
            orientation = left @ right.transpose()
            if orientation.determinant() < 0.0:
                for row in ti.static(range(3)):
                    left[row, 2] = -left[row, 2]
                orientation = left @ right.transpose()

            self.velocity[None] = velocity
            self.angular_velocity[None] = angular_velocity
            self.mass_center[None] = center
            self.rotate_center[None] = center
            self.orientation[None] = orientation

    @ti.kernel
    def _update_rigid_wall(self, start_index: int, end_index: int, wall: ti.template()):
        center = self.mass_center[None]
        orientation = self.orientation[None]
        velocity = self.velocity[None]
        angular_velocity = self.angular_velocity[None]
        for wall_id in range(start_index, end_index):
            local_id = wall_id - start_index
            old_v1 = wall[wall_id].vertice1
            old_v2 = wall[wall_id].vertice2
            old_v3 = wall[wall_id].vertice3
            new_v1 = center + orientation @ self.local_v1[local_id]
            new_v2 = center + orientation @ self.local_v2[local_id]
            new_v3 = center + orientation @ self.local_v3[local_id]
            displacement1 = new_v1 - old_v1
            displacement2 = new_v2 - old_v2
            displacement3 = new_v3 - old_v3
            displacement = displacement1
            if displacement2.dot(displacement2) > displacement.dot(displacement):
                displacement = displacement2
            if displacement3.dot(displacement3) > displacement.dot(displacement):
                displacement = displacement3
            wall[wall_id].vertice1 = new_v1
            wall[wall_id].vertice2 = new_v2
            wall[wall_id].vertice3 = new_v3
            wall[wall_id].norm = orientation @ self.local_norm[local_id]
            wall_center = (new_v1 + new_v2 + new_v3) / 3.0
            wall[wall_id].v = velocity + angular_velocity.cross(wall_center - center)
            wall[wall_id].verletDisp += displacement

    def step_rigid(self, start_index, end_index, dt, wall):
        self._reset_contact_resultant()
        self._accumulate_contact_resultant(start_index, end_index, wall)
        self._integrate_rigid_state(float(dt))
        self._update_rigid_wall(start_index, end_index, wall)

    @ti.kernel
    def translate_prescribed(
        self,
        start_index: int,
        end_index: int,
        dt: ti.template(),
        wall: ti.template(),
    ):
        displacement = self.velocity[None] * dt[None]
        self.rotate_center[None] += displacement
        for wall_id in range(start_index, end_index):
            wall[wall_id]._move(displacement)

    @ti.kernel
    def rotate_prescribed(
        self,
        start_index: int,
        end_index: int,
        dt: ti.template(),
        wall: ti.template(),
    ):
        angular_velocity = self.angular_velocity[None]
        magnitude = angular_velocity.norm()
        if magnitude > 1.0e-14:
            angle = magnitude * dt[None]
            axis = angular_velocity / magnitude
            cosine = ti.cos(angle)
            sine = ti.sin(angle)
            center = self.rotate_center[None]
            for wall_id in range(start_index, end_index):
                old_v1 = wall[wall_id].vertice1
                old_v2 = wall[wall_id].vertice2
                old_v3 = wall[wall_id].vertice3
                relative1 = old_v1 - center
                relative2 = old_v2 - center
                relative3 = old_v3 - center
                normal = wall[wall_id].norm
                new_relative1 = (
                    relative1 * cosine + axis.cross(relative1) * sine + axis * axis.dot(relative1) * (1.0 - cosine)
                )
                new_relative2 = (
                    relative2 * cosine + axis.cross(relative2) * sine + axis * axis.dot(relative2) * (1.0 - cosine)
                )
                new_relative3 = (
                    relative3 * cosine + axis.cross(relative3) * sine + axis * axis.dot(relative3) * (1.0 - cosine)
                )
                new_normal = normal * cosine + axis.cross(normal) * sine + axis * axis.dot(normal) * (1.0 - cosine)
                displacement1 = new_relative1 - relative1
                displacement2 = new_relative2 - relative2
                displacement3 = new_relative3 - relative3
                displacement = displacement1
                if displacement2.dot(displacement2) > displacement.dot(displacement):
                    displacement = displacement2
                if displacement3.dot(displacement3) > displacement.dot(displacement):
                    displacement = displacement3
                wall[wall_id].vertice1 = center + new_relative1
                wall[wall_id].vertice2 = center + new_relative2
                wall[wall_id].vertice3 = center + new_relative3
                wall[wall_id].norm = new_normal
                wall[wall_id].v = angular_velocity.cross(wall[wall_id]._get_center() - center)
                wall[wall_id].verletDisp += displacement

    @ti.kernel
    def move_center(self, displacement: ti.types.vector(3, float)):
        self.mass_center[None] += displacement
        self.rotate_center[None] += displacement
