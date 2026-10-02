import taichi as ti
import numpy as np

from src.utils.linalg import square_norm
from src.utils.ObjectIO import DictIO


@ti.data_oriented
class _GeometryDeviceState:
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
    def _accumulate_contact_resultant(
        self, start_index: int, end_index: int, wall: ti.template()
    ):
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
                ti.atomic_add(
                    self.total_torque[None][component], torque[component]
                )

    @ti.kernel
    def _integrate_rigid_state(self, dt: float):
        mass = self.mass[None]
        if mass > 0.0:
            velocity = self.velocity[None] + dt * self.total_force[None] / mass
            center = self.mass_center[None] + dt * velocity
            orientation = self.orientation[None]
            inverse_inertia_world = (
                orientation
                @ self.inertia_body_inv[None]
                @ orientation.transpose()
            )
            angular_velocity = self.angular_velocity[None] + dt * (
                inverse_inertia_world @ self.total_torque[None]
            )

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
                rotation = (
                    identity
                    + ti.sin(angle) * skew
                    + (1.0 - ti.cos(angle)) * (skew @ skew)
                )
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
    def _update_rigid_wall(
        self, start_index: int, end_index: int, wall: ti.template()
    ):
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
            wall[wall_id].v = velocity + angular_velocity.cross(
                wall_center - center
            )
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
                    relative1 * cosine
                    + axis.cross(relative1) * sine
                    + axis * axis.dot(relative1) * (1.0 - cosine)
                )
                new_relative2 = (
                    relative2 * cosine
                    + axis.cross(relative2) * sine
                    + axis * axis.dot(relative2) * (1.0 - cosine)
                )
                new_relative3 = (
                    relative3 * cosine
                    + axis.cross(relative3) * sine
                    + axis * axis.dot(relative3) * (1.0 - cosine)
                )
                new_normal = (
                    normal * cosine
                    + axis.cross(normal) * sine
                    + axis * axis.dot(normal) * (1.0 - cosine)
                )
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
                wall[wall_id].v = angular_velocity.cross(
                    wall[wall_id]._get_center() - center
                )
                wall[wall_id].verletDisp += displacement

    @ti.kernel
    def move_center(self, displacement: ti.types.vector(3, float)):
        self.mass_center[None] += displacement
        self.rotate_center[None] += displacement


@ti.data_oriented
class Geometry:
    def __init__(self):
        self.num = 0

        self.start_index = []
        self.end_index = []

        # kinematic setting
        self.rotate_center = []
        self.velocity = []
        self.angular_velocity = []
        self.translate = []
        self.rotate = []

        # rigid-body setting
        self.rigid = []
        self.density = []
        self.mass = []

        # body-frame inertia tensor
        self.inertia_body = []
        self.inertia_body_inv = []

        # world-frame state
        self.mass_center = []
        self.orientation = []

        # local geometry relative to mass center
        self.local_vertice1 = []
        self.local_vertice2 = []
        self.local_vertice3 = []
        self.local_norm = []

        # external loads
        self.external_force = []
        self.external_torque = []
        self.applied_point = []
        self.device_state = []

    # ----------------------------------------------------------------------
    # public API
    # ----------------------------------------------------------------------

    def append(self, start_index, end_index, wall_dict, wall):
        self.start_index.append(start_index)
        self.end_index.append(end_index)

        self.settings(self.num, wall_dict, wall)
        self.num += 1

    def modify(self, geometryID, wall_dict=None, wall=None, **kwargs):
        """Modify geometry configuration without creating a host backend.

        A complete dictionary keeps the historical reinitialization API.
        The keyword form used by ``SceneManager`` updates only kinematics and
        writes the new values directly into the persistent Taichi state.
        """
        if wall_dict is not None:
            self.settings(geometryID, wall_dict, wall)
            return
        if wall is None:
            raise ValueError("Geometry.modify requires the patch wall field")
        state = self.device_state[geometryID]
        if "velocity" in kwargs:
            velocity = np.asarray(kwargs["velocity"], dtype=float)
            self.velocity[geometryID] = velocity
            self.translate[geometryID] = int(square_norm(velocity) > 0.0)
            state.velocity[None] = velocity
        if "rotate_center" in kwargs:
            rotate_center = np.asarray(kwargs["rotate_center"], dtype=float)
            self.rotate_center[geometryID] = rotate_center
            state.rotate_center[None] = rotate_center
        if "angular_velocity" in kwargs:
            angular_velocity = np.asarray(
                kwargs["angular_velocity"], dtype=float
            )
            self.angular_velocity[geometryID] = angular_velocity
            self.rotate[geometryID] = int(
                square_norm(angular_velocity) > 0.0
            )
            state.angular_velocity[None] = angular_velocity
        if not self.rigid[geometryID]:
            start_index = self.start_index[geometryID]
            end_index = self.end_index[geometryID]
            self.reset(start_index, end_index, wall)
            if self.translate[geometryID]:
                self.translate_initialize(
                    start_index,
                    end_index,
                    state.velocity[None],
                    wall,
                )
            if self.rotate[geometryID]:
                self.rotate_initialize(
                    start_index,
                    end_index,
                    state.rotate_center[None],
                    state.angular_velocity[None],
                    wall,
                )

    def reload(self, geometry_dict: dict, wall):
        for geometryID, geometry_info in geometry_dict.items():
            self.append(
                geometry_info["start_index"],
                geometry_info["end_index"],
                geometry_info,
                wall
            )

    def write(self):
        for geometry_id in range(self.num):
            self._sync_device_state(geometry_id)
        return {
            "geometry_num": self.num,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "rotate_center": self.rotate_center,
            "velocity": self.velocity,
            "angular_velocity": self.angular_velocity,
            "translate": self.translate,
            "rotate": self.rotate,
            "rigid": self.rigid,
            "density": self.density,
            "mass": self.mass,
            "inertia_body": self.inertia_body,
            "mass_center": self.mass_center,
            "orientation": self.orientation,
            "external_force": self.external_force,
            "external_torque": self.external_torque,
            "applied_point": self.applied_point,
        }

    def print(self, geometryID=None):
        if geometryID is None:
            geometryID = self.num - 1
        self._sync_device_state(geometryID)

        print(f"Geometry {geometryID}:")
        print(f"  Start Index: {self.start_index[geometryID]}")
        print(f"  End Index: {self.end_index[geometryID]}")
        print(f"  Rotate Center: {self.rotate_center[geometryID]}")
        print(f"  Velocity: {self.velocity[geometryID]}")
        print(f"  Angular Velocity: {self.angular_velocity[geometryID]}")
        print(f"  Translate: {self.translate[geometryID]}")
        print(f"  Rotate: {self.rotate[geometryID]}")
        print(f"  Rigid: {self.rigid[geometryID]}")
        print(f"  Density: {self.density[geometryID]}")
        print(f"  Mass: {self.mass[geometryID]}")
        print(f"  Inertia Body: {self.inertia_body[geometryID]}")
        print(f"  Mass Center: {self.mass_center[geometryID]}")
        print(f"  Orientation: {self.orientation[geometryID]}")
        print(f"  External Force: {self.external_force[geometryID]}")
        print(f"  External Torque: {self.external_torque[geometryID]}")
        print(f"  Applied Point: {self.applied_point[geometryID]}")

    def move(self, geometryID, disp, wall):
        start_index = self.start_index[geometryID]
        end_index = self.end_index[geometryID]

        disp = np.asarray(disp, dtype=float)

        self.kernel_move_patch_wall(start_index, end_index, disp, wall)

        self.device_state[geometryID].move_center(disp)
        self.mass_center[geometryID] = (
            np.asarray(self.mass_center[geometryID], dtype=float) + disp
        )
        self.rotate_center[geometryID] = (
            np.asarray(self.rotate_center[geometryID], dtype=float) + disp
        )

    def _sync_device_state(self, geometry_id):
        """Explicit recorder/restart boundary for one geometry."""
        if geometry_id < 0 or geometry_id >= len(self.device_state):
            return
        state = self.device_state[geometry_id]
        self.mass[geometry_id] = float(state.mass[None])
        self.mass_center[geometry_id] = np.asarray(
            state.mass_center[None], dtype=float
        )
        self.rotate_center[geometry_id] = np.asarray(
            state.rotate_center[None], dtype=float
        )
        self.orientation[geometry_id] = np.asarray(
            state.orientation[None], dtype=float
        )
        self.velocity[geometry_id] = np.asarray(
            state.velocity[None], dtype=float
        )
        self.angular_velocity[geometry_id] = np.asarray(
            state.angular_velocity[None], dtype=float
        )

    def delete(self, geometryID):
        raise NotImplementedError("Geometry.delete() is not implemented yet.")

    # ----------------------------------------------------------------------
    # setting and initialization
    # ----------------------------------------------------------------------

    def settings(self, index, wall_dict, wall):
        rotate_center = DictIO.GetAlternative(
            wall_dict,
            "RotateCenter",
            [0.0, 0.0, 0.0]
        )
        velocity = DictIO.GetAlternative(
            wall_dict,
            "Velocity",
            [0.0, 0.0, 0.0]
        )
        angular_velocity = DictIO.GetAlternative(
            wall_dict,
            "AngularVelocity",
            [0.0, 0.0, 0.0]
        )

        rotate_center = np.asarray(rotate_center, dtype=float)
        velocity = np.asarray(velocity, dtype=float)
        angular_velocity = np.asarray(angular_velocity, dtype=float)

        is_translate = 1 if square_norm(velocity) > 0.0 else 0
        is_rotate = 1 if square_norm(angular_velocity) > 0.0 else 0

        density = DictIO.GetAlternative(wall_dict, "Density", 0.0)
        density = float(density)
        is_rigid = 1 if density > 0.0 else 0

        start_index = self.start_index[index]
        end_index = self.end_index[index]

        volume, inertia_body, mass_center, local_data = self.rigid_init(
            start_index,
            end_index,
            density,
            wall
        )

        mass = density * volume

        if mass <= 0.0:
            is_rigid = 0

        if is_rigid:
            if np.linalg.det(inertia_body) == 0.0:
                inertia_body_inv = np.zeros((3, 3), dtype=float)
            else:
                inertia_body_inv = np.linalg.inv(inertia_body)
        else:
            inertia_body_inv = np.zeros((3, 3), dtype=float)

        external_force = DictIO.GetAlternative(
            wall_dict,
            "ExternalForce",
            [0.0, 0.0, 0.0]
        )
        external_torque = DictIO.GetAlternative(
            wall_dict,
            "ExternalTorque",
            [0.0, 0.0, 0.0]
        )
        applied_point = DictIO.GetAlternative(
            wall_dict,
            "AppliedPoint",
            mass_center
        )

        external_force = np.asarray(external_force, dtype=float)
        external_torque = np.asarray(external_torque, dtype=float)
        applied_point = np.asarray(applied_point, dtype=float)

        # convert off-center external force into torque about mass center
        external_torque = self.torque_init(
            external_force,
            applied_point,
            mass_center,
            external_torque
        )

        # for free rigid body, rotation center should usually be center of mass
        if is_rigid:
            rotate_center = np.asarray(mass_center, dtype=float)

        if index == self.num:
            self.rotate_center.append(rotate_center)
            self.velocity.append(velocity)
            self.angular_velocity.append(angular_velocity)
            self.translate.append(is_translate)
            self.rotate.append(is_rotate)

            self.rigid.append(is_rigid)
            self.density.append(density)
            self.mass.append(mass)
            self.inertia_body.append(inertia_body)
            self.inertia_body_inv.append(inertia_body_inv)
            self.mass_center.append(mass_center)
            self.orientation.append(np.eye(3, dtype=float))

            self.local_vertice1.append(local_data["v1"])
            self.local_vertice2.append(local_data["v2"])
            self.local_vertice3.append(local_data["v3"])
            self.local_norm.append(local_data["norm"])

            self.external_force.append(external_force)
            self.external_torque.append(external_torque)
            self.applied_point.append(applied_point)

        else:
            self.rotate_center[index] = rotate_center
            self.velocity[index] = velocity
            self.angular_velocity[index] = angular_velocity
            self.translate[index] = is_translate
            self.rotate[index] = is_rotate

            self.rigid[index] = is_rigid
            self.density[index] = density
            self.mass[index] = mass
            self.inertia_body[index] = inertia_body
            self.inertia_body_inv[index] = inertia_body_inv
            self.mass_center[index] = mass_center
            self.orientation[index] = np.eye(3, dtype=float)

            self.local_vertice1[index] = local_data["v1"]
            self.local_vertice2[index] = local_data["v2"]
            self.local_vertice3[index] = local_data["v3"]
            self.local_norm[index] = local_data["norm"]

            self.external_force[index] = external_force
            self.external_torque[index] = external_torque
            self.applied_point[index] = applied_point

            self.reset(start_index, end_index, wall)

        state = _GeometryDeviceState(
            mass=mass,
            mass_center=mass_center,
            orientation=np.eye(3, dtype=float),
            velocity=velocity,
            angular_velocity=angular_velocity,
            inertia_body=inertia_body,
            inertia_body_inv=inertia_body_inv,
            rotate_center=rotate_center,
            external_force=external_force,
            external_torque=external_torque,
            local_v1=local_data["v1"],
            local_v2=local_data["v2"],
            local_v3=local_data["v3"],
            local_norm=local_data["norm"],
        )
        if index == self.num:
            self.device_state.append(state)
        else:
            self.device_state[index] = state

        # initialize wall velocity for non-rigid prescribed motion
        if not is_rigid:
            if self.translate[index]:
                self.translate_initialize(start_index, end_index, velocity, wall)

            if self.rotate[index]:
                self.rotate_initialize(
                    start_index,
                    end_index,
                    rotate_center,
                    angular_velocity,
                    wall
                )

    def rigid_init(self, start_index, end_index, density, wall):
        vertices1 = np.ascontiguousarray(
            wall.vertice1.to_numpy()[start_index:end_index],
            dtype=float
        )
        vertices2 = np.ascontiguousarray(
            wall.vertice2.to_numpy()[start_index:end_index],
            dtype=float
        )
        vertices3 = np.ascontiguousarray(
            wall.vertice3.to_numpy()[start_index:end_index],
            dtype=float
        )
        shell_thickness = np.ascontiguousarray(
            wall.offset.to_numpy()[start_index:end_index],
            dtype=float
        )

        n = vertices1.shape[0]

        if n == 0:
            local_data = {
                "v1": np.zeros((0, 3), dtype=float),
                "v2": np.zeros((0, 3), dtype=float),
                "v3": np.zeros((0, 3), dtype=float),
                "norm": np.zeros((0, 3), dtype=float),
            }
            return (
                0.0,
                np.zeros((3, 3), dtype=float),
                np.zeros(3, dtype=float),
                local_data
            )

        edge1 = vertices2 - vertices1
        edge2 = vertices3 - vertices1
        cross = np.cross(edge1, edge2)
        area = 0.5 * np.linalg.norm(cross, axis=1)

        tri_center = (vertices1 + vertices2 + vertices3) / 3.0

        shell_thickness = np.maximum(shell_thickness, 0.0)
        volume_weight = area * shell_thickness
        total_volume = np.sum(volume_weight)

        total_area = np.sum(area)

        if total_volume > 0.0:
            mass_center = (
                np.sum(volume_weight[:, None] * tri_center, axis=0)
                / total_volume
            )
        elif total_area > 0.0:
            mass_center = (
                np.sum(area[:, None] * tri_center, axis=0)
                / total_area
            )
        else:
            mass_center = np.mean(tri_center, axis=0)

        local_v1 = vertices1 - mass_center
        local_v2 = vertices2 - mass_center
        local_v3 = vertices3 - mass_center

        # normals
        norm_len = np.linalg.norm(cross, axis=1)
        normals = np.zeros_like(cross)
        valid = norm_len > 1e-14
        normals[valid] = cross[valid] / norm_len[valid, None]

        local_data = {
            "v1": local_v1,
            "v2": local_v2,
            "v3": local_v3,
            "norm": normals,
        }

        if total_volume <= 0.0 or density <= 0.0:
            return (
                total_volume,
                np.zeros((3, 3), dtype=float),
                mass_center,
                local_data
            )

        # second-order triangle quadrature
        barycentric = np.array(
            [
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ],
            dtype=float
        )

        tri_vertices = np.stack((vertices1, vertices2, vertices3), axis=1)
        sample_points = np.einsum("ab,nbc->nac", barycentric, tri_vertices)

        rel = sample_points - mass_center

        rel_sq = np.einsum("nij,nij->ni", rel, rel)
        outer = np.einsum("nij,nik->nijk", rel, rel)

        eye = np.eye(3, dtype=float)

        # dV = thickness * dA
        # dm = density * dV
        # 3 quadrature points per triangle
        weights = density * volume_weight / 3.0

        weighted_tensor = weights[:, None, None, None] * (
            rel_sq[:, :, None, None] * eye[None, None, :, :] - outer
        )

        inertia_body = np.sum(weighted_tensor, axis=(0, 1))
        inertia_body = 0.5 * (inertia_body + inertia_body.T)

        return total_volume, inertia_body, mass_center, local_data

    def torque_init(
        self,
        external_force,
        applied_point,
        mass_center,
        external_torque
    ):
        external_force = np.asarray(external_force, dtype=float)
        applied_point = np.asarray(applied_point, dtype=float)
        mass_center = np.asarray(mass_center, dtype=float)
        external_torque = np.asarray(external_torque, dtype=float)

        torque_from_force = np.cross(
            applied_point - mass_center,
            external_force
        )

        return torque_from_force + external_torque

    # ----------------------------------------------------------------------
    # time stepping
    # ----------------------------------------------------------------------

    def go(self, dt, delta, wall):
        for i in range(self.num):
            start_index = self.start_index[i]
            end_index = self.end_index[i]

            if self.rigid[i]:
                self.device_state[i].step_rigid(
                    start_index, end_index, delta, wall
                )
            else:
                if self.translate[i]:
                    self.device_state[i].translate_prescribed(
                        start_index, end_index, dt, wall
                    )
                if self.rotate[i]:
                    self.device_state[i].rotate_prescribed(
                        start_index, end_index, dt, wall
                    )

    def _step_prescribed_motion(
        self,
        i,
        start_index,
        end_index,
        dt,
        dt_value,
        wall
    ):
        if self.translate[i]:
            self.device_state[i].translate_prescribed(
                start_index, end_index, dt, wall
            )
        if self.rotate[i]:
            self.device_state[i].rotate_prescribed(
                start_index, end_index, dt, wall
            )

    # ----------------------------------------------------------------------
    # Taichi kernels
    # ----------------------------------------------------------------------

    @ti.kernel
    def reset(
        self,
        start_index: int,
        end_index: int,
        wall: ti.template()
    ):
        for nw in range(start_index, end_index):
            wall[nw].v = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def translate_initialize(
        self,
        start_index: int,
        end_index: int,
        velocity: ti.types.vector(3, float),
        wall: ti.template()
    ):
        for nw in range(start_index, end_index):
            wall[nw].v += velocity

    @ti.kernel
    def rotate_initialize(
        self,
        start_index: int,
        end_index: int,
        rotate_center: ti.types.vector(3, float),
        angular_velocity: ti.types.vector(3, float),
        wall: ti.template()
    ):
        for nw in range(start_index, end_index):
            omega_mag = angular_velocity.norm()

            if omega_mag > 1e-14:
                vec = wall[nw]._get_center() - rotate_center
                omega_dir = angular_velocity / omega_mag
                velocity = angular_velocity.cross(
                    vec - vec.dot(omega_dir) * omega_dir
                )
                wall[nw].v += velocity

    @ti.kernel
    def kernel_move_patch_wall(
        self,
        start_index: int,
        end_index: int,
        disp: ti.types.vector(3, float),
        wall: ti.template()
    ):
        for nw in range(start_index, end_index):
            wall[nw]._move(disp)
