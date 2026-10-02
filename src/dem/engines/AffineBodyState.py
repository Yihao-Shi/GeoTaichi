"""Authoritative host state for affine-body dynamics."""

import numpy as np

from src.physics_model.contact_model.ipc.ContactMeasure import lumped_vertex_measures

from .AffineBodyOperator import _asarray3, _euler_to_matrix, _field_scalar


class AffineBodyState(object):
    def __init__(self, bodies, gravity):
        self.bodies = bodies
        self.gravity = np.asarray(gravity, dtype=np.float64).reshape(3)
        self.body_num = len(bodies)
        self.control_num = self.body_num * 4
        self.y = np.asarray([body["y"] for body in bodies], dtype=np.float64)
        self.v_y = np.asarray([body["v_y"] for body in bodies], dtype=np.float64)
        self.y_n1 = self.y.copy()
        self.v_y_n1 = self.v_y.copy()
        self.hat_y = self.y.copy()
        self.tilde_y = self.y.copy()

    @staticmethod
    def from_scene(scene, sims):
        bodies = []
        material_field = scene.material.to_numpy() if scene.material is not None else None
        for body_id, spec in enumerate(scene.affine_bodies):
            template = spec["template"]
            material = scene.material_parameters.get(spec["materialID"], {})
            density = float(material.get("Density", 1.0))
            young_value = spec.get("YoungModulus", None)
            if young_value is None:
                young_value = material.get("YoungModulus", sims.affine_young_modulus)
            young = float(young_value)
            force_damp = spec.get("ForceLocalDamping", None)
            if force_damp is None:
                force_damp = getattr(sims, "affine_force_local_damping", None)
            if force_damp is None:
                force_damp = _field_scalar(material_field, "fdamp", spec["materialID"], 0.0)
            torque_damp = spec.get("TorqueLocalDamping", None)
            if torque_damp is None:
                torque_damp = getattr(sims, "affine_torque_local_damping", None)
            if torque_damp is None:
                torque_damp = _field_scalar(material_field, "tdamp", spec["materialID"], 0.0)
            scale = float(spec["scale"])
            rotation = _euler_to_matrix(spec.get("orientation"))
            body_point = _asarray3(spec["body_point"])
            init_v = _asarray3(spec.get("initial_velocity"))
            init_w = _asarray3(spec.get("initial_angular_velocity"))

            local_vertices = np.asarray(template.vertices, dtype=np.float64) * scale
            material_coord = local_vertices + 0.25
            basis = np.column_stack(
                (
                    1.0 - material_coord[:, 0] - material_coord[:, 1] - material_coord[:, 2],
                    material_coord[:, 0],
                    material_coord[:, 1],
                    material_coord[:, 2],
                )
            )
            origin = body_point - rotation @ np.array([0.25, 0.25, 0.25], dtype=np.float64)
            y = np.vstack(
                (
                    origin,
                    origin + rotation[:, 0],
                    origin + rotation[:, 1],
                    origin + rotation[:, 2],
                )
            )
            v_y = np.zeros((4, 3), dtype=np.float64)
            for i in range(4):
                v_y[i] = init_v + np.cross(init_w, y[i] - body_point)
            volume = float(template.volume * scale**3)
            mass_matrix = AffineBodyState._mass_matrix(local_vertices, template.faces, basis, density, volume)
            bodies.append(
                {
                    "body_id": body_id,
                    "template": template,
                    "contact_representation": getattr(template, "contact_representation", "TriangleMesh"),
                    "levelset": getattr(template, "levelset", None),
                    "scale": scale,
                    "vertices0": local_vertices,
                    "faces": np.asarray(template.faces, dtype=np.int32),
                    "basis": basis,
                    "volume": volume,
                    "mass_matrix": mass_matrix,
                    "young": young,
                    "force_damping": max(float(force_damp), 0.0),
                    "torque_damping": max(float(torque_damp), 0.0),
                    "mu": float(spec.get("Friction", 0.0)),
                    "materialID": int(spec["materialID"]),
                    "groupID": int(spec["groupID"]),
                    "y": y,
                    "v_y": v_y,
                }
            )
        return AffineBodyState(bodies, sims.gravity)

    @staticmethod
    def _mass_matrix(vertices, faces, basis, density, volume):
        node_measure = lumped_vertex_measures(vertices, faces)
        if np.sum(node_measure) <= 1.0e-30:
            node_measure[:] = 1.0 / max(vertices.shape[0], 1)
        else:
            node_measure /= np.sum(node_measure)
        total_mass = max(float(density) * abs(float(volume)), 1.0e-16)
        nodal_mass = total_mass * node_measure
        matrix = basis.T @ (nodal_mass[:, None] * basis)
        matrix += np.eye(4) * (1.0e-12 * total_mass)
        return matrix

    def pack(self, y=None):
        if y is None:
            y = self.y
        return np.asarray(y, dtype=np.float64).reshape(-1)

    def unpack(self, values):
        return np.asarray(values, dtype=np.float64).reshape((self.body_num, 4, 3))

    def world_vertices(self, y=None):
        if y is None:
            y = self.y
        return [body["basis"] @ y[body_id] for body_id, body in enumerate(self.bodies)]

    def surface_mesh(self, y=None):
        vertices = []
        faces = []
        body_ids = []
        group_ids = []
        offset = 0
        for body_id, body_vertices in enumerate(self.world_vertices(y)):
            body = self.bodies[body_id]
            vertices.append(body_vertices)
            faces.append(body["faces"] + offset)
            body_ids.extend([body_id] * body_vertices.shape[0])
            group_ids.extend([body["groupID"]] * body_vertices.shape[0])
            offset += body_vertices.shape[0]
        if not vertices:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32), np.zeros(0), np.zeros(0)
        return (
            np.vstack(vertices),
            np.vstack(faces).astype(np.int32),
            np.asarray(body_ids, dtype=np.int32),
            np.asarray(group_ids, dtype=np.int32),
        )

    def begin_step(self, dt):
        self.tilde_y = self.y + dt * self.v_y
        self.hat_y = self.y.copy()

    def accept_step(self, y_new, dt):
        old_y = self.y.copy()
        old_v = self.v_y.copy()
        self.v_y = (y_new - old_y) / dt
        self.y_n1 = old_y
        self.v_y_n1 = old_v
        self.y = y_new.copy()


__all__ = ["AffineBodyState"]
