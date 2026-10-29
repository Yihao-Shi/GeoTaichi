"""Host adapters for testing production ABD contact laws."""

import numpy as np
import taichi as ti
from src.physics_model.contact_model.ipc.ContactDistance import (
    point_triangle_distance_grad,
    point_triangle_distance_grad_hess,
    edge_edge_distance_grad,
    edge_edge_distance_grad_hess,
)
from src.physics_model.contact_model.ipc.ContactMollifier import (
    edge_edge_mollifier,
    edge_edge_mollifier_grad,
    edge_edge_mollifier_threshold,
)
from src.physics_model.contact_model.ipc.IPC import ipc_fully_implicit_point_plane_stribeck_friction


def surface_direction_inf_norm(self, direction):
    """Maximum correction of the actual collision surface vertices."""
    controls = np.asarray(direction, dtype=np.float64).reshape((self.body_num, 4, 3))
    vertex_directions = np.sum(
        self.basis_np[:, :, None] * controls[self.node2body_np],
        axis=1,
    )
    if vertex_directions.size == 0:
        return 0.0
    return float(np.max(np.abs(vertex_directions)))


@ti.data_oriented
class AffineContactProbe:
    def __init__(self, operator):
        self.operator = operator

    @ti.kernel
    def _evaluate_fully_implicit_pt_local_kernel(
        self,
        positions_np: ti.types.ndarray(),
        hats_np: ti.types.ndarray(),
        dhat: float,
        kappa: float,
        pair_mu: float,
        area: float,
        residual_out: ti.types.ndarray(),
        jacobian_out: ti.types.ndarray(),
    ):
        positions = ti.Vector.zero(float, 12)
        hats = ti.Vector.zero(float, 12)
        for i in range(12):
            positions[i] = positions_np[i]
            hats[i] = hats_np[i]
        p = ti.Vector([positions[0], positions[1], positions[2]])
        a = ti.Vector([positions[3], positions[4], positions[5]])
        b = ti.Vector([positions[6], positions[7], positions[8]])
        c = ti.Vector([positions[9], positions[10], positions[11]])
        dist2, grad_d, hess_d, unused_type = point_triangle_distance_grad_hess(p, a, b, c)
        local_residual = ti.Vector.zero(float, 12)
        local_jacobian = ti.Matrix.zero(float, 12, 12)
        if dist2 > 0.0 and dist2 < dhat * dhat:
            unused_energy, db, ddb = self.operator._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
            mollifier_gradient = ti.Vector.zero(float, 12)
            local_residual, local_jacobian = self.operator._fully_implicit_local_friction_residual_jacobian(
                positions,
                hats,
                dist2,
                grad_d,
                hess_d,
                db,
                ddb,
                pair_mu,
                area,
                1.0,
                mollifier_gradient,
                False,
            )
        for i in range(12):
            residual_out[i] = local_residual[i]
            for j in range(12):
                jacobian_out[i, j] = local_jacobian[i, j]

    @ti.kernel
    def _evaluate_fully_implicit_ee_local_kernel(
        self,
        positions_np: ti.types.ndarray(),
        hats_np: ti.types.ndarray(),
        rest_np: ti.types.ndarray(),
        dhat: float,
        kappa: float,
        pair_mu: float,
        area: float,
        residual_out: ti.types.ndarray(),
        jacobian_out: ti.types.ndarray(),
    ):
        positions = ti.Vector.zero(float, 12)
        hats = ti.Vector.zero(float, 12)
        rest = ti.Vector.zero(float, 12)
        for i in range(12):
            positions[i] = positions_np[i]
            hats[i] = hats_np[i]
            rest[i] = rest_np[i]
        p0 = ti.Vector([positions[0], positions[1], positions[2]])
        p1 = ti.Vector([positions[3], positions[4], positions[5]])
        q0 = ti.Vector([positions[6], positions[7], positions[8]])
        q1 = ti.Vector([positions[9], positions[10], positions[11]])
        rp0 = ti.Vector([rest[0], rest[1], rest[2]])
        rp1 = ti.Vector([rest[3], rest[4], rest[5]])
        rq0 = ti.Vector([rest[6], rest[7], rest[8]])
        rq1 = ti.Vector([rest[9], rest[10], rest[11]])
        dist2, grad_d, hess_d, unused_type = edge_edge_distance_grad_hess(p0, p1, q0, q1)
        local_residual = ti.Vector.zero(float, 12)
        local_jacobian = ti.Matrix.zero(float, 12, 12)
        if dist2 > 0.0 and dist2 < dhat * dhat:
            unused_energy, db, ddb = self.operator._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
            eps_x = edge_edge_mollifier_threshold(rp0, rp1, rq0, rq1)
            mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
            mollifier_gradient = edge_edge_mollifier_grad(p0, p1, q0, q1, eps_x)
            local_residual, local_jacobian = self.operator._fully_implicit_local_friction_residual_jacobian(
                positions,
                hats,
                dist2,
                grad_d,
                hess_d,
                db,
                ddb,
                pair_mu,
                area,
                mollifier,
                mollifier_gradient,
                True,
            )
        for i in range(12):
            residual_out[i] = local_residual[i]
            for j in range(12):
                jacobian_out[i, j] = local_jacobian[i, j]

    @ti.kernel
    def _evaluate_fully_implicit_pt_local_residual_kernel(
        self,
        positions_np: ti.types.ndarray(),
        hats_np: ti.types.ndarray(),
        dhat: float,
        kappa: float,
        pair_mu: float,
        area: float,
        residual_out: ti.types.ndarray(),
    ):
        positions = ti.Vector.zero(float, 12)
        hats = ti.Vector.zero(float, 12)
        for i in range(12):
            positions[i] = positions_np[i]
            hats[i] = hats_np[i]
        p = ti.Vector([positions[0], positions[1], positions[2]])
        a = ti.Vector([positions[3], positions[4], positions[5]])
        b = ti.Vector([positions[6], positions[7], positions[8]])
        c = ti.Vector([positions[9], positions[10], positions[11]])
        dist2, grad_d, unused_type = point_triangle_distance_grad(p, a, b, c)
        local_residual = ti.Vector.zero(float, 12)
        if dist2 > 0.0 and dist2 < dhat * dhat:
            unused_energy, db, unused_ddb = self.operator._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
            local_residual = self.operator._fully_implicit_local_friction_residual(
                positions,
                hats,
                dist2,
                grad_d,
                db,
                pair_mu,
                area,
                1.0,
                False,
            )
        for i in range(12):
            residual_out[i] = local_residual[i]

    @ti.kernel
    def _evaluate_fully_implicit_ee_local_residual_kernel(
        self,
        positions_np: ti.types.ndarray(),
        hats_np: ti.types.ndarray(),
        rest_np: ti.types.ndarray(),
        dhat: float,
        kappa: float,
        pair_mu: float,
        area: float,
        residual_out: ti.types.ndarray(),
    ):
        positions = ti.Vector.zero(float, 12)
        hats = ti.Vector.zero(float, 12)
        rest = ti.Vector.zero(float, 12)
        for i in range(12):
            positions[i] = positions_np[i]
            hats[i] = hats_np[i]
            rest[i] = rest_np[i]
        p0 = ti.Vector([positions[0], positions[1], positions[2]])
        p1 = ti.Vector([positions[3], positions[4], positions[5]])
        q0 = ti.Vector([positions[6], positions[7], positions[8]])
        q1 = ti.Vector([positions[9], positions[10], positions[11]])
        rp0 = ti.Vector([rest[0], rest[1], rest[2]])
        rp1 = ti.Vector([rest[3], rest[4], rest[5]])
        rq0 = ti.Vector([rest[6], rest[7], rest[8]])
        rq1 = ti.Vector([rest[9], rest[10], rest[11]])
        dist2, grad_d, unused_type = edge_edge_distance_grad(p0, p1, q0, q1)
        local_residual = ti.Vector.zero(float, 12)
        if dist2 > 0.0 and dist2 < dhat * dhat:
            unused_energy, db, unused_ddb = self.operator._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
            eps_x = edge_edge_mollifier_threshold(rp0, rp1, rq0, rq1)
            mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
            local_residual = self.operator._fully_implicit_local_friction_residual(
                positions,
                hats,
                dist2,
                grad_d,
                db,
                pair_mu,
                area,
                mollifier,
                True,
            )
        for i in range(12):
            residual_out[i] = local_residual[i]

    @ti.kernel
    def _evaluate_fully_implicit_plane_local_kernel(
        self,
        position_np: ti.types.ndarray(),
        hat_position_np: ti.types.ndarray(),
        plane_point_np: ti.types.ndarray(),
        normal_np: ti.types.ndarray(),
        dhat: float,
        kappa: float,
        pair_mu: float,
        area: float,
        residual_out: ti.types.ndarray(),
        jacobian_out: ti.types.ndarray(),
    ):
        for i, j in ti.ndrange(3, 3):
            jacobian_out[i, j] = 0.0
        for i in range(3):
            residual_out[i] = 0.0
        position = ti.Vector([position_np[0], position_np[1], position_np[2]])
        hat_position = ti.Vector([hat_position_np[0], hat_position_np[1], hat_position_np[2]])
        plane_point = ti.Vector([plane_point_np[0], plane_point_np[1], plane_point_np[2]])
        normal = ti.Vector([normal_np[0], normal_np[1], normal_np[2]])
        gap = (position - plane_point).dot(normal)
        if gap > 0.0 and gap < dhat:
            unused_energy, dphi, ddphi = self.operator._ipc_barrier_gap(gap, dhat, kappa)
            normal_force = area * ti.max(-dphi, 0.0)
            normal_force_gradient = -area * ddphi * normal
            mu_dynamic, mu_static = self.operator._resolved_fully_implicit_friction(pair_mu)
            force, jacobian = ipc_fully_implicit_point_plane_stribeck_friction(
                (position - hat_position) * self.operator.endpoint_velocity_factor(),
                normal,
                normal_force,
                normal_force_gradient,
                mu_dynamic,
                mu_static,
                self.operator.fully_mu_viscous,
                self.operator.fully_stribeck_velocity,
                self.operator.epsv,
                self.operator.fully_profile_id,
                self.operator.endpoint_velocity_factor(),
            )
            scale = self.operator.scale_device[None] * self.operator.friction_scale[0]
            for row in ti.static(range(3)):
                residual_out[row] = scale * force[row]
                for column in ti.static(range(3)):
                    jacobian_out[row, column] = scale * jacobian[row, column]

    def evaluate_fully_implicit_plane_local(
        self,
        position,
        hat_position,
        plane_point,
        normal,
        dhat,
        kappa,
        pair_mu,
        area=1.0,
    ):
        """Evaluate the exact production plane-wall residual/Jacobian."""
        position = np.ascontiguousarray(np.asarray(position, dtype=np.float64).reshape(3))
        hat_position = np.ascontiguousarray(np.asarray(hat_position, dtype=np.float64).reshape(3))
        plane_point = np.ascontiguousarray(np.asarray(plane_point, dtype=np.float64).reshape(3))
        normal = np.ascontiguousarray(np.asarray(normal, dtype=np.float64).reshape(3))
        if not np.all(np.isfinite(normal)) or not np.isclose(np.linalg.norm(normal), 1.0, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("plane normal must be finite and unit length")
        residual = np.zeros(3, dtype=np.float64)
        jacobian = np.zeros((3, 3), dtype=np.float64)
        self._evaluate_fully_implicit_plane_local_kernel(
            position,
            hat_position,
            plane_point,
            normal,
            float(dhat),
            float(kappa),
            float(pair_mu),
            float(area),
            residual,
            jacobian,
        )
        return residual, jacobian

    def evaluate_fully_implicit_contact_local(
        self,
        kind,
        positions,
        hat_positions,
        dhat,
        kappa,
        pair_mu,
        area=1.0,
        rest_positions=None,
        need_matrix=True,
    ):
        """Evaluate the exact production PT/EE residual, optionally with J."""
        positions = np.ascontiguousarray(np.asarray(positions, dtype=np.float64).reshape(12))
        hats = np.ascontiguousarray(np.asarray(hat_positions, dtype=np.float64).reshape(12))
        residual = np.zeros(12, dtype=np.float64)
        jacobian = np.zeros((12, 12), dtype=np.float64) if need_matrix else None
        key = str(kind).strip().replace("-", "_").lower()
        if key in ("pt", "point_triangle", "pointtriangle"):
            if need_matrix:
                self._evaluate_fully_implicit_pt_local_kernel(
                    positions,
                    hats,
                    float(dhat),
                    float(kappa),
                    float(pair_mu),
                    float(area),
                    residual,
                    jacobian,
                )
            else:
                self._evaluate_fully_implicit_pt_local_residual_kernel(
                    positions,
                    hats,
                    float(dhat),
                    float(kappa),
                    float(pair_mu),
                    float(area),
                    residual,
                )
        elif key in ("ee", "edge_edge", "edgeedge"):
            if rest_positions is None:
                raise ValueError("rest_positions are required for mollified EE friction")
            rest = np.ascontiguousarray(np.asarray(rest_positions, dtype=np.float64).reshape(12))
            if need_matrix:
                self._evaluate_fully_implicit_ee_local_kernel(
                    positions,
                    hats,
                    rest,
                    float(dhat),
                    float(kappa),
                    float(pair_mu),
                    float(area),
                    residual,
                    jacobian,
                )
            else:
                self._evaluate_fully_implicit_ee_local_residual_kernel(
                    positions,
                    hats,
                    rest,
                    float(dhat),
                    float(kappa),
                    float(pair_mu),
                    float(area),
                    residual,
                )
        else:
            raise ValueError("kind must be 'pt' or 'ee'")
        return residual, jacobian
