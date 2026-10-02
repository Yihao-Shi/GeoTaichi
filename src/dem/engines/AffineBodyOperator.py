"""Affine-body contact assembly and device operators."""

import os
import time

import numpy as np
import taichi as ti

from src.contact_detection.continuous_contact_detection import (
    ccd_mode_parameters,
    edge_edge_accd,
    edge_edge_ccd,
    linear_gap_accd,
    linear_gap_ccd,
    point_triangle_accd,
    point_triangle_ccd,
)
from src.physics_model.contact_model.ipc.ContactDistance import (
    H_EE,
    H_PE3D,
    edge_edge_distance2_from_type,
    edge_edge_distance_grad,
    edge_edge_distance_grad_hess,
    edge_edge_distance_grad_hess_by_type,
    edge_edge_distance_type,
    g_EE,
    g_PE3D,
    line_line_distance2,
    point_line_distance2,
    point_point_distance2,
    point_point_grad_hess,
    point_triangle_distance2_from_type,
    point_triangle_distance_grad,
    point_triangle_distance_grad_hess,
    point_triangle_distance_grad_hess_by_type,
    point_triangle_distance_type,
)
from src.physics_model.contact_model.ipc.ContactMollifier import (
    edge_edge_mollifier,
    edge_edge_mollifier_grad,
    edge_edge_mollifier_grad_hess,
    edge_edge_mollifier_threshold,
)
from src.physics_model.contact_model.ipc.ContactGeometry import (
    closest_point_triangle,
    edge_edge_contact_frame,
    point_triangle_contact_frame,
)
from src.physics_model.contact_model.ipc.ContactMeasure import (
    surface_vertex_edge_measures,
)
from src.physics_model.contact_model.ipc.ContactAssembly import (
    psd_project_nd,
)
from src.physics_model.contact_model.ipc.IPC import (
    ipc_fully_implicit_point_plane_stribeck_friction,
    ipc_fully_implicit_profile_over_speed,
    ipc_fully_implicit_profile_over_speed_derivative,
    ipc_stribeck_falloff,
    ipc_stribeck_falloff_derivative,
    ipc_toolkit_barrier_distance2_terms,
    ipc_toolkit_barrier_distance_terms,
    ipc_friction_f0,
    ipc_friction_f1_over_speed,
    normalize_ipc_model,
    semi_ipc_find,
    semi_ipc_find_or_insert,
    semi_ipc_terms,
    semi_ipc_update_multiplier,
)
from src.dem.neighbor.AffineBodyNeighbor import make_affine_body_neighbor
from src.utils.ScalarFunction import linearize3D

MATRIX_COO = 1
MATRIX_HASH_TRIPLET = 2
MATRIX_CONTACT_DAMPING = 3


def _normalize_affine_assemble_type(assemble_type):
    key = str(assemble_type).replace("_", "").replace("-", "").lower()
    aliases = {
        "matrixfree": "MatrixFree",
        "coo": "COO",
        "csr": "COO",
        "coordinate": "COO",
        "coordinatematrix": "COO",
        "hashtriplet": "HashTriplet",
        "triplet": "HashTriplet",
    }
    if key not in aliases:
        raise RuntimeError("AffineBody assemble_type must be one of ['MatrixFree', 'COO', 'HashTriplet']")
    return aliases[key]


def _normalize_affine_friction_mode(mode, error_type=RuntimeError):
    key = str(mode).strip().replace("-", "_").lower()
    aliases = {
        "lagged": "lagged",
        "lag": "lagged",
        "fully_implicit": "fully_implicit",
        "fullyimplicit": "fully_implicit",
    }
    if key not in aliases:
        raise error_type("AffineBody friction_mode must be 'lagged' or " "'fully_implicit'")
    return aliases[key]


def _coordination_value(value, default=16):
    if isinstance(value, (list, tuple, np.ndarray)):
        return max(int(v) for v in value) if len(value) > 0 else int(default)
    return int(value) if value is not None else int(default)


def _asarray3(value, default=None):
    if value is None:
        value = default if default is not None else [0.0, 0.0, 0.0]
    return np.asarray(value, dtype=np.float64).reshape(3)


def _field_scalar(field_numpy, name, index, default=0.0):
    if field_numpy is None:
        return default
    try:
        return float(field_numpy[name][index])
    except (KeyError, IndexError, TypeError, ValueError):
        return default


def _property_value(properties, names, default):
    for name in names:
        if name in properties:
            return float(properties[name])
    return default


def _euler_to_matrix(orientation):
    if orientation is None or isinstance(orientation, str):
        return np.eye(3)
    theta = np.radians(np.asarray(orientation, dtype=np.float64).reshape(3))
    cx, cy, cz = np.cos(theta)
    sx, sy, sz = np.sin(theta)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rz @ ry @ rx


def _triangle_normal(a, b, c):
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n)
    if norm <= 1.0e-30:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return n / norm


def _read_affine_walls(scene, sims):
    if scene.wall is None or int(scene.wallNum[0]) == 0:
        return []
    raw = scene.wall.to_numpy()
    walls = []
    count = int(scene.wallNum[0])
    if sims.wall_type == 0:
        for i in range(count):
            if raw["active"][i] == 0:
                continue
            normal = np.asarray(raw["norm"][i], dtype=np.float64)
            norm = np.linalg.norm(normal)
            normal = normal / norm if norm > 0.0 else normal
            walls.append(
                {
                    "type": "plane",
                    "point": np.asarray(raw["point"][i], dtype=np.float64),
                    "normal": normal,
                    "materialID": int(raw["materialID"][i]),
                }
            )
    elif sims.wall_type in (1, 2):
        for i in range(count):
            if raw["active"][i] == 0:
                continue
            walls.append(
                {
                    "type": "facet",
                    "vertices": np.vstack((raw["vertice1"][i], raw["vertice2"][i], raw["vertice3"][i])).astype(
                        np.float64
                    ),
                    "materialID": int(raw["materialID"][i]),
                }
            )
    return walls


@ti.data_oriented
class TaichiAffineBodyOperator(object):
    @property
    def is_semi(self):
        return getattr(self, "_is_semi", False)

    @is_semi.setter
    def is_semi(self, value):
        self._is_semi = bool(value)

    def __init__(self, state, sims, scene, initialization_only=False):
        self.state = state
        self.initialization_only = bool(initialization_only)
        self.body_num = state.body_num
        self.control_num = state.control_num
        self.dof = self.control_num * 3
        self.control_mass_matrix_np = np.zeros((self.dof, self.dof), dtype=np.float64)
        for body_id, body in enumerate(state.bodies):
            body_mass = np.asarray(body["mass_matrix"], dtype=np.float64)
            for component in range(3):
                indices = body_id * 12 + 3 * np.arange(4, dtype=np.int64) + component
                self.control_mass_matrix_np[np.ix_(indices, indices)] = body_mass
        mass_eigenvalue_floors = [
            float(
                np.linalg.eigvalsh(
                    0.5
                    * (
                        np.asarray(body["mass_matrix"], dtype=np.float64)
                        + np.asarray(body["mass_matrix"], dtype=np.float64).T
                    )
                ).min()
            )
            for body in state.bodies
        ]
        self.inertia_eigenvalue_floor = min(mass_eigenvalue_floors)
        if not np.isfinite(self.inertia_eigenvalue_floor) or self.inertia_eigenvalue_floor <= 0.0:
            raise ValueError("AffineBody control-space mass matrix must be positive " "definite")
        self.material_num = max(int(sims.max_material_num), 1)
        self.dt = float(sims.dt[None])
        self.scale = self.dt * self.dt
        self.dt_device = ti.field(float, shape=())
        self.scale_device = ti.field(float, shape=())
        self.dt_device[None] = self.dt
        self.scale_device[None] = self.scale
        self.default_dhat = float(sims.affine_dhat)
        self.default_kappa = float(sims.affine_barrier_stiffness)
        self.contact_model = normalize_ipc_model(getattr(sims, "affine_contact_model", "BarrierIPC"))
        self.is_semi = self.contact_model == "SemiIPC"
        configured_penalty = getattr(sims, "affine_penalty", None)
        self.semi_penalty = self.default_kappa / self.scale if configured_penalty is None else float(configured_penalty)
        self.semi_penalty_scale = self.semi_penalty / self.default_kappa
        self.semi_constraint_tolerance = float(getattr(sims, "affine_constraint_tolerance", 1.0e-6))
        if not np.isfinite(self.semi_constraint_tolerance) or self.semi_constraint_tolerance < 0.0:
            raise ValueError("AffineBody constraint_tolerance must be finite and non-negative")
        self.default_contact_damping_stiffness = float(sims.affine_contact_damping_stiffness)
        self.epsv = float(sims.affine_friction_epsv)
        self.hessian_shift = float(sims.affine_hessian_shift)
        # The generic Hessian shift belongs to the conservative lagged
        # optimization.  Fully implicit friction differentiates a residual,
        # so its production Jacobian is exact by default; users may opt into
        # this separate regularization explicitly.
        self.fully_implicit_jacobian_shift = float(getattr(sims, "affine_fully_implicit_jacobian_shift", 0.0))
        mode = _normalize_affine_friction_mode(
            getattr(sims, "affine_friction_mode", "lagged"),
            error_type=ValueError,
        )
        self.fully_implicit = mode == "fully_implicit"
        if self.is_semi and self.fully_implicit:
            raise ValueError("AffineBody SemiIPC currently requires friction_mode='lagged'")
        self.fully_mu_dynamic = float(getattr(sims, "affine_dynamic_friction", -1.0))
        self.fully_mu_static = float(getattr(sims, "affine_static_friction", -1.0))
        self.fully_mu_viscous = float(getattr(sims, "affine_viscous_friction", 0.0))
        stribeck_velocity = float(getattr(sims, "affine_stribeck_velocity", -1.0))
        if not np.isfinite(stribeck_velocity):
            raise ValueError("AffineBody stribeck_velocity must be finite")
        if stribeck_velocity < 0.0 and stribeck_velocity != -1.0:
            raise ValueError(
                "AffineBody stribeck_velocity must be non-negative or " "-1 for the 10 * friction_epsv fallback"
            )
        self.fully_stribeck_velocity = 10.0 * self.epsv if stribeck_velocity < 0.0 else stribeck_velocity
        profile = str(getattr(sims, "affine_friction_profile", "quadratic")).strip().replace("-", "_").lower()
        profile_aliases = {
            "quadratic": 0,
            "c1": 0,
            "ipc": 0,
            "stabilized": 1,
            "stabilised": 1,
            "cinfinity": 1,
            "c_infinity": 1,
        }
        if profile not in profile_aliases:
            raise ValueError("AffineBody friction_profile must be 'quadratic' or " "'stabilized'")
        self.fully_profile_id = profile_aliases[profile]
        for name, value in (
            ("dt", self.dt),
            ("dhat", self.default_dhat),
            ("barrier_stiffness", self.default_kappa),
            ("friction_epsv", self.epsv),
            ("penalty", self.semi_penalty),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"AffineBody {name} must be finite and positive")
        for name, value in (
            (
                "contact_damping_stiffness",
                self.default_contact_damping_stiffness,
            ),
            ("hessian_shift", self.hessian_shift),
            (
                "fully_implicit_jacobian_shift",
                self.fully_implicit_jacobian_shift,
            ),
            ("viscous_friction", self.fully_mu_viscous),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"AffineBody {name} must be finite and non-negative")
        if not np.isfinite(self.fully_mu_dynamic) or not np.isfinite(self.fully_mu_static):
            raise ValueError("AffineBody static/dynamic friction overrides must be " "finite")
        if (self.fully_mu_dynamic < 0.0 and self.fully_mu_dynamic != -1.0) or (
            self.fully_mu_static < 0.0 and self.fully_mu_static != -1.0
        ):
            raise ValueError(
                "AffineBody static/dynamic friction overrides must be " "non-negative or -1 for the material-pair value"
            )
        if not np.isfinite(self.fully_stribeck_velocity) or self.fully_stribeck_velocity < 0.0:
            raise ValueError("AffineBody stribeck_velocity must be finite and " "non-negative")
        if (
            self.fully_implicit
            and self.fully_stribeck_velocity <= 0.0
            and self.fully_mu_static >= 0.0
            and (self.fully_mu_dynamic < 0.0 or self.fully_mu_static != self.fully_mu_dynamic)
        ):
            raise ValueError(
                "AffineBody stribeck_velocity must be positive when " "static_friction can differ from dynamic_friction"
            )
        if not self.fully_implicit:
            if self.fully_mu_viscous > 0.0:
                raise ValueError(
                    "AffineBody lagged friction does not support "
                    "viscous_friction; use friction_mode='fully_implicit'"
                )
            if self.fully_profile_id != 0:
                raise ValueError("AffineBody lagged friction supports only the quadratic IPC profile")
            if self.fully_mu_static >= 0.0 and (
                self.fully_mu_dynamic < 0.0 or self.fully_mu_static != self.fully_mu_dynamic
            ):
                raise ValueError(
                    "AffineBody lagged friction supports one Coulomb "
                    "coefficient; unequal static/dynamic coefficients "
                    "require friction_mode='fully_implicit'"
                )
        else:
            try:
                requested_iterations_value = float(getattr(sims, "affine_friction_iterations", 1))
                requested_iterations = int(requested_iterations_value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("AffineBody friction_iterations must be an integer") from exc
            if not np.isfinite(requested_iterations_value) or requested_iterations_value != requested_iterations:
                raise ValueError("AffineBody friction_iterations must be an integer")
            if requested_iterations != 1:
                raise ValueError(
                    "AffineBody fully implicit friction updates geometry "
                    "inside Newton and therefore requires "
                    "friction_iterations=1; lagged outer fixed-point "
                    "iterations are not used"
                )

        self._build_surface_storage()
        self._build_joint_storage(scene, sims)
        representations = {str(body.get("contact_representation", "TriangleMesh")) for body in self.state.bodies}
        if len(representations) > 1:
            raise ValueError(
                "AffineBody does not mix TriangleMesh and LevelSet contact " "representations in one solve"
            )
        self.levelset_contact = representations == {"LevelSet"}
        walls = _read_affine_walls(scene, sims)
        self.wall_num = len(walls)
        self._build_contact_property_arrays(scene, walls)
        if self.levelset_contact:
            requested_friction = (
                any(float(body.get("mu", 0.0)) > 0.0 for body in state.bodies)
                or np.any(self.pp_mu_np > 0.0)
                or np.any(self.pw_mu_np > 0.0)
                or self.fully_mu_dynamic > 0.0
                or self.fully_mu_static > 0.0
                or self.fully_mu_viscous > 0.0
            )
            if not self.initialization_only and self.fully_implicit and requested_friction:
                raise ValueError(
                    "LevelSet AffineBody IPC friction currently uses the "
                    "lagged IPC potential; set friction_mode='lagged' or "
                    "set all friction coefficients to zero"
                )
            if not self.initialization_only and self.max_contact_damping_stiffness > 0.0:
                raise ValueError("LevelSet AffineBody IPC does not yet support the " "mesh contact-damping operator")
        self.dhat = self.max_dhat
        self.kappa = self.default_kappa
        self.contact_damping_stiffness = self.max_contact_damping_stiffness
        self.neighbor = make_affine_body_neighbor(sims, self.vertex_num, self.face_num, self.edge_num, self.dhat)
        self.last_candidate_pairs = 0

        self.y = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.tilde_y = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.hat_y = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.velocity_y = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.previous_y = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.previous_velocity_y = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.state_y_vjp = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.state_velocity_vjp = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.accepted_translation_velocity = ti.Vector.field(3, dtype=float, shape=())
        self.translation_correction_vjp = ti.Vector.field(3, dtype=float, shape=())
        initial_y = np.ascontiguousarray(state.y.reshape((self.control_num, 3)), dtype=np.float64)
        initial_velocity = np.ascontiguousarray(state.v_y.reshape((self.control_num, 3)), dtype=np.float64)
        self.y.from_numpy(initial_y)
        self.tilde_y.from_numpy(initial_y)
        self.hat_y.from_numpy(initial_y)
        self.previous_y.from_numpy(initial_y)
        self.velocity_y.from_numpy(initial_velocity)
        self.previous_velocity_y.from_numpy(initial_velocity)
        # Frozen geometry for the lagged level-set friction potential.  It is
        # refreshed once per outer lagging iteration, exactly like the
        # closest-feature coordinates and tangent frames of mesh IPC.
        self.levelset_friction_y = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        # CUDA nonlinear solves keep the complete iterate on the device.  A
        # separate base state lets CCD/Armijo construct every trial from the
        # same accepted Newton iterate without a y.to_numpy()/from_numpy()
        # round trip at each backtracking iteration.
        self.line_search_base_y = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.step_start_y = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.direction_y = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.grad = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        self.x = ti.Vector.field(3, dtype=float, shape=max(self.vertex_num, 1))
        self.rest_x = ti.Vector.field(3, dtype=float, shape=max(self.vertex_num, 1))
        self.hat_x = ti.Vector.field(3, dtype=float, shape=max(self.vertex_num, 1))
        self.dx = ti.Vector.field(3, dtype=float, shape=max(self.vertex_num, 1))
        self.basis = ti.field(dtype=float, shape=(max(self.vertex_num, 1), 4))
        self.node2body = ti.field(dtype=ti.i32, shape=max(self.vertex_num, 1))
        self.body_vertex_start = ti.field(dtype=ti.i32, shape=max(self.body_num, 1))
        self.body_vertex_count = ti.field(dtype=ti.i32, shape=max(self.body_num, 1))
        self.faces = ti.Vector.field(3, dtype=ti.i32, shape=max(self.face_num, 1))
        self.face2body = ti.field(dtype=ti.i32, shape=max(self.face_num, 1))
        self.edges = ti.Vector.field(2, dtype=ti.i32, shape=max(self.edge_num, 1))
        self.edge2body = ti.field(dtype=ti.i32, shape=max(self.edge_num, 1))
        self.node_area = ti.field(dtype=float, shape=max(self.vertex_num, 1))
        self.edge_area = ti.field(dtype=float, shape=max(self.edge_num, 1))
        self.mass = ti.field(dtype=float, shape=(max(self.body_num, 1), 4, 4))
        self.body_mass = ti.field(dtype=float, shape=max(self.body_num, 1))
        self.volume = ti.field(dtype=float, shape=max(self.body_num, 1))
        self.young = ti.field(dtype=float, shape=max(self.body_num, 1))
        self.force_damp = ti.field(dtype=float, shape=max(self.body_num, 1))
        self.torque_damp = ti.field(dtype=float, shape=max(self.body_num, 1))
        self.body_mu = ti.field(dtype=float, shape=max(self.body_num, 1))
        self.body_material = ti.field(dtype=ti.i32, shape=max(self.body_num, 1))
        self.body_contact_type = ti.field(dtype=ti.i32, shape=max(self.body_num, 1))
        self.body_scale = ti.field(dtype=float, shape=max(self.body_num, 1))
        self.external_generalized_force = ti.Vector.field(3, dtype=float, shape=max(self.control_num, 1))
        joint_capacity = max(self.joint_num, 1)
        self.joint_body = ti.Vector.field(2, dtype=ti.i32, shape=joint_capacity)
        self.joint_anchor_weight = ti.Vector.field(4, dtype=float, shape=(joint_capacity, 2))
        self.joint_axis_weight = ti.Vector.field(4, dtype=float, shape=(joint_capacity, 2))
        self.joint_u_weight = ti.Vector.field(4, dtype=float, shape=(joint_capacity, 2))
        self.joint_v_weight = ti.Vector.field(4, dtype=float, shape=(joint_capacity, 2))
        self.joint_world_anchor = ti.Vector.field(3, dtype=float, shape=joint_capacity)
        self.joint_world_axis = ti.Vector.field(3, dtype=float, shape=joint_capacity)
        self.joint_world_u = ti.Vector.field(3, dtype=float, shape=joint_capacity)
        self.joint_world_v = ti.Vector.field(3, dtype=float, shape=joint_capacity)
        self.joint_position_stiffness = ti.field(dtype=float, shape=joint_capacity)
        self.joint_axis_stiffness = ti.field(dtype=float, shape=joint_capacity)
        self.joint_motor_stiffness = ti.field(dtype=float, shape=joint_capacity)
        self.joint_target_angle = ti.field(dtype=float, shape=joint_capacity)
        self.joint_limit_enabled = ti.field(dtype=ti.i32, shape=joint_capacity)
        self.joint_limit_lower = ti.field(dtype=float, shape=joint_capacity)
        self.joint_limit_upper = ti.field(dtype=float, shape=joint_capacity)
        self.joint_limit_stiffness = ti.field(dtype=float, shape=joint_capacity)
        self.joint_damping = ti.field(dtype=float, shape=joint_capacity)
        self.joint_collision_disabled = ti.field(
            dtype=ti.i32,
            shape=(max(self.body_num, 1), max(self.body_num, 1)),
        )
        self.levelset_grid_start = ti.field(dtype=ti.i32, shape=max(self.body_num, 1))
        self.levelset_grid_shape = ti.Vector.field(3, dtype=ti.i32, shape=max(self.body_num, 1))
        self.levelset_grid_origin = ti.Vector.field(3, dtype=float, shape=max(self.body_num, 1))
        self.levelset_grid_spacing = ti.field(dtype=float, shape=max(self.body_num, 1))
        self.levelset_template_to_grid = ti.Matrix.field(3, 3, dtype=float, shape=max(self.body_num, 1))
        self.levelset_template_to_grid_offset = ti.Vector.field(3, dtype=float, shape=max(self.body_num, 1))
        self.levelset_lipschitz = ti.field(dtype=float, shape=max(self.body_num, 1))
        self.levelset_grid_value = ti.field(dtype=float, shape=max(self.levelset_grid_num, 1))
        self.levelset_active_contacts = ti.field(dtype=ti.i32, shape=())
        self.levelset_minimum_gap = ti.field(dtype=float, shape=())
        self.levelset_friction_contacts = ti.field(dtype=ti.i32, shape=())
        self.wall_type = ti.field(dtype=ti.i32, shape=max(self.wall_num, 1))
        self.wall_material = ti.field(dtype=ti.i32, shape=max(self.wall_num, 1))
        self.wall_point = ti.Vector.field(3, dtype=float, shape=max(self.wall_num, 1))
        self.wall_normal = ti.Vector.field(3, dtype=float, shape=max(self.wall_num, 1))
        self.wall_v0 = ti.Vector.field(3, dtype=float, shape=max(self.wall_num, 1))
        self.wall_v1 = ti.Vector.field(3, dtype=float, shape=max(self.wall_num, 1))
        self.wall_v2 = ti.Vector.field(3, dtype=float, shape=max(self.wall_num, 1))
        self.wall_mu = ti.field(dtype=float, shape=max(self.wall_num, 1))
        self.gravity = ti.Vector.field(3, dtype=float, shape=())
        self.pp_dhat = ti.field(dtype=float, shape=(self.material_num, self.material_num))
        self.pp_kappa = ti.field(dtype=float, shape=(self.material_num, self.material_num))
        self.pp_contact_damping = ti.field(dtype=float, shape=(self.material_num, self.material_num))
        self.pp_mu = ti.field(dtype=float, shape=(self.material_num, self.material_num))
        self.pw_dhat = ti.field(dtype=float, shape=(self.material_num, self.material_num))
        self.pw_kappa = ti.field(dtype=float, shape=(self.material_num, self.material_num))
        self.pw_contact_damping = ti.field(dtype=float, shape=(self.material_num, self.material_num))
        self.pw_mu = ti.field(dtype=float, shape=(self.material_num, self.material_num))
        self.energy = ti.field(dtype=float, shape=())
        self.ccd_alpha = ti.field(dtype=float, shape=())
        self.edge_type_count = ti.field(dtype=ti.i32, shape=9)
        self.edge_mollifier_type_count = ti.field(dtype=ti.i32, shape=9)
        self.body_pair_point_type_count = ti.field(dtype=ti.i32, shape=7)
        self.body_pair_edge_type_count = ti.field(dtype=ti.i32, shape=9)
        # ponytail: dense pair slots make candidate dedup and Hessian lookup
        # collision-free; use a device hash if O(body_count^2) storage matters.
        body_pair_capacity = max(self.body_num * self.body_num, 1)
        self.body_pair_segment_base = ti.field(dtype=ti.i32, shape=body_pair_capacity)
        self.active_body_pair = ti.field(dtype=ti.i32, shape=body_pair_capacity)
        self.active_body_pair_list = ti.field(dtype=ti.i32, shape=body_pair_capacity)
        self.active_body_pair_count = ti.field(dtype=ti.i32, shape=())
        levelset_pair_capacity = min(
            self.body_num * max(self.body_num - 1, 0) // 2,
            self.neighbor.candidate_capacity + self.neighbor.edge_candidate_capacity,
        )
        levelset_vertex_tasks = 2 * self.max_body_vertex_count * levelset_pair_capacity if self.levelset_contact else 0
        semi_contacts = (
            self.neighbor.candidate_capacity
            + self.neighbor.edge_candidate_capacity
            + self.vertex_num * self.wall_num
            + levelset_vertex_tasks
        )
        self.semi_capacity = 1 << (max(2 * int(semi_contacts), 1) - 1).bit_length()
        self.semi_state = ti.field(ti.i32, shape=self.semi_capacity)
        self.semi_key = ti.Vector.field(4, ti.i32, shape=self.semi_capacity)
        self.semi_multiplier = ti.field(float, shape=self.semi_capacity)
        self.semi_count = ti.field(ti.i32, shape=())
        self.semi_overflow = ti.field(ti.i32, shape=())
        self.semi_constraint_violation = ti.field(float, shape=())
        self.semi_state.fill(2)
        self.friction_contact_capacity = max(
            self.neighbor.candidate_capacity + self.neighbor.edge_candidate_capacity + self.vertex_num * self.wall_num,
            1,
        )
        self.friction_contact_count = ti.field(dtype=ti.i32, shape=1)
        self.friction_contact_overflow = ti.field(dtype=ti.i32, shape=1)
        self.friction_contact_bodies = ti.Vector.field(4, dtype=ti.i32, shape=self.friction_contact_capacity)
        self.friction_contact_vertices = ti.Vector.field(4, dtype=ti.i32, shape=self.friction_contact_capacity)
        self.friction_contact_weights = ti.Vector.field(4, dtype=float, shape=self.friction_contact_capacity)
        self.friction_contact_normal = ti.Vector.field(3, dtype=float, shape=self.friction_contact_capacity)
        self.friction_contact_hat_rel = ti.Vector.field(3, dtype=float, shape=self.friction_contact_capacity)
        self.friction_contact_coeff = ti.field(dtype=float, shape=self.friction_contact_capacity)
        self.adjoint_friction_contact_count = ti.field(dtype=ti.i32, shape=())
        self.adjoint_friction_contact_bodies = ti.Vector.field(4, dtype=ti.i32, shape=self.friction_contact_capacity)
        self.adjoint_friction_contact_vertices = ti.Vector.field(4, dtype=ti.i32, shape=self.friction_contact_capacity)
        self.adjoint_friction_contact_weights = ti.Vector.field(4, dtype=float, shape=self.friction_contact_capacity)
        self.adjoint_friction_contact_normal = ti.Vector.field(3, dtype=float, shape=self.friction_contact_capacity)
        self.adjoint_friction_contact_hat_rel = ti.Vector.field(3, dtype=float, shape=self.friction_contact_capacity)
        self.adjoint_friction_contact_coeff = ti.field(dtype=float, shape=self.friction_contact_capacity)
        self.friction_scale = ti.field(dtype=float, shape=1)
        self.friction_scale[0] = 1.0
        hash_triplet_capacity = self._estimate_hash_triplet_capacity(sims)
        self.max_hash_triplets = int(os.environ.get("GT_AFFINE_MAX_HASH_TRIPLETS", str(hash_triplet_capacity)))
        coo_capacity = 9 * self._estimate_hash_triplet_capacity(sims, safety_factor=1.0, minimum_capacity=1)
        self.max_coo_entries = int(os.environ.get("GT_AFFINE_MAX_COO_ENTRIES", str(max(coo_capacity, 1))))
        if self.max_coo_entries <= 0:
            raise ValueError("AffineBody COO capacity must be positive")
        self.coo_matrix = None
        self.hash_triplet = None
        self.hash_store_full_symmetric_input = False
        self.K_coo_rows = None
        self.K_coo_cols = None
        self.K_coo_values = None
        self.K_coo_diag = None
        self.coo_entry_count = ti.field(dtype=ti.i32, shape=())
        self.coo_overflow = ti.field(dtype=ti.i32, shape=())
        # The contact-damping operator is frozen for one Newton solve, so it
        # must persist across matrix resets.  Keep only its upper 3x3 block
        # triplets; the old dense dof-by-dof field wasted quadratic memory.
        self.contact_damping_capacity = self._estimate_contact_damping_capacity(sims)
        self.contact_damping_count = ti.field(dtype=ti.i32, shape=())
        self.contact_damping_overflow = ti.field(dtype=ti.i32, shape=())
        self.contact_damping_block_i = ti.field(dtype=ti.i32, shape=self.contact_damping_capacity)
        self.contact_damping_block_j = ti.field(dtype=ti.i32, shape=self.contact_damping_capacity)
        self.contact_damping_block_h = ti.Vector.field(9, dtype=float, shape=self.contact_damping_capacity)
        self.linear_rhs = ti.field(dtype=float, shape=max(self.dof, 1))
        self.linear_x = ti.field(dtype=float, shape=max(self.dof, 1))
        self.linear_M = ti.field(dtype=float, shape=max(self.dof, 1))
        self.gravity_vjp = ti.Vector.field(3, dtype=float, shape=())
        self.young_vjp = ti.field(dtype=float, shape=max(self.body_num, 1))
        self.joint_target_vjp = ti.field(dtype=float, shape=joint_capacity)
        self.joint_damping_vjp = ti.field(dtype=float, shape=joint_capacity)
        self.friction_scale_vjp = ti.field(dtype=float, shape=())

        self._load_constant_fields(walls, scene)
        self.gravity[None] = ti.Vector(list(self.state.gravity))

    def set_timestep(self, timestep):
        self.dt = float(timestep)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("AffineBody timestep must be finite and positive")
        self.scale = self.dt * self.dt
        self.dt_device[None] = self.dt
        self.scale_device[None] = self.scale

    def set_joint_target_angle(self, joint_id, target_angle):
        joint_id = int(joint_id)
        if joint_id < 0 or joint_id >= self.joint_num:
            raise IndexError(f"AffineBody joint ID {joint_id} is out of range")
        target_angle = float(target_angle)
        if not np.isfinite(target_angle):
            raise ValueError("AffineBody joint target angle must be finite")
        self.joint_target_angle[joint_id] = np.deg2rad(target_angle)

    def reset_semi_state(self):
        if not self.is_semi:
            return
        self.semi_state.fill(2)
        self.semi_multiplier.fill(0.0)
        self.semi_count[None] = 0
        self.semi_overflow[None] = 0
        self.semi_constraint_violation[None] = 0.0

    def _estimate_contact_damping_capacity(self, sims):
        """Bound frozen upper 3x3 blocks from configured contact resources."""
        if self.contact_damping_stiffness <= 0.0 or self.levelset_contact:
            return 1
        mesh_candidates = (
            self.neighbor.candidate_capacity + self.neighbor.edge_candidate_capacity if self.body_num > 1 else 0
        )
        active_body_pairs = min(
            self.body_num * max(self.body_num - 1, 0) // 2,
            mesh_candidates,
        )
        wall_coord = _coordination_value(sims.wall_coordination_number, self.wall_num)
        wall_candidates = self.vertex_num * min(max(wall_coord, 0), self.wall_num) if self.wall_num > 0 else 0
        # One projected body pair has 8 affine controls (36 upper blocks);
        # one point-wall stencil has 4 controls (10 upper blocks).
        return max(36 * active_body_pairs + 10 * wall_candidates, 1)

    def _estimate_hash_triplet_capacity(
        self,
        sims,
        *,
        vf_candidate_capacity=None,
        ee_candidate_capacity=None,
        full_symmetric_input=None,
        safety_factor=None,
        minimum_capacity=None,
    ):
        """Estimate raw 3x3 block scatters before reduction.

        Standalone affine solves use the neighbor buffers as their bounds.
        Coupled solvers may provide tighter bounds after accounting for the
        configured body coordination and the number of bodies that actually
        exist; same-body broad-phase candidates never enter the Hessian.
        """
        vf_candidates = max(
            int(self.neighbor.candidate_capacity if vf_candidate_capacity is None else vf_candidate_capacity),
            0,
        )
        ee_candidates = max(
            int(self.neighbor.edge_candidate_capacity if ee_candidate_capacity is None else ee_candidate_capacity),
            0,
        )
        wall_coord = _coordination_value(sims.wall_coordination_number, self.wall_num)
        wall_candidates = self.vertex_num * min(max(wall_coord, 0), self.wall_num) if self.wall_num > 0 else 0

        # Friction is emitted per four-site stencil (16 affine controls).
        # One projected body-pair barrier reserves an 8x8 block segment plus
        # an equally sized in-array Jacobi workspace. A wall point maps to
        # four controls.
        four_site_scatter = int(getattr(sims, "affine_contact_block_capacity", 256))
        if four_site_scatter <= 0:
            raise ValueError("AffineBody contact block capacity must be positive")
        body_pair_scatter = 2 * (2 * 4) ** 2
        one_point_scatter = 4 * 4
        mesh_candidates = vf_candidates + ee_candidates if self.body_num > 1 else 0
        active_body_pair_bound = min(
            self.body_num * max(self.body_num - 1, 0) // 2,
            mesh_candidates,
        )
        mesh_hessian_triplets = active_body_pair_bound * body_pair_scatter + mesh_candidates * four_site_scatter
        wall_hessian_triplets = 2 * one_point_scatter
        base_triplets_per_body = 52
        levelset_hessian_triplets = 2 * (8 * 8)
        base_triplets = self.body_num * base_triplets_per_body
        # Anchor, axis, motor, one active limit and damping each scatter at
        # most one dense 8x8 affine-control block.
        joint_triplets = self.joint_num * 5 * (2 * 4) ** 2
        levelset_pair_bound = min(
            self.body_num * max(self.body_num - 1, 0) // 2,
            vf_candidates + ee_candidates,
        )
        levelset_candidates = (
            2 * self.max_body_vertex_count * levelset_pair_bound if getattr(self, "levelset_contact", False) else 0
        )
        damping_triplets = 0
        if self.contact_damping_stiffness > 0.0:
            # COO and nonsymmetric HashTriplet store both triangles.  The
            # frozen cache itself stores only the upper triangle.
            damping_triplets = 2 * self._estimate_contact_damping_capacity(sims)
        estimated = (
            base_triplets
            + joint_triplets
            + mesh_hessian_triplets
            + wall_candidates * wall_hessian_triplets
            + levelset_candidates * levelset_hessian_triplets
            + damping_triplets
        )
        safety = (
            float(os.environ.get("GT_AFFINE_HASH_TRIPLET_SAFETY", "1.25"))
            if safety_factor is None
            else float(safety_factor)
        )
        minimum = 256 if minimum_capacity is None else int(minimum_capacity)
        if not np.isfinite(safety) or safety < 1.0:
            raise ValueError("affine HashTriplet safety factor must be finite and >= 1")
        if minimum <= 0:
            raise ValueError("affine HashTriplet minimum capacity must be positive")
        return max(minimum, int(np.ceil(safety * estimated)))

    def bind_coo_matrix(self, coo_matrix):
        if int(coo_matrix.capacity) < self.max_coo_entries:
            raise ValueError(
                "AffineBody COO matrix capacity is smaller than its fixed "
                f"assembly bound ({coo_matrix.capacity} < {self.max_coo_entries})"
            )
        self.coo_matrix = coo_matrix
        self.K_coo_rows = coo_matrix.rows
        self.K_coo_cols = coo_matrix.cols
        self.K_coo_values = coo_matrix.data
        if self.K_coo_diag is None:
            self.K_coo_diag = ti.field(dtype=float, shape=max(self.dof, 1))

    def finalize_coo_assembly(self):
        used = int(self.coo_entry_count[None])
        if int(self.coo_overflow[None]) != 0 or used > self.max_coo_entries:
            raise RuntimeError(
                "AffineBody COO triplet buffer overflow: "
                f"used {used}, capacity {self.max_coo_entries}; increase the "
                "body/wall coordination number or GT_AFFINE_MAX_COO_ENTRIES"
            )
        if self.coo_matrix is not None:
            self.coo_matrix.linear_operator.update_nnz(used)
        return used

    def bind_hash_triplet(self, hash_triplet, *, full_symmetric_input=None):
        if int(hash_triplet.dim) != 3:
            raise ValueError("AffineBody HashTriplet must use 3x3 blocks")
        self.hash_triplet = hash_triplet
        self.hash_store_full_symmetric_input = (
            bool(hash_triplet.full_symmetric_input) if full_symmetric_input is None else bool(full_symmetric_input)
        )

    def _build_contact_property_arrays(self, scene, walls):
        shape = (self.material_num, self.material_num)
        self.pp_dhat_np = np.full(shape, self.default_dhat, dtype=np.float64)
        self.pp_kappa_np = np.full(shape, self.default_kappa, dtype=np.float64)
        self.pp_contact_damping_np = np.full(shape, self.default_contact_damping_stiffness, dtype=np.float64)
        self.pp_mu_np = np.full(shape, -1.0, dtype=np.float64)
        self.pw_dhat_np = np.full(shape, self.default_dhat, dtype=np.float64)
        self.pw_kappa_np = np.full(shape, self.default_kappa, dtype=np.float64)
        self.pw_contact_damping_np = np.full(shape, self.default_contact_damping_stiffness, dtype=np.float64)
        self.pw_mu_np = np.full(shape, -1.0, dtype=np.float64)

        for key, properties in scene.affine_contact_properties.items():
            mat_i, mat_j, dtype = key
            mat_i = int(mat_i)
            mat_j = int(mat_j)
            if mat_i < 0 or mat_i >= self.material_num or mat_j < 0 or mat_j >= self.material_num:
                continue
            target = None
            if dtype == "particle-particle":
                target = (self.pp_dhat_np, self.pp_kappa_np, self.pp_contact_damping_np, self.pp_mu_np)
            elif dtype == "particle-wall":
                target = (self.pw_dhat_np, self.pw_kappa_np, self.pw_contact_damping_np, self.pw_mu_np)
            if target is None:
                continue
            dhat, kappa, contact_damping, mu = target
            dhat[mat_i, mat_j] = _property_value(
                properties, ("Dhat", "dhat", "BarrierDistance", "barrier_distance"), dhat[mat_i, mat_j]
            )
            kappa[mat_i, mat_j] = _property_value(
                properties, ("BarrierStiffness", "barrier_stiffness", "Kappa", "kappa"), kappa[mat_i, mat_j]
            )
            contact_damping[mat_i, mat_j] = _property_value(
                properties,
                (
                    "ContactDampingStiffness",
                    "contact_damping_stiffness",
                    "ContactDamping",
                    "contact_damping",
                    "k_CD",
                    "k_cd",
                ),
                contact_damping[mat_i, mat_j],
            )
            mu_value = _property_value(
                properties, ("Friction", "friction", "StaticFriction", "static_friction"), mu[mat_i, mat_j]
            )
            mu[mat_i, mat_j] = mu_value
            if dtype == "particle-particle":
                dhat[mat_j, mat_i] = dhat[mat_i, mat_j]
                kappa[mat_j, mat_i] = kappa[mat_i, mat_j]
                contact_damping[mat_j, mat_i] = contact_damping[mat_i, mat_j]
                mu[mat_j, mat_i] = mu[mat_i, mat_j]

        for name, values in (
            ("particle-particle dhat", self.pp_dhat_np),
            ("particle-wall dhat", self.pw_dhat_np),
            ("particle-particle barrier stiffness", self.pp_kappa_np),
            ("particle-wall barrier stiffness", self.pw_kappa_np),
        ):
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(f"AffineBody {name} values must be finite and positive")
        for name, values in (
            (
                "particle-particle contact damping",
                self.pp_contact_damping_np,
            ),
            ("particle-wall contact damping", self.pw_contact_damping_np),
        ):
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(f"AffineBody {name} values must be finite and " "non-negative")
        for name, values in (
            ("particle-particle friction", self.pp_mu_np),
            ("particle-wall friction", self.pw_mu_np),
        ):
            valid = (values >= 0.0) | (values == -1.0)
            if not np.all(np.isfinite(values)) or not np.all(valid):
                raise ValueError(
                    f"AffineBody {name} values must be finite and either " "non-negative or -1 for fallback"
                )

        self.max_dhat = max(
            self.default_dhat,
            float(np.max(self.pp_dhat_np)) if self.pp_dhat_np.size else self.default_dhat,
            float(np.max(self.pw_dhat_np)) if len(walls) > 0 and self.pw_dhat_np.size else self.default_dhat,
        )
        self.max_contact_damping_stiffness = max(
            self.default_contact_damping_stiffness,
            (
                float(np.max(self.pp_contact_damping_np))
                if self.pp_contact_damping_np.size
                else self.default_contact_damping_stiffness
            ),
            (
                float(np.max(self.pw_contact_damping_np))
                if len(walls) > 0 and self.pw_contact_damping_np.size
                else self.default_contact_damping_stiffness
            ),
        )

    def _build_surface_storage(self):
        basis = []
        rest_vertices = []
        node2body = []
        faces = []
        face2body = []
        edges = []
        edge2body = []
        node_area = []
        edge_area = []
        body_contact_type = []
        body_scale = []
        levelset_grid_start = []
        levelset_grid_shape = []
        levelset_grid_origin = []
        levelset_grid_spacing = []
        levelset_template_to_grid = []
        levelset_template_to_grid_offset = []
        levelset_lipschitz = []
        levelset_grid_value = []
        levelset_lipschitz_cache = {}
        grid_offset = 0
        offset = 0
        for body_id, body in enumerate(self.state.bodies):
            body_basis = np.asarray(body["basis"], dtype=np.float64)
            body_faces = np.asarray(body["faces"], dtype=np.int32)
            body_rest_vertices = body_basis @ np.asarray(body["y"], dtype=np.float64)
            body_node_area, body_edges, body_edge_area = surface_vertex_edge_measures(body_rest_vertices, body_faces)
            basis.append(body_basis)
            rest_vertices.append(body_rest_vertices)
            node2body.extend([body_id] * body_basis.shape[0])
            faces.append(body_faces + offset)
            face2body.extend([body_id] * body_faces.shape[0])
            node_area.append(body_node_area)
            for local_edge, local_measure in zip(body_edges, body_edge_area):
                edges.append([int(local_edge[0]) + offset, int(local_edge[1]) + offset])
                edge2body.append(body_id)
                edge_area.append(float(local_measure))
            representation = str(body.get("contact_representation", "TriangleMesh"))
            is_levelset = representation == "LevelSet"
            body_contact_type.append(1 if is_levelset else 0)
            body_scale.append(float(body.get("scale", 1.0)))
            levelset_grid_start.append(grid_offset)
            if is_levelset:
                levelset = body.get("levelset")
                if levelset is None:
                    raise ValueError("LevelSet affine body is missing its signed-distance " "field")
                values = np.asarray(levelset.values, dtype=np.float64).reshape(-1)
                levelset_grid_shape.append(np.asarray(levelset.shape, dtype=np.int32))
                levelset_grid_origin.append(np.asarray(levelset.origin, dtype=np.float64))
                levelset_grid_spacing.append(float(levelset.spacing))
                linear = np.asarray(levelset.template_to_grid, dtype=np.float64)
                levelset_template_to_grid.append(linear)
                levelset_template_to_grid_offset.append(
                    np.asarray(
                        levelset.template_to_grid_offset,
                        dtype=np.float64,
                    )
                )
                cache_key = id(levelset)
                if cache_key not in levelset_lipschitz_cache:
                    shape = np.asarray(levelset.shape, dtype=np.int32)
                    grid_values = values.reshape(tuple(shape), order="F")
                    axis_slopes = [
                        np.max(np.abs(np.diff(grid_values, axis=axis))) / float(levelset.spacing) for axis in range(3)
                    ]
                    levelset_lipschitz_cache[cache_key] = float(
                        np.linalg.norm(linear, ord=2) * np.linalg.norm(axis_slopes)
                    )
                levelset_lipschitz.append(levelset_lipschitz_cache[cache_key])
                levelset_grid_value.append(values)
                grid_offset += int(values.size)
            else:
                levelset_grid_shape.append(np.ones(3, dtype=np.int32) * 2)
                levelset_grid_origin.append(np.zeros(3, dtype=np.float64))
                levelset_grid_spacing.append(1.0)
                levelset_template_to_grid.append(np.eye(3, dtype=np.float64))
                levelset_template_to_grid_offset.append(np.zeros(3, dtype=np.float64))
                levelset_lipschitz.append(1.0)
            offset += body_basis.shape[0]
        self.basis_np = (
            np.ascontiguousarray(np.vstack(basis), dtype=np.float64) if basis else np.zeros((0, 4), dtype=np.float64)
        )
        self.rest_x_np = (
            np.ascontiguousarray(np.vstack(rest_vertices), dtype=np.float64)
            if rest_vertices
            else np.zeros((0, 3), dtype=np.float64)
        )
        self.node2body_np = np.asarray(node2body, dtype=np.int32)
        self.body_vertex_count_np = np.bincount(
            self.node2body_np,
            minlength=max(self.body_num, 1),
        ).astype(np.int32)
        self.body_vertex_start_np = np.zeros(max(self.body_num, 1), dtype=np.int32)
        if self.body_num > 1:
            self.body_vertex_start_np[1 : self.body_num] = np.cumsum(self.body_vertex_count_np[: self.body_num - 1])
        self.max_body_vertex_count = max(
            int(np.max(self.body_vertex_count_np[: self.body_num], initial=0)),
            1,
        )
        self.faces_np = (
            np.ascontiguousarray(np.vstack(faces), dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
        )
        self.face2body_np = np.asarray(face2body, dtype=np.int32)
        self.edges_np = (
            np.ascontiguousarray(np.asarray(edges, dtype=np.int32).reshape((-1, 2)))
            if edges
            else np.zeros((0, 2), dtype=np.int32)
        )
        self.edge2body_np = np.asarray(edge2body, dtype=np.int32)
        self.node_area_np = (
            np.ascontiguousarray(np.concatenate(node_area), dtype=np.float64)
            if node_area
            else np.zeros(0, dtype=np.float64)
        )
        self.edge_area_np = (
            np.ascontiguousarray(np.asarray(edge_area, dtype=np.float64), dtype=np.float64)
            if edge_area
            else np.zeros(0, dtype=np.float64)
        )
        self.body_contact_type_np = np.asarray(body_contact_type, dtype=np.int32)
        self.body_scale_np = np.asarray(body_scale, dtype=np.float64)
        self.levelset_grid_start_np = np.asarray(levelset_grid_start, dtype=np.int32)
        self.levelset_grid_shape_np = np.ascontiguousarray(
            np.asarray(levelset_grid_shape, dtype=np.int32).reshape((-1, 3))
        )
        self.levelset_grid_origin_np = np.ascontiguousarray(
            np.asarray(levelset_grid_origin, dtype=np.float64).reshape((-1, 3))
        )
        self.levelset_grid_spacing_np = np.asarray(levelset_grid_spacing, dtype=np.float64)
        self.levelset_template_to_grid_np = np.ascontiguousarray(
            np.asarray(levelset_template_to_grid, dtype=np.float64).reshape((-1, 3, 3))
        )
        self.levelset_template_to_grid_offset_np = np.ascontiguousarray(
            np.asarray(levelset_template_to_grid_offset, dtype=np.float64).reshape((-1, 3))
        )
        self.levelset_lipschitz_np = np.asarray(levelset_lipschitz, dtype=np.float64)
        self.levelset_grid_value_np = (
            np.ascontiguousarray(np.concatenate(levelset_grid_value), dtype=np.float64)
            if levelset_grid_value
            else np.zeros(0, dtype=np.float64)
        )
        self.levelset_grid_num = int(self.levelset_grid_value_np.size)
        self.vertex_num = int(self.basis_np.shape[0])
        self.face_num = int(self.faces_np.shape[0])
        self.edge_num = int(self.edges_np.shape[0])

    def _build_joint_storage(self, scene, sims):
        joints = list(getattr(scene, "affine_joints", []))
        self.joint_num = len(joints)
        capacity = max(self.joint_num, 1)
        self.joint_body_np = np.full((capacity, 2), -1, dtype=np.int32)
        self.joint_anchor_weight_np = np.zeros((capacity, 2, 4), dtype=np.float64)
        self.joint_axis_weight_np = np.zeros((capacity, 2, 4), dtype=np.float64)
        self.joint_u_weight_np = np.zeros((capacity, 2, 4), dtype=np.float64)
        self.joint_v_weight_np = np.zeros((capacity, 2, 4), dtype=np.float64)
        self.joint_world_anchor_np = np.zeros((capacity, 3), dtype=np.float64)
        self.joint_world_axis_np = np.zeros((capacity, 3), dtype=np.float64)
        self.joint_world_u_np = np.zeros((capacity, 3), dtype=np.float64)
        self.joint_world_v_np = np.zeros((capacity, 3), dtype=np.float64)
        self.joint_position_stiffness_np = np.zeros(capacity, dtype=np.float64)
        self.joint_axis_stiffness_np = np.zeros(capacity, dtype=np.float64)
        self.joint_motor_stiffness_np = np.zeros(capacity, dtype=np.float64)
        self.joint_target_angle_np = np.zeros(capacity, dtype=np.float64)
        self.joint_limit_enabled_np = np.zeros(capacity, dtype=np.int32)
        self.joint_limit_lower_np = np.zeros(capacity, dtype=np.float64)
        self.joint_limit_upper_np = np.zeros(capacity, dtype=np.float64)
        self.joint_limit_stiffness_np = np.zeros(capacity, dtype=np.float64)
        self.joint_damping_np = np.zeros(capacity, dtype=np.float64)
        self.joint_collision_disabled_np = np.zeros((max(self.body_num, 1), max(self.body_num, 1)), dtype=np.int32)
        self.excluded_body_pairs = set()

        missing = object()

        def get(spec, names, default=missing):
            for name in names:
                if name in spec:
                    return spec[name]
            if default is missing:
                raise KeyError(f"AffineBody joint requires '{names[0]}'")
            return default

        def point_weights(local):
            return np.array([1.0 - np.sum(local), local[0], local[1], local[2]], dtype=np.float64)

        def vector_weights(local):
            return np.array([-np.sum(local), local[0], local[1], local[2]], dtype=np.float64)

        initial_y = np.asarray(self.state.y, dtype=np.float64)
        default_stiffness = float(getattr(sims, "affine_young_modulus", 1.0e6))
        for joint_id, spec in enumerate(joints):
            if not isinstance(spec, dict):
                raise TypeError("AffineBody joint must be a dictionary")
            joint_type = str(get(spec, ("JointType", "joint_type", "Type"), "Revolute"))
            joint_type = joint_type.strip().replace("_", "").replace("-", "").lower()
            if joint_type not in ("revolute", "hinge"):
                raise ValueError("AffineBody currently supports only Revolute joints")
            body_a = int(get(spec, ("BodyID1", "body_id1", "ParentBodyID")))
            body_b = int(get(spec, ("BodyID2", "body_id2", "ChildBodyID"), -1))
            if body_a < 0 or body_a >= self.body_num:
                raise ValueError(f"AffineBody joint {joint_id} BodyID1 is out of range")
            if body_b < -1 or body_b >= self.body_num or body_b == body_a:
                raise ValueError(f"AffineBody joint {joint_id} BodyID2 is invalid")

            anchor = np.asarray(get(spec, ("WorldAnchor", "world_anchor", "Anchor")), dtype=np.float64).reshape(3)
            axis = np.asarray(get(spec, ("WorldAxis", "world_axis", "Axis")), dtype=np.float64).reshape(3)
            if not np.all(np.isfinite(anchor)) or not np.all(np.isfinite(axis)):
                raise ValueError(f"AffineBody joint {joint_id} anchor and axis must be finite")
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= 1.0e-12:
                raise ValueError(f"AffineBody joint {joint_id} axis must be nonzero")
            axis /= axis_norm
            coordinate = np.eye(3, dtype=np.float64)[int(np.argmin(np.abs(axis)))]
            world_u = coordinate - np.dot(coordinate, axis) * axis
            world_u /= np.linalg.norm(world_u)
            world_v = np.cross(axis, world_u)

            base_stiffness = float(get(spec, ("Stiffness", "stiffness"), default_stiffness))
            position_stiffness = float(get(spec, ("PositionStiffness", "position_stiffness"), base_stiffness))
            axis_stiffness = float(get(spec, ("AxisStiffness", "axis_stiffness"), base_stiffness))
            motor_stiffness = float(get(spec, ("MotorStiffness", "motor_stiffness"), 0.0))
            target_angle = float(get(spec, ("TargetAngle", "target_angle"), 0.0))
            damping = float(get(spec, ("Damping", "damping"), 0.0))
            limits = get(spec, ("AngleLimit", "angle_limit", "Limits", "limits"), None)
            limit_stiffness = float(
                get(spec, ("LimitStiffness", "limit_stiffness"), base_stiffness if limits is not None else 0.0)
            )
            values = np.array(
                [
                    base_stiffness,
                    position_stiffness,
                    axis_stiffness,
                    motor_stiffness,
                    target_angle,
                    limit_stiffness,
                    damping,
                ],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"AffineBody joint {joint_id} parameters must be finite")
            if position_stiffness <= 0.0 or axis_stiffness <= 0.0:
                raise ValueError(f"AffineBody joint {joint_id} position/axis stiffness must be positive")
            if motor_stiffness < 0.0 or limit_stiffness < 0.0 or damping < 0.0:
                raise ValueError(f"AffineBody joint {joint_id} motor/limit/damping values must be non-negative")

            self.joint_body_np[joint_id] = [body_a, body_b]
            self.joint_world_anchor_np[joint_id] = anchor
            self.joint_world_axis_np[joint_id] = axis
            self.joint_world_u_np[joint_id] = world_u
            self.joint_world_v_np[joint_id] = world_v
            for side, body_id in enumerate((body_a, body_b)):
                if body_id < 0:
                    continue
                controls = initial_y[body_id]
                F = np.column_stack((controls[1] - controls[0], controls[2] - controls[0], controls[3] - controls[0]))
                if abs(float(np.linalg.det(F))) <= 1.0e-12:
                    raise ValueError(f"AffineBody joint {joint_id} body {body_id} has a singular affine frame")
                local_anchor = np.linalg.solve(F, anchor - controls[0])
                self.joint_anchor_weight_np[joint_id, side] = point_weights(local_anchor)
                self.joint_axis_weight_np[joint_id, side] = vector_weights(np.linalg.solve(F, axis))
                self.joint_u_weight_np[joint_id, side] = vector_weights(np.linalg.solve(F, world_u))
                self.joint_v_weight_np[joint_id, side] = vector_weights(np.linalg.solve(F, world_v))

            self.joint_position_stiffness_np[joint_id] = position_stiffness
            self.joint_axis_stiffness_np[joint_id] = axis_stiffness
            self.joint_motor_stiffness_np[joint_id] = motor_stiffness
            self.joint_target_angle_np[joint_id] = np.deg2rad(target_angle)
            self.joint_limit_stiffness_np[joint_id] = limit_stiffness
            self.joint_damping_np[joint_id] = damping
            if limits is not None:
                limits = np.asarray(limits, dtype=np.float64).reshape(-1)
                if limits.size != 2 or not np.all(np.isfinite(limits)):
                    raise ValueError(f"AffineBody joint {joint_id} AngleLimit must contain two finite degrees")
                if limits[0] >= limits[1] or limits[0] < -180.0 or limits[1] > 180.0:
                    raise ValueError(
                        f"AffineBody joint {joint_id} AngleLimit must satisfy -180 <= lower < upper <= 180"
                    )
                self.joint_limit_enabled_np[joint_id] = 1
                self.joint_limit_lower_np[joint_id] = np.deg2rad(limits[0])
                self.joint_limit_upper_np[joint_id] = np.deg2rad(limits[1])

            collide = bool(get(spec, ("CollideConnected", "collide_connected"), False))
            if body_b >= 0 and not collide:
                self.joint_collision_disabled_np[body_a, body_b] = 1
                self.joint_collision_disabled_np[body_b, body_a] = 1
                self.excluded_body_pairs.add(tuple(sorted((body_a, body_b))))

    def _load_constant_fields(self, walls, scene):
        if self.body_num > 0:
            self.body_vertex_start.from_numpy(self.body_vertex_start_np)
            self.body_vertex_count.from_numpy(self.body_vertex_count_np)
            self.body_contact_type.from_numpy(self.body_contact_type_np)
            self.body_scale.from_numpy(self.body_scale_np)
            self.levelset_grid_start.from_numpy(self.levelset_grid_start_np)
            self.levelset_grid_shape.from_numpy(self.levelset_grid_shape_np)
            self.levelset_grid_origin.from_numpy(self.levelset_grid_origin_np)
            self.levelset_grid_spacing.from_numpy(self.levelset_grid_spacing_np)
            self.levelset_template_to_grid.from_numpy(self.levelset_template_to_grid_np)
            self.levelset_template_to_grid_offset.from_numpy(self.levelset_template_to_grid_offset_np)
            self.levelset_lipschitz.from_numpy(self.levelset_lipschitz_np)
        if self.levelset_grid_num > 0:
            self.levelset_grid_value.from_numpy(self.levelset_grid_value_np)
        if self.vertex_num > 0:
            self.basis.from_numpy(self.basis_np)
            self.rest_x.from_numpy(self.rest_x_np)
            self.node2body.from_numpy(self.node2body_np)
            self.node_area.from_numpy(self.node_area_np)
        if self.face_num > 0:
            self.faces.from_numpy(self.faces_np)
            self.face2body.from_numpy(self.face2body_np)
        if self.edge_num > 0:
            self.edges.from_numpy(self.edges_np)
            self.edge2body.from_numpy(self.edge2body_np)
            self.edge_area.from_numpy(self.edge_area_np)

        mass_np = np.zeros((max(self.body_num, 1), 4, 4), dtype=np.float64)
        body_mass_np = np.zeros(max(self.body_num, 1), dtype=np.float64)
        volume_np = np.zeros(max(self.body_num, 1), dtype=np.float64)
        young_np = np.zeros(max(self.body_num, 1), dtype=np.float64)
        force_damp_np = np.zeros(max(self.body_num, 1), dtype=np.float64)
        torque_damp_np = np.zeros(max(self.body_num, 1), dtype=np.float64)
        body_mu_np = np.zeros(max(self.body_num, 1), dtype=np.float64)
        body_material_np = np.zeros(max(self.body_num, 1), dtype=np.int32)
        for body_id, body in enumerate(self.state.bodies):
            mass_np[body_id] = body["mass_matrix"]
            body_mass_np[body_id] = np.sum(body["mass_matrix"])
            volume_np[body_id] = body["volume"]
            young_np[body_id] = body["young"]
            force_damp_np[body_id] = body.get("force_damping", 0.0)
            torque_damp_np[body_id] = body.get("torque_damping", 0.0)
            body_mu_np[body_id] = body.get("mu", 0.0)
            body_material_np[body_id] = int(body["materialID"])
        self.mass.from_numpy(mass_np)
        self.body_mass.from_numpy(body_mass_np)
        self.volume.from_numpy(volume_np)
        self.young.from_numpy(young_np)
        self.force_damp.from_numpy(force_damp_np)
        self.torque_damp.from_numpy(torque_damp_np)
        self.body_mu.from_numpy(body_mu_np)
        self.body_material.from_numpy(body_material_np)
        if self.joint_num > 0:
            self.joint_body.from_numpy(self.joint_body_np)
            self.joint_anchor_weight.from_numpy(self.joint_anchor_weight_np)
            self.joint_axis_weight.from_numpy(self.joint_axis_weight_np)
            self.joint_u_weight.from_numpy(self.joint_u_weight_np)
            self.joint_v_weight.from_numpy(self.joint_v_weight_np)
            self.joint_world_anchor.from_numpy(self.joint_world_anchor_np)
            self.joint_world_axis.from_numpy(self.joint_world_axis_np)
            self.joint_world_u.from_numpy(self.joint_world_u_np)
            self.joint_world_v.from_numpy(self.joint_world_v_np)
            self.joint_position_stiffness.from_numpy(self.joint_position_stiffness_np)
            self.joint_axis_stiffness.from_numpy(self.joint_axis_stiffness_np)
            self.joint_motor_stiffness.from_numpy(self.joint_motor_stiffness_np)
            self.joint_target_angle.from_numpy(self.joint_target_angle_np)
            self.joint_limit_enabled.from_numpy(self.joint_limit_enabled_np)
            self.joint_limit_lower.from_numpy(self.joint_limit_lower_np)
            self.joint_limit_upper.from_numpy(self.joint_limit_upper_np)
            self.joint_limit_stiffness.from_numpy(self.joint_limit_stiffness_np)
            self.joint_damping.from_numpy(self.joint_damping_np)
        self.joint_collision_disabled.from_numpy(self.joint_collision_disabled_np)

        wall_type_np = np.zeros(max(self.wall_num, 1), dtype=np.int32)
        wall_material_np = np.zeros(max(self.wall_num, 1), dtype=np.int32)
        wall_point_np = np.zeros((max(self.wall_num, 1), 3), dtype=np.float64)
        wall_normal_np = np.zeros((max(self.wall_num, 1), 3), dtype=np.float64)
        wall_v0_np = np.zeros((max(self.wall_num, 1), 3), dtype=np.float64)
        wall_v1_np = np.zeros((max(self.wall_num, 1), 3), dtype=np.float64)
        wall_v2_np = np.zeros((max(self.wall_num, 1), 3), dtype=np.float64)
        wall_mu_np = np.zeros(max(self.wall_num, 1), dtype=np.float64)
        for wall_id, wall in enumerate(walls):
            if wall["type"] == "plane":
                wall_type_np[wall_id] = 0
                wall_point_np[wall_id] = wall["point"]
                wall_normal_np[wall_id] = wall["normal"]
            else:
                wall_type_np[wall_id] = 1
                wall_v0_np[wall_id], wall_v1_np[wall_id], wall_v2_np[wall_id] = wall["vertices"]
                wall_normal_np[wall_id] = _triangle_normal(
                    wall_v0_np[wall_id], wall_v1_np[wall_id], wall_v2_np[wall_id]
                )
            wall_material_np[wall_id] = int(wall.get("materialID", 0))
            wall_prop = scene.affine_contact_properties.get((0, wall_material_np[wall_id], "particle-wall"), {})
            wall_mu_np[wall_id] = _property_value(
                wall_prop, ("Friction", "friction", "StaticFriction", "static_friction"), 0.0
            )
        self.wall_type.from_numpy(wall_type_np)
        self.wall_material.from_numpy(wall_material_np)
        self.wall_point.from_numpy(wall_point_np)
        self.wall_normal.from_numpy(wall_normal_np)
        self.wall_v0.from_numpy(wall_v0_np)
        self.wall_v1.from_numpy(wall_v1_np)
        self.wall_v2.from_numpy(wall_v2_np)
        self.wall_mu.from_numpy(wall_mu_np)
        self.pp_dhat.from_numpy(self.pp_dhat_np)
        self.pp_kappa.from_numpy(self.pp_kappa_np)
        self.pp_contact_damping.from_numpy(self.pp_contact_damping_np)
        self.pp_mu.from_numpy(self.pp_mu_np)
        self.pw_dhat.from_numpy(self.pw_dhat_np)
        self.pw_kappa.from_numpy(self.pw_kappa_np)
        self.pw_contact_damping.from_numpy(self.pw_contact_damping_np)
        self.pw_mu.from_numpy(self.pw_mu_np)

    def initialize_contact_damping(self, y, hat_y):
        self.y.from_numpy(np.ascontiguousarray(y.reshape((self.control_num, 3)), dtype=np.float64))
        self.hat_y.from_numpy(np.ascontiguousarray(hat_y.reshape((self.control_num, 3)), dtype=np.float64))
        self._initialize_contact_damping_loaded()

    def initialize_contact_damping_device(self):
        """Refresh frozen contact data from the current device iterate."""
        self._initialize_contact_damping_loaded()

    def _initialize_contact_damping_loaded(self):
        self._clear_contact_damping_matrix()
        self._reconstruct_vertices()
        if self.levelset_contact:
            self.friction_contact_count.fill(0)
            self.friction_contact_overflow.fill(0)
            if self.body_num > 1:
                self._refresh_levelset_body_pairs(swept=False, margin=self.dhat)
                self._freeze_levelset_lagged_friction_geometry()
            if not self.fully_implicit and self.wall_num > 0:
                # Body/body level-set contacts are assembled directly from
                # the frozen affine controls; planar/facet walls reuse the
                # existing mesh-vertex lagged-friction cache.
                self._initialize_lagged_wall_friction()
                if int(self.friction_contact_overflow[0]) != 0:
                    raise RuntimeError(
                        "Affine lagged-friction contact buffer overflow: "
                        f"used {int(self.friction_contact_count[0])}, "
                        f"capacity {self.friction_contact_capacity}"
                    )
            return
        self.last_candidate_pairs = self.neighbor.update(
            self.x,
            self.dx,
            self.faces,
            self.edges,
            self.node2body,
            self.face2body,
            self.edge2body,
            self.dhat,
            swept=False,
        )
        self.friction_contact_count.fill(0)
        self.friction_contact_overflow.fill(0)
        if not self.fully_implicit:
            self._initialize_lagged_mesh_friction(
                self.neighbor.candidate_count,
                self.neighbor.candidate_vertex,
                self.neighbor.candidate_face,
                self.neighbor.edge_candidate_count,
                self.neighbor.candidate_edge0,
                self.neighbor.candidate_edge1,
            )
            self._initialize_lagged_wall_friction()
            if int(self.friction_contact_overflow[0]) != 0:
                raise RuntimeError(
                    "Affine lagged-friction contact buffer overflow: "
                    f"used {int(self.friction_contact_count[0])}, "
                    f"capacity {self.friction_contact_capacity}"
                )
        if self.contact_damping_stiffness <= 0.0:
            return
        self._assemble_body_pair_barrier_hessian(
            self.neighbor.candidate_count,
            self.neighbor.candidate_vertex,
            self.neighbor.candidate_face,
            self.neighbor.edge_candidate_count,
            self.neighbor.candidate_edge0,
            self.neighbor.candidate_edge1,
            1.0,
            MATRIX_CONTACT_DAMPING,
        )
        self._assemble_wall_contact_damping_hessian()
        self.finalize_contact_damping_assembly()

    def finalize_contact_damping_assembly(self):
        used = int(self.contact_damping_count[None])
        if int(self.contact_damping_overflow[None]) != 0 or used > self.contact_damping_capacity:
            raise RuntimeError(
                "Affine contact-damping triplet buffer overflow: "
                f"used {used}, capacity {self.contact_damping_capacity}; "
                "increase the body/wall coordination number"
            )
        return used

    def refresh_neighbor_candidates(self, y, margin=None):
        self.y.from_numpy(np.ascontiguousarray(y.reshape((self.control_num, 3)), dtype=np.float64))
        self._reconstruct_vertices()
        self.last_candidate_pairs = self.neighbor.update(
            self.x,
            self.dx,
            self.faces,
            self.edges,
            self.node2body,
            self.face2body,
            self.edge2body,
            self.dhat if margin is None else float(margin),
            swept=False,
        )
        return self.last_candidate_pairs

    def _refresh_levelset_body_pairs(self, swept, margin):
        self.neighbor.update(
            self.x,
            self.dx,
            self.faces,
            self.edges,
            self.node2body,
            self.face2body,
            self.edge2body,
            float(margin),
            swept=bool(swept),
        )
        self._reset_active_body_pairs()
        self._build_levelset_body_pairs(
            self.neighbor.candidate_count,
            self.neighbor.candidate_vertex,
            self.neighbor.candidate_face,
            self.neighbor.edge_candidate_count,
            self.neighbor.candidate_edge0,
            self.neighbor.candidate_edge1,
        )
        self.last_candidate_pairs = int(self.active_body_pair_count[None])
        return self.last_candidate_pairs

    @ti.kernel
    def _build_levelset_body_pairs(
        self,
        candidate_count: ti.template(),
        candidate_vertex: ti.template(),
        candidate_face: ti.template(),
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
    ):
        for candidate in range(candidate_count[None]):
            body_i = self.node2body[candidate_vertex[candidate]]
            body_j = self.face2body[candidate_face[candidate]]
            if (
                self._body_pair_allowed(body_i, body_j)
                and self.body_contact_type[body_i] == 1
                and self.body_contact_type[body_j] == 1
            ):
                self._mark_active_body_pair(body_i, body_j)
        for candidate in range(edge_candidate_count[None]):
            body_i = self.edge2body[candidate_edge0[candidate]]
            body_j = self.edge2body[candidate_edge1[candidate]]
            if (
                self._body_pair_allowed(body_i, body_j)
                and self.body_contact_type[body_i] == 1
                and self.body_contact_type[body_j] == 1
            ):
                self._mark_active_body_pair(body_i, body_j)

    def assemble(
        self,
        y,
        tilde_y,
        hat_y,
        need_matrix=True,
        matrix_mode=MATRIX_COO,
        include_inertia_matrix=True,
        project_spd=None,
    ):
        self.y.from_numpy(np.ascontiguousarray(y.reshape((self.control_num, 3)), dtype=np.float64))
        self.tilde_y.from_numpy(np.ascontiguousarray(tilde_y.reshape((self.control_num, 3)), dtype=np.float64))
        self.hat_y.from_numpy(np.ascontiguousarray(hat_y.reshape((self.control_num, 3)), dtype=np.float64))
        self._assemble_loaded(
            bool(need_matrix),
            int(matrix_mode),
            device_resident=False,
            include_inertia_matrix=bool(include_inertia_matrix),
            project_spd=(not self.fully_implicit if project_spd is None else bool(project_spd)),
        )
        grad = self.grad.to_numpy()[: self.control_num].reshape(-1)
        return float(self.energy[None]), grad

    def assemble_device(
        self,
        need_matrix=True,
        matrix_mode=MATRIX_HASH_TRIPLET,
        project_spd=None,
        solver_shift=True,
    ):
        """Assemble from device-resident ``y/tilde_y/hat_y`` fields.

        Only the scalar energy is synchronized.  The residual and optional
        Jacobian remain in Taichi fields for the CUDA nonlinear/Krylov path.
        """
        self._assemble_loaded(
            bool(need_matrix),
            int(matrix_mode),
            device_resident=True,
            include_inertia_matrix=True,
            project_spd=(not self.fully_implicit if project_spd is None else bool(project_spd)),
            solver_shift=bool(solver_shift),
        )
        return float(self.energy[None])

    def load_device_step_state(self, y, tilde_y, hat_y):
        """Upload the three step states once before a CUDA nonlinear solve."""
        shape = (self.control_num, 3)
        self.y.from_numpy(np.ascontiguousarray(np.asarray(y).reshape(shape), dtype=np.float64))
        self.tilde_y.from_numpy(np.ascontiguousarray(np.asarray(tilde_y).reshape(shape), dtype=np.float64))
        self.hat_y.from_numpy(np.ascontiguousarray(np.asarray(hat_y).reshape(shape), dtype=np.float64))

    def _assemble_loaded(
        self,
        need_matrix,
        matrix_mode,
        device_resident,
        include_inertia_matrix=True,
        project_spd=True,
        solver_shift=True,
    ):
        if self.is_semi:
            self.semi_constraint_violation[None] = 0.0
            self.semi_overflow[None] = 0
        self._clear_system(bool(need_matrix), int(matrix_mode))
        self._reconstruct_vertices()
        if self.levelset_contact:
            if self.body_num > 1:
                self._refresh_levelset_body_pairs(swept=False, margin=self.dhat)
        else:
            self.last_candidate_pairs = self.neighbor.update(
                self.x,
                self.dx,
                self.faces,
                self.edges,
                self.node2body,
                self.face2body,
                self.edge2body,
                self.dhat,
                swept=False,
            )
        # Small host-side lagged systems keep the exact, well-conditioned
        # control mass matrix separate from the projected non-inertial
        # Hessian. Adding O(1) mass entries atomically to an O(1e17) contact
        # block erases them in f64 before the dense solver can see them.
        self._assemble_inertia(
            bool(need_matrix and include_inertia_matrix),
            int(matrix_mode),
        )
        if device_resident:
            self._assemble_body_force_device(bool(need_matrix))
        else:
            self._assemble_body_force(self.state.gravity, bool(need_matrix))
        self._assemble_local_damping(bool(need_matrix), int(matrix_mode))
        self._assemble_rigidity(bool(need_matrix), int(matrix_mode), bool(project_spd))
        self._assemble_joints(bool(need_matrix), int(matrix_mode))
        if self.levelset_contact:
            if self.body_num > 1:
                self._assemble_levelset_contacts(bool(need_matrix), int(matrix_mode))
                if not self.fully_implicit and np.any(self.pp_mu_np > 0.0):
                    self._assemble_levelset_lagged_friction(bool(need_matrix), int(matrix_mode))
                self.last_candidate_pairs = int(self.levelset_active_contacts[None])
            else:
                self.last_candidate_pairs = 0
            if not self.fully_implicit and self.wall_num > 0:
                self._assemble_lagged_friction(bool(need_matrix), int(matrix_mode))
        else:
            if self.is_semi:
                self._assemble_semi_particle_contacts(
                    bool(need_matrix),
                    self.neighbor.candidate_count,
                    self.neighbor.candidate_vertex,
                    self.neighbor.candidate_face,
                    int(matrix_mode),
                )
                self._assemble_semi_edge_contacts(
                    bool(need_matrix),
                    self.neighbor.edge_candidate_count,
                    self.neighbor.candidate_edge0,
                    self.neighbor.candidate_edge1,
                    int(matrix_mode),
                )
            else:
                self._assemble_particle_contacts(
                    False,
                    self.neighbor.candidate_count,
                    self.neighbor.candidate_vertex,
                    self.neighbor.candidate_face,
                    int(matrix_mode),
                )
                self._assemble_edge_contacts(
                    False,
                    self.neighbor.edge_candidate_count,
                    self.neighbor.candidate_edge0,
                    self.neighbor.candidate_edge1,
                )
                if need_matrix:
                    self._assemble_body_pair_barrier_hessian(
                        self.neighbor.candidate_count,
                        self.neighbor.candidate_vertex,
                        self.neighbor.candidate_face,
                        self.neighbor.edge_candidate_count,
                        self.neighbor.candidate_edge0,
                        self.neighbor.candidate_edge1,
                        self.scale,
                        int(matrix_mode),
                        project_pd=bool(project_spd),
                    )
            if self.fully_implicit:
                self._assemble_fully_implicit_friction(bool(need_matrix), int(matrix_mode))
            else:
                self._assemble_lagged_friction(bool(need_matrix), int(matrix_mode))
        if self.is_semi:
            self._assemble_semi_wall_contacts(bool(need_matrix), int(matrix_mode))
        else:
            self._assemble_wall_contacts(bool(need_matrix), int(matrix_mode), bool(project_spd))
        if self.contact_damping_stiffness > 0.0:
            self._assemble_contact_damping(bool(need_matrix), int(matrix_mode))
        jacobian_shift = self._active_jacobian_shift()
        if need_matrix and solver_shift and jacobian_shift > 0.0:
            self._add_hessian_shift(int(matrix_mode), jacobian_shift)
        if need_matrix and matrix_mode == MATRIX_COO:
            self.finalize_coo_assembly()
        if self.is_semi and int(self.semi_overflow[None]) != 0:
            raise RuntimeError("AffineBody SemiIPC multiplier hash capacity is too small")

    def _active_jacobian_shift(self):
        """Return the mode-specific, explicitly requested regularization."""
        if self.fully_implicit:
            return self.fully_implicit_jacobian_shift
        return self.hessian_shift

    def init_step_size(self, y, direction, ccd_type="ccd", eta=0.2, accd_tolerance=1.0e-7, max_iteration=10000):
        y_np = np.ascontiguousarray(y.reshape((self.control_num, 3)), dtype=np.float64)
        direction_np = np.ascontiguousarray(direction.reshape((self.control_num, 3)), dtype=np.float64)
        self.y.from_numpy(y_np)
        self.direction_y.from_numpy(direction_np)
        return self.init_step_size_device(
            ccd_type=ccd_type,
            eta=eta,
            accd_tolerance=accd_tolerance,
            max_iteration=max_iteration,
        )

    def init_step_size_device(self, ccd_type="ccd", eta=0.2, accd_tolerance=1.0e-7, max_iteration=10000):
        """Compute IPC CCD from device-resident state and direction fields."""
        mode, eta, thickness = ccd_mode_parameters(ccd_type, eta, accd_tolerance)
        if mode in ("none", "off"):
            return 1.0
        self._reconstruct_vertices()
        self._reconstruct_vertex_directions()
        if self.levelset_contact:
            if self.body_num > 1:
                self._refresh_levelset_body_pairs(swept=True, margin=thickness)
            # The mesh candidates only seed the compact body-pair set.  Keep
            # the actual level-set CCD law authoritative; this shared kernel
            # should process walls only on this branch.
            self.neighbor.candidate_count[None] = 0
            self.neighbor.edge_candidate_count[None] = 0
            self._init_ccd_alpha()
            # Reuse the wall portion of the mesh CCD kernel with empty
            # particle candidate lists, then add the implicit level-set CCD.
            self._compute_ccd_alpha(
                eta,
                float(accd_tolerance),
                int(max_iteration),
                mode == "accd",
                self.neighbor.candidate_count,
                self.neighbor.candidate_vertex,
                self.neighbor.candidate_face,
                self.neighbor.edge_candidate_count,
                self.neighbor.candidate_edge0,
                self.neighbor.candidate_edge1,
            )
            if self.body_num > 1:
                self._compute_levelset_ccd_alpha(
                    eta,
                    float(accd_tolerance),
                    min(int(max_iteration), 10000),
                    mode == "accd",
                )
            alpha = float(self.ccd_alpha[None])
            return max(0.0, min(1.0, alpha))
        self.last_candidate_pairs = self.neighbor.update(
            self.x,
            self.dx,
            self.faces,
            self.edges,
            self.node2body,
            self.face2body,
            self.edge2body,
            thickness,
            swept=True,
        )
        self._init_ccd_alpha()
        if mode == "accd":
            self._compute_ccd_alpha(
                eta,
                float(accd_tolerance),
                int(max_iteration),
                True,
                self.neighbor.candidate_count,
                self.neighbor.candidate_vertex,
                self.neighbor.candidate_face,
                self.neighbor.edge_candidate_count,
                self.neighbor.candidate_edge0,
                self.neighbor.candidate_edge1,
            )
        else:
            self._compute_ccd_alpha(
                eta,
                float(accd_tolerance),
                int(max_iteration),
                False,
                self.neighbor.candidate_count,
                self.neighbor.candidate_vertex,
                self.neighbor.candidate_face,
                self.neighbor.edge_candidate_count,
                self.neighbor.candidate_edge0,
                self.neighbor.candidate_edge1,
            )
        alpha = float(self.ccd_alpha[None])
        return max(0.0, min(1.0, alpha))

    @ti.kernel
    def _clear_system(self, need_matrix: ti.template(), matrix_mode: ti.template()):
        self.energy[None] = 0.0
        for i in range(self.control_num):
            self.grad[i] = ti.Vector.zero(float, 3)
        if ti.static(need_matrix):
            if ti.static(matrix_mode == MATRIX_COO):
                self.coo_entry_count[None] = 0
                self.coo_overflow[None] = 0
                for idx in range(self.max_coo_entries):
                    self.K_coo_rows[idx] = 0
                    self.K_coo_cols[idx] = 0
                    self.K_coo_values[idx] = 0.0
                for i in range(self.dof):
                    self.K_coo_diag[i] = 0.0
            elif ti.static(matrix_mode == MATRIX_HASH_TRIPLET):
                pass

    @ti.kernel
    def _clear_contact_damping_matrix(self):
        self.contact_damping_count[None] = 0
        self.contact_damping_overflow[None] = 0

    @ti.kernel
    def _reconstruct_vertices(self):
        for v in range(self.vertex_num):
            body = self.node2body[v]
            x_v = ti.Vector.zero(float, 3)
            hx_v = ti.Vector.zero(float, 3)
            for a in range(4):
                w = self.basis[v, a]
                x_v += w * self.y[body * 4 + a]
                hx_v += w * self.hat_y[body * 4 + a]
            self.x[v] = x_v
            self.hat_x[v] = hx_v

    @ti.kernel
    def _reconstruct_vertex_directions(self):
        for v in range(self.vertex_num):
            body = self.node2body[v]
            dx_v = ti.Vector.zero(float, 3)
            for a in range(4):
                dx_v += self.basis[v, a] * self.direction_y[body * 4 + a]
            self.dx[v] = dx_v

    @ti.kernel
    def _assemble_inertia(self, need_matrix: ti.template(), matrix_target: ti.template()):
        for b, a, c in ti.ndrange(self.body_num, 4, 4):
            row = b * 4 + a
            col = b * 4 + c
            m = self.mass[b, a, c]
            for d in ti.static(range(3)):
                diff_row = self.y[row][d] - self.tilde_y[row][d]
                diff_col = self.y[col][d] - self.tilde_y[col][d]
                ti.atomic_add(self.energy[None], 0.5 * diff_row * m * diff_col)
                ti.atomic_add(self.grad[row][d], m * diff_col)
            if ti.static(need_matrix):
                self._add_matrix_block(
                    row,
                    col,
                    m * ti.Matrix.identity(float, 3),
                    matrix_target,
                )

    @ti.kernel
    def _assemble_body_force(self, gravity: ti.types.ndarray(), need_matrix: ti.template()):
        for b, a, d in ti.ndrange(self.body_num, 4, 3):
            lumped = 0.0
            for c in range(4):
                lumped += self.mass[b, a, c]
            row = b * 4 + a
            coeff = -self.scale_device[None] * (lumped * gravity[d] + self.external_generalized_force[row][d])
            ti.atomic_add(self.energy[None], coeff * self.y[row][d])
            ti.atomic_add(self.grad[row][d], coeff)

    @ti.kernel
    def _assemble_body_force_device(self, need_matrix: ti.template()):
        for b, a, d in ti.ndrange(self.body_num, 4, 3):
            lumped = 0.0
            for c in range(4):
                lumped += self.mass[b, a, c]
            row = b * 4 + a
            coeff = -self.scale_device[None] * (
                lumped * self.gravity[None][d] + self.external_generalized_force[row][d]
            )
            ti.atomic_add(self.energy[None], coeff * self.y[row][d])
            ti.atomic_add(self.grad[row][d], coeff)

    @ti.kernel
    def _assemble_local_damping(self, need_matrix: ti.template(), matrix_target: ti.template()):
        for b in range(self.body_num):
            mass = self.body_mass[b]
            fdamp = self.force_damp[b]
            tdamp = self.torque_damp[b]
            if mass > 0.0 and (fdamp > 0.0 or tdamp > 0.0):
                force_coeff = self.dt_device[None] * mass * fdamp
                torque_coeff = 0.25 * self.dt_device[None] * mass * tdamp
                for d in ti.static(range(3)):
                    delta0 = self.y[b * 4 + 0][d] - self.hat_y[b * 4 + 0][d]
                    delta1 = self.y[b * 4 + 1][d] - self.hat_y[b * 4 + 1][d]
                    delta2 = self.y[b * 4 + 2][d] - self.hat_y[b * 4 + 2][d]
                    delta3 = self.y[b * 4 + 3][d] - self.hat_y[b * 4 + 3][d]
                    mean_delta = 0.25 * (delta0 + delta1 + delta2 + delta3)
                    rel0 = delta0 - mean_delta
                    rel1 = delta1 - mean_delta
                    rel2 = delta2 - mean_delta
                    rel3 = delta3 - mean_delta
                    ti.atomic_add(self.energy[None], 0.5 * force_coeff * mean_delta * mean_delta)
                    ti.atomic_add(
                        self.energy[None],
                        0.5 * torque_coeff * (rel0 * rel0 + rel1 * rel1 + rel2 * rel2 + rel3 * rel3),
                    )
                    ti.atomic_add(self.grad[b * 4 + 0][d], 0.25 * force_coeff * mean_delta + torque_coeff * rel0)
                    ti.atomic_add(self.grad[b * 4 + 1][d], 0.25 * force_coeff * mean_delta + torque_coeff * rel1)
                    ti.atomic_add(self.grad[b * 4 + 2][d], 0.25 * force_coeff * mean_delta + torque_coeff * rel2)
                    ti.atomic_add(self.grad[b * 4 + 3][d], 0.25 * force_coeff * mean_delta + torque_coeff * rel3)
                if ti.static(need_matrix):
                    for a, c in ti.ndrange(4, 4):
                        projection = (1.0 if a == c else 0.0) - 0.25
                        value = 0.0625 * force_coeff + torque_coeff * projection
                        self._add_matrix_block(
                            b * 4 + a,
                            b * 4 + c,
                            value * ti.Matrix.identity(float, 3),
                            matrix_target,
                        )

    @ti.kernel
    def _assemble_rigidity(
        self,
        need_matrix: ti.template(),
        matrix_target: ti.template(),
        project_spd: ti.template(),
    ):
        for b in range(self.body_num):
            F = ti.Matrix.zero(float, 3, 3)
            for col in ti.static(range(3)):
                edge = self.y[b * 4 + col + 1] - self.y[b * 4]
                for row in ti.static(range(3)):
                    F[row, col] = edge[row]
            strain = F.transpose() @ F - ti.Matrix.identity(float, 3)
            coeff = self.volume[b] * self.young[b] * self.scale_device[None] / 8.0
            e = 0.0
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    e += strain[i, j] * strain[i, j]
            ti.atomic_add(self.energy[None], coeff * e)

            gF = 4.0 * coeff * F @ strain
            for d in ti.static(range(3)):
                ti.atomic_add(self.grad[b * 4 + 0][d], -gF[d, 0] - gF[d, 1] - gF[d, 2])
                ti.atomic_add(self.grad[b * 4 + 1][d], gF[d, 0])
                ti.atomic_add(self.grad[b * 4 + 2][d], gF[d, 1])
                ti.atomic_add(self.grad[b * 4 + 3][d], gF[d, 2])

            if ti.static(need_matrix):
                hessian_F = ti.Matrix.zero(float, 9, 9)
                for di, col_i, dj, col_j in ti.ndrange(3, 3, 3, 3):
                    hF = 0.0
                    if di == dj:
                        hF += strain[col_j, col_i]
                    hF += F[di, col_j] * F[dj, col_i]
                    if col_i == col_j:
                        dot_row = 0.0
                        for k in ti.static(range(3)):
                            dot_row += F[di, k] * F[dj, k]
                        hF += dot_row
                    hessian_F[di * 3 + col_i, dj * 3 + col_j] = 4.0 * coeff * hF
                local_hessian = ti.Matrix.zero(float, 12, 12)
                for ci in range(4):
                    for cj in range(4):
                        for col_i in range(3):
                            wi = -1.0 if ci == 0 else (1.0 if ci == col_i + 1 else 0.0)
                            for col_j in range(3):
                                wj = -1.0 if cj == 0 else (1.0 if cj == col_j + 1 else 0.0)
                                if wi != 0.0 and wj != 0.0:
                                    for di in range(3):
                                        for dj in range(3):
                                            local_hessian[ci * 3 + di, cj * 3 + dj] += (
                                                wi
                                                * wj
                                                * hessian_F[
                                                    di * 3 + col_i,
                                                    dj * 3 + col_j,
                                                ]
                                            )
                # Projects rigidity after the F-to-control pullback,
                # i.e. on the complete 12 affine DOFs of one body.
                if ti.static(project_spd):
                    local_hessian = psd_project_nd(local_hessian)
                for ci in range(4):
                    for cj in range(4):
                        block = ti.Matrix.zero(float, 3, 3)
                        for di, dj in ti.static(ti.ndrange(3, 3)):
                            block[di, dj] = local_hessian[ci * 3 + di, cj * 3 + dj]
                        self._add_matrix_block(
                            b * 4 + ci,
                            b * 4 + cj,
                            block,
                            matrix_target,
                        )

    @ti.func
    def _joint_direction_u(self, joint_id, side, positions: ti.template()):
        value = ti.Vector.zero(float, 3)
        body = self.joint_body[joint_id][side]
        if body >= 0:
            for control in range(4):
                value += self.joint_u_weight[joint_id, side][control] * positions[body * 4 + control]
        return value

    @ti.func
    def _joint_direction_v(self, joint_id, side, positions: ti.template()):
        value = ti.Vector.zero(float, 3)
        body = self.joint_body[joint_id][side]
        if body >= 0:
            for control in range(4):
                value += self.joint_v_weight[joint_id, side][control] * positions[body * 4 + control]
        return value

    @ti.func
    def _joint_angle_compact(self, joint_id, positions: ti.template()):
        body_a = self.joint_body[joint_id][0]
        body_b = self.joint_body[joint_id][1]
        u_a = ti.Vector.zero(float, 3)
        v_a = ti.Vector.zero(float, 3)
        u_b = ti.Vector.zero(float, 3)
        if body_a >= 0:
            for control_a in ti.static(range(4)):
                position = positions[body_a * 4 + control_a]
                u_a += self.joint_u_weight[joint_id, 0][control_a] * position
                v_a += self.joint_v_weight[joint_id, 0][control_a] * position
        cosine, sine = 0.0, 0.0
        if body_b >= 0:
            for control_b in ti.static(range(4)):
                u_b += self.joint_u_weight[joint_id, 1][control_b] * positions[body_b * 4 + control_b]
            cosine = u_b.dot(u_a)
            sine = u_b.dot(v_a)
        else:
            cosine = u_a.dot(self.joint_world_u[joint_id])
            sine = u_a.dot(self.joint_world_v[joint_id])
        return ti.atan2(sine, cosine)

    @ti.func
    def _joint_angle_gradient_compact(self, joint_id, local, positions: ti.template()):
        gradient = ti.Vector.zero(float, 3)
        side, control = local // 4, local % 4
        body_a = self.joint_body[joint_id][0]
        body_b = self.joint_body[joint_id][1]
        u_a = ti.Vector.zero(float, 3)
        v_a = ti.Vector.zero(float, 3)
        u_b = ti.Vector.zero(float, 3)
        if body_a >= 0:
            for control_a in ti.static(range(4)):
                position = positions[body_a * 4 + control_a]
                u_a += self.joint_u_weight[joint_id, 0][control_a] * position
                v_a += self.joint_v_weight[joint_id, 0][control_a] * position
        if body_b >= 0:
            for control_b in ti.static(range(4)):
                u_b += self.joint_u_weight[joint_id, 1][control_b] * positions[body_b * 4 + control_b]
            cosine, sine = u_b.dot(u_a), u_b.dot(v_a)
            denominator = cosine * cosine + sine * sine
            grad_cosine = ti.Vector.zero(float, 3)
            grad_sine = ti.Vector.zero(float, 3)
            if denominator > 1.0e-20:
                if side == 0:
                    grad_cosine = self.joint_u_weight[joint_id, 0][control] * u_b
                    grad_sine = self.joint_v_weight[joint_id, 0][control] * u_b
                else:
                    weight = self.joint_u_weight[joint_id, 1][control]
                    grad_cosine = weight * u_a
                    grad_sine = weight * v_a
                gradient = (cosine * grad_sine - sine * grad_cosine) / denominator
        elif side == 0:
            cosine = u_a.dot(self.joint_world_u[joint_id])
            sine = u_a.dot(self.joint_world_v[joint_id])
            denominator = cosine * cosine + sine * sine
            if denominator > 1.0e-20:
                weight = self.joint_u_weight[joint_id, 0][control]
                gradient = (
                    weight * (cosine * self.joint_world_v[joint_id] - sine * self.joint_world_u[joint_id]) / denominator
                )
        return gradient

    @ti.func
    def _joint_direction_perturbation(
        self,
        joint_id,
        side,
        family: ti.template(),
        source: ti.template(),
    ):
        value = ti.Vector.zero(float, 3)
        body = self.joint_body[joint_id][side]
        if body >= 0:
            for control in range(4):
                weight = self.joint_u_weight[joint_id, side][control]
                if ti.static(family == 1):
                    weight = self.joint_v_weight[joint_id, side][control]
                for component in ti.static(range(3)):
                    direction = 0.0
                    if ti.static(source == 0):
                        direction = self.linear_x[12 * body + 3 * control + component]
                    else:
                        direction = self.y[4 * body + control][component] - self.hat_y[4 * body + control][component]
                    value[component] += weight * direction
        return value

    @ti.func
    def _joint_angle_hessian_product_compact(
        self,
        joint_id,
        local,
        positions: ti.template(),
        source: ti.template(),
    ):
        result = ti.Vector.zero(float, 3)
        side, control = local // 4, local % 4
        body_b = self.joint_body[joint_id][1]
        u_a = self._joint_direction_u(joint_id, 0, positions)
        zu_a = self._joint_direction_perturbation(joint_id, 0, 0, source)
        if body_b >= 0:
            v_a = self._joint_direction_v(joint_id, 0, positions)
            u_b = self._joint_direction_u(joint_id, 1, positions)
            zv_a = self._joint_direction_perturbation(joint_id, 0, 1, source)
            zu_b = self._joint_direction_perturbation(joint_id, 1, 0, source)
            cosine, sine = u_b.dot(u_a), u_b.dot(v_a)
            denominator = cosine * cosine + sine * sine
            if denominator > 1.0e-20:
                cosine_gradient = ti.Vector.zero(float, 3)
                sine_gradient = ti.Vector.zero(float, 3)
                cosine_hessian_product = ti.Vector.zero(float, 3)
                sine_hessian_product = ti.Vector.zero(float, 3)
                if side == 0:
                    cosine_gradient = self.joint_u_weight[joint_id, 0][control] * u_b
                    sine_gradient = self.joint_v_weight[joint_id, 0][control] * u_b
                    cosine_hessian_product = self.joint_u_weight[joint_id, 0][control] * zu_b
                    sine_hessian_product = self.joint_v_weight[joint_id, 0][control] * zu_b
                else:
                    cosine_gradient = self.joint_u_weight[joint_id, 1][control] * u_a
                    sine_gradient = self.joint_u_weight[joint_id, 1][control] * v_a
                    cosine_hessian_product = self.joint_u_weight[joint_id, 1][control] * zu_a
                    sine_hessian_product = self.joint_u_weight[joint_id, 1][control] * zv_a
                cosine_direction = zu_b.dot(u_a) + u_b.dot(zu_a)
                sine_direction = zu_b.dot(v_a) + u_b.dot(zv_a)
                angle_gradient = (cosine * sine_gradient - sine * cosine_gradient) / denominator
                result = (
                    cosine_direction * sine_gradient
                    + cosine * sine_hessian_product
                    - sine_direction * cosine_gradient
                    - sine * cosine_hessian_product
                ) / denominator
                result -= angle_gradient * (2.0 * (cosine * cosine_direction + sine * sine_direction) / denominator)
        elif side == 0:
            cosine = u_a.dot(self.joint_world_u[joint_id])
            sine = u_a.dot(self.joint_world_v[joint_id])
            denominator = cosine * cosine + sine * sine
            if denominator > 1.0e-20:
                cosine_gradient = self.joint_u_weight[joint_id, 0][control] * self.joint_world_u[joint_id]
                sine_gradient = self.joint_u_weight[joint_id, 0][control] * self.joint_world_v[joint_id]
                cosine_direction = zu_a.dot(self.joint_world_u[joint_id])
                sine_direction = zu_a.dot(self.joint_world_v[joint_id])
                angle_gradient = (cosine * sine_gradient - sine * cosine_gradient) / denominator
                result = (cosine_direction * sine_gradient - sine_direction * cosine_gradient) / denominator
                result -= angle_gradient * (2.0 * (cosine * cosine_direction + sine * sine_direction) / denominator)
        return result

    @ti.func
    def _joint_linear_weight(self, joint_id, term: ti.template(), angle, local):
        side, control = local // 4, local % 4
        weight = 0.0
        if ti.static(term == 0):
            weight = self.joint_anchor_weight[joint_id, side][control]
            if side == 1:
                weight = -weight
        elif ti.static(term == 1):
            weight = self.joint_axis_weight[joint_id, side][control]
            if side == 1:
                weight = -weight
        else:
            if self.joint_body[joint_id][1] >= 0:
                if side == 0:
                    weight = -(
                        ti.cos(angle) * self.joint_u_weight[joint_id, 0][control]
                        + ti.sin(angle) * self.joint_v_weight[joint_id, 0][control]
                    )
                else:
                    weight = self.joint_u_weight[joint_id, 1][control]
            elif side == 0:
                weight = self.joint_u_weight[joint_id, 0][control]
        return weight

    @ti.func
    def _scatter_joint_linear_term(
        self,
        joint_id,
        term: ti.template(),
        angle,
        target,
        coefficient,
        need_matrix: ti.template(),
        matrix_target: ti.template(),
    ):
        residual = -target
        for local in range(8):
            side, control = local // 4, local % 4
            body = self.joint_body[joint_id][side]
            if body >= 0:
                residual += self._joint_linear_weight(joint_id, term, angle, local) * self.y[body * 4 + control]
        ti.atomic_add(self.energy[None], 0.5 * coefficient * residual.dot(residual))
        for local in range(8):
            side, control = local // 4, local % 4
            body = self.joint_body[joint_id][side]
            if body >= 0:
                weight = self._joint_linear_weight(joint_id, term, angle, local)
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        self.grad[body * 4 + control][component],
                        coefficient * weight * residual[component],
                    )
        if ti.static(need_matrix):
            for local_i, local_j in ti.ndrange(8, 8):
                side_i, control_i = local_i // 4, local_i % 4
                side_j, control_j = local_j // 4, local_j % 4
                body_i = self.joint_body[joint_id][side_i]
                body_j = self.joint_body[joint_id][side_j]
                if body_i >= 0 and body_j >= 0:
                    value = (
                        coefficient
                        * self._joint_linear_weight(joint_id, term, angle, local_i)
                        * self._joint_linear_weight(joint_id, term, angle, local_j)
                    )
                    self._add_matrix_block(
                        body_i * 4 + control_i,
                        body_j * 4 + control_j,
                        value * ti.Matrix.identity(float, 3),
                        matrix_target,
                    )

    @ti.kernel
    def _differentiate_gravity_parameter(self):
        self.gravity_vjp[None] = ti.Vector.zero(float, 3)
        for body, control in ti.ndrange(self.body_num, 4):
            lumped = 0.0
            for column in range(4):
                lumped += self.mass[body, control, column]
            for component in ti.static(range(3)):
                ti.atomic_add(
                    self.gravity_vjp[None][component],
                    self.scale_device[None] * lumped * self.linear_x[12 * body + 3 * control + component],
                )

    @ti.kernel
    def _differentiate_young_parameter(self):
        for body in range(self.body_num):
            F = ti.Matrix.zero(float, 3, 3)
            for column in ti.static(range(3)):
                edge = self.y[body * 4 + column + 1] - self.y[body * 4]
                for row in ti.static(range(3)):
                    F[row, column] = edge[row]
            strain = F.transpose() @ F - ti.Matrix.identity(float, 3)
            gradient_F = 0.5 * self.volume[body] * self.scale_device[None] * F @ strain
            derivative = 0.0
            for component in ti.static(range(3)):
                derivative += self.linear_x[12 * body + component] * (
                    gradient_F[component, 0] + gradient_F[component, 1] + gradient_F[component, 2]
                )
                for column in ti.static(range(3)):
                    derivative -= (
                        self.linear_x[12 * body + 3 * (column + 1) + component] * gradient_F[component, column]
                    )
            self.young_vjp[body] = derivative

    @ti.kernel
    def _differentiate_joint_target_parameters(self):
        for joint_id in range(self.joint_num):
            self.joint_target_vjp[joint_id] = 0.0
            stiffness = self.joint_motor_stiffness[joint_id]
            if stiffness > 0.0:
                angle = self.joint_target_angle[joint_id]
                cosine, sine = ti.cos(angle), ti.sin(angle)
                body_b = self.joint_body[joint_id][1]
                target = ti.Vector.zero(float, 3)
                target_derivative = ti.Vector.zero(float, 3)
                if body_b < 0:
                    target = cosine * self.joint_world_u[joint_id] + sine * self.joint_world_v[joint_id]
                    target_derivative = -sine * self.joint_world_u[joint_id] + cosine * self.joint_world_v[joint_id]
                residual = -target
                residual_derivative = -target_derivative
                for local in range(8):
                    side, control = local // 4, local % 4
                    body = self.joint_body[joint_id][side]
                    if body >= 0:
                        weight = self._joint_linear_weight(joint_id, 2, angle, local)
                        weight_derivative = 0.0
                        if side == 0 and body_b >= 0:
                            weight_derivative = (
                                sine * self.joint_u_weight[joint_id, 0][control]
                                - cosine * self.joint_v_weight[joint_id, 0][control]
                            )
                        residual += weight * self.y[body * 4 + control]
                        residual_derivative += weight_derivative * self.y[body * 4 + control]
                derivative = 0.0
                coefficient = self.scale_device[None] * stiffness
                for local in range(8):
                    side, control = local // 4, local % 4
                    body = self.joint_body[joint_id][side]
                    if body >= 0:
                        weight = self._joint_linear_weight(joint_id, 2, angle, local)
                        weight_derivative = 0.0
                        if side == 0 and body_b >= 0:
                            weight_derivative = (
                                sine * self.joint_u_weight[joint_id, 0][control]
                                - cosine * self.joint_v_weight[joint_id, 0][control]
                            )
                        residual_partial = coefficient * (weight_derivative * residual + weight * residual_derivative)
                        adjoint = ti.Vector.zero(float, 3)
                        for component in ti.static(range(3)):
                            adjoint[component] = self.linear_x[12 * body + 3 * control + component]
                        derivative -= adjoint.dot(residual_partial)
                self.joint_target_vjp[joint_id] = derivative

    @ti.kernel
    def _differentiate_friction_scale_parameter(self):
        self.friction_scale_vjp[None] = 0.0
        contact_count = ti.min(self.friction_contact_count[0], self.friction_contact_capacity)
        for contact in range(contact_count):
            bodies = self.friction_contact_bodies[contact]
            vertices = self.friction_contact_vertices[contact]
            weights = self.friction_contact_weights[contact]
            rel = ti.Vector.zero(float, 3)
            adjoint_rel = ti.Vector.zero(float, 3)
            for site in ti.static(range(4)):
                rel += weights[site] * self.x[vertices[site]]
                for control in range(4):
                    basis = self.basis[vertices[site], control]
                    for component in ti.static(range(3)):
                        adjoint_rel[component] += (
                            weights[site] * basis * self.linear_x[12 * bodies[site] + 3 * control + component]
                        )
            normal = self.friction_contact_normal[contact]
            unit_normal = normal / normal.norm()
            projection = ti.Matrix.identity(float, 3) - unit_normal.outer_product(unit_normal)
            velocity = projection @ ((rel - self.friction_contact_hat_rel[contact]) / self.dt_device[None])
            force = (
                self.friction_contact_coeff[contact]
                * self._friction_f1_div_vbarnorm(velocity.norm(), self.epsv)
                * (projection @ velocity)
            )
            ti.atomic_add(self.friction_scale_vjp[None], -adjoint_rel.dot(force))

    @ti.kernel
    def _differentiate_joint_damping_parameters(self):
        for joint_id in range(self.joint_num):
            body_a = self.joint_body[joint_id][0]
            body_b = self.joint_body[joint_id][1]
            u_a = ti.Vector.zero(float, 3)
            v_a = ti.Vector.zero(float, 3)
            u_b = ti.Vector.zero(float, 3)
            for control in ti.static(range(4)):
                if body_a >= 0:
                    position = self.hat_y[body_a * 4 + control]
                    u_a += self.joint_u_weight[joint_id, 0][control] * position
                    v_a += self.joint_v_weight[joint_id, 0][control] * position
                if body_b >= 0:
                    u_b += self.joint_u_weight[joint_id, 1][control] * self.hat_y[body_b * 4 + control]
            angle = 0.0
            target = ti.Vector.zero(float, 3)
            if body_b >= 0:
                angle = ti.atan2(u_b.dot(v_a), u_b.dot(u_a))
                target = u_b - ti.cos(angle) * u_a - ti.sin(angle) * v_a
            elif body_a >= 0:
                target = u_a
            residual = -target
            for local in range(8):
                side, control = local // 4, local % 4
                body = self.joint_body[joint_id][side]
                if body >= 0:
                    residual += self._joint_linear_weight(joint_id, 2, angle, local) * self.y[body * 4 + control]
            derivative = 0.0
            for local in range(8):
                side, control = local // 4, local % 4
                body = self.joint_body[joint_id][side]
                if body >= 0:
                    adjoint = ti.Vector.zero(float, 3)
                    for component in ti.static(range(3)):
                        adjoint[component] = self.linear_x[12 * body + 3 * control + component]
                    derivative += adjoint.dot(self._joint_linear_weight(joint_id, 2, angle, local) * residual)
            self.joint_damping_vjp[joint_id] = -self.dt_device[None] * derivative

    def differentiate_step_parameters(self):
        self._differentiate_gravity_parameter()
        self._differentiate_young_parameter()
        self._differentiate_joint_target_parameters()
        self._differentiate_joint_damping_parameters()
        self._differentiate_friction_scale_parameter()
        return (
            np.asarray(self.gravity_vjp[None], dtype=np.float64),
            self.young_vjp.to_numpy()[: self.body_num].copy(),
            self.joint_target_vjp.to_numpy()[: self.joint_num].copy(),
        )

    @ti.kernel
    def _assemble_joint_constraints(self, need_matrix: ti.template(), matrix_target: ti.template()):
        for joint_id in range(self.joint_num):
            body_b = self.joint_body[joint_id][1]
            anchor_target = ti.Vector.zero(float, 3)
            axis_target = ti.Vector.zero(float, 3)
            if body_b < 0:
                anchor_target = self.joint_world_anchor[joint_id]
                axis_target = self.joint_world_axis[joint_id]
            self._scatter_joint_linear_term(
                joint_id,
                0,
                0.0,
                anchor_target,
                self.scale_device[None] * self.joint_position_stiffness[joint_id],
                need_matrix,
                matrix_target,
            )
            self._scatter_joint_linear_term(
                joint_id,
                1,
                0.0,
                axis_target,
                self.scale_device[None] * self.joint_axis_stiffness[joint_id],
                need_matrix,
                matrix_target,
            )

    @ti.kernel
    def _assemble_joint_motors(self, need_matrix: ti.template(), matrix_target: ti.template()):
        for joint_id in range(self.joint_num):
            stiffness = self.joint_motor_stiffness[joint_id]
            if stiffness > 0.0:
                angle = self.joint_target_angle[joint_id]
                target = ti.Vector.zero(float, 3)
                if self.joint_body[joint_id][1] < 0:
                    target = ti.cos(angle) * self.joint_world_u[joint_id] + ti.sin(angle) * self.joint_world_v[joint_id]
                self._scatter_joint_linear_term(
                    joint_id,
                    2,
                    angle,
                    target,
                    self.scale_device[None] * stiffness,
                    need_matrix,
                    matrix_target,
                )

    @ti.kernel
    def _assemble_joint_limits(self, need_matrix: ti.template(), matrix_target: ti.template()):
        for joint_id in range(self.joint_num):
            stiffness = self.joint_limit_stiffness[joint_id]
            if self.joint_limit_enabled[joint_id] != 0 and stiffness > 0.0:
                angle = self._joint_angle_compact(joint_id, self.y)
                target_angle = 0.0
                active = 0
                if angle < self.joint_limit_lower[joint_id]:
                    target_angle = self.joint_limit_lower[joint_id]
                    active = 1
                elif angle > self.joint_limit_upper[joint_id]:
                    target_angle = self.joint_limit_upper[joint_id]
                    active = 1
                if active != 0:
                    target = ti.Vector.zero(float, 3)
                    if self.joint_body[joint_id][1] < 0:
                        target = (
                            ti.cos(target_angle) * self.joint_world_u[joint_id]
                            + ti.sin(target_angle) * self.joint_world_v[joint_id]
                        )
                    self._scatter_joint_linear_term(
                        joint_id,
                        2,
                        target_angle,
                        target,
                        self.scale_device[None] * stiffness,
                        need_matrix,
                        matrix_target,
                    )

    @ti.kernel
    def _assemble_joint_damping_compact(self, need_matrix: ti.template(), matrix_target: ti.template()):
        for joint_id in range(self.joint_num):
            damping = self.joint_damping[joint_id]
            if damping > 0.0:
                body_a = self.joint_body[joint_id][0]
                body_b = self.joint_body[joint_id][1]
                u_a = ti.Vector.zero(float, 3)
                v_a = ti.Vector.zero(float, 3)
                u_b = ti.Vector.zero(float, 3)
                if body_a >= 0:
                    for control in ti.static(range(4)):
                        position = self.hat_y[body_a * 4 + control]
                        u_a += self.joint_u_weight[joint_id, 0][control] * position
                        v_a += self.joint_v_weight[joint_id, 0][control] * position
                angle = 0.0
                target = ti.Vector.zero(float, 3)
                if body_b >= 0:
                    for control in ti.static(range(4)):
                        u_b += self.joint_u_weight[joint_id, 1][control] * self.hat_y[body_b * 4 + control]
                    cosine = u_b.dot(u_a)
                    sine = u_b.dot(v_a)
                    angle = ti.atan2(sine, cosine)
                    target = u_b - ti.cos(angle) * u_a - ti.sin(angle) * v_a
                elif body_a >= 0:
                    target = u_a
                self._scatter_joint_linear_term(
                    joint_id,
                    2,
                    angle,
                    target,
                    self.dt_device[None] * damping,
                    need_matrix,
                    matrix_target,
                )

    def _assemble_joints(self, need_matrix, matrix_target):
        if self.joint_num == 0:
            return
        self._assemble_joint_constraints(bool(need_matrix), int(matrix_target))
        self._assemble_joint_motors(bool(need_matrix), int(matrix_target))
        self._assemble_joint_limits(bool(need_matrix), int(matrix_target))
        self._assemble_joint_damping_compact(bool(need_matrix), int(matrix_target))

    @ti.func
    def _body_pair_allowed(self, body_i, body_j):
        return body_i != body_j and self.joint_collision_disabled[body_i, body_j] == 0

    @ti.func
    def _material_id(self, material):
        return ti.min(ti.max(material, 0), self.material_num - 1)

    @ti.func
    def _pp_dhat(self, body_i, body_j):
        mi = self._material_id(self.body_material[body_i])
        mj = self._material_id(self.body_material[body_j])
        return self.pp_dhat[mi, mj]

    @ti.func
    def _pp_kappa(self, body_i, body_j):
        mi = self._material_id(self.body_material[body_i])
        mj = self._material_id(self.body_material[body_j])
        return self.pp_kappa[mi, mj]

    @ti.func
    def _pp_penalty(self, body_i, body_j):
        return self.pp_kappa[
            self._material_id(self.body_material[body_i]),
            self._material_id(self.body_material[body_j]),
        ] * ti.static(self.semi_penalty_scale)

    @ti.func
    def _pp_contact_damping(self, body_i, body_j):
        mi = self._material_id(self.body_material[body_i])
        mj = self._material_id(self.body_material[body_j])
        return self.pp_contact_damping[mi, mj]

    @ti.func
    def _pp_mu(self, body_i, body_j):
        mi = self._material_id(self.body_material[body_i])
        mj = self._material_id(self.body_material[body_j])
        mu = self.pp_mu[mi, mj]
        if mu < 0.0:
            mu = ti.max(self.body_mu[body_i], self.body_mu[body_j])
        if ti.static(not self.fully_implicit and self.fully_mu_dynamic >= 0.0):
            mu = self.fully_mu_dynamic
        return mu

    @ti.func
    def _pw_dhat(self, body_i, wall_id):
        mi = self._material_id(self.body_material[body_i])
        mj = self._material_id(self.wall_material[wall_id])
        return self.pw_dhat[mi, mj]

    @ti.func
    def _pw_kappa(self, body_i, wall_id):
        mi = self._material_id(self.body_material[body_i])
        mj = self._material_id(self.wall_material[wall_id])
        return self.pw_kappa[mi, mj]

    @ti.func
    def _pw_penalty(self, body_i, wall_id):
        return self.pw_kappa[
            self._material_id(self.body_material[body_i]),
            self._material_id(self.wall_material[wall_id]),
        ] * ti.static(self.semi_penalty_scale)

    @ti.func
    def _pw_contact_damping(self, body_i, wall_id):
        mi = self._material_id(self.body_material[body_i])
        mj = self._material_id(self.wall_material[wall_id])
        return self.pw_contact_damping[mi, mj]

    @ti.func
    def _pw_mu(self, body_i, wall_id):
        mi = self._material_id(self.body_material[body_i])
        mj = self._material_id(self.wall_material[wall_id])
        mu = self.pw_mu[mi, mj]
        if mu < 0.0:
            mu = ti.max(self.body_mu[body_i], self.wall_mu[wall_id])
        if ti.static(not self.fully_implicit and self.fully_mu_dynamic >= 0.0):
            mu = self.fully_mu_dynamic
        return mu

    @ti.func
    def _closest_point_triangle(self, p, a, b, c):
        return closest_point_triangle(p, a, b, c)

    @ti.func
    def _scatter_distance2_barrier_hessian_four(
        self,
        body0,
        vertex0,
        weight0,
        body1,
        vertex1,
        weight1,
        body2,
        vertex2,
        weight2,
        body3,
        vertex3,
        weight3,
        delta,
        db,
        ddb,
        matrix_target: ti.template(),
    ):
        self._scatter_distance2_barrier_hessian_pair(
            body0, vertex0, weight0, body0, vertex0, weight0, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body0, vertex0, weight0, body1, vertex1, weight1, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body0, vertex0, weight0, body2, vertex2, weight2, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body0, vertex0, weight0, body3, vertex3, weight3, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body1, vertex1, weight1, body0, vertex0, weight0, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body1, vertex1, weight1, body1, vertex1, weight1, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body1, vertex1, weight1, body2, vertex2, weight2, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body1, vertex1, weight1, body3, vertex3, weight3, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body2, vertex2, weight2, body0, vertex0, weight0, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body2, vertex2, weight2, body1, vertex1, weight1, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body2, vertex2, weight2, body2, vertex2, weight2, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body2, vertex2, weight2, body3, vertex3, weight3, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body3, vertex3, weight3, body0, vertex0, weight0, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body3, vertex3, weight3, body1, vertex1, weight1, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body3, vertex3, weight3, body2, vertex2, weight2, delta, db, ddb, matrix_target
        )
        self._scatter_distance2_barrier_hessian_pair(
            body3, vertex3, weight3, body3, vertex3, weight3, delta, db, ddb, matrix_target
        )

    @ti.func
    def _barrier(self, gap):
        energy, dphi, ddphi = self._ipc_barrier_gap(gap, self.dhat, self.kappa)
        energy *= self.scale_device[None]
        dphi *= self.scale_device[None]
        ddphi *= self.scale_device[None]
        return energy, dphi, ddphi

    @ti.func
    def _semi_terms(self, key, gap, penalty):
        slot = semi_ipc_find_or_insert(
            self.semi_state,
            self.semi_key,
            self.semi_multiplier,
            self.semi_count,
            key,
            ti.static(self.semi_capacity),
        )
        multiplier = 0.0
        if slot >= 0:
            multiplier = self.semi_multiplier[slot]
        else:
            self.semi_overflow[None] = 1
        energy, gradient, hessian = semi_ipc_terms(gap, multiplier, penalty)
        ti.atomic_max(self.semi_constraint_violation[None], ti.max(-gap, 0.0))
        return energy, gradient, hessian

    @ti.func
    def _semi_find_terms(self, key, gap, penalty):
        slot = semi_ipc_find(
            self.semi_state,
            self.semi_key,
            key,
            ti.static(self.semi_capacity),
        )
        multiplier = 0.0
        if slot >= 0:
            multiplier = self.semi_multiplier[slot]
        return semi_ipc_terms(gap, multiplier, penalty)

    @ti.kernel
    def _update_semi_multipliers(self):
        self.semi_constraint_violation[None] = 0.0
        offset = ti.Vector([0.25, 0.25, 0.25])
        for slot in range(self.semi_capacity):
            if self.semi_state[slot] == 0:
                key = self.semi_key[slot]
                gap = 0.0
                penalty = self.semi_penalty
                valid = 1
                if key[2] == 0:
                    vertex_id, face_id = key[0], key[1]
                    body_i, body_j = self.node2body[vertex_id], self.face2body[face_id]
                    face = self.faces[face_id]
                    dist2 = point_triangle_distance2_from_type(
                        self.x[vertex_id],
                        self.x[face[0]],
                        self.x[face[1]],
                        self.x[face[2]],
                        point_triangle_distance_type(
                            self.x[vertex_id],
                            self.x[face[0]],
                            self.x[face[1]],
                            self.x[face[2]],
                        ),
                    )
                    gap = ti.sqrt(ti.max(dist2, 0.0)) - self._pp_dhat(body_i, body_j)
                    penalty = self._pp_penalty(body_i, body_j)
                elif key[2] == 1:
                    edge_i, edge_j = key[0], key[1]
                    body_i, body_j = self.edge2body[edge_i], self.edge2body[edge_j]
                    ei, ej = self.edges[edge_i], self.edges[edge_j]
                    dist2 = edge_edge_distance2_from_type(
                        self.x[ei[0]],
                        self.x[ei[1]],
                        self.x[ej[0]],
                        self.x[ej[1]],
                        edge_edge_distance_type(
                            self.x[ei[0]],
                            self.x[ei[1]],
                            self.x[ej[0]],
                            self.x[ej[1]],
                        ),
                    )
                    gap = ti.sqrt(ti.max(dist2, 0.0)) - self._pp_dhat(body_i, body_j)
                    penalty = self._pp_penalty(body_i, body_j)
                elif key[2] == 2:
                    vertex_id, wall_id = key[0], key[1]
                    body = self.node2body[vertex_id]
                    point = self.x[vertex_id]
                    if self.wall_type[wall_id] == 0:
                        gap = (point - self.wall_point[wall_id]).dot(self.wall_normal[wall_id])
                    else:
                        closest, unused_barycentric = self._closest_point_triangle(
                            point,
                            self.wall_v0[wall_id],
                            self.wall_v1[wall_id],
                            self.wall_v2[wall_id],
                        )
                        gap = (point - closest).norm()
                    gap -= self._pw_dhat(body, wall_id)
                    penalty = self._pw_penalty(body, wall_id)
                else:
                    vertex_id, target_body = key[0], key[1]
                    source_body = self.node2body[vertex_id]
                    target_A = self._affine_levelset_matrix(target_body)
                    if ti.abs(target_A.determinant()) > 1.0e-12:
                        material = target_A.inverse() @ (self.x[vertex_id] - self.y[target_body * 4])
                        scale = self.body_scale[target_body]
                        coordinate = (material - offset) / scale
                        phi, unused_gradient, unused_hessian, inside = self._sample_affine_levelset(
                            target_body, coordinate
                        )
                        if inside:
                            gap = scale * phi - self._pp_dhat(source_body, target_body)
                        else:
                            valid = 0
                    else:
                        valid = 0
                    penalty = self._pp_penalty(source_body, target_body)
                if valid != 0:
                    self.semi_multiplier[slot] = semi_ipc_update_multiplier(gap, self.semi_multiplier[slot], penalty)
                    ti.atomic_max(self.semi_constraint_violation[None], ti.max(-gap, 0.0))
                else:
                    self.semi_multiplier[slot] = 0.0

    def accept_semi_update_device(self):
        if not self.is_semi:
            return
        self._reconstruct_vertices()
        self._update_semi_multipliers()

    def semi_contact_converged(self):
        return not self.is_semi or float(self.semi_constraint_violation[None]) <= self.semi_constraint_tolerance

    @ti.func
    def _ipc_barrier_gap(self, gap, active_gap, kappa):
        energy, gradient, hessian = ipc_toolkit_barrier_distance_terms(gap, active_gap, kappa)
        # Unlike an unsigned point/triangle distance, a level-set or plane
        # gap is signed.  Squaring a negative gap would otherwise make an
        # interpenetrating trial look feasible to the IPC line search.
        if gap <= 0.0:
            energy = ti.math.inf
        return energy, gradient, hessian

    @ti.func
    def _barrier_distance2(self, dist2):
        energy, db, ddb = self._ipc_barrier_distance2(dist2, self.dhat * self.dhat, self.kappa)
        energy *= self.scale_device[None]
        db *= self.scale_device[None]
        ddb *= self.scale_device[None]
        return energy, db, ddb

    @ti.func
    def _ipc_barrier_distance2(self, dist2, active_gap2, kappa):
        return ipc_toolkit_barrier_distance2_terms(dist2, active_gap2, kappa)

    @ti.func
    def _point_triangle_tangent(self, p, t0, t1, t2, dtype):
        _, barycentric, normal = point_triangle_contact_frame(p, t0, t1, t2, dtype)
        return barycentric[1], barycentric[2], normal

    @ti.func
    def _edge_edge_tangent(self, ea0, ea1, eb0, eb1, dtype):
        _, _, parameter_a, parameter_b, normal = edge_edge_contact_frame(ea0, ea1, eb0, eb1, dtype)
        return parameter_a, parameter_b, normal

    @ti.func
    def _scatter_local_gradient_hessian(
        self,
        bodies: ti.template(),
        vertices: ti.template(),
        local_grad: ti.template(),
        local_hess: ti.template(),
        need_matrix: ti.template(),
        matrix_target: ti.template(),
    ):
        self._scatter_local_gradient(bodies, vertices, local_grad)
        if ti.static(need_matrix):
            self._scatter_local_hessian(bodies, vertices, local_hess, matrix_target)

    @ti.func
    def _scatter_local_gradient(self, bodies: ti.template(), vertices: ti.template(), local_grad: ti.template()):
        for li in range(4):
            body_i = bodies[li]
            vertex_i = vertices[li]
            for ai in range(4):
                wi = self.basis[vertex_i, ai]
                ci = body_i * 4 + ai
                for di in range(3):
                    ti.atomic_add(self.grad[ci][di], wi * local_grad[li * 3 + di])

    @ti.func
    def _scatter_local_hessian(
        self, bodies: ti.template(), vertices: ti.template(), local_hess: ti.template(), matrix_target: ti.template()
    ):
        for li in range(4):
            for lj in range(4):
                block = ti.Matrix.zero(float, 3, 3)
                for di, dj in ti.static(ti.ndrange(3, 3)):
                    block[di, dj] = local_hess[li * 3 + di, lj * 3 + dj]
                self._scatter_local_hessian_block(bodies, vertices, li, lj, block, matrix_target)

    @ti.func
    def _scatter_local_hessian_block(
        self,
        bodies: ti.template(),
        vertices: ti.template(),
        local_i,
        local_j,
        block: ti.template(),
        matrix_target: ti.template(),
    ):
        body_i = bodies[local_i]
        body_j = bodies[local_j]
        vertex_i = vertices[local_i]
        vertex_j = vertices[local_j]
        for ai in range(4):
            wi = self.basis[vertex_i, ai]
            ci = body_i * 4 + ai
            for aj in range(4):
                wj = self.basis[vertex_j, aj]
                cj = body_j * 4 + aj
                self._add_matrix_block(ci, cj, wi * wj * block, matrix_target)

    @ti.func
    def _reserve_body_pair_segment(
        self,
        storage_target: ti.template(),
    ):
        base = 0
        if ti.static(storage_target == MATRIX_HASH_TRIPLET):
            base = ti.atomic_add(self.hash_triplet.raw_non_diag_count[0], 128)
            if base + 128 > self.hash_triplet.non_diag.blockH.shape[0]:
                self.hash_triplet.overflow[0] = 1
        else:
            base = ti.atomic_add(self.coo_entry_count[None], 1152)
            if base + 1152 > self.max_coo_entries:
                self.coo_overflow[None] = 1
        return base

    @ti.func
    def _initialize_body_pair_segment(
        self,
        base,
        body_i,
        body_j,
        storage_target: ti.template(),
    ):
        """Initialize H and its Jacobi workspace inside the final raw array."""
        if ti.static(storage_target == MATRIX_HASH_TRIPLET):
            if base + 128 <= self.hash_triplet.non_diag.blockH.shape[0]:
                block_row = 0
                while block_row < 8:
                    block_column = 0
                    while block_column < 8:
                        local_block = block_row * 8 + block_column
                        global_i = (body_i if block_row < 4 else body_j) * 4 + block_row % 4
                        global_j = (body_i if block_column < 4 else body_j) * 4 + block_column % 4
                        self.hash_triplet.initialize_raw_block_slot(base + local_block, global_i, global_j)
                        self.hash_triplet.initialize_raw_block_slot(base + 64 + local_block, 0, 1)
                        if block_row == block_column:
                            identity = ti.Matrix.identity(float, 3)
                            self.hash_triplet.atomic_add_raw_block_slot(base + 64 + local_block, identity)
                        block_column += 1
                    block_row += 1
                self.hash_triplet.non_diag.blockI[base + 64] = -1
        else:
            if base + 1152 <= self.max_coo_entries:
                row = 0
                while row < 24:
                    column = 0
                    while column < 24:
                        local_entry = row * 24 + column
                        global_row = (body_i if row < 12 else body_j) * 12 + row % 12
                        global_column = (body_i if column < 12 else body_j) * 12 + column % 12
                        self.K_coo_rows[base + local_entry] = global_row
                        self.K_coo_cols[base + local_entry] = global_column
                        self.K_coo_values[base + local_entry] = 0.0
                        workspace = base + 576 + local_entry
                        self.K_coo_rows[workspace] = 0
                        self.K_coo_cols[workspace] = 0
                        self.K_coo_values[workspace] = ti.cast(row == column, float)
                        column += 1
                    row += 1
                self.K_coo_rows[base + 576] = -1

    @ti.func
    def _body_pair_dense_get(self, base, row, column, storage_target: ti.template()):
        value = 0.0
        if ti.static(storage_target == MATRIX_HASH_TRIPLET):
            block = base + (row // 3) * 8 + column // 3
            component = (row % 3) * 3 + column % 3
            value = self.hash_triplet.non_diag.blockH[block][component]
        else:
            value = self.K_coo_values[base + row * 24 + column]
        return value

    @ti.func
    def _body_pair_dense_set(self, base, row, column, value, storage_target: ti.template()):
        if ti.static(storage_target == MATRIX_HASH_TRIPLET):
            block = base + (row // 3) * 8 + column // 3
            component = (row % 3) * 3 + column % 3
            self.hash_triplet.non_diag.blockH[block][component] = value
        else:
            self.K_coo_values[base + row * 24 + column] = value

    @ti.func
    def _body_pair_dense_add(self, base, row, column, value, storage_target: ti.template()):
        if ti.static(storage_target == MATRIX_HASH_TRIPLET):
            block = base + (row // 3) * 8 + column // 3
            component = (row % 3) * 3 + column % 3
            ti.atomic_add(self.hash_triplet.non_diag.blockH[block][component], value)
        else:
            ti.atomic_add(self.K_coo_values[base + row * 24 + column], value)

    @ti.func
    def _project_body_pair_segment(
        self,
        base,
        storage_target: ti.template(),
        output_target: ti.template(),
        project_pd: ti.template(),
    ):
        """Jacobi-clamp one raw 24x24 segment entirely on the device."""
        workspace = base + 576
        if ti.static(storage_target == MATRIX_HASH_TRIPLET):
            workspace = base + 64
        if ti.static(project_pd):
            row = 0
            while row < 24:
                column = row + 1
                while column < 24:
                    value = 0.5 * (
                        self._body_pair_dense_get(base, row, column, storage_target)
                        + self._body_pair_dense_get(base, column, row, storage_target)
                    )
                    self._body_pair_dense_set(base, row, column, value, storage_target)
                    self._body_pair_dense_set(base, column, row, value, storage_target)
                    column += 1
                row += 1

            sweep = 0
            while sweep < 24:
                p = 0
                while p < 24:
                    q = p + 1
                    while q < 24:
                        app = self._body_pair_dense_get(base, p, p, storage_target)
                        aqq = self._body_pair_dense_get(base, q, q, storage_target)
                        apq = self._body_pair_dense_get(base, p, q, storage_target)
                        pivot_scale = ti.max(ti.abs(app) + ti.abs(aqq), 1.0e-30)
                        if ti.abs(apq) > 1.0e-14 * pivot_scale:
                            tau = (aqq - app) / (2.0 * apq)
                            sign_tau = 1.0
                            if tau < 0.0:
                                sign_tau = -1.0
                            tangent = sign_tau / (ti.abs(tau) + ti.sqrt(1.0 + tau * tau))
                            cosine = 1.0 / ti.sqrt(1.0 + tangent * tangent)
                            sine = tangent * cosine

                            k = 0
                            while k < 24:
                                if k != p and k != q:
                                    akp = self._body_pair_dense_get(base, k, p, storage_target)
                                    akq = self._body_pair_dense_get(base, k, q, storage_target)
                                    rotated_kp = cosine * akp - sine * akq
                                    rotated_kq = sine * akp + cosine * akq
                                    self._body_pair_dense_set(base, k, p, rotated_kp, storage_target)
                                    self._body_pair_dense_set(base, p, k, rotated_kp, storage_target)
                                    self._body_pair_dense_set(base, k, q, rotated_kq, storage_target)
                                    self._body_pair_dense_set(base, q, k, rotated_kq, storage_target)
                                vkp = self._body_pair_dense_get(workspace, k, p, storage_target)
                                vkq = self._body_pair_dense_get(workspace, k, q, storage_target)
                                self._body_pair_dense_set(
                                    workspace,
                                    k,
                                    p,
                                    cosine * vkp - sine * vkq,
                                    storage_target,
                                )
                                self._body_pair_dense_set(
                                    workspace,
                                    k,
                                    q,
                                    sine * vkp + cosine * vkq,
                                    storage_target,
                                )
                                k += 1

                            self._body_pair_dense_set(
                                base,
                                p,
                                p,
                                app - tangent * apq,
                                storage_target,
                            )
                            self._body_pair_dense_set(
                                base,
                                q,
                                q,
                                aqq + tangent * apq,
                                storage_target,
                            )
                            self._body_pair_dense_set(base, p, q, 0.0, storage_target)
                            self._body_pair_dense_set(base, q, p, 0.0, storage_target)
                        q += 1
                    p += 1
                sweep += 1

            # Reconstruct off-diagonals while H's diagonal still stores all
            # eigenvalues.  Then stage each projected diagonal in its own V
            # row, which is no longer needed by any other diagonal.
            row = 0
            while row < 24:
                column = 0
                while column < 24:
                    if row != column:
                        value = 0.0
                        eigenmode = 0
                        while eigenmode < 24:
                            value += (
                                ti.max(
                                    self._body_pair_dense_get(
                                        base,
                                        eigenmode,
                                        eigenmode,
                                        storage_target,
                                    ),
                                    0.0,
                                )
                                * self._body_pair_dense_get(workspace, row, eigenmode, storage_target)
                                * self._body_pair_dense_get(
                                    workspace,
                                    column,
                                    eigenmode,
                                    storage_target,
                                )
                            )
                            eigenmode += 1
                        self._body_pair_dense_set(base, row, column, value, storage_target)
                    column += 1
                row += 1

            row = 0
            while row < 24:
                value = 0.0
                eigenmode = 0
                while eigenmode < 24:
                    component = self._body_pair_dense_get(workspace, row, eigenmode, storage_target)
                    value += (
                        ti.max(
                            self._body_pair_dense_get(
                                base,
                                eigenmode,
                                eigenmode,
                                storage_target,
                            ),
                            0.0,
                        )
                        * component
                        * component
                    )
                    eigenmode += 1
                self._body_pair_dense_set(workspace, row, 0, value, storage_target)
                row += 1
            row = 0
            while row < 24:
                self._body_pair_dense_set(
                    base,
                    row,
                    row,
                    self._body_pair_dense_get(workspace, row, 0, storage_target),
                    storage_target,
                )
                row += 1

        if ti.static(storage_target == MATRIX_HASH_TRIPLET):
            block_row = 0
            while block_row < 8:
                block_column = 0
                while block_column < 8:
                    raw_index = base + block_row * 8 + block_column
                    block_i = self.hash_triplet.non_diag.blockI[raw_index]
                    block_j = self.hash_triplet.non_diag.blockJ[raw_index]
                    clear = 0
                    if ti.static(output_target == MATRIX_CONTACT_DAMPING):
                        if block_row <= block_column:
                            block = ti.Matrix.zero(float, 3, 3)
                            for component_i, component_j in ti.static(ti.ndrange(3, 3)):
                                block[component_i, component_j] = self.hash_triplet.non_diag.blockH[raw_index][
                                    component_i * 3 + component_j
                                ]
                            self._append_contact_damping_block(block_i, block_j, block)
                        clear = 1
                    elif block_row == block_column:
                        ti.atomic_add(
                            self.hash_triplet.diag[block_i],
                            self.hash_triplet.non_diag.blockH[raw_index],
                        )
                        clear = 1
                    elif ti.static(self.hash_triplet.matrix_symmetric and not self.hash_triplet.full_symmetric_input):
                        if block_row > block_column:
                            clear = 1
                    if clear != 0:
                        self.hash_triplet.non_diag.blockI[raw_index] = 0
                        self.hash_triplet.non_diag.blockJ[raw_index] = 1
                        self.hash_triplet.non_diag.blockH[raw_index] = ti.Vector.zero(float, 9)
                    block_column += 1
                block_row += 1
            workspace_block = 0
            while workspace_block < 64:
                raw_index = workspace + workspace_block
                self.hash_triplet.non_diag.blockI[raw_index] = 0
                self.hash_triplet.non_diag.blockJ[raw_index] = 1
                self.hash_triplet.non_diag.blockH[raw_index] = ti.Vector.zero(float, 9)
                workspace_block += 1
        else:
            if ti.static(output_target == MATRIX_CONTACT_DAMPING):
                block_row = 0
                while block_row < 8:
                    block_column = block_row
                    while block_column < 8:
                        row = block_row * 3
                        column = block_column * 3
                        index = base + row * 24 + column
                        block = ti.Matrix.zero(float, 3, 3)
                        for component_i, component_j in ti.static(ti.ndrange(3, 3)):
                            block[component_i, component_j] = self.K_coo_values[
                                base + (row + component_i) * 24 + column + component_j
                            ]
                        self._append_contact_damping_block(
                            self.K_coo_rows[index] // 3,
                            self.K_coo_cols[index] // 3,
                            block,
                        )
                        block_column += 1
                    block_row += 1
                entry = 0
                while entry < 576:
                    self.K_coo_values[base + entry] = 0.0
                    entry += 1
            else:
                entry = 0
                while entry < 576:
                    index = base + entry
                    if self.K_coo_rows[index] == self.K_coo_cols[index]:
                        ti.atomic_add(
                            self.K_coo_diag[self.K_coo_rows[index]],
                            self.K_coo_values[index],
                        )
                    entry += 1
            entry = 0
            while entry < 576:
                index = workspace + entry
                self.K_coo_rows[index] = 0
                self.K_coo_cols[index] = 0
                self.K_coo_values[index] = 0.0
                entry += 1

    @ti.func
    def _add_matrix_block(self, block_i, block_j, block, matrix_target: ti.template()):
        if ti.static(matrix_target == MATRIX_CONTACT_DAMPING):
            self._append_contact_damping_block(block_i, block_j, block)
        elif ti.static(matrix_target == MATRIX_HASH_TRIPLET):
            if ti.static(self.fully_implicit or self.hash_store_full_symmetric_input):
                self.hash_triplet.add_block_entry(block_i, block_j, block)
            elif block_i <= block_j:
                self.hash_triplet.add_block_entry(block_i, block_j, block)
        else:
            for row, column in ti.static(ti.ndrange(3, 3)):
                self._add_matrix_entry(
                    block_i * 3 + row,
                    block_j * 3 + column,
                    block[row, column],
                    matrix_target,
                )

    @ti.func
    def _append_contact_damping_block(self, block_i, block_j, block):
        if block_i <= block_j:
            index = ti.atomic_add(self.contact_damping_count[None], 1)
            if index < self.contact_damping_capacity:
                self.contact_damping_block_i[index] = block_i
                self.contact_damping_block_j[index] = block_j
                values = ti.Vector.zero(float, 9)
                for row, column in ti.static(ti.ndrange(3, 3)):
                    values[row * 3 + column] = block[row, column]
                self.contact_damping_block_h[index] = values
            else:
                self.contact_damping_overflow[None] = 1

    @ti.func
    def _add_matrix_entry(self, row, col, value, matrix_target: ti.template()):
        if ti.static(matrix_target == MATRIX_COO):
            idx = ti.atomic_add(self.coo_entry_count[None], 1)
            if idx < self.max_coo_entries:
                self.K_coo_rows[idx] = row
                self.K_coo_cols[idx] = col
                self.K_coo_values[idx] = value
                if row == col:
                    ti.atomic_add(self.K_coo_diag[row], value)
            else:
                self.coo_overflow[None] = 1

    @ti.func
    def _scatter_local_barrier_hessian(
        self,
        bodies: ti.template(),
        vertices: ti.template(),
        grad_d: ti.template(),
        hess_d: ti.template(),
        hess_coeff,
        outer_coeff,
        matrix_target: ti.template(),
    ):
        local_hessian = ti.Matrix.zero(float, 12, 12)
        for row, column in ti.ndrange(12, 12):
            local_hessian[row, column] = hess_coeff * hess_d[row, column] + outer_coeff * grad_d[row] * grad_d[column]
        if ti.static(not self.fully_implicit):
            local_hessian = psd_project_nd(local_hessian)
        self._scatter_local_hessian(bodies, vertices, local_hessian, matrix_target)

    @ti.func
    def _assemble_edge_barrier_hessian_compact(
        self,
        edge_i,
        edge_j,
        body_i,
        body_j,
        ei: ti.template(),
        ej: ti.template(),
        p0,
        p1,
        q0,
        q1,
        dist2,
        ids: ti.template(),
        n: ti.template(),
        grad_d: ti.template(),
        hess_d: ti.template(),
        matrix_scale,
        matrix_target: ti.template(),
    ):
        dhat = self._pp_dhat(body_i, body_j)
        kappa = self._pp_kappa(body_i, body_j)
        pair_scale = matrix_scale
        if ti.static(matrix_target == MATRIX_CONTACT_DAMPING):
            pair_scale = self.dt_device[None] * self._pp_contact_damping(body_i, body_j)
        active_gap2 = dhat * dhat
        if pair_scale > 0.0 and dist2 < active_gap2:
            base_coeff = pair_scale * 0.25 * (self.edge_area[edge_i] + self.edge_area[edge_j])
            energy, db, ddb = self._ipc_barrier_distance2(dist2, active_gap2, kappa)
            eps_x = edge_edge_mollifier_threshold(
                self.rest_x[ei[0]], self.rest_x[ei[1]], self.rest_x[ej[0]], self.rest_x[ej[1]]
            )
            mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
            if mollifier >= 1.0:
                local_hessian = ti.Matrix.zero(float, 12, 12)
                for compact_i in range(n):
                    local_i = ids[compact_i]
                    for compact_j in range(n):
                        local_j = ids[compact_j]
                        for di, dj in ti.static(ti.ndrange(3, 3)):
                            ii = compact_i * 3 + di
                            jj = compact_j * 3 + dj
                            row = local_i * 3 + di
                            column = local_j * 3 + dj
                            local_hessian[row, column] = base_coeff * (
                                db * hess_d[ii, jj] + ddb * grad_d[ii] * grad_d[jj]
                            )
                if ti.static(not self.fully_implicit):
                    local_hessian = psd_project_nd(local_hessian)
                bodies = ti.Vector([body_i, body_i, body_j, body_j])
                vertices = ti.Vector([ei[0], ei[1], ej[0], ej[1]])
                self._scatter_local_hessian(bodies, vertices, local_hessian, matrix_target)

    @ti.func
    def _scatter_contact(self, body_id, vertex_id, local_weight, normal, dphi, ddphi, need_matrix: ti.template()):
        for a in range(4):
            weight = local_weight * self.basis[vertex_id, a]
            control = body_id * 4 + a
            for d in range(3):
                ti.atomic_add(self.grad[control][d], dphi * weight * normal[d])

    @ti.func
    def _scatter_hessian_pair(
        self, body_i, vertex_i, local_i, body_j, vertex_j, local_j, normal, ddphi, matrix_target: ti.template()
    ):
        for ai in range(4):
            wi = local_i * self.basis[vertex_i, ai]
            for aj in range(4):
                wj = local_j * self.basis[vertex_j, aj]
                ci = body_i * 4 + ai
                cj = body_j * 4 + aj
                self._add_matrix_block(
                    ci,
                    cj,
                    ddphi * wi * wj * normal.outer_product(normal),
                    matrix_target,
                )

    @ti.func
    def _scatter_gradient_vec(self, body_id, vertex_id, local_weight, vec):
        for a in range(4):
            w = local_weight * self.basis[vertex_id, a]
            control = body_id * 4 + a
            for d in range(3):
                ti.atomic_add(self.grad[control][d], w * vec[d])

    @ti.func
    def _scatter_hessian_block(
        self, body_i, vertex_i, local_i, body_j, vertex_j, local_j, H, matrix_target: ti.template()
    ):
        for ai in range(4):
            wi = local_i * self.basis[vertex_i, ai]
            for aj in range(4):
                wj = local_j * self.basis[vertex_j, aj]
                ci = body_i * 4 + ai
                cj = body_j * 4 + aj
                self._add_matrix_block(ci, cj, wi * wj * H, matrix_target)

    @ti.func
    def _scatter_hessian_pair_contact_damping(
        self, body_i, vertex_i, local_i, body_j, vertex_j, local_j, normal, ddphi
    ):
        for ai in range(4):
            wi = local_i * self.basis[vertex_i, ai]
            for aj in range(4):
                wj = local_j * self.basis[vertex_j, aj]
                ci = body_i * 4 + ai
                cj = body_j * 4 + aj
                self._add_matrix_block(
                    ci,
                    cj,
                    ddphi * wi * wj * normal.outer_product(normal),
                    MATRIX_CONTACT_DAMPING,
                )

    @ti.func
    def _scatter_hessian_block_contact_damping(self, body_i, vertex_i, local_i, body_j, vertex_j, local_j, H):
        for ai in range(4):
            wi = local_i * self.basis[vertex_i, ai]
            for aj in range(4):
                wj = local_j * self.basis[vertex_j, aj]
                ci = body_i * 4 + ai
                cj = body_j * 4 + aj
                self._add_matrix_block(ci, cj, wi * wj * H, MATRIX_CONTACT_DAMPING)

    @ti.func
    def _scatter_distance2_barrier_hessian_pair(
        self, body_i, vertex_i, local_i, body_j, vertex_j, local_j, delta, db, ddb, matrix_target: ti.template()
    ):
        H = 4.0 * ddb * delta.outer_product(delta)
        H += 2.0 * db * ti.Matrix.identity(float, 3)
        self._scatter_hessian_block(body_i, vertex_i, local_i, body_j, vertex_j, local_j, H, matrix_target)

    @ti.func
    def _friction_f0(self, vbarnorm, epsv, hat_h):
        return ipc_friction_f0(vbarnorm, epsv, hat_h)

    @ti.func
    def _friction_f1_div_vbarnorm(self, vbarnorm, epsv):
        return ipc_friction_f1_over_speed(vbarnorm, epsv)

    @ti.func
    def _lagged_friction_hessian(self, rel, hat_rel, normal, coeff):
        unit_normal = normal / normal.norm()
        projection = ti.Matrix.identity(float, 3) - unit_normal.outer_product(unit_normal)
        velocity = projection @ ((rel - hat_rel) / self.dt_device[None])
        speed = velocity.norm()
        f1 = self._friction_f1_div_vbarnorm(speed, self.epsv)
        axis = ti.Vector([1.0, 0.0, 0.0])
        if ti.abs(unit_normal[0]) >= 0.9:
            axis = ti.Vector([0.0, 1.0, 0.0])
        tangent0 = unit_normal.cross(axis)
        tangent0 /= tangent0.norm()
        tangent1 = unit_normal.cross(tangent0)
        tangent1 /= tangent1.norm()
        hessian = coeff * f1 * (tangent0.outer_product(tangent0) + tangent1.outer_product(tangent1))
        if speed > 0.0:
            direction = velocity / speed
            transverse = unit_normal.cross(direction)
            transverse /= transverse.norm()
            radial = 0.0
            if speed < self.epsv:
                radial = 2.0 * (self.epsv - speed) / (self.epsv * self.epsv)
            hessian = coeff * (f1 * transverse.outer_product(transverse) + radial * direction.outer_product(direction))
        return hessian / self.dt_device[None]

    @ti.kernel
    def _freeze_levelset_lagged_friction_geometry(self):
        for control in range(self.control_num):
            self.levelset_friction_y[control] = self.y[control]

    @ti.func
    def _scatter_levelset_friction_gradient(self, source_body, target_body, weights, vector):
        for site in ti.static(range(8)):
            body = source_body if site < 4 else target_body
            control = site if site < 4 else site - 4
            for component in ti.static(range(3)):
                ti.atomic_add(
                    self.grad[body * 4 + control][component],
                    weights[site] * vector[component],
                )

    @ti.func
    def _scatter_levelset_friction_hessian(
        self,
        source_body,
        target_body,
        weights,
        local_hessian,
        matrix_target: ti.template(),
    ):
        for site_i, site_j in ti.ndrange(8, 8):
            body_i = source_body if site_i < 4 else target_body
            control_i = site_i if site_i < 4 else site_i - 4
            body_j = source_body if site_j < 4 else target_body
            control_j = site_j if site_j < 4 else site_j - 4
            self._add_matrix_block(
                body_i * 4 + control_i,
                body_j * 4 + control_j,
                weights[site_i] * weights[site_j] * local_hessian,
                matrix_target,
            )

    @ti.func
    def _assemble_levelset_local_friction(
        self,
        source_body,
        target_body,
        weights,
        relative,
        reference_relative,
        normal,
        coefficient,
        need_matrix: ti.template(),
        matrix_target: ti.template(),
    ):
        normal_norm = normal.norm()
        if coefficient > 0.0 and normal_norm > 1.0e-14:
            unit_normal = normal / normal_norm
            projector = ti.Matrix.identity(float, 3) - unit_normal.outer_product(unit_normal)
            tangential_velocity = projector @ ((relative - reference_relative) / self.dt_device[None])
            speed = tangential_velocity.norm()
            ti.atomic_add(
                self.energy[None],
                coefficient * self._friction_f0(speed, self.epsv, self.dt_device[None]),
            )
            f1_over_speed = self._friction_f1_div_vbarnorm(speed, self.epsv)
            local_gradient = coefficient * f1_over_speed * (projector @ tangential_velocity)
            self._scatter_levelset_friction_gradient(
                source_body,
                target_body,
                weights,
                local_gradient,
            )
            if ti.static(need_matrix):
                axis = ti.Vector([1.0, 0.0, 0.0])
                if ti.abs(unit_normal[0]) >= 0.9:
                    axis = ti.Vector([0.0, 1.0, 0.0])
                tangent0 = unit_normal.cross(axis)
                tangent0 /= tangent0.norm()
                tangent1 = unit_normal.cross(tangent0)
                tangent1 /= tangent1.norm()
                local_hessian = (
                    coefficient * f1_over_speed * (tangent0.outer_product(tangent0) + tangent1.outer_product(tangent1))
                )
                if speed > 0.0:
                    direction = tangential_velocity / speed
                    transverse = unit_normal.cross(direction)
                    transverse /= transverse.norm()
                    radial_eigenvalue = 0.0
                    if speed < self.epsv:
                        radial_eigenvalue = 2.0 * (self.epsv - speed) / (self.epsv * self.epsv)
                    local_hessian = coefficient * (
                        f1_over_speed * transverse.outer_product(transverse)
                        + radial_eigenvalue * direction.outer_product(direction)
                    )
                local_hessian /= self.dt_device[None]
                self._scatter_levelset_friction_hessian(
                    source_body,
                    target_body,
                    weights,
                    local_hessian,
                    matrix_target,
                )

    @ti.func
    def _scatter_friction_pair(
        self, body_i, vertex_i, local_i, body_j, vertex_j, local_j, H, matrix_target: ti.template()
    ):
        self._scatter_hessian_block(body_i, vertex_i, local_i, body_j, vertex_j, local_j, H, matrix_target)

    @ti.func
    def _store_lagged_friction_contact(self, bodies, vertices, weights, normal, hat_rel, coeff):
        contact = ti.atomic_add(self.friction_contact_count[0], 1)
        if contact < self.friction_contact_capacity:
            self.friction_contact_bodies[contact] = bodies
            self.friction_contact_vertices[contact] = vertices
            self.friction_contact_weights[contact] = weights
            self.friction_contact_normal[contact] = normal
            self.friction_contact_hat_rel[contact] = hat_rel
            self.friction_contact_coeff[contact] = coeff
        else:
            self.friction_contact_overflow[0] = 1

    @ti.kernel
    def device_backup_lagged_friction_for_adjoint(self):
        count = ti.min(self.friction_contact_count[0], self.friction_contact_capacity)
        self.adjoint_friction_contact_count[None] = count
        for contact in range(count):
            self.adjoint_friction_contact_bodies[contact] = self.friction_contact_bodies[contact]
            self.adjoint_friction_contact_vertices[contact] = self.friction_contact_vertices[contact]
            self.adjoint_friction_contact_weights[contact] = self.friction_contact_weights[contact]
            self.adjoint_friction_contact_normal[contact] = self.friction_contact_normal[contact]
            self.adjoint_friction_contact_hat_rel[contact] = self.friction_contact_hat_rel[contact]
            self.adjoint_friction_contact_coeff[contact] = self.friction_contact_coeff[contact]

    @ti.kernel
    def device_restore_lagged_friction_for_adjoint(self):
        count = self.adjoint_friction_contact_count[None]
        self.friction_contact_count[0] = count
        self.friction_contact_overflow[0] = 0
        for contact in range(count):
            self.friction_contact_bodies[contact] = self.adjoint_friction_contact_bodies[contact]
            self.friction_contact_vertices[contact] = self.adjoint_friction_contact_vertices[contact]
            self.friction_contact_weights[contact] = self.adjoint_friction_contact_weights[contact]
            self.friction_contact_normal[contact] = self.adjoint_friction_contact_normal[contact]
            self.friction_contact_hat_rel[contact] = self.adjoint_friction_contact_hat_rel[contact]
            self.friction_contact_coeff[contact] = self.adjoint_friction_contact_coeff[contact]

    @ti.kernel
    def _initialize_lagged_mesh_friction(
        self,
        candidate_count: ti.template(),
        candidate_vertex: ti.template(),
        candidate_face: ti.template(),
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
    ):
        self.friction_contact_count[0] = 0
        self.friction_contact_overflow[0] = 0
        for candidate_id in range(candidate_count[None]):
            vertex_id = candidate_vertex[candidate_id]
            face_id = candidate_face[candidate_id]
            body_i = self.node2body[vertex_id]
            body_j = self.face2body[face_id]
            if self._body_pair_allowed(body_i, body_j):
                face = self.faces[face_id]
                p = self.x[vertex_id]
                a = self.x[face[0]]
                b = self.x[face[1]]
                c = self.x[face[2]]
                dist2, unused_grad, dtype = point_triangle_distance_grad(p, a, b, c)
                dhat = self._pp_dhat(body_i, body_j)
                mu = self._pp_mu(body_i, body_j)
                if mu > 0.0 and dist2 < dhat * dhat:
                    u, v, normal = self._point_triangle_tangent(p, a, b, c, dtype)
                    weights = ti.Vector([1.0, -(1.0 - u - v), -u, -v])
                    vertices = ti.Vector([vertex_id, face[0], face[1], face[2]])
                    bodies = ti.Vector([body_i, body_j, body_j, body_j])
                    hat_rel = (
                        weights[0] * self.hat_x[vertices[0]]
                        + weights[1] * self.hat_x[vertices[1]]
                        + weights[2] * self.hat_x[vertices[2]]
                        + weights[3] * self.hat_x[vertices[3]]
                    )
                    distance = ti.sqrt(ti.max(dist2, 1.0e-30))
                    normal_force = 0.0
                    if ti.static(self.is_semi):
                        unused_energy, first, unused_hessian = self._semi_terms(
                            ti.Vector([vertex_id, face_id, 0, -1]),
                            distance - dhat,
                            self._pp_penalty(body_i, body_j),
                        )
                        normal_force = ti.max(-first, 0.0)
                    else:
                        unused_energy, db, unused_hessian = self._ipc_barrier_distance2(
                            dist2, dhat * dhat, self._pp_kappa(body_i, body_j)
                        )
                        normal_force = ti.max(-2.0 * db * distance, 0.0)
                    coeff = mu * normal_force * 0.25 * self.node_area[vertex_id] * self.scale_device[None]
                    self._store_lagged_friction_contact(bodies, vertices, weights, normal, hat_rel, coeff)

        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            body_i = self.edge2body[edge_i]
            body_j = self.edge2body[edge_j]
            if self._body_pair_allowed(body_i, body_j):
                ei = self.edges[edge_i]
                ej = self.edges[edge_j]
                p0 = self.x[ei[0]]
                p1 = self.x[ei[1]]
                q0 = self.x[ej[0]]
                q1 = self.x[ej[1]]
                dtype = edge_edge_distance_type(p0, p1, q0, q1)
                dist2 = edge_edge_distance2_from_type(p0, p1, q0, q1, dtype)
                dhat = self._pp_dhat(body_i, body_j)
                mu = self._pp_mu(body_i, body_j)
                eps_x = edge_edge_mollifier_threshold(
                    self.rest_x[ei[0]],
                    self.rest_x[ei[1]],
                    self.rest_x[ej[0]],
                    self.rest_x[ej[1]],
                )
                mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                if mu > 0.0 and dist2 < dhat * dhat and mollifier >= 1.0:
                    s, t, normal = self._edge_edge_tangent(p0, p1, q0, q1, dtype)
                    weights = ti.Vector([1.0 - s, s, -(1.0 - t), -t])
                    vertices = ti.Vector([ei[0], ei[1], ej[0], ej[1]])
                    bodies = ti.Vector([body_i, body_i, body_j, body_j])
                    hat_rel = (
                        weights[0] * self.hat_x[vertices[0]]
                        + weights[1] * self.hat_x[vertices[1]]
                        + weights[2] * self.hat_x[vertices[2]]
                        + weights[3] * self.hat_x[vertices[3]]
                    )
                    distance = ti.sqrt(ti.max(dist2, 1.0e-30))
                    normal_force = 0.0
                    if ti.static(self.is_semi):
                        unused_energy, first, unused_hessian = self._semi_terms(
                            ti.Vector(
                                [
                                    ti.min(edge_i, edge_j),
                                    ti.max(edge_i, edge_j),
                                    1,
                                    -1,
                                ]
                            ),
                            distance - dhat,
                            self._pp_penalty(body_i, body_j),
                        )
                        normal_force = ti.max(-first, 0.0)
                    else:
                        unused_energy, db, unused_hessian = self._ipc_barrier_distance2(
                            dist2, dhat * dhat, self._pp_kappa(body_i, body_j)
                        )
                        normal_force = ti.max(-2.0 * db * distance, 0.0)
                    coeff = (
                        mu
                        * normal_force
                        * 0.25
                        * (self.edge_area[edge_i] + self.edge_area[edge_j])
                        * self.scale_device[None]
                    )
                    self._store_lagged_friction_contact(bodies, vertices, weights, normal, hat_rel, coeff)

    @ti.kernel
    def _initialize_lagged_wall_friction(self):
        """Freeze wall normals and normal-force magnitudes for one inner solve."""
        for vertex_id, wall_id in ti.ndrange(self.vertex_num, self.wall_num):
            body_i = self.node2body[vertex_id]
            p = self.x[vertex_id]
            dhat = self._pw_dhat(body_i, wall_id)
            mu = self._pw_mu(body_i, wall_id)
            normal = self.wall_normal[wall_id]
            normal_force = 0.0
            active = False
            if mu > 0.0:
                if self.wall_type[wall_id] == 0:
                    gap = (p - self.wall_point[wall_id]).dot(normal)
                    if gap < dhat:
                        dphi = 0.0
                        if ti.static(self.is_semi):
                            unused_energy, dphi, unused_hessian = self._semi_terms(
                                ti.Vector([vertex_id, wall_id, 2, -1]),
                                gap - dhat,
                                self._pw_penalty(body_i, wall_id),
                            )
                        else:
                            unused_energy, dphi, unused_hessian = self._ipc_barrier_gap(
                                gap, dhat, self._pw_kappa(body_i, wall_id)
                            )
                        normal_force = ti.max(-self.node_area[vertex_id] * dphi, 0.0)
                        active = True
                else:
                    dist2, unused_grad, dtype = point_triangle_distance_grad(
                        p,
                        self.wall_v0[wall_id],
                        self.wall_v1[wall_id],
                        self.wall_v2[wall_id],
                    )
                    if dist2 < dhat * dhat:
                        db = 0.0
                        unused_u, unused_v, normal = self._point_triangle_tangent(
                            p,
                            self.wall_v0[wall_id],
                            self.wall_v1[wall_id],
                            self.wall_v2[wall_id],
                            dtype,
                        )
                        distance = ti.sqrt(ti.max(dist2, 1.0e-30))
                        if ti.static(self.is_semi):
                            unused_energy, first, unused_hessian = self._semi_terms(
                                ti.Vector([vertex_id, wall_id, 2, -1]),
                                distance - dhat,
                                self._pw_penalty(body_i, wall_id),
                            )
                            normal_force = ti.max(-self.node_area[vertex_id] * first, 0.0)
                        else:
                            unused_energy, db, unused_hessian = self._ipc_barrier_distance2(
                                dist2,
                                dhat * dhat,
                                self._pw_kappa(body_i, wall_id),
                            )
                            normal_force = ti.max(
                                -self.node_area[vertex_id] * db * 2.0 * distance,
                                0.0,
                            )
                        active = True
            if active and normal_force > 0.0:
                bodies = ti.Vector([body_i, body_i, body_i, body_i])
                vertices = ti.Vector([vertex_id, vertex_id, vertex_id, vertex_id])
                weights = ti.Vector([1.0, 0.0, 0.0, 0.0])
                self._store_lagged_friction_contact(
                    bodies,
                    vertices,
                    weights,
                    normal,
                    self.hat_x[vertex_id],
                    mu * normal_force * self.scale_device[None],
                )

    @ti.kernel
    def _assemble_lagged_friction(self, need_matrix: ti.template(), matrix_target: ti.template()):
        contact_count = ti.min(self.friction_contact_count[0], self.friction_contact_capacity)
        for contact in range(contact_count):
            bodies = self.friction_contact_bodies[contact]
            vertices = self.friction_contact_vertices[contact]
            weights = self.friction_contact_weights[contact]
            rel = (
                weights[0] * self.x[vertices[0]]
                + weights[1] * self.x[vertices[1]]
                + weights[2] * self.x[vertices[2]]
                + weights[3] * self.x[vertices[3]]
            )
            self._assemble_local_friction(
                bodies[0],
                vertices[0],
                weights[0],
                bodies[1],
                vertices[1],
                weights[1],
                bodies[2],
                vertices[2],
                weights[2],
                bodies[3],
                vertices[3],
                weights[3],
                rel,
                self.friction_contact_hat_rel[contact],
                self.friction_contact_normal[contact],
                self.friction_scale[0] * self.friction_contact_coeff[contact],
                need_matrix,
                matrix_target,
            )

    @ti.func
    def _assemble_local_friction(
        self,
        body0,
        vertex0,
        w0,
        body1,
        vertex1,
        w1,
        body2,
        vertex2,
        w2,
        body3,
        vertex3,
        w3,
        rel,
        hat_rel,
        normal,
        coeff,
        need_matrix: ti.template(),
        matrix_target: ti.template(),
    ):
        if coeff > 0.0:
            unit_normal = normal / normal.norm()
            P = ti.Matrix.identity(float, 3) - unit_normal.outer_product(unit_normal)
            vbar = P @ ((rel - hat_rel) / self.dt_device[None])
            vbarnorm = vbar.norm()
            ti.atomic_add(
                self.energy[None],
                coeff * self._friction_f0(vbarnorm, self.epsv, self.dt_device[None]),
            )
            f1 = self._friction_f1_div_vbarnorm(vbarnorm, self.epsv)
            grad_f = coeff * f1 * (P @ vbar)
            if w0 != 0.0:
                self._scatter_gradient_vec(body0, vertex0, w0, grad_f)
            if w1 != 0.0:
                self._scatter_gradient_vec(body1, vertex1, w1, grad_f)
            if w2 != 0.0:
                self._scatter_gradient_vec(body2, vertex2, w2, grad_f)
            if w3 != 0.0:
                self._scatter_gradient_vec(body3, vertex3, w3, grad_f)
            if ti.static(need_matrix):
                H = self._lagged_friction_hessian(rel, hat_rel, normal, coeff)
                if w0 != 0.0:
                    self._scatter_friction_pair(body0, vertex0, w0, body0, vertex0, w0, H, matrix_target)
                    if w1 != 0.0:
                        self._scatter_friction_pair(body0, vertex0, w0, body1, vertex1, w1, H, matrix_target)
                    if w2 != 0.0:
                        self._scatter_friction_pair(body0, vertex0, w0, body2, vertex2, w2, H, matrix_target)
                    if w3 != 0.0:
                        self._scatter_friction_pair(body0, vertex0, w0, body3, vertex3, w3, H, matrix_target)
                if w1 != 0.0:
                    if w0 != 0.0:
                        self._scatter_friction_pair(body1, vertex1, w1, body0, vertex0, w0, H, matrix_target)
                    self._scatter_friction_pair(body1, vertex1, w1, body1, vertex1, w1, H, matrix_target)
                    if w2 != 0.0:
                        self._scatter_friction_pair(body1, vertex1, w1, body2, vertex2, w2, H, matrix_target)
                    if w3 != 0.0:
                        self._scatter_friction_pair(body1, vertex1, w1, body3, vertex3, w3, H, matrix_target)
                if w2 != 0.0:
                    if w0 != 0.0:
                        self._scatter_friction_pair(body2, vertex2, w2, body0, vertex0, w0, H, matrix_target)
                    if w1 != 0.0:
                        self._scatter_friction_pair(body2, vertex2, w2, body1, vertex1, w1, H, matrix_target)
                    self._scatter_friction_pair(body2, vertex2, w2, body2, vertex2, w2, H, matrix_target)
                    if w3 != 0.0:
                        self._scatter_friction_pair(body2, vertex2, w2, body3, vertex3, w3, H, matrix_target)
                if w3 != 0.0:
                    if w0 != 0.0:
                        self._scatter_friction_pair(body3, vertex3, w3, body0, vertex0, w0, H, matrix_target)
                    if w1 != 0.0:
                        self._scatter_friction_pair(body3, vertex3, w3, body1, vertex1, w1, H, matrix_target)
                    if w2 != 0.0:
                        self._scatter_friction_pair(body3, vertex3, w3, body2, vertex2, w2, H, matrix_target)
                    self._scatter_friction_pair(body3, vertex3, w3, body3, vertex3, w3, H, matrix_target)

    @ti.func
    def _resolved_fully_implicit_friction(self, pair_mu):
        """Resolve global paper parameters against one material-pair value."""
        mu_dynamic = pair_mu
        if ti.static(self.fully_mu_dynamic >= 0.0):
            mu_dynamic = self.fully_mu_dynamic
        mu_static = mu_dynamic
        if ti.static(self.fully_mu_static >= 0.0):
            mu_static = self.fully_mu_static
        return mu_dynamic, mu_static

    @ti.func
    def _fully_implicit_local_friction_residual(
        self,
        positions,
        hat_positions,
        distance2,
        distance_gradient,
        barrier_gradient,
        pair_mu,
        area,
        mollifier,
        edge_contact: ti.template(),
    ):
        """Paper friction residual without constructing a 12x12 derivative."""
        local_residual = ti.Vector.zero(float, 12)
        distance = ti.sqrt(ti.max(distance2, 1.0e-30))
        base_normal_force = ti.max(-2.0 * barrier_gradient * distance, 0.0)
        normal_force = area * mollifier * base_normal_force
        mu_dynamic, mu_static = self._resolved_fully_implicit_friction(pair_mu)
        if normal_force > 0.0 and (mu_dynamic > 0.0 or mu_static > 0.0 or ti.static(self.fully_mu_viscous > 0.0)):
            normal = ti.Vector.zero(float, 3)
            for component in ti.static(range(3)):
                numerator = distance_gradient[component]
                if ti.static(edge_contact):
                    numerator += distance_gradient[3 + component]
                normal[component] = numerator / (2.0 * distance)

            weights = ti.Vector.zero(float, 4)
            for weight_site in range(4):
                numerator = 0.0
                for component in ti.static(range(3)):
                    numerator += distance_gradient[3 * weight_site + component] * normal[component]
                weights[weight_site] = numerator / (2.0 * distance)

            inverse_dt = 1.0 / self.dt_device[None]
            relative_velocity = ti.Vector.zero(float, 3)
            velocity_site = 0
            while velocity_site < 4:
                for component in ti.static(range(3)):
                    velocity = (
                        positions[3 * velocity_site + component] - hat_positions[3 * velocity_site + component]
                    ) * inverse_dt
                    relative_velocity[component] += weights[velocity_site] * velocity
                velocity_site += 1

            tangent = ti.Matrix.identity(float, 3) - normal.outer_product(normal)
            tangential_velocity = tangent @ relative_velocity
            speed = tangential_velocity.norm()
            profile = ipc_fully_implicit_profile_over_speed(speed, self.epsv, self.fully_profile_id)
            mu_difference = mu_static - mu_dynamic
            falloff = 0.0
            if mu_difference != 0.0:
                falloff = ipc_stribeck_falloff(speed, self.fully_stribeck_velocity)
            effective_mu = mu_dynamic + mu_difference * falloff
            radial_factor = normal_force * effective_mu * profile + self.fully_mu_viscous
            scale = self.scale_device[None] * self.friction_scale[0]
            resistance = scale * radial_factor * tangential_velocity
            for residual_site in range(4):
                for component in ti.static(range(3)):
                    local_residual[3 * residual_site + component] = weights[residual_site] * resistance[component]
        return local_residual

    @ti.func
    def _fully_implicit_local_friction_residual_jacobian(
        self,
        positions,
        hat_positions,
        distance2,
        distance_gradient,
        distance_hessian,
        barrier_gradient,
        barrier_hessian,
        pair_mu,
        area,
        mollifier,
        mollifier_gradient,
        edge_contact: ti.template(),
    ):
        """Paper friction residual and complete current-geometry Jacobian.

        The local coordinates are ``[p,t0,t1,t2]`` for PT and
        ``[a0,a1,b0,b1]`` for EE.  Signed closest-point weights are recovered
        from the exact squared-distance gradient.  Their derivatives and the
        normal derivative follow from the exact distance Hessian, which is
        precisely the chain rule for the paper's configuration-dependent
        ``Gamma(q)`` and tangential map ``T(q)``.

        For the IPC near-parallel EE region, ``normal_force`` is extended as
        ``area * m(q) [-b'(d)]``.  The ``dm/dq`` term below makes this a smooth,
        fully-implicit extension of the mollified normal barrier.
        Keeping area inside this normal-force measure, rather than outside the
        complete friction law, gives ``mu_viscous`` the same measure-independent
        meaning as the shared MPM/IGA fully implicit law.
        The lagged IPC path intentionally continues to skip these
        tangential contacts.
        """
        local_residual = ti.Vector.zero(float, 12)
        local_jacobian = ti.Matrix.zero(float, 12, 12)
        distance = ti.sqrt(ti.max(distance2, 1.0e-30))
        base_normal_force = ti.max(-2.0 * barrier_gradient * distance, 0.0)
        normal_force = area * mollifier * base_normal_force
        mu_dynamic, mu_static = self._resolved_fully_implicit_friction(pair_mu)
        if normal_force > 0.0 and (mu_dynamic > 0.0 or mu_static > 0.0 or ti.static(self.fully_mu_viscous > 0.0)):
            # sum(grad d2) over the first object equals 2 * closest_delta.
            # For PT the first object is the leading point; for EE it is the
            # first edge (sites 0 and 1).
            normal = ti.Vector.zero(float, 3)
            for component in ti.static(range(3)):
                numerator = distance_gradient[component]
                if ti.static(edge_contact):
                    numerator += distance_gradient[3 + component]
                normal[component] = numerator / (2.0 * distance)

            weights = ti.Vector.zero(float, 4)
            for weight_site in range(4):
                numerator = 0.0
                for component in ti.static(range(3)):
                    numerator += distance_gradient[3 * weight_site + component] * normal[component]
                weights[weight_site] = numerator / (2.0 * distance)

            inverse_dt = 1.0 / self.dt_device[None]
            site_velocity = ti.Matrix.zero(float, 4, 3)
            relative_velocity = ti.Vector.zero(float, 3)
            velocity_site = 0
            while velocity_site < 4:
                for component in ti.static(range(3)):
                    velocity = (
                        positions[3 * velocity_site + component] - hat_positions[3 * velocity_site + component]
                    ) * inverse_dt
                    site_velocity[velocity_site, component] = velocity
                    relative_velocity[component] += weights[velocity_site] * velocity
                velocity_site += 1

            tangent = ti.Matrix.identity(float, 3) - normal.outer_product(normal)
            tangential_velocity = tangent @ relative_velocity
            speed = tangential_velocity.norm()
            profile = ipc_fully_implicit_profile_over_speed(speed, self.epsv, self.fully_profile_id)
            profile_derivative = ipc_fully_implicit_profile_over_speed_derivative(
                speed, self.epsv, self.fully_profile_id
            )
            mu_difference = mu_static - mu_dynamic
            falloff = 0.0
            falloff_derivative = 0.0
            if mu_difference != 0.0:
                falloff = ipc_stribeck_falloff(speed, self.fully_stribeck_velocity)
                falloff_derivative = ipc_stribeck_falloff_derivative(speed, self.fully_stribeck_velocity)
            effective_mu = mu_dynamic + mu_difference * falloff
            factor_per_normal_force = effective_mu * profile
            radial_factor = normal_force * factor_per_normal_force + self.fully_mu_viscous
            radial_speed_derivative = normal_force * (
                mu_difference * falloff_derivative * profile + effective_mu * profile_derivative
            )
            scale = self.scale_device[None] * self.friction_scale[0]
            resistance = scale * radial_factor * tangential_velocity
            for residual_site in range(4):
                for component in ti.static(range(3)):
                    local_residual[3 * residual_site + component] = weights[residual_site] * resistance[component]

            if ti.static(True):
                distance_derivative = ti.Vector.zero(float, 12)
                normal_derivative = ti.Matrix.zero(float, 3, 12)
                weight_derivative = ti.Matrix.zero(float, 4, 12)
                for column in range(12):
                    distance_derivative[column] = distance_gradient[column] / (2.0 * distance)
                    for component in ti.static(range(3)):
                        numerator_derivative = distance_hessian[component, column]
                        if ti.static(edge_contact):
                            numerator_derivative += distance_hessian[3 + component, column]
                        normal_derivative[component, column] = (
                            numerator_derivative / (2.0 * distance)
                            - normal[component] * distance_derivative[column] / distance
                        )
                for derivative_site in range(4):
                    for column in range(12):
                        numerator_derivative = 0.0
                        for component in ti.static(range(3)):
                            numerator_derivative += (
                                distance_hessian[3 * derivative_site + component, column] * normal[component]
                                + distance_gradient[3 * derivative_site + component]
                                * normal_derivative[component, column]
                            )
                        weight_derivative[derivative_site, column] = (
                            numerator_derivative / (2.0 * distance)
                            - weights[derivative_site] * distance_derivative[column] / distance
                        )

                relative_velocity_derivative = ti.Matrix.zero(float, 3, 12)
                for column in range(12):
                    column_site = column // 3
                    column_component = column - 3 * column_site
                    for component in ti.static(range(3)):
                        velocity_weight_site = 0
                        while velocity_weight_site < 4:
                            relative_velocity_derivative[component, column] += (
                                weight_derivative[velocity_weight_site, column]
                                * site_velocity[velocity_weight_site, component]
                            )
                            velocity_weight_site += 1
                        if component == column_component:
                            relative_velocity_derivative[component, column] += weights[column_site] * inverse_dt

                tangential_derivative = ti.Matrix.zero(float, 3, 12)
                speed_derivative = ti.Vector.zero(float, 12)
                for column in range(12):
                    dn = ti.Vector.zero(float, 3)
                    dv = ti.Vector.zero(float, 3)
                    for component in ti.static(range(3)):
                        dn[component] = normal_derivative[component, column]
                        dv[component] = relative_velocity_derivative[component, column]
                    dz = tangent @ dv - dn * normal.dot(relative_velocity) - normal * dn.dot(relative_velocity)
                    for component in ti.static(range(3)):
                        tangential_derivative[component, column] = dz[component]
                    if speed > 0.0:
                        speed_derivative[column] = tangential_velocity.dot(dz) / speed

                for column in range(12):
                    base_normal_force_derivative = -2.0 * (
                        barrier_hessian * distance_gradient[column] * distance
                        + barrier_gradient * distance_derivative[column]
                    )
                    normal_force_derivative = area * (
                        mollifier * base_normal_force_derivative + base_normal_force * mollifier_gradient[column]
                    )
                    radial_derivative = (
                        factor_per_normal_force * normal_force_derivative
                        + radial_speed_derivative * speed_derivative[column]
                    )
                    for jacobian_site in range(4):
                        for component in ti.static(range(3)):
                            resistance_derivative = scale * (
                                radial_factor * tangential_derivative[component, column]
                                + radial_derivative * tangential_velocity[component]
                            )
                            local_jacobian[3 * jacobian_site + component, column] = (
                                weight_derivative[jacobian_site, column] * resistance[component]
                                + weights[jacobian_site] * resistance_derivative
                            )
        return local_residual, local_jacobian

    @ti.kernel
    def _assemble_fully_implicit_pt_friction(
        self,
        need_matrix: ti.template(),
        matrix_target: ti.template(),
        candidate_count: ti.template(),
        candidate_vertex: ti.template(),
        candidate_face: ti.template(),
    ):
        for candidate_id in range(candidate_count[None]):
            vertex_id = candidate_vertex[candidate_id]
            face_id = candidate_face[candidate_id]
            body_i = self.node2body[vertex_id]
            body_j = self.face2body[face_id]
            if self._body_pair_allowed(body_i, body_j):
                face = self.faces[face_id]
                p = self.x[vertex_id]
                a = self.x[face[0]]
                b = self.x[face[1]]
                c = self.x[face[2]]
                if ti.static(need_matrix):
                    dist2, grad_d, hess_d, unused_type = point_triangle_distance_grad_hess(p, a, b, c)
                    dhat = self._pp_dhat(body_i, body_j)
                    if dist2 > 0.0 and dist2 < dhat * dhat:
                        unused_energy, db, ddb = self._ipc_barrier_distance2(
                            dist2,
                            dhat * dhat,
                            self._pp_kappa(body_i, body_j),
                        )
                        positions = ti.Vector.zero(float, 12)
                        hats = ti.Vector.zero(float, 12)
                        vertices = ti.Vector([vertex_id, face[0], face[1], face[2]])
                        bodies = ti.Vector([body_i, body_j, body_j, body_j])
                        for site in range(4):
                            for component in ti.static(range(3)):
                                positions[3 * site + component] = self.x[vertices[site]][component]
                                hats[3 * site + component] = self.hat_x[vertices[site]][component]
                        mollifier_gradient = ti.Vector.zero(float, 12)
                        local_residual, local_jacobian = self._fully_implicit_local_friction_residual_jacobian(
                            positions,
                            hats,
                            dist2,
                            grad_d,
                            hess_d,
                            db,
                            ddb,
                            self._pp_mu(body_i, body_j),
                            0.25 * self.node_area[vertex_id],
                            1.0,
                            mollifier_gradient,
                            False,
                        )
                        self._scatter_local_gradient(bodies, vertices, local_residual)
                        self._scatter_local_hessian(
                            bodies,
                            vertices,
                            local_jacobian,
                            matrix_target,
                        )
                else:
                    dist2, grad_d, unused_type = point_triangle_distance_grad(p, a, b, c)
                    dhat = self._pp_dhat(body_i, body_j)
                    if dist2 > 0.0 and dist2 < dhat * dhat:
                        unused_energy, db, unused_ddb = self._ipc_barrier_distance2(
                            dist2,
                            dhat * dhat,
                            self._pp_kappa(body_i, body_j),
                        )
                        positions = ti.Vector.zero(float, 12)
                        hats = ti.Vector.zero(float, 12)
                        vertices = ti.Vector([vertex_id, face[0], face[1], face[2]])
                        bodies = ti.Vector([body_i, body_j, body_j, body_j])
                        for site in range(4):
                            for component in ti.static(range(3)):
                                positions[3 * site + component] = self.x[vertices[site]][component]
                                hats[3 * site + component] = self.hat_x[vertices[site]][component]
                        local_residual = self._fully_implicit_local_friction_residual(
                            positions,
                            hats,
                            dist2,
                            grad_d,
                            db,
                            self._pp_mu(body_i, body_j),
                            0.25 * self.node_area[vertex_id],
                            1.0,
                            False,
                        )
                        self._scatter_local_gradient(bodies, vertices, local_residual)

    @ti.kernel
    def _assemble_fully_implicit_ee_friction(
        self,
        need_matrix: ti.template(),
        matrix_target: ti.template(),
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
    ):
        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            body_i = self.edge2body[edge_i]
            body_j = self.edge2body[edge_j]
            if self._body_pair_allowed(body_i, body_j):
                ei = self.edges[edge_i]
                ej = self.edges[edge_j]
                p0 = self.x[ei[0]]
                p1 = self.x[ei[1]]
                q0 = self.x[ej[0]]
                q1 = self.x[ej[1]]
                if ti.static(need_matrix):
                    dist2, grad_d, hess_d, unused_type = edge_edge_distance_grad_hess(p0, p1, q0, q1)
                    dhat = self._pp_dhat(body_i, body_j)
                    if dist2 > 0.0 and dist2 < dhat * dhat:
                        unused_energy, db, ddb = self._ipc_barrier_distance2(
                            dist2,
                            dhat * dhat,
                            self._pp_kappa(body_i, body_j),
                        )
                        eps_x = edge_edge_mollifier_threshold(
                            self.rest_x[ei[0]],
                            self.rest_x[ei[1]],
                            self.rest_x[ej[0]],
                            self.rest_x[ej[1]],
                        )
                        mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                        mollifier_gradient = edge_edge_mollifier_grad(p0, p1, q0, q1, eps_x)
                        positions = ti.Vector.zero(float, 12)
                        hats = ti.Vector.zero(float, 12)
                        vertices = ti.Vector([ei[0], ei[1], ej[0], ej[1]])
                        bodies = ti.Vector([body_i, body_i, body_j, body_j])
                        for site in range(4):
                            for component in ti.static(range(3)):
                                positions[3 * site + component] = self.x[vertices[site]][component]
                                hats[3 * site + component] = self.hat_x[vertices[site]][component]
                        local_residual, local_jacobian = self._fully_implicit_local_friction_residual_jacobian(
                            positions,
                            hats,
                            dist2,
                            grad_d,
                            hess_d,
                            db,
                            ddb,
                            self._pp_mu(body_i, body_j),
                            0.25 * (self.edge_area[edge_i] + self.edge_area[edge_j]),
                            mollifier,
                            mollifier_gradient,
                            True,
                        )
                        self._scatter_local_gradient(bodies, vertices, local_residual)
                        self._scatter_local_hessian(
                            bodies,
                            vertices,
                            local_jacobian,
                            matrix_target,
                        )
                else:
                    dist2, grad_d, unused_type = edge_edge_distance_grad(p0, p1, q0, q1)
                    dhat = self._pp_dhat(body_i, body_j)
                    if dist2 > 0.0 and dist2 < dhat * dhat:
                        unused_energy, db, unused_ddb = self._ipc_barrier_distance2(
                            dist2,
                            dhat * dhat,
                            self._pp_kappa(body_i, body_j),
                        )
                        eps_x = edge_edge_mollifier_threshold(
                            self.rest_x[ei[0]],
                            self.rest_x[ei[1]],
                            self.rest_x[ej[0]],
                            self.rest_x[ej[1]],
                        )
                        mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                        positions = ti.Vector.zero(float, 12)
                        hats = ti.Vector.zero(float, 12)
                        vertices = ti.Vector([ei[0], ei[1], ej[0], ej[1]])
                        bodies = ti.Vector([body_i, body_i, body_j, body_j])
                        for site in range(4):
                            for component in ti.static(range(3)):
                                positions[3 * site + component] = self.x[vertices[site]][component]
                                hats[3 * site + component] = self.hat_x[vertices[site]][component]
                        local_residual = self._fully_implicit_local_friction_residual(
                            positions,
                            hats,
                            dist2,
                            grad_d,
                            db,
                            self._pp_mu(body_i, body_j),
                            0.25 * (self.edge_area[edge_i] + self.edge_area[edge_j]),
                            mollifier,
                            True,
                        )
                        self._scatter_local_gradient(bodies, vertices, local_residual)

    @ti.kernel
    def _assemble_fully_implicit_wall_friction(self, need_matrix: ti.template(), matrix_target: ti.template()):
        for vertex_id, wall_id in ti.ndrange(self.vertex_num, self.wall_num):
            body_i = self.node2body[vertex_id]
            p = self.x[vertex_id]
            pair_mu = self._pw_mu(body_i, wall_id)
            dhat = self._pw_dhat(body_i, wall_id)
            kappa = self._pw_kappa(body_i, wall_id)
            if self.wall_type[wall_id] == 0:
                normal = self.wall_normal[wall_id]
                gap = (p - self.wall_point[wall_id]).dot(normal)
                if gap > 0.0 and gap < dhat:
                    unused_energy, dphi, ddphi = self._ipc_barrier_gap(gap, dhat, kappa)
                    contact_area = self.node_area[vertex_id]
                    normal_force = contact_area * ti.max(-dphi, 0.0)
                    normal_force_gradient = -contact_area * ddphi * normal
                    mu_dynamic, mu_static = self._resolved_fully_implicit_friction(pair_mu)
                    force, jacobian = ipc_fully_implicit_point_plane_stribeck_friction(
                        (p - self.hat_x[vertex_id]) / self.dt_device[None],
                        normal,
                        normal_force,
                        normal_force_gradient,
                        mu_dynamic,
                        mu_static,
                        self.fully_mu_viscous,
                        self.fully_stribeck_velocity,
                        self.epsv,
                        self.fully_profile_id,
                        1.0 / self.dt_device[None],
                    )
                    scale = self.scale_device[None] * self.friction_scale[0]
                    self._scatter_gradient_vec(body_i, vertex_id, 1.0, scale * force)
                    if ti.static(need_matrix):
                        self._scatter_hessian_block(
                            body_i,
                            vertex_id,
                            1.0,
                            body_i,
                            vertex_id,
                            1.0,
                            scale * jacobian,
                            matrix_target,
                        )
            else:
                a = self.wall_v0[wall_id]
                b = self.wall_v1[wall_id]
                c = self.wall_v2[wall_id]
                if ti.static(need_matrix):
                    dist2, grad_d, hess_d, unused_type = point_triangle_distance_grad_hess(p, a, b, c)
                    if dist2 > 0.0 and dist2 < dhat * dhat:
                        unused_energy, db, ddb = self._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
                        positions = ti.Vector.zero(float, 12)
                        hats = ti.Vector.zero(float, 12)
                        for component in ti.static(range(3)):
                            positions[component] = p[component]
                            hats[component] = self.hat_x[vertex_id][component]
                            positions[3 + component] = a[component]
                            positions[6 + component] = b[component]
                            positions[9 + component] = c[component]
                            hats[3 + component] = a[component]
                            hats[6 + component] = b[component]
                            hats[9 + component] = c[component]
                        mollifier_gradient = ti.Vector.zero(float, 12)
                        local_residual, local_jacobian = self._fully_implicit_local_friction_residual_jacobian(
                            positions,
                            hats,
                            dist2,
                            grad_d,
                            hess_d,
                            db,
                            ddb,
                            pair_mu,
                            self.node_area[vertex_id],
                            1.0,
                            mollifier_gradient,
                            False,
                        )
                        point_residual = ti.Vector.zero(float, 3)
                        point_jacobian = ti.Matrix.zero(float, 3, 3)
                        for row in ti.static(range(3)):
                            point_residual[row] = local_residual[row]
                            for column in ti.static(range(3)):
                                point_jacobian[row, column] = local_jacobian[row, column]
                        self._scatter_gradient_vec(body_i, vertex_id, 1.0, point_residual)
                        self._scatter_hessian_block(
                            body_i,
                            vertex_id,
                            1.0,
                            body_i,
                            vertex_id,
                            1.0,
                            point_jacobian,
                            matrix_target,
                        )
                else:
                    dist2, grad_d, unused_type = point_triangle_distance_grad(p, a, b, c)
                    if dist2 > 0.0 and dist2 < dhat * dhat:
                        unused_energy, db, unused_ddb = self._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
                        positions = ti.Vector.zero(float, 12)
                        hats = ti.Vector.zero(float, 12)
                        for component in ti.static(range(3)):
                            positions[component] = p[component]
                            hats[component] = self.hat_x[vertex_id][component]
                            positions[3 + component] = a[component]
                            positions[6 + component] = b[component]
                            positions[9 + component] = c[component]
                            hats[3 + component] = a[component]
                            hats[6 + component] = b[component]
                            hats[9 + component] = c[component]
                        local_residual = self._fully_implicit_local_friction_residual(
                            positions,
                            hats,
                            dist2,
                            grad_d,
                            db,
                            pair_mu,
                            self.node_area[vertex_id],
                            1.0,
                            False,
                        )
                        point_residual = ti.Vector.zero(float, 3)
                        for row in ti.static(range(3)):
                            point_residual[row] = local_residual[row]
                        self._scatter_gradient_vec(body_i, vertex_id, 1.0, point_residual)

    def _assemble_fully_implicit_friction(self, need_matrix, assemble_type=MATRIX_COO):
        """Scatter current PT/EE/wall friction residual and Jacobian.

        This is the stable entry point shared by standalone AffineBody and
        SoftAffine.  The caller must already have reconstructed current/hat
        vertices and refreshed the current broad-phase candidates.
        """
        self._assemble_fully_implicit_pt_friction(
            bool(need_matrix),
            int(assemble_type),
            self.neighbor.candidate_count,
            self.neighbor.candidate_vertex,
            self.neighbor.candidate_face,
        )
        self._assemble_fully_implicit_ee_friction(
            bool(need_matrix),
            int(assemble_type),
            self.neighbor.edge_candidate_count,
            self.neighbor.candidate_edge0,
            self.neighbor.candidate_edge1,
        )
        self._assemble_fully_implicit_wall_friction(bool(need_matrix), int(assemble_type))

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
            unused_energy, db, ddb = self._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
            mollifier_gradient = ti.Vector.zero(float, 12)
            local_residual, local_jacobian = self._fully_implicit_local_friction_residual_jacobian(
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
            unused_energy, db, ddb = self._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
            eps_x = edge_edge_mollifier_threshold(rp0, rp1, rq0, rq1)
            mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
            mollifier_gradient = edge_edge_mollifier_grad(p0, p1, q0, q1, eps_x)
            local_residual, local_jacobian = self._fully_implicit_local_friction_residual_jacobian(
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
            unused_energy, db, unused_ddb = self._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
            local_residual = self._fully_implicit_local_friction_residual(
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
            unused_energy, db, unused_ddb = self._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
            eps_x = edge_edge_mollifier_threshold(rp0, rp1, rq0, rq1)
            mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
            local_residual = self._fully_implicit_local_friction_residual(
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
            unused_energy, dphi, ddphi = self._ipc_barrier_gap(gap, dhat, kappa)
            normal_force = area * ti.max(-dphi, 0.0)
            normal_force_gradient = -area * ddphi * normal
            mu_dynamic, mu_static = self._resolved_fully_implicit_friction(pair_mu)
            force, jacobian = ipc_fully_implicit_point_plane_stribeck_friction(
                (position - hat_position) / self.dt_device[None],
                normal,
                normal_force,
                normal_force_gradient,
                mu_dynamic,
                mu_static,
                self.fully_mu_viscous,
                self.fully_stribeck_velocity,
                self.epsv,
                self.fully_profile_id,
                1.0 / self.dt_device[None],
            )
            scale = self.scale_device[None] * self.friction_scale[0]
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

    def _assemble_body_pair_barrier_hessian(
        self,
        candidate_count,
        candidate_vertex,
        candidate_face,
        edge_candidate_count,
        candidate_edge0,
        candidate_edge1,
        matrix_scale,
        matrix_target,
        project_pd=None,
    ):
        """ABD pair projection using final raw sparse storage."""
        if self.body_num < 2:
            return
        storage_target = int(matrix_target)
        if storage_target == MATRIX_CONTACT_DAMPING:
            if self.hash_triplet is not None:
                self.hash_triplet.reset_system()
                storage_target = MATRIX_HASH_TRIPLET
            elif self.K_coo_values is not None:
                self.coo_entry_count.fill(0)
                self.coo_overflow.fill(0)
                storage_target = MATRIX_COO
            else:
                raise RuntimeError("Affine contact damping needs a bound COO or HashTriplet " "workspace")
        tick = time.time()
        self.body_pair_point_type_count.fill(0)
        self.body_pair_edge_type_count.fill(0)
        self._reset_active_body_pairs()
        self._prepare_body_pair_barrier_segments(
            candidate_count,
            candidate_vertex,
            candidate_face,
            edge_candidate_count,
            candidate_edge0,
            candidate_edge1,
            float(matrix_scale),
            int(storage_target),
            int(matrix_target),
        )
        self._allocate_active_body_pair_segments(int(storage_target))
        point_mask = int(self._body_pair_point_dispatch_mask())
        for contact_type in range(7):
            if point_mask & (1 << contact_type):
                self._collect_body_pair_point_barrier_hessian_type(
                    contact_type,
                    candidate_count,
                    candidate_vertex,
                    candidate_face,
                    float(matrix_scale),
                    int(storage_target),
                    int(matrix_target),
                )
        edge_mask = int(self._body_pair_edge_dispatch_mask())
        for contact_type in range(9):
            if edge_mask & (1 << contact_type):
                self._collect_body_pair_edge_barrier_hessian_type(
                    contact_type,
                    edge_candidate_count,
                    candidate_edge0,
                    candidate_edge1,
                    float(matrix_scale),
                    int(storage_target),
                    int(matrix_target),
                )
        if bool(int(os.environ.get("GT_SOFT_AFFINE_PROFILE", "0"))):
            ti.sync()
            now = time.time()
            print(f"[SoftAffineIPC] affine_collect_barrier_hessian: {now - tick:.6f}s", flush=True)
            tick = now
        if project_pd is None:
            project_pd = not self.fully_implicit
        self._project_body_pair_segments(
            int(storage_target),
            int(matrix_target),
            bool(project_pd),
        )
        if bool(int(os.environ.get("GT_SOFT_AFFINE_PROFILE", "0"))):
            ti.sync()
            print(f"[SoftAffineIPC] affine_project_barrier_hessian: {time.time() - tick:.6f}s", flush=True)

    @ti.kernel
    def _prepare_body_pair_barrier_segments(
        self,
        candidate_count: ti.template(),
        candidate_vertex: ti.template(),
        candidate_face: ti.template(),
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
        matrix_scale: ti.template(),
        storage_target: ti.template(),
        output_target: ti.template(),
    ):
        for candidate_id in range(candidate_count[None]):
            vertex_id = candidate_vertex[candidate_id]
            face_id = candidate_face[candidate_id]
            body_i = self.node2body[vertex_id]
            body_j = self.face2body[face_id]
            if self._body_pair_allowed(body_i, body_j):
                face = self.faces[face_id]
                point = self.x[vertex_id]
                a, b, c = self.x[face[0]], self.x[face[1]], self.x[face[2]]
                contact_type = point_triangle_distance_type(point, a, b, c)
                dist2 = point_triangle_distance2_from_type(point, a, b, c, contact_type)
                pair_scale = matrix_scale
                if ti.static(output_target == MATRIX_CONTACT_DAMPING):
                    pair_scale = self.dt_device[None] * self._pp_contact_damping(body_i, body_j)
                dhat = self._pp_dhat(body_i, body_j)
                if pair_scale > 0.0 and dist2 < dhat * dhat:
                    self._mark_active_body_pair(body_i, body_j)
                    ti.atomic_add(self.body_pair_point_type_count[contact_type], 1)

        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            body_i = self.edge2body[edge_i]
            body_j = self.edge2body[edge_j]
            if self._body_pair_allowed(body_i, body_j):
                edge0, edge1 = self.edges[edge_i], self.edges[edge_j]
                p0, p1 = self.x[edge0[0]], self.x[edge0[1]]
                q0, q1 = self.x[edge1[0]], self.x[edge1[1]]
                contact_type = edge_edge_distance_type(p0, p1, q0, q1)
                dist2 = edge_edge_distance2_from_type(p0, p1, q0, q1, contact_type)
                pair_scale = matrix_scale
                if ti.static(output_target == MATRIX_CONTACT_DAMPING):
                    pair_scale = self.dt_device[None] * self._pp_contact_damping(body_i, body_j)
                dhat = self._pp_dhat(body_i, body_j)
                if pair_scale > 0.0 and dist2 < dhat * dhat:
                    self._mark_active_body_pair(body_i, body_j)
                    ti.atomic_add(self.body_pair_edge_type_count[contact_type], 1)

    @ti.func
    def _mark_active_body_pair(self, body_i, body_j):
        pair_i, pair_j = ti.min(body_i, body_j), ti.max(body_i, body_j)
        key = pair_i * self.body_num + pair_j
        if ti.atomic_max(self.active_body_pair[key], 1) == 0:
            self.body_pair_segment_base[key] = -1
            slot = ti.atomic_add(self.active_body_pair_count[None], 1)
            self.active_body_pair_list[slot] = key

    @ti.kernel
    def _reset_active_body_pairs(self):
        previous_count = self.active_body_pair_count[None]
        for active_id in range(previous_count):
            key = self.active_body_pair_list[active_id]
            self.active_body_pair[key] = 0
            self.body_pair_segment_base[key] = -1
        self.active_body_pair_count[None] = 0

    @ti.kernel
    def _allocate_active_body_pair_segments(self, storage_target: ti.template()):
        for active_id in range(self.active_body_pair_count[None]):
            key = self.active_body_pair_list[active_id]
            body_i, body_j = key // self.body_num, key % self.body_num
            base = self._reserve_body_pair_segment(storage_target)
            fits = 0
            if ti.static(storage_target == MATRIX_HASH_TRIPLET):
                fits = ti.cast(base + 128 <= self.hash_triplet.non_diag.blockH.shape[0], ti.i32)
            else:
                fits = ti.cast(base + 1152 <= self.max_coo_entries, ti.i32)
            if fits != 0:
                self._initialize_body_pair_segment(base, body_i, body_j, storage_target)
                self.body_pair_segment_base[key] = base

    @ti.kernel
    def _body_pair_point_dispatch_mask(self) -> ti.i32:
        mask = 0
        for contact_type in range(7):
            if self.body_pair_point_type_count[contact_type] > 0:
                mask += 1 << contact_type
        return mask

    @ti.kernel
    def _body_pair_edge_dispatch_mask(self) -> ti.i32:
        mask = 0
        for contact_type in range(9):
            if self.body_pair_edge_type_count[contact_type] > 0:
                mask += 1 << contact_type
        return mask

    @ti.func
    def _accumulate_body_pair_point_hessian(
        self, base, body_i, bodies, vertices, grad_d, hess_d, db, ddb, coefficient, storage_target: ti.template()
    ):
        for local_i, local_j in ti.ndrange(4, 4):
            side_i = ti.cast(bodies[local_i] != body_i, ti.i32)
            side_j = ti.cast(bodies[local_j] != body_i, ti.i32)
            vertex_i, vertex_j = vertices[local_i], vertices[local_j]
            for affine_i, affine_j in ti.ndrange(4, 4):
                weight = self.basis[vertex_i, affine_i] * self.basis[vertex_j, affine_j]
                for row, column in ti.static(ti.ndrange(3, 3)):
                    local_row = 3 * local_i + row
                    local_column = 3 * local_j + column
                    self._body_pair_dense_add(
                        base,
                        side_i * 12 + affine_i * 3 + row,
                        side_j * 12 + affine_j * 3 + column,
                        weight
                        * coefficient
                        * (ddb * grad_d[local_row] * grad_d[local_column] + db * hess_d[local_row, local_column]),
                        storage_target,
                    )

    @ti.kernel
    def _collect_body_pair_point_barrier_hessian_type(
        self,
        contact_type: ti.template(),
        candidate_count: ti.template(),
        candidate_vertex: ti.template(),
        candidate_face: ti.template(),
        matrix_scale: ti.template(),
        storage_target: ti.template(),
        output_target: ti.template(),
    ):
        for candidate_id in range(candidate_count[None]):
            vertex_id = candidate_vertex[candidate_id]
            face_id = candidate_face[candidate_id]
            point_body = self.node2body[vertex_id]
            face_body = self.face2body[face_id]
            body_i, body_j = ti.min(point_body, face_body), ti.max(point_body, face_body)
            base = self.body_pair_segment_base[body_i * self.body_num + body_j]
            if point_body != face_body and base >= 0:
                face = self.faces[face_id]
                point = self.x[vertex_id]
                a, b, c = self.x[face[0]], self.x[face[1]], self.x[face[2]]
                if point_triangle_distance_type(point, a, b, c) == contact_type:
                    dist2, grad_d, hess_d = point_triangle_distance_grad_hess_by_type(point, a, b, c, contact_type)
                    pair_scale = matrix_scale
                    if ti.static(output_target == MATRIX_CONTACT_DAMPING):
                        pair_scale = self.dt_device[None] * self._pp_contact_damping(point_body, face_body)
                    dhat = self._pp_dhat(point_body, face_body)
                    if pair_scale > 0.0 and dist2 < dhat * dhat:
                        unused_energy, db, ddb = self._ipc_barrier_distance2(
                            dist2, dhat * dhat, self._pp_kappa(point_body, face_body)
                        )
                        bodies = ti.Vector([point_body, face_body, face_body, face_body])
                        vertices = ti.Vector([vertex_id, face[0], face[1], face[2]])
                        self._accumulate_body_pair_point_hessian(
                            base,
                            body_i,
                            bodies,
                            vertices,
                            grad_d,
                            hess_d,
                            db,
                            ddb,
                            pair_scale * 0.25 * self.node_area[vertex_id],
                            storage_target,
                        )

    @ti.func
    def _accumulate_body_pair_edge_hessian(
        self,
        base,
        body_i,
        bodies,
        vertices,
        grad_d,
        hess_d,
        grad_m,
        hess_m,
        energy,
        db,
        ddb,
        mollifier,
        coefficient,
        storage_target: ti.template(),
    ):
        for local_i, local_j in ti.ndrange(4, 4):
            side_i = ti.cast(bodies[local_i] != body_i, ti.i32)
            side_j = ti.cast(bodies[local_j] != body_i, ti.i32)
            vertex_i, vertex_j = vertices[local_i], vertices[local_j]
            for affine_i, affine_j in ti.ndrange(4, 4):
                weight = self.basis[vertex_i, affine_i] * self.basis[vertex_j, affine_j]
                for row, column in ti.static(ti.ndrange(3, 3)):
                    local_row = 3 * local_i + row
                    local_column = 3 * local_j + column
                    value = (
                        mollifier
                        * (ddb * grad_d[local_row] * grad_d[local_column] + db * hess_d[local_row, local_column])
                        + energy * hess_m[local_row, local_column]
                        + db * (grad_m[local_row] * grad_d[local_column] + grad_d[local_row] * grad_m[local_column])
                    )
                    self._body_pair_dense_add(
                        base,
                        side_i * 12 + affine_i * 3 + row,
                        side_j * 12 + affine_j * 3 + column,
                        weight * coefficient * value,
                        storage_target,
                    )

    @ti.kernel
    def _collect_body_pair_edge_barrier_hessian_type(
        self,
        contact_type: ti.template(),
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
        matrix_scale: ti.template(),
        storage_target: ti.template(),
        output_target: ti.template(),
    ):
        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            edge_body_i = self.edge2body[edge_i]
            edge_body_j = self.edge2body[edge_j]
            body_i, body_j = ti.min(edge_body_i, edge_body_j), ti.max(edge_body_i, edge_body_j)
            base = self.body_pair_segment_base[body_i * self.body_num + body_j]
            if edge_body_i != edge_body_j and base >= 0:
                edge0, edge1 = self.edges[edge_i], self.edges[edge_j]
                p0, p1 = self.x[edge0[0]], self.x[edge0[1]]
                q0, q1 = self.x[edge1[0]], self.x[edge1[1]]
                if edge_edge_distance_type(p0, p1, q0, q1) == contact_type:
                    dist2, grad_d, hess_d = edge_edge_distance_grad_hess_by_type(p0, p1, q0, q1, contact_type)
                    pair_scale = matrix_scale
                    if ti.static(output_target == MATRIX_CONTACT_DAMPING):
                        pair_scale = self.dt_device[None] * self._pp_contact_damping(edge_body_i, edge_body_j)
                    dhat = self._pp_dhat(edge_body_i, edge_body_j)
                    if pair_scale > 0.0 and dist2 < dhat * dhat:
                        energy, db, ddb = self._ipc_barrier_distance2(
                            dist2, dhat * dhat, self._pp_kappa(edge_body_i, edge_body_j)
                        )
                        eps_x = edge_edge_mollifier_threshold(
                            self.rest_x[edge0[0]],
                            self.rest_x[edge0[1]],
                            self.rest_x[edge1[0]],
                            self.rest_x[edge1[1]],
                        )
                        mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                        grad_m = ti.Vector.zero(float, 12)
                        hess_m = ti.Matrix.zero(float, 12, 12)
                        if mollifier < 1.0:
                            grad_m, hess_m = edge_edge_mollifier_grad_hess(p0, p1, q0, q1, eps_x)
                        bodies = ti.Vector([edge_body_i, edge_body_i, edge_body_j, edge_body_j])
                        vertices = ti.Vector([edge0[0], edge0[1], edge1[0], edge1[1]])
                        self._accumulate_body_pair_edge_hessian(
                            base,
                            body_i,
                            bodies,
                            vertices,
                            grad_d,
                            hess_d,
                            grad_m,
                            hess_m,
                            energy,
                            db,
                            ddb,
                            mollifier,
                            pair_scale * 0.25 * (self.edge_area[edge_i] + self.edge_area[edge_j]),
                            storage_target,
                        )

    @ti.kernel
    def _project_body_pair_segments(
        self,
        storage_target: ti.template(),
        output_target: ti.template(),
        project_pd: ti.template(),
    ):
        if ti.static(storage_target == MATRIX_HASH_TRIPLET):
            for marker in range(self.hash_triplet.non_diag.blockI.shape[0]):
                if marker < self.hash_triplet.raw_non_diag_count[0] and self.hash_triplet.non_diag.blockI[marker] == -1:
                    self._project_body_pair_segment(
                        marker - 64,
                        storage_target,
                        output_target,
                        project_pd,
                    )
        else:
            for marker in range(self.max_coo_entries):
                if marker < self.coo_entry_count[None] and self.K_coo_rows[marker] == -1:
                    self._project_body_pair_segment(
                        marker - 576,
                        storage_target,
                        output_target,
                        project_pd,
                    )

    @ti.kernel
    def _assemble_semi_particle_contacts(
        self,
        need_matrix: ti.template(),
        candidate_count: ti.template(),
        candidate_vertex: ti.template(),
        candidate_face: ti.template(),
        matrix_target: ti.template(),
    ):
        for candidate_id in range(candidate_count[None]):
            vertex_id = candidate_vertex[candidate_id]
            face_id = candidate_face[candidate_id]
            body_i = self.node2body[vertex_id]
            body_j = self.face2body[face_id]
            if self._body_pair_allowed(body_i, body_j):
                face = self.faces[face_id]
                p, a, b, c = (
                    self.x[vertex_id],
                    self.x[face[0]],
                    self.x[face[1]],
                    self.x[face[2]],
                )
                dist2, grad_d, unused_type = point_triangle_distance_grad(p, a, b, c)
                distance = ti.sqrt(ti.max(dist2, 1.0e-30))
                gap = distance - self._pp_dhat(body_i, body_j)
                key = ti.Vector([vertex_id, face_id, 0, -1])
                slot = semi_ipc_find(self.semi_state, self.semi_key, key, ti.static(self.semi_capacity))
                multiplier = 0.0
                if slot >= 0:
                    multiplier = self.semi_multiplier[slot]
                if multiplier - self._pp_penalty(body_i, body_j) * gap >= 0.0:
                    energy, first, second = self._semi_terms(key, gap, self._pp_penalty(body_i, body_j))
                    coefficient = self.scale_device[None] * 0.25 * self.node_area[vertex_id]
                    ti.atomic_add(self.energy[None], coefficient * energy)
                    gap_gradient = grad_d / (2.0 * distance)
                    local_gradient = coefficient * first * gap_gradient
                    bodies = ti.Vector([body_i, body_j, body_j, body_j])
                    vertices = ti.Vector([vertex_id, face[0], face[1], face[2]])
                    self._scatter_local_gradient(bodies, vertices, local_gradient)
                    if ti.static(need_matrix):
                        local_hessian = coefficient * second * gap_gradient.outer_product(gap_gradient)
                        self._scatter_local_hessian(bodies, vertices, local_hessian, matrix_target)

    @ti.kernel
    def _assemble_semi_edge_contacts(
        self,
        need_matrix: ti.template(),
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
        matrix_target: ti.template(),
    ):
        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            body_i = self.edge2body[edge_i]
            body_j = self.edge2body[edge_j]
            if self._body_pair_allowed(body_i, body_j):
                ei, ej = self.edges[edge_i], self.edges[edge_j]
                p0, p1 = self.x[ei[0]], self.x[ei[1]]
                q0, q1 = self.x[ej[0]], self.x[ej[1]]
                dist2, grad_d, unused_type = edge_edge_distance_grad(p0, p1, q0, q1)
                distance = ti.sqrt(ti.max(dist2, 1.0e-30))
                gap = distance - self._pp_dhat(body_i, body_j)
                key = ti.Vector([ti.min(edge_i, edge_j), ti.max(edge_i, edge_j), 1, -1])
                slot = semi_ipc_find(self.semi_state, self.semi_key, key, ti.static(self.semi_capacity))
                multiplier = 0.0
                if slot >= 0:
                    multiplier = self.semi_multiplier[slot]
                if multiplier - self._pp_penalty(body_i, body_j) * gap >= 0.0:
                    energy, first, second = self._semi_terms(key, gap, self._pp_penalty(body_i, body_j))
                    eps_x = edge_edge_mollifier_threshold(
                        self.rest_x[ei[0]],
                        self.rest_x[ei[1]],
                        self.rest_x[ej[0]],
                        self.rest_x[ej[1]],
                    )
                    mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                    coefficient = (
                        self.scale_device[None] * 0.25 * (self.edge_area[edge_i] + self.edge_area[edge_j]) * mollifier
                    )
                    ti.atomic_add(self.energy[None], coefficient * energy)
                    gap_gradient = grad_d / (2.0 * distance)
                    local_gradient = coefficient * first * gap_gradient
                    bodies = ti.Vector([body_i, body_i, body_j, body_j])
                    vertices = ti.Vector([ei[0], ei[1], ej[0], ej[1]])
                    self._scatter_local_gradient(bodies, vertices, local_gradient)
                    if ti.static(need_matrix):
                        local_hessian = coefficient * second * gap_gradient.outer_product(gap_gradient)
                        self._scatter_local_hessian(bodies, vertices, local_hessian, matrix_target)

    @ti.kernel
    def _assemble_particle_contacts(
        self,
        need_matrix: ti.template(),
        candidate_count: ti.template(),
        candidate_vertex: ti.template(),
        candidate_face: ti.template(),
        matrix_target: ti.template(),
    ):
        for candidate_id in range(candidate_count[None]):
            vertex_id = candidate_vertex[candidate_id]
            face_id = candidate_face[candidate_id]
            body_i = self.node2body[vertex_id]
            body_j = self.face2body[face_id]
            if self._body_pair_allowed(body_i, body_j):
                face = self.faces[face_id]
                p = self.x[vertex_id]
                a = self.x[face[0]]
                b = self.x[face[1]]
                c = self.x[face[2]]
                if ti.static(need_matrix):
                    dist2, grad_d, hess_d, dtype = point_triangle_distance_grad_hess(p, a, b, c)
                    dhat = self._pp_dhat(body_i, body_j)
                    kappa = self._pp_kappa(body_i, body_j)
                    active_gap2 = dhat * dhat
                    if dist2 < active_gap2:
                        base_coeff = self.scale_device[None] * 0.25 * self.node_area[vertex_id]
                        energy, db, ddb = self._ipc_barrier_distance2(dist2, active_gap2, kappa)
                        ti.atomic_add(self.energy[None], base_coeff * energy)
                        local_grad = ti.Vector.zero(float, 12)
                        for i in range(12):
                            local_grad[i] = base_coeff * db * grad_d[i]
                        bodies = ti.Vector([body_i, body_j, body_j, body_j])
                        vertices = ti.Vector([vertex_id, face[0], face[1], face[2]])
                        self._scatter_local_gradient(bodies, vertices, local_grad)
                        local_hessian = ti.Matrix.zero(float, 12, 12)
                        for row, column in ti.ndrange(12, 12):
                            local_hessian[row, column] = base_coeff * (
                                ddb * grad_d[row] * grad_d[column] + db * hess_d[row, column]
                            )
                        if ti.static(not self.fully_implicit):
                            local_hessian = psd_project_nd(local_hessian)
                        self._scatter_local_hessian(bodies, vertices, local_hessian, matrix_target)
                else:
                    dist2, grad_d, dtype = point_triangle_distance_grad(p, a, b, c)
                    dhat = self._pp_dhat(body_i, body_j)
                    kappa = self._pp_kappa(body_i, body_j)
                    active_gap2 = dhat * dhat
                    if dist2 < active_gap2:
                        base_coeff = self.scale_device[None] * 0.25 * self.node_area[vertex_id]
                        energy, db, unused_ddb = self._ipc_barrier_distance2(dist2, active_gap2, kappa)
                        ti.atomic_add(self.energy[None], base_coeff * energy)
                        local_grad = ti.Vector.zero(float, 12)
                        for i in range(12):
                            local_grad[i] = base_coeff * db * grad_d[i]
                        bodies = ti.Vector([body_i, body_j, body_j, body_j])
                        vertices = ti.Vector([vertex_id, face[0], face[1], face[2]])
                        self._scatter_local_gradient(bodies, vertices, local_grad)

    @ti.kernel
    def _assemble_particle_contact_damping_hessian(
        self, candidate_count: ti.template(), candidate_vertex: ti.template(), candidate_face: ti.template()
    ):
        for candidate_id in range(candidate_count[None]):
            vertex_id = candidate_vertex[candidate_id]
            face_id = candidate_face[candidate_id]
            body_i = self.node2body[vertex_id]
            body_j = self.face2body[face_id]
            if self._body_pair_allowed(body_i, body_j):
                face = self.faces[face_id]
                p = self.x[vertex_id]
                a = self.x[face[0]]
                b = self.x[face[1]]
                c = self.x[face[2]]
                dist2, grad_d, hess_d, dtype = point_triangle_distance_grad_hess(p, a, b, c)
                dhat = self._pp_dhat(body_i, body_j)
                kappa = self._pp_kappa(body_i, body_j)
                damping_scale = self.dt_device[None] * self._pp_contact_damping(body_i, body_j)
                active_gap2 = dhat * dhat
                if damping_scale > 0.0 and dist2 < active_gap2:
                    base_coeff = damping_scale * 0.25 * self.node_area[vertex_id]
                    energy, db, ddb = self._ipc_barrier_distance2(dist2, active_gap2, kappa)
                    bodies = ti.Vector([body_i, body_j, body_j, body_j])
                    vertices = ti.Vector([vertex_id, face[0], face[1], face[2]])
                    self._scatter_local_barrier_hessian(
                        bodies, vertices, grad_d, hess_d, base_coeff * db, base_coeff * ddb, MATRIX_CONTACT_DAMPING
                    )

    @ti.kernel
    def _assemble_edge_contacts(
        self,
        need_matrix: ti.template(),
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
    ):
        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            body_i = self.edge2body[edge_i]
            body_j = self.edge2body[edge_j]
            if self._body_pair_allowed(body_i, body_j):
                ei = self.edges[edge_i]
                ej = self.edges[edge_j]
                p0 = self.x[ei[0]]
                p1 = self.x[ei[1]]
                q0 = self.x[ej[0]]
                q1 = self.x[ej[1]]
                dist2, grad_d, dtype = edge_edge_distance_grad(p0, p1, q0, q1)
                dhat = self._pp_dhat(body_i, body_j)
                kappa = self._pp_kappa(body_i, body_j)
                active_gap2 = dhat * dhat
                if dist2 < active_gap2:
                    base_coeff = self.scale_device[None] * 0.25 * (self.edge_area[edge_i] + self.edge_area[edge_j])
                    energy, db, ddb = self._ipc_barrier_distance2(dist2, active_gap2, kappa)
                    eps_x = edge_edge_mollifier_threshold(
                        self.rest_x[ei[0]], self.rest_x[ei[1]], self.rest_x[ej[0]], self.rest_x[ej[1]]
                    )
                    mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                    ti.atomic_add(self.energy[None], base_coeff * mollifier * energy)
                    local_grad = ti.Vector.zero(float, 12)
                    for i in range(12):
                        local_grad[i] = base_coeff * mollifier * db * grad_d[i]
                    grad_m = ti.Vector.zero(float, 12)
                    if mollifier < 1.0:
                        grad_m = edge_edge_mollifier_grad(p0, p1, q0, q1, eps_x)
                        for i in range(12):
                            local_grad[i] += base_coeff * energy * grad_m[i]
                    bodies = ti.Vector([body_i, body_i, body_j, body_j])
                    vertices = ti.Vector([ei[0], ei[1], ej[0], ej[1]])
                    self._scatter_local_gradient(bodies, vertices, local_grad)

    @ti.kernel
    def _assemble_edge_barrier_hessian(
        self,
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
        matrix_scale: ti.template(),
        matrix_target: ti.template(),
    ):
        """Assemble and project the complete EE contact Hessian."""
        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            body_i = self.edge2body[edge_i]
            body_j = self.edge2body[edge_j]
            if self._body_pair_allowed(body_i, body_j):
                ei = self.edges[edge_i]
                ej = self.edges[edge_j]
                p0 = self.x[ei[0]]
                p1 = self.x[ei[1]]
                q0 = self.x[ej[0]]
                q1 = self.x[ej[1]]
                dist2, grad_d, hess_d, unused_type = edge_edge_distance_grad_hess(p0, p1, q0, q1)
                dhat = self._pp_dhat(body_i, body_j)
                active_gap2 = dhat * dhat
                pair_scale = matrix_scale
                if ti.static(matrix_target == MATRIX_CONTACT_DAMPING):
                    pair_scale = self.dt_device[None] * self._pp_contact_damping(body_i, body_j)
                if pair_scale > 0.0 and dist2 < active_gap2:
                    base_coeff = pair_scale * 0.25 * (self.edge_area[edge_i] + self.edge_area[edge_j])
                    energy, db, ddb = self._ipc_barrier_distance2(dist2, active_gap2, self._pp_kappa(body_i, body_j))
                    eps_x = edge_edge_mollifier_threshold(
                        self.rest_x[ei[0]], self.rest_x[ei[1]], self.rest_x[ej[0]], self.rest_x[ej[1]]
                    )
                    mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                    grad_m = ti.Vector.zero(float, 12)
                    hess_m = ti.Matrix.zero(float, 12, 12)
                    if mollifier < 1.0:
                        grad_m, hess_m = edge_edge_mollifier_grad_hess(p0, p1, q0, q1, eps_x)
                    local_hessian = ti.Matrix.zero(float, 12, 12)
                    for row, column in ti.ndrange(12, 12):
                        local_hessian[row, column] = base_coeff * (
                            mollifier * (ddb * grad_d[row] * grad_d[column] + db * hess_d[row, column])
                            + energy * hess_m[row, column]
                            + db * (grad_m[row] * grad_d[column] + grad_d[row] * grad_m[column])
                        )
                    if ti.static(not self.fully_implicit):
                        local_hessian = psd_project_nd(local_hessian)
                    bodies = ti.Vector([body_i, body_i, body_j, body_j])
                    vertices = ti.Vector([ei[0], ei[1], ej[0], ej[1]])
                    self._scatter_local_hessian(bodies, vertices, local_hessian, matrix_target)

    @ti.kernel
    def _classify_active_edge_contacts(
        self, edge_candidate_count: ti.template(), candidate_edge0: ti.template(), candidate_edge1: ti.template()
    ):
        for edge_type in range(9):
            self.edge_type_count[edge_type] = 0
            self.edge_mollifier_type_count[edge_type] = 0
        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            body_i = self.edge2body[edge_i]
            body_j = self.edge2body[edge_j]
            if self._body_pair_allowed(body_i, body_j):
                ei = self.edges[edge_i]
                ej = self.edges[edge_j]
                p0 = self.x[ei[0]]
                p1 = self.x[ei[1]]
                q0 = self.x[ej[0]]
                q1 = self.x[ej[1]]
                dtype = edge_edge_distance_type(p0, p1, q0, q1)
                dist2 = edge_edge_distance2_from_type(p0, p1, q0, q1, dtype)
                dhat = self._pp_dhat(body_i, body_j)
                if dist2 < dhat * dhat:
                    ti.atomic_add(self.edge_type_count[dtype], 1)
                    eps_x = edge_edge_mollifier_threshold(
                        self.rest_x[ei[0]], self.rest_x[ei[1]], self.rest_x[ej[0]], self.rest_x[ej[1]]
                    )
                    mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                    if mollifier < 1.0:
                        ti.atomic_add(self.edge_mollifier_type_count[dtype], 1)

    @ti.kernel
    def _assemble_edge_barrier_hessian_type(
        self,
        edge_type: ti.template(),
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
        matrix_scale: ti.template(),
        matrix_target: ti.template(),
    ):
        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            body_i = self.edge2body[edge_i]
            body_j = self.edge2body[edge_j]
            if self._body_pair_allowed(body_i, body_j):
                ei = self.edges[edge_i]
                ej = self.edges[edge_j]
                p0 = self.x[ei[0]]
                p1 = self.x[ei[1]]
                q0 = self.x[ej[0]]
                q1 = self.x[ej[1]]
                dtype = edge_edge_distance_type(p0, p1, q0, q1)
                if dtype == edge_type:
                    if ti.static(edge_type == 0):
                        dist2 = point_point_distance2(p0, q0)
                        grad_d, hess_d = point_point_grad_hess(p0, q0)
                        ids = ti.Vector([0, 2])
                        self._assemble_edge_barrier_hessian_compact(
                            edge_i,
                            edge_j,
                            body_i,
                            body_j,
                            ei,
                            ej,
                            p0,
                            p1,
                            q0,
                            q1,
                            dist2,
                            ids,
                            2,
                            grad_d,
                            hess_d,
                            matrix_scale,
                            matrix_target,
                        )
                    elif ti.static(edge_type == 1):
                        dist2 = point_point_distance2(p0, q1)
                        grad_d, hess_d = point_point_grad_hess(p0, q1)
                        ids = ti.Vector([0, 3])
                        self._assemble_edge_barrier_hessian_compact(
                            edge_i,
                            edge_j,
                            body_i,
                            body_j,
                            ei,
                            ej,
                            p0,
                            p1,
                            q0,
                            q1,
                            dist2,
                            ids,
                            2,
                            grad_d,
                            hess_d,
                            matrix_scale,
                            matrix_target,
                        )
                    elif ti.static(edge_type == 2):
                        dist2 = point_line_distance2(p0, q0, q1)
                        grad_d = g_PE3D(p0, q0, q1)
                        hess_d = H_PE3D(p0, q0, q1)
                        ids = ti.Vector([0, 2, 3])
                        self._assemble_edge_barrier_hessian_compact(
                            edge_i,
                            edge_j,
                            body_i,
                            body_j,
                            ei,
                            ej,
                            p0,
                            p1,
                            q0,
                            q1,
                            dist2,
                            ids,
                            3,
                            grad_d,
                            hess_d,
                            matrix_scale,
                            matrix_target,
                        )
                    elif ti.static(edge_type == 3):
                        dist2 = point_point_distance2(p1, q0)
                        grad_d, hess_d = point_point_grad_hess(p1, q0)
                        ids = ti.Vector([1, 2])
                        self._assemble_edge_barrier_hessian_compact(
                            edge_i,
                            edge_j,
                            body_i,
                            body_j,
                            ei,
                            ej,
                            p0,
                            p1,
                            q0,
                            q1,
                            dist2,
                            ids,
                            2,
                            grad_d,
                            hess_d,
                            matrix_scale,
                            matrix_target,
                        )
                    elif ti.static(edge_type == 4):
                        dist2 = point_point_distance2(p1, q1)
                        grad_d, hess_d = point_point_grad_hess(p1, q1)
                        ids = ti.Vector([1, 3])
                        self._assemble_edge_barrier_hessian_compact(
                            edge_i,
                            edge_j,
                            body_i,
                            body_j,
                            ei,
                            ej,
                            p0,
                            p1,
                            q0,
                            q1,
                            dist2,
                            ids,
                            2,
                            grad_d,
                            hess_d,
                            matrix_scale,
                            matrix_target,
                        )
                    elif ti.static(edge_type == 5):
                        dist2 = point_line_distance2(p1, q0, q1)
                        grad_d = g_PE3D(p1, q0, q1)
                        hess_d = H_PE3D(p1, q0, q1)
                        ids = ti.Vector([1, 2, 3])
                        self._assemble_edge_barrier_hessian_compact(
                            edge_i,
                            edge_j,
                            body_i,
                            body_j,
                            ei,
                            ej,
                            p0,
                            p1,
                            q0,
                            q1,
                            dist2,
                            ids,
                            3,
                            grad_d,
                            hess_d,
                            matrix_scale,
                            matrix_target,
                        )
                    elif ti.static(edge_type == 6):
                        dist2 = point_line_distance2(q0, p0, p1)
                        grad_d = g_PE3D(q0, p0, p1)
                        hess_d = H_PE3D(q0, p0, p1)
                        ids = ti.Vector([2, 0, 1])
                        self._assemble_edge_barrier_hessian_compact(
                            edge_i,
                            edge_j,
                            body_i,
                            body_j,
                            ei,
                            ej,
                            p0,
                            p1,
                            q0,
                            q1,
                            dist2,
                            ids,
                            3,
                            grad_d,
                            hess_d,
                            matrix_scale,
                            matrix_target,
                        )
                    elif ti.static(edge_type == 7):
                        dist2 = point_line_distance2(q1, p0, p1)
                        grad_d = g_PE3D(q1, p0, p1)
                        hess_d = H_PE3D(q1, p0, p1)
                        ids = ti.Vector([3, 0, 1])
                        self._assemble_edge_barrier_hessian_compact(
                            edge_i,
                            edge_j,
                            body_i,
                            body_j,
                            ei,
                            ej,
                            p0,
                            p1,
                            q0,
                            q1,
                            dist2,
                            ids,
                            3,
                            grad_d,
                            hess_d,
                            matrix_scale,
                            matrix_target,
                        )
                    else:
                        dist2 = line_line_distance2(p0, p1, q0, q1)
                        grad_d = g_EE(p0, p1, q0, q1)
                        hess_d = H_EE(p0, p1, q0, q1)
                        ids = ti.Vector([0, 1, 2, 3])
                        self._assemble_edge_barrier_hessian_compact(
                            edge_i,
                            edge_j,
                            body_i,
                            body_j,
                            ei,
                            ej,
                            p0,
                            p1,
                            q0,
                            q1,
                            dist2,
                            ids,
                            4,
                            grad_d,
                            hess_d,
                            matrix_scale,
                            matrix_target,
                        )

    @ti.kernel
    def _assemble_edge_mollifier_hessian_type(
        self,
        edge_type: ti.template(),
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
        matrix_scale: ti.template(),
        matrix_target: ti.template(),
    ):
        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            body_i = self.edge2body[edge_i]
            body_j = self.edge2body[edge_j]
            if self._body_pair_allowed(body_i, body_j):
                ei = self.edges[edge_i]
                ej = self.edges[edge_j]
                p0 = self.x[ei[0]]
                p1 = self.x[ei[1]]
                q0 = self.x[ej[0]]
                q1 = self.x[ej[1]]
                dtype = edge_edge_distance_type(p0, p1, q0, q1)
                if dtype == edge_type:
                    dist2, grad_d, hess_d = edge_edge_distance_grad_hess_by_type(p0, p1, q0, q1, edge_type)
                    dhat = self._pp_dhat(body_i, body_j)
                    kappa = self._pp_kappa(body_i, body_j)
                    pair_scale = matrix_scale
                    if ti.static(matrix_target == MATRIX_CONTACT_DAMPING):
                        pair_scale = self.dt_device[None] * self._pp_contact_damping(body_i, body_j)
                    active_gap2 = dhat * dhat
                    if pair_scale > 0.0 and dist2 < active_gap2:
                        eps_x = edge_edge_mollifier_threshold(
                            self.rest_x[ei[0]], self.rest_x[ei[1]], self.rest_x[ej[0]], self.rest_x[ej[1]]
                        )
                        mollifier = edge_edge_mollifier(p0, p1, q0, q1, eps_x)
                        if mollifier < 1.0:
                            base_coeff = pair_scale * 0.25 * (self.edge_area[edge_i] + self.edge_area[edge_j])
                            energy, db, ddb = self._ipc_barrier_distance2(dist2, active_gap2, kappa)
                            grad_m, hess_m = edge_edge_mollifier_grad_hess(p0, p1, q0, q1, eps_x)
                            local_hessian = ti.Matrix.zero(float, 12, 12)
                            for row, column in ti.ndrange(12, 12):
                                local_hessian[row, column] = base_coeff * (
                                    mollifier * (ddb * grad_d[row] * grad_d[column] + db * hess_d[row, column])
                                    + energy * hess_m[row, column]
                                    + db * (grad_m[row] * grad_d[column] + grad_d[row] * grad_m[column])
                                )
                            if ti.static(not self.fully_implicit):
                                local_hessian = psd_project_nd(local_hessian)
                            bodies = ti.Vector([body_i, body_i, body_j, body_j])
                            vertices = ti.Vector([ei[0], ei[1], ej[0], ej[1]])
                            self._scatter_local_hessian(bodies, vertices, local_hessian, matrix_target)

    @ti.kernel
    def _assemble_wall_contact_damping_hessian(self):
        for vertex_id, wall_id in ti.ndrange(self.vertex_num, self.wall_num):
            body_i = self.node2body[vertex_id]
            p = self.x[vertex_id]
            normal = self.wall_normal[wall_id]
            dhat = self._pw_dhat(body_i, wall_id)
            kappa = self._pw_kappa(body_i, wall_id)
            damping_scale = self.dt_device[None] * self._pw_contact_damping(body_i, wall_id)
            if self.wall_type[wall_id] == 0:
                gap = (p - self.wall_point[wall_id]).dot(normal)
                if damping_scale > 0.0 and gap < dhat:
                    coeff = damping_scale * self.node_area[vertex_id]
                    energy, dphi, ddphi = self._ipc_barrier_gap(gap, dhat, kappa)
                    self._scatter_hessian_pair_contact_damping(
                        body_i, vertex_id, 1.0, body_i, vertex_id, 1.0, normal, coeff * ddphi
                    )
            else:
                dist2, grad_d, hess_d, dtype = point_triangle_distance_grad_hess(
                    p, self.wall_v0[wall_id], self.wall_v1[wall_id], self.wall_v2[wall_id]
                )
                active_gap2 = dhat * dhat
                if damping_scale > 0.0 and dist2 < active_gap2:
                    coeff = damping_scale * self.node_area[vertex_id]
                    energy, db, ddb = self._ipc_barrier_distance2(dist2, active_gap2, kappa)
                    point_hess = ti.Matrix.zero(float, 3, 3)
                    for di in ti.static(range(3)):
                        for dj in ti.static(range(3)):
                            point_hess[di, dj] = coeff * (ddb * grad_d[di] * grad_d[dj] + db * hess_d[di, dj])
                    if ti.static(not self.fully_implicit):
                        point_hess = psd_project_nd(point_hess)
                    self._scatter_hessian_block_contact_damping(
                        body_i, vertex_id, 1.0, body_i, vertex_id, 1.0, point_hess
                    )

    @ti.kernel
    def _assemble_contact_damping(self, need_matrix: ti.template(), matrix_target: ti.template()):
        for index in range(self.contact_damping_capacity):
            if index < self.contact_damping_count[None]:
                control_i = self.contact_damping_block_i[index]
                control_j = self.contact_damping_block_j[index]
                values = self.contact_damping_block_h[index]
                block = ti.Matrix.zero(float, 3, 3)
                for row, column in ti.static(ti.ndrange(3, 3)):
                    block[row, column] = values[row * 3 + column]
                vi = self.y[control_i] - self.hat_y[control_i]
                vj = self.y[control_j] - self.hat_y[control_j]
                product_i = block @ vj
                ti.atomic_add(self.grad[control_i], product_i)
                ti.atomic_add(self.energy[None], 0.5 * vi.dot(product_i))
                if ti.static(need_matrix):
                    self._add_matrix_block(control_i, control_j, block, matrix_target)
                if control_i != control_j:
                    transpose = block.transpose()
                    product_j = transpose @ vi
                    ti.atomic_add(self.grad[control_j], product_j)
                    ti.atomic_add(self.energy[None], 0.5 * vj.dot(product_j))
                    if ti.static(need_matrix):
                        self._add_matrix_block(
                            control_j,
                            control_i,
                            transpose,
                            matrix_target,
                        )

    @ti.kernel
    def _add_hessian_shift(self, matrix_target: ti.template(), shift: float):
        for control in range(self.control_num):
            self._add_matrix_block(
                control,
                control,
                shift * ti.Matrix.identity(float, 3),
                matrix_target,
            )

    def prepare_linear_fields(self, rhs, min_preconditioner=1.0e-12):
        self.linear_rhs.from_numpy(np.ascontiguousarray(rhs.reshape(-1), dtype=np.float64))
        self.linear_x.fill(0.0)
        self._build_coo_preconditioner(float(min_preconditioner))

    def coo_solution_numpy(self):
        return self.linear_x.to_numpy()[: self.dof].copy()

    @ti.kernel
    def device_set_predictor_direction(self):
        for control in range(self.control_num):
            self.direction_y[control] = self.tilde_y[control] - self.y[control]

    @ti.kernel
    def device_begin_step(self, dt: float):
        for control in range(self.control_num):
            self.hat_y[control] = self.y[control]
            self.tilde_y[control] = self.y[control] + dt * self.velocity_y[control]

    @ti.kernel
    def device_set_body_translation_velocity(
        self,
        body_id: ti.i32,
        vx: float,
        vy: float,
        vz: float,
    ):
        """Prescribe one AffineBody as a translating, shape-locked platen."""
        velocity = ti.Vector([vx, vy, vz])
        for local_control in range(4):
            self.velocity_y[4 * body_id + local_control] = velocity

    @ti.kernel
    def device_apply_body_translation_prediction(self, body_id: ti.i32):
        """Move a prescribed platen to its implicit end-of-step position."""
        for local_control in range(4):
            control = 4 * body_id + local_control
            self.y[control] = self.tilde_y[control]

    @ti.kernel
    def device_translate_wall(self, wall_id: ti.i32, dx: float, dy: float, dz: float):
        """Translate one IPC wall; moving it outside the domain removes it."""
        offset = ti.Vector([dx, dy, dz])
        self.wall_point[wall_id] += offset
        self.wall_v0[wall_id] += offset
        self.wall_v1[wall_id] += offset
        self.wall_v2[wall_id] += offset

    @ti.kernel
    def device_clear_external_generalized_force(self):
        for control in range(self.control_num):
            self.external_generalized_force[control] = ti.Vector.zero(float, 3)

    @ti.kernel
    def device_accept_step(self, dt: float):
        self.accepted_translation_velocity[None] = ti.Vector.zero(float, 3)
        for control in range(self.control_num):
            self.previous_y[control] = self.hat_y[control]
            self.previous_velocity_y[control] = self.velocity_y[control]
            self.velocity_y[control] = (self.y[control] - self.hat_y[control]) / dt

    @ti.kernel
    def device_enforce_free_translation_momentum(self, dt: float):
        previous_momentum = ti.Vector.zero(float, 3)
        accepted_momentum = ti.Vector.zero(float, 3)
        external_force = ti.Vector.zero(float, 3)
        total_mass = 0.0
        ti.loop_config(serialize=True)
        for body in range(self.body_num):
            total_mass += self.body_mass[body]
            for control in ti.static(range(4)):
                external_force += self.external_generalized_force[body * 4 + control]
            for row, column in ti.static(ti.ndrange(4, 4)):
                coefficient = self.mass[body, row, column]
                previous_momentum += coefficient * self.previous_velocity_y[body * 4 + column]
                accepted_momentum += coefficient * self.velocity_y[body * 4 + column]
        target_momentum = previous_momentum + dt * (total_mass * self.gravity[None] + external_force)
        translation_velocity = (target_momentum - accepted_momentum) / total_mass
        self.accepted_translation_velocity[None] = translation_velocity
        for control in range(self.control_num):
            self.velocity_y[control] += translation_velocity
            self.y[control] += dt * translation_velocity

    @ti.kernel
    def device_enter_equilibrium_adjoint(self):
        correction = self.dt_device[None] * self.accepted_translation_velocity[None]
        for control in range(self.control_num):
            self.y[control] -= correction

    @ti.kernel
    def device_leave_equilibrium_adjoint(self):
        correction = self.dt_device[None] * self.accepted_translation_velocity[None]
        for control in range(self.control_num):
            self.y[control] += correction

    @ti.kernel
    def device_prepare_step_state_adjoint(self, free_translation: ti.i32):
        total_mass = 0.0
        correction_vjp = ti.Vector.zero(float, 3)
        ti.loop_config(serialize=True)
        for body in range(self.body_num):
            total_mass += self.body_mass[body]
        if free_translation != 0:
            ti.loop_config(serialize=True)
            for control in range(self.control_num):
                correction_vjp += self.dt_device[None] * self.state_y_vjp[control] + self.state_velocity_vjp[control]
        self.translation_correction_vjp[None] = correction_vjp
        for body, control in ti.ndrange(self.body_num, 4):
            lumped = 0.0
            for row in range(4):
                lumped += self.mass[body, row, control]
            fraction = 0.0
            if free_translation != 0:
                fraction = lumped / total_mass
            index = 4 * body + control
            for component in ti.static(range(3)):
                velocity_vjp = self.state_velocity_vjp[index][component] - fraction * correction_vjp[component]
                self.linear_rhs[3 * index + component] = (
                    self.state_y_vjp[index][component] + velocity_vjp / self.dt_device[None]
                )
                self.state_y_vjp[index][component] = -velocity_vjp / self.dt_device[None]
                self.state_velocity_vjp[index][component] = fraction * correction_vjp[component]

    @ti.kernel
    def device_propagate_step_state_adjoint(self):
        for body, control in ti.ndrange(self.body_num, 4):
            mass = self.body_mass[body]
            force_coeff = self.dt_device[None] * mass * self.force_damp[body]
            torque_coeff = 0.25 * self.dt_device[None] * mass * self.torque_damp[body]
            index = 4 * body + control
            for component in ti.static(range(3)):
                inertia_vjp = 0.0
                damping_vjp = 0.0
                for row in range(4):
                    adjoint = self.linear_x[12 * body + 3 * row + component]
                    inertia_vjp += self.mass[body, row, control] * adjoint
                    projection = (1.0 if row == control else 0.0) - 0.25
                    damping_vjp += (0.0625 * force_coeff + torque_coeff * projection) * adjoint
                self.state_y_vjp[index][component] += inertia_vjp + damping_vjp
                self.state_velocity_vjp[index][component] += self.dt_device[None] * inertia_vjp

        for joint_id in range(self.joint_num):
            coefficient = self.dt_device[None] * self.joint_damping[joint_id]
            if coefficient > 0.0:
                body_a = self.joint_body[joint_id][0]
                body_b = self.joint_body[joint_id][1]
                angle = self._joint_angle_compact(joint_id, self.hat_y)
                delta_u_a = self._joint_direction_perturbation(joint_id, 0, 0, 1)
                delta_v_a = self._joint_direction_perturbation(joint_id, 0, 1, 1)
                residual = ti.Vector.zero(float, 3)
                adjoint_residual = ti.Vector.zero(float, 3)
                for local in range(8):
                    side, control = local // 4, local % 4
                    body = self.joint_body[joint_id][side]
                    if body >= 0:
                        weight = self._joint_linear_weight(joint_id, 2, angle, local)
                        residual += weight * (self.y[4 * body + control] - self.hat_y[4 * body + control])
                        adjoint = ti.Vector.zero(float, 3)
                        for component in ti.static(range(3)):
                            adjoint[component] = self.linear_x[12 * body + 3 * control + component]
                        adjoint_residual += weight * adjoint

                # The forward damping energy is 1/2*k*||W(hat_y)(y-hat_y)||^2.
                # Pull back its frozen linear residual, including the dependence
                # of W on the previous-step joint angle.
                angle_residual = ti.Vector.zero(float, 3)
                angle_adjoint = 0.0
                if body_b >= 0:
                    angle_residual = ti.sin(angle) * delta_u_a - ti.cos(angle) * delta_v_a
                    for control in ti.static(range(4)):
                        weight_derivative = (
                            ti.sin(angle) * self.joint_u_weight[joint_id, 0][control]
                            - ti.cos(angle) * self.joint_v_weight[joint_id, 0][control]
                        )
                        adjoint_a = ti.Vector.zero(float, 3)
                        for component in ti.static(range(3)):
                            adjoint_a[component] = self.linear_x[12 * body_a + 3 * control + component]
                        angle_adjoint += weight_derivative * adjoint_a.dot(residual)
                    angle_adjoint += adjoint_residual.dot(angle_residual)

                for local in range(8):
                    side, control = local // 4, local % 4
                    body = self.joint_body[joint_id][side]
                    if body >= 0:
                        base = ti.Vector.zero(float, 3)
                        if body_b >= 0:
                            if side == 0:
                                base = (
                                    -self.joint_u_weight[joint_id, 0][control] * ti.cos(angle)
                                    - self.joint_v_weight[joint_id, 0][control] * ti.sin(angle)
                                ) * adjoint_residual
                            else:
                                base = self.joint_u_weight[joint_id, 1][control] * adjoint_residual
                        else:
                            base = self.joint_u_weight[joint_id, 0][control] * adjoint_residual
                        gradient = ti.Vector.zero(float, 3)
                        if body_b >= 0:
                            gradient = self._joint_angle_gradient_compact(joint_id, local, self.hat_y)
                        for component in ti.static(range(3)):
                            ti.atomic_add(
                                self.state_y_vjp[4 * body + control][component],
                                coefficient * (base[component] - angle_adjoint * gradient[component]),
                            )

    @ti.kernel
    def device_propagate_lagged_friction_state_adjoint(self):
        contact_count = ti.min(self.friction_contact_count[0], self.friction_contact_capacity)
        for contact in range(contact_count):
            bodies = self.friction_contact_bodies[contact]
            vertices = self.friction_contact_vertices[contact]
            weights = self.friction_contact_weights[contact]
            rel = ti.Vector.zero(float, 3)
            adjoint_rel = ti.Vector.zero(float, 3)
            for site in ti.static(range(4)):
                rel += weights[site] * self.x[vertices[site]]
                for control in range(4):
                    basis = self.basis[vertices[site], control]
                    for component in ti.static(range(3)):
                        adjoint_rel[component] += (
                            weights[site] * basis * self.linear_x[12 * bodies[site] + 3 * control + component]
                        )
            response = (
                self._lagged_friction_hessian(
                    rel,
                    self.friction_contact_hat_rel[contact],
                    self.friction_contact_normal[contact],
                    self.friction_scale[0] * self.friction_contact_coeff[contact],
                )
                @ adjoint_rel
            )
            for site in ti.static(range(4)):
                for control in range(4):
                    factor = weights[site] * self.basis[vertices[site], control]
                    for component in ti.static(range(3)):
                        ti.atomic_add(
                            self.state_y_vjp[4 * bodies[site] + control][component],
                            factor * response[component],
                        )

    @ti.kernel
    def device_add_translation_gravity_vjp(self):
        self.gravity_vjp[None] += self.dt_device[None] * self.translation_correction_vjp[None]

    def sync_output_state(self):
        """Download an accepted snapshot only for recorder/user output."""
        shape = (self.body_num, 4, 3)
        self.state.y = self.y.to_numpy()[: self.control_num].reshape(shape)
        self.state.v_y = self.velocity_y.to_numpy()[: self.control_num].reshape(shape)
        self.state.y_n1 = self.previous_y.to_numpy()[: self.control_num].reshape(shape)
        self.state.v_y_n1 = self.previous_velocity_y.to_numpy()[: self.control_num].reshape(shape)

    @ti.kernel
    def device_backup_line_search_base(self):
        for control in range(self.control_num):
            self.line_search_base_y[control] = self.y[control]

    @ti.kernel
    def device_backup_step_start(self):
        for control in range(self.control_num):
            self.step_start_y[control] = self.y[control]

    @ti.kernel
    def device_restore_step_start(self):
        for control in range(self.control_num):
            self.y[control] = self.step_start_y[control]

    @ti.kernel
    def device_set_line_search_trial(self, alpha: float):
        for control in range(self.control_num):
            self.y[control] = self.line_search_base_y[control] + alpha * self.direction_y[control]

    @ti.kernel
    def device_restore_line_search_base(self):
        for control in range(self.control_num):
            self.y[control] = self.line_search_base_y[control]

    @ti.kernel
    def device_load_negative_gradient(self, rhs: ti.template()):
        for control in range(self.control_num):
            rhs[control] = -self.grad[control]

    @ti.kernel
    def device_copy_hash_solution_to_direction(self, solution: ti.template()):
        for control in range(self.control_num):
            self.direction_y[control] = solution[control]

    @ti.kernel
    def device_load_negative_gradient_scalar(self, rhs: ti.template()):
        for dof in range(self.dof):
            control = dof // 3
            component = dof - 3 * control
            rhs[dof] = -self.grad[control][component]

    @ti.kernel
    def device_copy_scalar_solution_to_direction(self, solution: ti.template()):
        for dof in range(self.dof):
            control = dof // 3
            component = dof - 3 * control
            self.direction_y[control][component] = solution[dof]

    @ti.kernel
    def device_copy_direction_to_hash_vector(self, vector: ti.template()):
        for control in range(self.control_num):
            vector[control] = self.direction_y[control]

    @ti.kernel
    def device_scale_direction(self, scale: float):
        for control in range(self.control_num):
            self.direction_y[control] *= scale

    @ti.kernel
    def device_direction_inf_norm(self) -> float:
        result = 0.0
        for control, component in ti.ndrange(self.control_num, 3):
            ti.atomic_max(result, ti.abs(self.direction_y[control][component]))
        return result

    @ti.kernel
    def device_surface_direction_inf_norm(self) -> float:
        """Maximum physical surface-coordinate correction on device."""
        result = 0.0
        for vertex in range(self.vertex_num):
            body = self.node2body[vertex]
            direction = ti.Vector.zero(float, 3)
            for local_control in range(4):
                direction += self.basis[vertex, local_control] * self.direction_y[body * 4 + local_control]
            for component in ti.static(range(3)):
                ti.atomic_max(result, ti.abs(direction[component]))
        return result

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

    @ti.kernel
    def device_gradient_inf_norm(self) -> float:
        result = 0.0
        for control, component in ti.ndrange(self.control_num, 3):
            ti.atomic_max(result, ti.abs(self.grad[control][component]))
        return result

    @ti.kernel
    def device_gradient_squared_norm(self) -> float:
        result = 0.0
        for control, component in ti.ndrange(self.control_num, 3):
            value = self.grad[control][component]
            result += value * value
        return result

    @ti.kernel
    def device_gradient_direction_dot(self) -> float:
        result = 0.0
        for control, component in ti.ndrange(self.control_num, 3):
            result += self.grad[control][component] * self.direction_y[control][component]
        return result

    @ti.kernel
    def device_residual_jacobian_direction_dot(self, matrix_direction: ti.template(), jacobian_shift: float) -> float:
        """Return R^T (Jp), excluding optional solver regularization."""
        result = 0.0
        for control, component in ti.ndrange(self.control_num, 3):
            jp = matrix_direction[control][component] - jacobian_shift * self.direction_y[control][component]
            result += self.grad[control][component] * jp
        return result

    @ti.kernel
    def device_direction_has_nonfinite(self) -> int:
        invalid = 0
        for control, component in ti.ndrange(self.control_num, 3):
            value = self.direction_y[control][component]
            if ti.math.isnan(value) or ti.math.isinf(value):
                ti.atomic_max(invalid, 1)
        return invalid

    @ti.kernel
    def device_gradient_has_nonfinite(self) -> int:
        invalid = 0
        for control, component in ti.ndrange(self.control_num, 3):
            value = self.grad[control][component]
            if ti.math.isnan(value) or ti.math.isinf(value):
                ti.atomic_max(invalid, 1)
        return invalid

    def matvec(self, x, Ax):
        if self.coo_matrix is None:
            raise RuntimeError("Affine COO matrix has not been bound.")
        self.coo_matrix.linear_operator.matvec(x, Ax)

    @ti.kernel
    def _build_coo_preconditioner(self, min_preconditioner: float):
        for i in range(self.dof):
            self.linear_M[i] = ti.max(ti.abs(self.K_coo_diag[i]), min_preconditioner)

    @ti.kernel
    def _coo_matvec(self, x: ti.template(), Ax: ti.template()):
        for i in range(self.dof):
            Ax[i] = 0.0
        for k in range(ti.min(self.coo_entry_count[None], self.max_coo_entries)):
            value = self.K_coo_values[k]
            if value != 0.0:
                ti.atomic_add(Ax[self.K_coo_rows[k]], value * x[self.K_coo_cols[k]])

    @ti.func
    def _sample_affine_levelset(self, body_id, template_point):
        """Trilinear SDF value and exact in-cell derivatives."""
        linear = self.levelset_template_to_grid[body_id]
        grid_point = linear @ template_point + self.levelset_template_to_grid_offset[body_id]
        origin = self.levelset_grid_origin[body_id]
        spacing = ti.max(self.levelset_grid_spacing[body_id], 1.0e-12)
        shape = self.levelset_grid_shape[body_id]
        reduced = (grid_point - origin) / spacing
        inside = (
            reduced[0] >= 0.0
            and reduced[1] >= 0.0
            and reduced[2] >= 0.0
            and reduced[0] <= shape[0] - 1
            and reduced[1] <= shape[1] - 1
            and reduced[2] <= shape[2] - 1
        )
        base = ti.Vector.zero(ti.i32, 3)
        fraction = ti.Vector.zero(float, 3)
        for d in ti.static(range(3)):
            base[d] = ti.min(shape[d] - 2, ti.max(0, ti.floor(reduced[d], ti.i32)))
            fraction[d] = ti.min(1.0, ti.max(0.0, reduced[d] - base[d]))

        phi = 0.0
        gradient_reduced = ti.Vector.zero(float, 3)
        hessian_reduced = ti.Matrix.zero(float, 3, 3)
        if inside:
            start = self.levelset_grid_start[body_id]
            for i in ti.static(range(2)):
                wx = (1.0 - fraction[0]) if i == 0 else fraction[0]
                dwx = -1.0 if i == 0 else 1.0
                for j in ti.static(range(2)):
                    wy = (1.0 - fraction[1]) if j == 0 else fraction[1]
                    dwy = -1.0 if j == 0 else 1.0
                    for k in ti.static(range(2)):
                        wz = (1.0 - fraction[2]) if k == 0 else fraction[2]
                        dwz = -1.0 if k == 0 else 1.0
                        node = (
                            linearize3D(
                                base[0] + i,
                                base[1] + j,
                                base[2] + k,
                                shape,
                            )
                            + start
                        )
                        nodal = self.levelset_grid_value[node]
                        phi += nodal * wx * wy * wz
                        gradient_reduced[0] += nodal * dwx * wy * wz
                        gradient_reduced[1] += nodal * wx * dwy * wz
                        gradient_reduced[2] += nodal * wx * wy * dwz
                        hessian_reduced[0, 1] += nodal * dwx * dwy * wz
                        hessian_reduced[0, 2] += nodal * dwx * wy * dwz
                        hessian_reduced[1, 2] += nodal * wx * dwy * dwz
        hessian_reduced[1, 0] = hessian_reduced[0, 1]
        hessian_reduced[2, 0] = hessian_reduced[0, 2]
        hessian_reduced[2, 1] = hessian_reduced[1, 2]
        gradient_grid = gradient_reduced / spacing
        hessian_grid = hessian_reduced / (spacing * spacing)
        gradient_template = linear.transpose() @ gradient_grid
        hessian_template = linear.transpose() @ hessian_grid @ linear
        return phi, gradient_template, hessian_template, inside

    @ti.func
    def _affine_levelset_matrix(self, body_id):
        y0 = self.y[body_id * 4]
        matrix = ti.Matrix.cols(
            [
                self.y[body_id * 4 + 1] - y0,
                self.y[body_id * 4 + 2] - y0,
                self.y[body_id * 4 + 3] - y0,
            ]
        )
        return matrix

    @ti.func
    def _affine_levelset_coefficient_dot(self, control, vector):
        value = 0.0
        if control == 0:
            value = -vector[0] - vector[1] - vector[2]
        elif control == 1:
            value = vector[0]
        elif control == 2:
            value = vector[1]
        else:
            value = vector[2]
        return value

    @ti.func
    def _affine_levelset_jacobian_column(self, jacobian, column):
        return ti.Vector(
            [
                jacobian[0, column],
                jacobian[1, column],
                jacobian[2, column],
            ]
        )

    @ti.kernel
    def _assemble_levelset_contacts(self, need_matrix: ti.template(), matrix_target: ti.template()):
        """Assemble symmetric surface-quadrature/SDF normal IPC."""
        self.levelset_active_contacts[None] = 0
        self.levelset_minimum_gap[None] = 1.0e30
        offset = ti.Vector([0.25, 0.25, 0.25])
        task_count = self.active_body_pair_count[None] * 2 * ti.static(self.max_body_vertex_count)
        for task in range(task_count):
            pair_slot = task // (2 * ti.static(self.max_body_vertex_count))
            pair_local = task - pair_slot * 2 * ti.static(self.max_body_vertex_count)
            orientation = pair_local // ti.static(self.max_body_vertex_count)
            local_vertex = pair_local - orientation * ti.static(self.max_body_vertex_count)
            pair_key = self.active_body_pair_list[pair_slot]
            body_i = pair_key // self.body_num
            body_j = pair_key - body_i * self.body_num
            source_body = body_i if orientation == 0 else body_j
            target_body = body_j if orientation == 0 else body_i
            vertex_id = -1
            if local_vertex < self.body_vertex_count[source_body]:
                vertex_id = self.body_vertex_start[source_body] + local_vertex
            if (
                vertex_id >= 0
                and self._body_pair_allowed(source_body, target_body)
                and self.body_contact_type[source_body] == 1
                and self.body_contact_type[target_body] == 1
            ):
                target_A = self._affine_levelset_matrix(target_body)
                determinant = target_A.determinant()
                if ti.abs(determinant) > 1.0e-12:
                    inverse_A = target_A.inverse()
                    target_y0 = self.y[target_body * 4]
                    material = inverse_A @ (self.x[vertex_id] - target_y0)
                    target_scale = self.body_scale[target_body]
                    target_coordinate = (material - offset) / target_scale
                    (
                        phi,
                        phi_gradient,
                        phi_hessian,
                        inside,
                    ) = self._sample_affine_levelset(target_body, target_coordinate)
                    if inside:
                        gap = target_scale * phi
                        ti.atomic_min(self.levelset_minimum_gap[None], gap)
                        dhat = self._pp_dhat(source_body, target_body)
                        key = ti.Vector([vertex_id, target_body, 3, -1])
                        active = gap < dhat
                        if ti.static(self.is_semi):
                            slot = semi_ipc_find(
                                self.semi_state,
                                self.semi_key,
                                key,
                                ti.static(self.semi_capacity),
                            )
                            multiplier = 0.0
                            if slot >= 0:
                                multiplier = self.semi_multiplier[slot]
                            active = multiplier - self._pp_penalty(source_body, target_body) * (gap - dhat) >= 0.0
                        if active:
                            ti.atomic_add(self.levelset_active_contacts[None], 1)
                            kappa = self._pp_kappa(source_body, target_body)
                            energy, dphi, ddphi = self._ipc_barrier_gap(gap, dhat, kappa)
                            if ti.static(self.is_semi):
                                energy, dphi, ddphi = self._semi_terms(
                                    key,
                                    gap - dhat,
                                    self._pp_penalty(source_body, target_body),
                                )
                            coefficient = 0.5 * self.scale_device[None] * self.node_area[vertex_id]
                            ti.atomic_add(
                                self.energy[None],
                                coefficient * energy,
                            )

                            site_weight = ti.Vector.zero(float, 8)
                            for site in range(4):
                                site_weight[site] = self.basis[vertex_id, site]
                            site_weight[4] = -(1.0 - material[0] - material[1] - material[2])
                            site_weight[5] = -material[0]
                            site_weight[6] = -material[1]
                            site_weight[7] = -material[2]
                            q_jacobian = ti.Matrix.zero(float, 3, 24)
                            gap_gradient = ti.Vector.zero(float, 24)
                            for local_dof in ti.static(range(24)):
                                site = local_dof // 3
                                component = local_dof - 3 * site
                                for d in ti.static(range(3)):
                                    q_jacobian[d, local_dof] = inverse_A[d, component] * site_weight[site]
                                jacobian_column = self._affine_levelset_jacobian_column(q_jacobian, local_dof)
                                gap_gradient[local_dof] = phi_gradient.dot(jacobian_column)
                                body = source_body if site < 4 else target_body
                                control = site if site < 4 else site - 4
                                ti.atomic_add(
                                    self.grad[body * 4 + control][component],
                                    coefficient * dphi * gap_gradient[local_dof],
                                )

                            if ti.static(need_matrix):
                                hessian_q = phi_hessian / target_scale
                                world_normal = inverse_A.transpose() @ phi_gradient
                                for site_i, site_j in ti.ndrange(8, 8):
                                    block = ti.Matrix.zero(float, 3, 3)
                                    for component_i, component_j in ti.static(ti.ndrange(3, 3)):
                                        local_i = 3 * site_i + component_i
                                        local_j = 3 * site_j + component_j
                                        jacobian_i = self._affine_levelset_jacobian_column(q_jacobian, local_i)
                                        jacobian_j = self._affine_levelset_jacobian_column(q_jacobian, local_j)
                                        gap_hessian = jacobian_i.dot(hessian_q @ jacobian_j)
                                        if site_i >= 4:
                                            gap_hessian -= world_normal[
                                                component_i
                                            ] * self._affine_levelset_coefficient_dot(
                                                site_i - 4,
                                                jacobian_j,
                                            )
                                        if site_j >= 4:
                                            gap_hessian -= world_normal[
                                                component_j
                                            ] * self._affine_levelset_coefficient_dot(
                                                site_j - 4,
                                                jacobian_i,
                                            )
                                        block[component_i, component_j] = (
                                            coefficient * ddphi * gap_gradient[local_i] * gap_gradient[local_j]
                                        )
                                        if ti.static(self.fully_implicit):
                                            block[component_i, component_j] += coefficient * dphi * gap_hessian
                                    body_i = source_body if site_i < 4 else target_body
                                    control_i = site_i if site_i < 4 else site_i - 4
                                    body_j = source_body if site_j < 4 else target_body
                                    control_j = site_j if site_j < 4 else site_j - 4
                                    self._add_matrix_block(
                                        body_i * 4 + control_i,
                                        body_j * 4 + control_j,
                                        block,
                                        matrix_target,
                                    )

    @ti.kernel
    def _assemble_levelset_lagged_friction(self, need_matrix: ti.template(), matrix_target: ti.template()):
        """Assemble lagged Coulomb friction for SDF contact stencils.

        The active set, target material coordinate, tangent plane, and
        normal-force magnitude are evaluated at ``levelset_friction_y`` and
        then held fixed during the inner Newton solve.  The current slip is
        the relative motion of the source quadrature point and that frozen
        target material point, pulled linearly to the two bodies' 24 affine
        DOFs.  This is the level-set counterpart of IPC's lagged PT/EE
        friction potential.
        """
        self.levelset_friction_contacts[None] = 0
        offset = ti.Vector([0.25, 0.25, 0.25])
        task_count = self.active_body_pair_count[None] * 2 * ti.static(self.max_body_vertex_count)
        for task in range(task_count):
            pair_slot = task // (2 * ti.static(self.max_body_vertex_count))
            pair_local = task - pair_slot * 2 * ti.static(self.max_body_vertex_count)
            orientation = pair_local // ti.static(self.max_body_vertex_count)
            local_vertex = pair_local - orientation * ti.static(self.max_body_vertex_count)
            pair_key = self.active_body_pair_list[pair_slot]
            body_i = pair_key // self.body_num
            body_j = pair_key - body_i * self.body_num
            source_body = body_i if orientation == 0 else body_j
            target_body = body_j if orientation == 0 else body_i
            vertex_id = -1
            if local_vertex < self.body_vertex_count[source_body]:
                vertex_id = self.body_vertex_start[source_body] + local_vertex
            if (
                vertex_id >= 0
                and self._body_pair_allowed(source_body, target_body)
                and self.body_contact_type[source_body] == 1
                and self.body_contact_type[target_body] == 1
            ):
                source_frozen = ti.Vector.zero(float, 3)
                for control in range(4):
                    source_frozen += (
                        self.basis[vertex_id, control] * self.levelset_friction_y[source_body * 4 + control]
                    )
                target_y0 = self.levelset_friction_y[target_body * 4]
                target_A = ti.Matrix.cols(
                    [
                        self.levelset_friction_y[target_body * 4 + 1] - target_y0,
                        self.levelset_friction_y[target_body * 4 + 2] - target_y0,
                        self.levelset_friction_y[target_body * 4 + 3] - target_y0,
                    ]
                )
                if ti.abs(target_A.determinant()) > 1.0e-12:
                    inverse_A = target_A.inverse()
                    material = inverse_A @ (source_frozen - target_y0)
                    scale = self.body_scale[target_body]
                    coordinate = (material - offset) / scale
                    phi, phi_gradient, unused_hessian, inside = self._sample_affine_levelset(target_body, coordinate)
                    gap = scale * phi
                    dhat = self._pp_dhat(source_body, target_body)
                    mu = self._pp_mu(source_body, target_body)
                    if inside and mu > 0.0 and ((0.0 < gap < dhat) or ti.static(self.is_semi)):
                        barrier_gradient = 0.0
                        if ti.static(self.is_semi):
                            unused_energy, barrier_gradient, unused_second = self._semi_terms(
                                ti.Vector([vertex_id, target_body, 3, -1]),
                                gap - dhat,
                                self._pp_penalty(source_body, target_body),
                            )
                        else:
                            unused_energy, barrier_gradient, unused_second = self._ipc_barrier_gap(
                                gap,
                                dhat,
                                self._pp_kappa(source_body, target_body),
                            )
                        normal = inverse_A.transpose() @ phi_gradient
                        if normal.norm() > 1.0e-14:
                            weights = ti.Vector.zero(float, 8)
                            for control in range(4):
                                weights[control] = self.basis[vertex_id, control]
                            weights[4] = -(1.0 - material.sum())
                            weights[5] = -material[0]
                            weights[6] = -material[1]
                            weights[7] = -material[2]
                            relative = ti.Vector.zero(float, 3)
                            reference_relative = ti.Vector.zero(float, 3)
                            for site in ti.static(range(8)):
                                body = source_body if site < 4 else target_body
                                control = site if site < 4 else site - 4
                                relative += weights[site] * self.y[body * 4 + control]
                                reference_relative += weights[site] * self.hat_y[body * 4 + control]
                            coefficient = (
                                self.friction_scale[0]
                                * mu
                                * ti.max(-barrier_gradient, 0.0)
                                * 0.5
                                * self.node_area[vertex_id]
                                * self.scale_device[None]
                            )
                            if coefficient > 0.0:
                                ti.atomic_add(
                                    self.levelset_friction_contacts[None],
                                    1,
                                )
                                self._assemble_levelset_local_friction(
                                    source_body,
                                    target_body,
                                    weights,
                                    relative,
                                    reference_relative,
                                    normal,
                                    coefficient,
                                    need_matrix,
                                    matrix_target,
                                )

    @ti.func
    def _affine_levelset_world_gap_at_step(self, source_point, source_direction, target_body, alpha):
        query = source_point + alpha * source_direction
        target_control = ti.Matrix.zero(float, 4, 3)
        for control, component in ti.static(ti.ndrange(4, 3)):
            target_control[control, component] = (
                self.y[target_body * 4 + control][component]
                + alpha * self.direction_y[target_body * 4 + control][component]
            )
        target_y0 = ti.Vector(
            [
                target_control[0, 0],
                target_control[0, 1],
                target_control[0, 2],
            ]
        )
        target_y1 = ti.Vector(
            [
                target_control[1, 0],
                target_control[1, 1],
                target_control[1, 2],
            ]
        )
        target_y2 = ti.Vector(
            [
                target_control[2, 0],
                target_control[2, 1],
                target_control[2, 2],
            ]
        )
        target_y3 = ti.Vector(
            [
                target_control[3, 0],
                target_control[3, 1],
                target_control[3, 2],
            ]
        )
        target_A = ti.Matrix.cols(
            [
                target_y1 - target_y0,
                target_y2 - target_y0,
                target_y3 - target_y0,
            ]
        )
        gap = self.dhat
        inside = False
        if ti.abs(target_A.determinant()) > 1.0e-12:
            material = target_A.inverse() @ (query - target_y0)
            coordinate = (material - ti.Vector([0.25, 0.25, 0.25])) / self.body_scale[target_body]
            phi, unused_gradient, unused_hessian, inside = self._sample_affine_levelset(target_body, coordinate)
            if inside:
                gap = self.body_scale[target_body] * phi
        return gap, inside

    @ti.func
    def _affine_levelset_gap_at_step(self, vertex_id, target_body, alpha):
        source_body = self.node2body[vertex_id]
        source_point = ti.Vector.zero(float, 3)
        source_direction = ti.Vector.zero(float, 3)
        for control in range(4):
            weight = self.basis[vertex_id, control]
            source_point += weight * self.y[source_body * 4 + control]
            source_direction += weight * self.direction_y[source_body * 4 + control]
        return self._affine_levelset_world_gap_at_step(
            source_point,
            source_direction,
            target_body,
            alpha,
        )

    @ti.func
    def _affine_levelset_target_path_bound(self, target_body):
        """Bound displacement of every material point in the SDF grid box."""
        inverse_grid_map = self.levelset_template_to_grid[target_body].inverse()
        grid_lower = self.levelset_grid_origin[target_body]
        grid_upper = grid_lower + self.levelset_grid_spacing[target_body] * (
            self.levelset_grid_shape[target_body].cast(float) - 1.0
        )
        grid_offset = self.levelset_template_to_grid_offset[target_body]
        material_offset = ti.Vector([0.25, 0.25, 0.25])
        direction0 = self.direction_y[target_body * 4]
        direction1 = self.direction_y[target_body * 4 + 1]
        direction2 = self.direction_y[target_body * 4 + 2]
        direction3 = self.direction_y[target_body * 4 + 3]
        maximum = 0.0
        for corner in ti.static(range(8)):
            grid_corner = grid_lower
            if ti.static(corner & 1):
                grid_corner[0] = grid_upper[0]
            if ti.static(corner & 2):
                grid_corner[1] = grid_upper[1]
            if ti.static(corner & 4):
                grid_corner[2] = grid_upper[2]
            template_coordinate = inverse_grid_map @ (grid_corner - grid_offset)
            material = material_offset + self.body_scale[target_body] * template_coordinate
            direction = (
                direction0
                + material[0] * (direction1 - direction0)
                + material[1] * (direction2 - direction0)
                + material[2] * (direction3 - direction0)
            )
            maximum = ti.max(maximum, direction.norm())
        return maximum

    @ti.kernel
    def _compute_levelset_ccd_alpha(
        self,
        eta: float,
        accd_tolerance: float,
        max_iteration: ti.i32,
        accd: ti.template(),
    ):
        task_count = self.active_body_pair_count[None] * 2 * ti.static(self.max_body_vertex_count)
        for task in range(task_count):
            pair_slot = task // (2 * ti.static(self.max_body_vertex_count))
            pair_local = task - pair_slot * 2 * ti.static(self.max_body_vertex_count)
            orientation = pair_local // ti.static(self.max_body_vertex_count)
            local_vertex = pair_local - orientation * ti.static(self.max_body_vertex_count)
            pair_key = self.active_body_pair_list[pair_slot]
            body_i = pair_key // self.body_num
            body_j = pair_key - body_i * self.body_num
            source_body = body_i if orientation == 0 else body_j
            target_body = body_j if orientation == 0 else body_i
            vertex_id = -1
            if local_vertex < self.body_vertex_count[source_body]:
                vertex_id = self.body_vertex_start[source_body] + local_vertex
            if (
                vertex_id >= 0
                and self._body_pair_allowed(source_body, target_body)
                and self.body_contact_type[source_body] == 1
                and self.body_contact_type[target_body] == 1
            ):
                gap0, inside0 = self._affine_levelset_gap_at_step(vertex_id, target_body, 0.0)
                if inside0 and gap0 <= 0.0:
                    if ti.static(not self.is_semi):
                        ti.atomic_min(self.ccd_alpha[None], 0.0)
                elif ti.static(accd):
                    if inside0:
                        clearance = ti.max(accd_tolerance, 0.0)
                        gap = gap0 - clearance
                        if gap <= 0.0:
                            ti.atomic_min(self.ccd_alpha[None], 0.0)
                        else:
                            source_direction = ti.Vector.zero(float, 3)
                            for control in range(4):
                                source_direction += (
                                    self.basis[vertex_id, control] * self.direction_y[source_body * 4 + control]
                                )
                            path_bound = self.levelset_lipschitz[target_body] * (
                                source_direction.norm() + self._affine_levelset_target_path_bound(target_body)
                            )
                            if path_bound > 1.0e-30:
                                target_gap = eta * gap
                                current_alpha = 0.0
                                active = True
                                iteration = 0
                                while active and iteration < max_iteration:
                                    lower_bound = (gap - target_gap) / path_bound
                                    if lower_bound <= 1.0e-12:
                                        active = False
                                    elif current_alpha + lower_bound >= 1.0:
                                        current_alpha = 1.0
                                        active = False
                                    else:
                                        current_alpha += lower_bound
                                        sample_gap, sample_inside = self._affine_levelset_gap_at_step(
                                            vertex_id,
                                            target_body,
                                            current_alpha,
                                        )
                                        if sample_inside:
                                            gap = sample_gap - clearance
                                            if gap <= target_gap:
                                                active = False
                                        else:
                                            # A padded SDF normally keeps the
                                            # complete contact path inside.  The
                                            # symmetric orientation guards a path
                                            # that leaves this target grid.
                                            active = False
                                    iteration += 1
                                ti.atomic_min(
                                    self.ccd_alpha[None],
                                    ti.max(0.0, ti.min(1.0, current_alpha)),
                                )
                else:
                    reference_gap = gap0 if inside0 else self._pp_dhat(source_body, target_body)
                    safe_gap = ti.max(
                        1.0e-12,
                        eta * reference_gap,
                    )

                    source_direction = ti.Vector.zero(float, 3)
                    for control in range(4):
                        source_direction += self.basis[vertex_id, control] * self.direction_y[source_body * 4 + control]
                    path_bound = self.levelset_lipschitz[target_body] * (
                        source_direction.norm() + self._affine_levelset_target_path_bound(target_body)
                    )
                    cell_size = ti.max(
                        1.0e-12,
                        self.body_scale[target_body] * self.levelset_grid_spacing[target_body],
                    )
                    segment_count = ti.max(
                        8,
                        int(ti.ceil(4.0 * path_bound / cell_size)),
                    )
                    lower = 0.0
                    upper = 1.0
                    bracketed = False
                    for segment in range(1, segment_count + 1):
                        sample_alpha = float(segment) / float(segment_count)
                        sample_gap, sample_inside = self._affine_levelset_gap_at_step(
                            vertex_id,
                            target_body,
                            sample_alpha,
                        )
                        if not bracketed and sample_inside and sample_gap < safe_gap:
                            lower = float(segment - 1) / float(segment_count)
                            upper = sample_alpha
                            bracketed = True
                    if bracketed:
                        for iteration in range(64):
                            if iteration < max_iteration:
                                midpoint = 0.5 * (lower + upper)
                                middle_gap, middle_inside = self._affine_levelset_gap_at_step(
                                    vertex_id,
                                    target_body,
                                    midpoint,
                                )
                                if not middle_inside or middle_gap >= safe_gap:
                                    lower = midpoint
                                else:
                                    upper = midpoint
                        ti.atomic_min(
                            self.ccd_alpha[None],
                            ti.max(
                                0.0,
                                ti.min(
                                    1.0,
                                    lower,
                                ),
                            ),
                        )

    def _assemble_wall_contacts(self, need_matrix, matrix_target, project_pd=None):
        if self.wall_num > 0:
            if project_pd is None:
                project_pd = not self.fully_implicit
            self._assemble_wall_contacts_kernel(bool(need_matrix), int(matrix_target), bool(project_pd))

    def _assemble_semi_wall_contacts(self, need_matrix, matrix_target):
        if self.wall_num > 0:
            self._assemble_semi_wall_contacts_kernel(bool(need_matrix), int(matrix_target))

    @ti.kernel
    def _assemble_semi_wall_contacts_kernel(self, need_matrix: ti.template(), matrix_target: ti.template()):
        for vertex_id, wall_id in ti.ndrange(self.vertex_num, self.wall_num):
            body = self.node2body[vertex_id]
            point = self.x[vertex_id]
            normal = self.wall_normal[wall_id]
            gap = 0.0
            if self.wall_type[wall_id] == 0:
                gap = (point - self.wall_point[wall_id]).dot(normal)
            else:
                closest, unused_barycentric = self._closest_point_triangle(
                    point,
                    self.wall_v0[wall_id],
                    self.wall_v1[wall_id],
                    self.wall_v2[wall_id],
                )
                delta = point - closest
                distance = delta.norm()
                if distance > 1.0e-15:
                    normal = delta / distance
                gap = distance
            gap -= self._pw_dhat(body, wall_id)
            key = ti.Vector([vertex_id, wall_id, 2, -1])
            slot = semi_ipc_find(self.semi_state, self.semi_key, key, ti.static(self.semi_capacity))
            multiplier = 0.0
            if slot >= 0:
                multiplier = self.semi_multiplier[slot]
            if multiplier - self._pw_penalty(body, wall_id) * gap >= 0.0:
                energy, first, second = self._semi_terms(key, gap, self._pw_penalty(body, wall_id))
                coefficient = self.scale_device[None] * self.node_area[vertex_id]
                ti.atomic_add(self.energy[None], coefficient * energy)
                self._scatter_contact(
                    body,
                    vertex_id,
                    1.0,
                    normal,
                    coefficient * first,
                    coefficient * second,
                    need_matrix,
                )
                if ti.static(need_matrix):
                    self._scatter_hessian_pair(
                        body,
                        vertex_id,
                        1.0,
                        body,
                        vertex_id,
                        1.0,
                        normal,
                        coefficient * second,
                        matrix_target,
                    )

    @ti.kernel
    def _assemble_wall_contacts_kernel(
        self,
        need_matrix: ti.template(),
        matrix_target: ti.template(),
        project_pd: ti.template(),
    ):
        for vertex_id, wall_id in ti.ndrange(self.vertex_num, self.wall_num):
            body_i = self.node2body[vertex_id]
            p = self.x[vertex_id]
            normal = self.wall_normal[wall_id]
            gap = 0.0
            delta = ti.Vector.zero(float, 3)
            active = False
            dhat = self._pw_dhat(body_i, wall_id)
            kappa = self._pw_kappa(body_i, wall_id)
            if self.wall_type[wall_id] == 0:
                gap = (p - self.wall_point[wall_id]).dot(normal)
                delta = gap * normal
                active = gap < dhat
            else:
                closest, bary = self._closest_point_triangle(
                    p, self.wall_v0[wall_id], self.wall_v1[wall_id], self.wall_v2[wall_id]
                )
                delta = p - closest
                gap = delta.norm()
                if gap > 1.0e-12:
                    normal = delta / gap
                active = gap < dhat
            if active:
                if self.wall_type[wall_id] == 0:
                    coeff = self.scale_device[None] * self.node_area[vertex_id]
                    energy, dphi, ddphi = self._ipc_barrier_gap(gap, dhat, kappa)
                    ti.atomic_add(self.energy[None], coeff * energy)
                    self._scatter_contact(body_i, vertex_id, 1.0, normal, coeff * dphi, coeff * ddphi, need_matrix)
                    if ti.static(need_matrix):
                        self._scatter_hessian_pair(
                            body_i, vertex_id, 1.0, body_i, vertex_id, 1.0, normal, coeff * ddphi, matrix_target
                        )
                else:
                    if ti.static(need_matrix):
                        dist2, grad_d, hess_d, dtype = point_triangle_distance_grad_hess(
                            p,
                            self.wall_v0[wall_id],
                            self.wall_v1[wall_id],
                            self.wall_v2[wall_id],
                        )
                        dist = ti.sqrt(ti.max(dist2, 1.0e-30))
                        coeff = self.scale_device[None] * self.node_area[vertex_id]
                        energy, db, ddb = self._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
                        ti.atomic_add(self.energy[None], coeff * energy)
                        point_grad = ti.Vector.zero(float, 3)
                        for di in ti.static(range(3)):
                            point_grad[di] = coeff * db * grad_d[di]
                        self._scatter_gradient_vec(body_i, vertex_id, 1.0, point_grad)
                        point_hess = ti.Matrix.zero(float, 3, 3)
                        for di in ti.static(range(3)):
                            for dj in ti.static(range(3)):
                                point_hess[di, dj] = coeff * (ddb * grad_d[di] * grad_d[dj] + db * hess_d[di, dj])
                        if ti.static(project_pd):
                            point_hess = psd_project_nd(point_hess)
                        self._scatter_hessian_block(
                            body_i,
                            vertex_id,
                            1.0,
                            body_i,
                            vertex_id,
                            1.0,
                            point_hess,
                            matrix_target,
                        )
                    else:
                        dist2, grad_d, dtype = point_triangle_distance_grad(
                            p,
                            self.wall_v0[wall_id],
                            self.wall_v1[wall_id],
                            self.wall_v2[wall_id],
                        )
                        dist = ti.sqrt(ti.max(dist2, 1.0e-30))
                        coeff = self.scale_device[None] * self.node_area[vertex_id]
                        energy, db, unused_ddb = self._ipc_barrier_distance2(dist2, dhat * dhat, kappa)
                        ti.atomic_add(self.energy[None], coeff * energy)
                        point_grad = ti.Vector.zero(float, 3)
                        for di in ti.static(range(3)):
                            point_grad[di] = coeff * db * grad_d[di]
                        self._scatter_gradient_vec(body_i, vertex_id, 1.0, point_grad)

    @ti.kernel
    def _init_ccd_alpha(self):
        self.ccd_alpha[None] = 1.0

    @ti.kernel
    def _compute_ccd_alpha(
        self,
        eta: float,
        accd_tolerance: float,
        max_iteration: ti.i32,
        accd: ti.template(),
        candidate_count: ti.template(),
        candidate_vertex: ti.template(),
        candidate_face: ti.template(),
        edge_candidate_count: ti.template(),
        candidate_edge0: ti.template(),
        candidate_edge1: ti.template(),
    ):
        thickness = 0.0
        if ti.static(accd):
            thickness = ti.max(accd_tolerance, 0.0)

        for vertex_id, wall_id in ti.ndrange(self.vertex_num, self.wall_num):
            p = self.x[vertex_id]
            dp = self.dx[vertex_id]
            normal = self.wall_normal[wall_id]
            alpha = 1.0
            if self.wall_type[wall_id] == 0:
                vn = dp.dot(normal)
                if vn < 0.0:
                    gap = (p - self.wall_point[wall_id]).dot(normal)
                    if ti.static(accd):
                        alpha = linear_gap_accd(
                            gap,
                            vn,
                            1.0 - 0.5 * eta,
                            thickness,
                        )
                    else:
                        alpha = linear_gap_ccd(gap, vn, 1.0 - 0.5 * eta)
            else:
                if ti.static(accd):
                    alpha = point_triangle_accd(
                        p,
                        self.wall_v0[wall_id],
                        self.wall_v1[wall_id],
                        self.wall_v2[wall_id],
                        dp,
                        ti.Vector.zero(float, 3),
                        ti.Vector.zero(float, 3),
                        ti.Vector.zero(float, 3),
                        eta,
                        thickness,
                        max_iteration,
                    )
                else:
                    alpha = point_triangle_ccd(
                        p,
                        self.wall_v0[wall_id],
                        self.wall_v1[wall_id],
                        self.wall_v2[wall_id],
                        dp,
                        ti.Vector.zero(float, 3),
                        ti.Vector.zero(float, 3),
                        ti.Vector.zero(float, 3),
                        eta,
                        max_iteration,
                    )
            ti.atomic_min(self.ccd_alpha[None], ti.max(0.0, ti.min(1.0, alpha)))

        for candidate_id in range(candidate_count[None]):
            vertex_id = candidate_vertex[candidate_id]
            face_id = candidate_face[candidate_id]
            body_i = self.node2body[vertex_id]
            body_j = self.face2body[face_id]
            if self._body_pair_allowed(body_i, body_j):
                face = self.faces[face_id]
                alpha = 1.0
                if ti.static(accd):
                    alpha = point_triangle_accd(
                        self.x[vertex_id],
                        self.x[face[0]],
                        self.x[face[1]],
                        self.x[face[2]],
                        self.dx[vertex_id],
                        self.dx[face[0]],
                        self.dx[face[1]],
                        self.dx[face[2]],
                        eta,
                        thickness,
                        max_iteration,
                    )
                else:
                    alpha = point_triangle_ccd(
                        self.x[vertex_id],
                        self.x[face[0]],
                        self.x[face[1]],
                        self.x[face[2]],
                        self.dx[vertex_id],
                        self.dx[face[0]],
                        self.dx[face[1]],
                        self.dx[face[2]],
                        eta,
                        max_iteration,
                    )
                ti.atomic_min(self.ccd_alpha[None], ti.max(0.0, ti.min(1.0, alpha)))

        for candidate_id in range(edge_candidate_count[None]):
            edge_i = candidate_edge0[candidate_id]
            edge_j = candidate_edge1[candidate_id]
            body_i = self.edge2body[edge_i]
            body_j = self.edge2body[edge_j]
            if self._body_pair_allowed(body_i, body_j):
                ei = self.edges[edge_i]
                ej = self.edges[edge_j]
                alpha = 1.0
                if ti.static(accd):
                    alpha = edge_edge_accd(
                        self.x[ei[0]],
                        self.x[ei[1]],
                        self.x[ej[0]],
                        self.x[ej[1]],
                        self.dx[ei[0]],
                        self.dx[ei[1]],
                        self.dx[ej[0]],
                        self.dx[ej[1]],
                        eta,
                        thickness,
                        max_iteration,
                    )
                else:
                    alpha = edge_edge_ccd(
                        self.x[ei[0]],
                        self.x[ei[1]],
                        self.x[ej[0]],
                        self.x[ej[1]],
                        self.dx[ei[0]],
                        self.dx[ei[1]],
                        self.dx[ej[0]],
                        self.dx[ej[1]],
                        eta,
                        max_iteration,
                    )
                ti.atomic_min(self.ccd_alpha[None], ti.max(0.0, ti.min(1.0, alpha)))


__all__ = [
    "MATRIX_CONTACT_DAMPING",
    "MATRIX_COO",
    "MATRIX_HASH_TRIPLET",
    "TaichiAffineBodyOperator",
    "_asarray3",
    "_euler_to_matrix",
    "_field_scalar",
    "_normalize_affine_assemble_type",
    "_normalize_affine_friction_mode",
]
