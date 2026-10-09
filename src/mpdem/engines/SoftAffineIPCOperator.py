"""Soft-particle/affine-body IPC state and Taichi operators."""

import os
import time
import math

import numpy as np
import taichi as ti
from src.physics_model.contact_model.ipc.ContactGeometry import aabb_overlap_with_clearance

from src.contact_detection.continuous_contact_detection import (
    ccd_mode_parameters,
    point_point_accd,
    point_point_ccd,
    point_triangle_accd,
    point_triangle_ccd,
)
from src.dem.engines.AffineBodyOperator import (
    MATRIX_CONTACT_DAMPING,
    MATRIX_HASH_TRIPLET,
    TaichiAffineBodyOperator,
)
from src.dem.engines.AffineBodyState import AffineBodyState
from src.fem.contact.BVHBroadPhase import DynamicBVHBroadPhase
from src.physics_model.contact_model.ipc.ContactDistance import (
    point_triangle_distance_grad,
    point_triangle_distance_grad_hess,
    point_triangle_distance_grad_by_type,
    point_triangle_distance_grad_hess_by_type,
    point_triangle_distance_type,
)
from src.physics_model.contact_model.ipc.ContactGeometry import (
    closest_point_triangle,
    normalized_contact_direction,
    point_point_stencil_weights,
    point_triangle_contact_frame,
)
from src.physics_model.contact_model.ipc.ContactAssembly import (
    psd_project_nd,
    pullback_relative_gradient4,
    pullback_relative_hessian4,
)
from src.physics_model.contact_model.ipc.ContactMeasure import symmetric_contact_measure
from src.physics_model.contact_model.ipc.IPC import (
    ipc_friction_f0,
    ipc_friction_f1_over_speed,
    ipc_friction_hessian_term,
    ipc_fully_implicit_profile_over_speed,
    ipc_fully_implicit_profile_over_speed_derivative,
    ipc_stribeck_falloff,
    ipc_stribeck_falloff_derivative,
    semi_ipc_find,
    semi_ipc_find_or_insert,
    semi_ipc_terms,
    semi_ipc_update_multiplier,
)
from src.linear_solver.BuildTriplet import BuildTriplet
from src.mpm.utils import add_field, copy_field
from src.utils.MatrixFunction import contraction, flatten_matrix, unflatten_matrix
from src.utils.PrefixSum import PrefixSumExecutor
from src.utils.Quaternion import SetToRotate
from src.utils.ScalarFunction import linearize3D
from src.utils.constants import Threshold, ZEROVEC3f
from src.utils.TypeDefination import vec3f, mat3x3
from third_party.pyevtk.hl import unstructuredGridToVTK
from third_party.pyevtk.vtk import VtkTriangle


def write_soft_affine_surface_vtu(vtk_path, snapshot):
    """Write the affine half of a soft-affine snapshot as triangle VTU."""
    vertices = np.asarray(snapshot["vertices"])
    faces = np.asarray(snapshot["faces"])
    if vertices.size == 0 or faces.size == 0:
        return False
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("soft-affine VTU vertices must have shape (n, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("soft-affine VTU faces must have shape (m, 3)")
    point_body_ids = np.asarray(snapshot["bodyID"], dtype=np.int32)
    point_group_ids = np.asarray(snapshot["groupID"], dtype=np.int32)
    if point_body_ids.shape != (vertices.shape[0],) or point_group_ids.shape != (vertices.shape[0],):
        raise ValueError("soft-affine VTU point labels must match the vertex count")
    face_body_ids = np.asarray(snapshot["faceBodyID"], dtype=np.int32)
    face_group_ids = np.asarray(snapshot["faceGroupID"], dtype=np.int32)
    face_material_ids = np.asarray(snapshot["faceMaterialID"], dtype=np.int32)
    face_count = faces.shape[0]
    for values in (face_body_ids, face_group_ids, face_material_ids):
        if values.shape != (face_count,):
            raise ValueError("soft-affine VTU cell labels must match the face count")
    unstructuredGridToVTK(
        str(vtk_path),
        np.ascontiguousarray(vertices[:, 0]),
        np.ascontiguousarray(vertices[:, 1]),
        np.ascontiguousarray(vertices[:, 2]),
        connectivity=np.ascontiguousarray(faces.reshape(-1).astype(np.int32)),
        offsets=np.ascontiguousarray(np.arange(3, 3 * face_count + 1, 3, dtype=np.int32)),
        cell_types=np.ascontiguousarray(np.full(face_count, VtkTriangle.tid, dtype=np.uint8)),
        pointData={
            "bodyID": np.ascontiguousarray(point_body_ids),
            "groupID": np.ascontiguousarray(point_group_ids),
        },
        cellData={
            "bodyID": np.ascontiguousarray(face_body_ids),
            "groupID": np.ascontiguousarray(face_group_ids),
            "materialID": np.ascontiguousarray(face_material_ids),
        },
    )
    return True


def soft_affine_friction_capabilities():
    """Return the nonlinear friction modes implemented by this driver."""
    return {
        "lagged": {
            "enabled": True,
            "soft_soft_pp": True,
            "mixed_point_triangle": True,
            "affine_self": True,
            "outer_fixed_point": True,
        },
        "fully_implicit": {
            "enabled": True,
            "soft_soft_pp": True,
            "mixed_point_triangle": True,
            "affine_self_pt_ee_wall": True,
            "current_contact_state": True,
            "full_nonsymmetric_jacobian": True,
            "residual_armijo": True,
            "taichi_device_hot_loop": True,
        },
    }


def _validate_soft_affine_lagged_friction_configuration(sims):
    friction_mode = (
        str(getattr(sims, "affine_friction_mode", getattr(sims, "friction_mode", "lagged")))
        .strip()
        .replace("-", "_")
        .lower()
    )
    if friction_mode in ("fullyimplicit", "fully_implicit"):
        value = getattr(
            sims,
            "affine_friction_iterations",
            getattr(
                sims,
                "friction_iterations",
                getattr(sims, "friction_fixed_point_iterations", 1),
            ),
        )
        try:
            iterations = int(value)
            exact_integer = float(value) == iterations
        except (TypeError, ValueError, OverflowError):
            iterations = None
            exact_integer = False
        if not exact_integer or iterations != 1:
            raise RuntimeError(
                "SoftAffineIPC fully implicit friction_iterations must be 1; "
                "the nonlinear friction law is solved inside the Newton system"
            )
        return "fully_implicit", 0
    if friction_mode not in ("lag", "lagged"):
        raise ValueError("SoftAffineIPC friction_mode must be 'lagged' or 'fully_implicit'")

    mu_dynamic = float(getattr(sims, "affine_dynamic_friction", -1.0))
    mu_static = float(getattr(sims, "affine_static_friction", -1.0))
    mu_viscous = float(getattr(sims, "affine_viscous_friction", 0.0))
    profile = str(getattr(sims, "affine_friction_profile", "quadratic")).strip().replace("-", "_").lower()
    if not all(np.isfinite(v) for v in (mu_dynamic, mu_static, mu_viscous)):
        raise RuntimeError("SoftAffineIPC friction coefficients must be finite")
    if mu_viscous != 0.0:
        raise RuntimeError(
            "SoftAffineIPC lagged friction does not support viscous_friction; " "use friction_mode='fully_implicit'"
        )
    if profile not in ("quadratic", "paper", "ipc", "c1"):
        raise RuntimeError(
            "SoftAffineIPC lagged friction only supports the quadratic IPC C1 "
            "profile; use friction_mode='fully_implicit' for stabilized friction"
        )
    if mu_static >= 0.0 and (mu_dynamic < 0.0 or mu_static != mu_dynamic):
        raise RuntimeError(
            "SoftAffineIPC lagged friction only supports one Coulomb coefficient; "
            "use friction_mode='fully_implicit' when static_friction differs "
            "from dynamic_friction"
        )
    if (mu_dynamic < 0.0 and mu_dynamic != -1.0) or (mu_static < 0.0 and mu_static != -1.0):
        raise RuntimeError("SoftAffineIPC friction overrides must be non-negative or -1")

    value = getattr(
        sims,
        "affine_friction_iterations",
        getattr(sims, "friction_iterations", getattr(sims, "friction_fixed_point_iterations", 1)),
    )
    try:
        iterations = int(value)
        exact_integer = float(value) == iterations
    except (TypeError, ValueError, OverflowError):
        iterations = None
        exact_integer = False
    if not exact_integer:
        raise RuntimeError("SoftAffineIPC friction_iterations must be an integer")
    return "lagged", -1 if iterations <= 0 else iterations


@ti.data_oriented
class SoftAffineIPCOperator(object):
    @property
    def is_semi(self):
        return getattr(self, "_is_semi", False)

    @is_semi.setter
    def is_semi(self, value):
        self._is_semi = bool(value)

    def __init__(self, scene, sims, soft_material):
        self.scene = scene
        self.sims = sims
        self.soft_material = soft_material
        if getattr(soft_material.matProps, "is_finite_strain_plastic", False):
            raise ValueError(
                "SoftAffineIPC supports hyperelastic soft materials only; "
                "plastic MPM-ABD coupling belongs to the ordinary Direct MPM path."
            )
        self.affine_state = AffineBodyState.from_scene(scene, sims)
        self.affine = TaichiAffineBodyOperator(self.affine_state, sims, scene)
        self.is_semi = self.affine.is_semi

        mode, _ = _validate_soft_affine_lagged_friction_configuration(sims)
        self.friction_mode = mode
        self.fully_implicit = mode == "fully_implicit"
        # Every supported Taichi architecture uses the same device-resident
        # nonlinear path.  Keep the historical attribute name internally so
        # downstream extensions do not silently select the host oracle.
        self.cuda_hot_loop = True

        self.dt = float(sims.dt[None])
        self.scale = self.dt * self.dt
        self.dt_device = ti.field(float, shape=())
        self.scale_device = ti.field(float, shape=())
        self.dt_device[None] = self.dt
        self.scale_device[None] = self.scale
        self.soft_num = max(int(scene.softNum[0]), 0)
        self.soft_point_num = max(int(scene.softPointNum[0]), 0)
        self.surface_num = max(int(scene.surfaceNum[0]), 0)
        self.soft_surface_point_num = max(
            (int(scene.soft[sb].surfacePointEnd) for sb in range(self.soft_num)),
            default=0,
        )
        self.soft_grid_num = max(int(scene.softGridNum[0]), 1)
        self.soft_velocity_constraint_num = max(int(scene.softVelocityConstraintNum[0]), 0)
        self.soft_shape_nodes = max(int(sims.soft_shape_nodes), 1)
        self.affine_dof = self.affine.dof
        self.max_soft_dof = 3 * self.soft_grid_num
        self.max_dof = max(self.affine_dof + self.max_soft_dof, 3)
        self.total_dof = self.affine_dof
        self.soft_active_nodes = 0

        self.prefix_sum = PrefixSumExecutor(self.soft_grid_num)
        self.soft_node2dof = ti.field(ti.i32, shape=max(self.prefix_sum.get_length(), self.soft_grid_num, 1))
        self.soft_dof2node = ti.field(ti.i32, shape=max(self.soft_grid_num, 1))
        self.soft_node_fixed = ti.field(ti.i32, shape=max(self.soft_grid_num, 1))
        self._initialize_soft_fixed_nodes()
        self.soft_disp = ti.field(float, shape=max(self.max_soft_dof, 1))
        self.soft_disp_base = ti.field(float, shape=max(self.max_soft_dof, 1))
        self.soft_direction = ti.field(float, shape=max(self.max_soft_dof, 1))
        # CUDA Newton/Armijo state.  The affine iterate already lives in
        # ``self.affine.y``; this field is its accepted line-search snapshot.
        # Keeping both coupled halves in Taichi avoids one full-vector
        # download/upload on every trial.
        self.affine_y_base = ti.Vector.field(3, float, shape=max(self.affine.control_num, 1))
        self.affine_y_step_start = ti.Vector.field(3, float, shape=max(self.affine.control_num, 1))
        self.soft_hat_x = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_frozen_point = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_current_point = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.global_grad = ti.field(float, shape=self.max_dof)
        self.energy = ti.field(float, shape=())
        self.device_status = ti.field(ti.i32, shape=())
        self.adjoint_rhs = ti.field(float, shape=self.max_dof)
        self.adjoint_solution = ti.field(float, shape=self.max_dof)
        self.gravity_vjp = ti.Vector.field(3, float, shape=())
        self.soft_young_vjp = ti.field(float, shape=())
        # (Young, Poisson, DP cohesion/friction angle or VM yield/hardening)
        self.material_parameter_vjp = ti.Vector.field(4, ti.f64, shape=())
        plastic_enabled = bool(getattr(self.soft_material.matProps, "is_finite_strain_plastic", False))
        plastic_capacity = max(
            self.soft_point_num if plastic_enabled else 1,
            1,
        )
        self.plastic_inverse_vjp = ti.Matrix.field(3, 3, float, shape=plastic_capacity)
        self.plastic_equivalent_strain_vjp = ti.field(float, shape=plastic_capacity)
        self.plastic_volumetric_strain_vjp = ti.field(float, shape=plastic_capacity)
        self.plastic_deformation_vjp = ti.Matrix.field(3, 3, float, shape=plastic_capacity)
        self.plastic_commit_deformation_vjp = ti.Matrix.field(3, 3, float, shape=plastic_capacity)
        self.plastic_commit_inverse_vjp = ti.Matrix.field(3, 3, float, shape=plastic_capacity)
        self.plastic_commit_equivalent_vjp = ti.field(float, shape=plastic_capacity)
        self.plastic_commit_volumetric_vjp = ti.field(float, shape=plastic_capacity)
        self.plastic_output_deformation_vjp = ti.Matrix.field(3, 3, float, shape=plastic_capacity)
        self.plastic_output_inverse_vjp = ti.Matrix.field(3, 3, float, shape=plastic_capacity)
        self.plastic_output_equivalent_vjp = ti.field(float, shape=plastic_capacity)
        self.plastic_output_volumetric_vjp = ti.field(float, shape=plastic_capacity)
        plastic_dof_capacity = self.max_dof if plastic_enabled else 1
        self.plastic_commit_grid_vjp = ti.field(float, shape=plastic_dof_capacity)
        self.plastic_step_rhs = ti.field(float, shape=plastic_dof_capacity)
        self.soft_position_vjp = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.soft_velocity_vjp = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.soft_output_position_vjp = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.soft_output_velocity_vjp = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.trajectory_rhs = ti.field(float, shape=self.max_dof)
        self.last_adjoint_result = None
        self.ccd_alpha = ti.field(float, shape=())
        self.mixed_pair_num = ti.field(ti.i32, shape=())
        self.mixed_pair = ti.Vector.field(2, ti.i32, shape=max(self.soft_num * self.affine_state.body_num, 1))
        self.mixed_pair_start = ti.field(ti.i32, shape=max(self.soft_num, 1))
        self.mixed_pair_end = ti.field(ti.i32, shape=max(self.soft_num, 1))
        affine_body_capacity = max(self.affine_state.body_num, 1)
        # CUDA code generation is unstable when a matrix field loaded in a
        # large contact kernel is immediately used in matrix/vector products.
        # Cache the nine scalars and assemble only the three values needed by
        # each row below.
        self.mixed_levelset_inverse = ti.field(float, shape=(affine_body_capacity, 3, 3))
        self.mixed_levelset_inverse_valid = ti.field(ti.i32, shape=affine_body_capacity)
        self.mixed_levelset_contact_count = ti.field(ti.i32, shape=())
        self.mixed_levelset_contact_active = ti.field(ti.i32, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_contact_material = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_contact_phi = ti.field(float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_contact_phi_gradient = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_contact_world_normal = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_contact_barrier = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_contact_coefficient = ti.field(float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_cell_base = ti.field(ti.i32, shape=(max(self.soft_point_num, 1), 3))
        self.mixed_levelset_cell_fraction = ti.field(float, shape=(max(self.soft_point_num, 1), 3))
        self.mixed_levelset_friction_inverse = ti.field(float, shape=(affine_body_capacity, 3, 3))
        self.mixed_levelset_friction_inverse_valid = ti.field(ti.i32, shape=affine_body_capacity)
        self.mixed_levelset_friction_active = ti.field(ti.i32, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_friction_material = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_friction_phi = ti.field(float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_friction_phi_gradient = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_friction_unit_normal = ti.Vector.field(3, float, shape=max(self.soft_point_num, 1))
        self.mixed_levelset_friction_coefficient = ti.field(float, shape=max(self.soft_point_num, 1))
        self.mixed_contact_type_count = ti.field(ti.i32, shape=7)
        self.soft_pair_num = ti.field(ti.i32, shape=())
        self.soft_pair = ti.Vector.field(
            2,
            ti.i32,
            shape=max(self.soft_num * max(self.soft_num - 1, 1) // 2, 1),
        )
        self.soft_pair_start = ti.field(ti.i32, shape=max(self.soft_num, 1))
        self.soft_pair_end = ti.field(ti.i32, shape=max(self.soft_num, 1))
        self.soft_pair_capacity = int(self.soft_pair.shape[0])
        self.mixed_pair_capacity = int(self.mixed_pair.shape[0])
        self.soft_min = ti.Vector.field(3, float, shape=max(self.soft_num, 1))
        self.soft_max = ti.Vector.field(3, float, shape=max(self.soft_num, 1))
        self.soft_center_disp = ti.Vector.field(3, float, shape=max(self.soft_num, 1))
        self.soft_center_direction = ti.Vector.field(3, float, shape=max(self.soft_num, 1))
        self.soft_body_mass = ti.field(float, shape=max(self.soft_num, 1))
        self.affine_min = ti.Vector.field(3, float, shape=max(self.affine_state.body_num, 1))
        self.affine_max = ti.Vector.field(3, float, shape=max(self.affine_state.body_num, 1))

        self.mixed_bvh = None
        self.mixed_bvh_node_count = self.soft_point_num + self.affine.vertex_num
        self.mixed_bvh_position = ti.Vector.field(3, float, shape=max(self.mixed_bvh_node_count, 1))
        self.mixed_bvh_end_position = ti.Vector.field(3, float, shape=max(self.mixed_bvh_node_count, 1))
        self.mixed_candidate_capacity = 0
        if not self.affine.levelset_contact and self.soft_surface_point_num > 0 and self.affine.face_num > 0:
            self.mixed_candidate_capacity = int(getattr(sims, "max_point_triangle_pairs", 0))
            if self.mixed_candidate_capacity <= 0:
                raise ValueError("SoftAffineIPC mesh coupling requires a positive " "max_point_triangle_pairs")
            surface_ids = np.ascontiguousarray(
                scene.soft_surface_point_id.to_numpy()[: self.soft_surface_point_num],
                dtype=np.int32,
            )
            if np.any(surface_ids < 0) or np.any(surface_ids >= self.soft_point_num):
                raise ValueError("SoftAffineIPC surface point IDs are out of range")
            reference = np.empty((self.mixed_bvh_node_count, 3), dtype=np.float64)
            reference[: self.soft_point_num] = scene.soft_point.x.to_numpy()[: self.soft_point_num]
            reference[self.soft_point_num :] = self.affine.rest_x_np
            faces = np.ascontiguousarray(self.affine.faces_np + self.soft_point_num, dtype=np.int32)
            self.mixed_bvh = DynamicBVHBroadPhase(
                faces,
                np.zeros((0, 2), dtype=np.int32),
                surface_ids,
                np.zeros(self.mixed_bvh_node_count, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                reference,
                max_point_triangle_pairs=self.mixed_candidate_capacity,
                max_edge_edge_pairs=1,
            )

        # The old ``points * min(neighbors, 16/64)`` values looked like
        # topology bounds but were not.  Instead expose a real memory budget:
        # derive bounded defaults from the number of scalar triplets consumed
        # by one complete lagged contact stencil, and fail with the exact
        # required contact count before assembly if the active set exceeds it.
        contact_triplet_budget = int(sims.soft_affine_contact_triplet_budget)
        if contact_triplet_budget < 2:
            raise ValueError("soft_affine_contact_triplet_budget must be at least 2")
        per_contact_kind_budget = max(contact_triplet_budget // 2, 1)
        soft_scalar_stencil = 3 * (2 * self.soft_shape_nodes)
        mixed_scalar_stencil = 3 * (self.soft_shape_nodes + 12)
        default_soft_contacts = max(per_contact_kind_budget // max(soft_scalar_stencil**2, 1), 1)
        default_mixed_contacts = max(per_contact_kind_budget // max(mixed_scalar_stencil**2, 1), 1)
        self.soft_friction_capacity = int(
            default_soft_contacts
            if sims.soft_affine_soft_friction_capacity is None
            else sims.soft_affine_soft_friction_capacity
        )
        self.mixed_friction_capacity = int(
            default_mixed_contacts
            if sims.soft_affine_mixed_friction_capacity is None
            else sims.soft_affine_mixed_friction_capacity
        )
        if self.soft_friction_capacity <= 0 or self.mixed_friction_capacity <= 0:
            raise ValueError("SoftAffineIPC friction contact capacities must be positive")
        # Barrier and current-state FI friction use the same activation set;
        # bounding that set by the same configured resources makes the matrix
        # estimate strict without allocating the full P^2 or P*F topology.
        self.soft_contact_capacity = int(
            self.soft_friction_capacity
            if sims.soft_affine_soft_contact_capacity is None
            else sims.soft_affine_soft_contact_capacity
        )
        self.mixed_contact_capacity = int(
            self.mixed_friction_capacity
            if sims.soft_affine_mixed_contact_capacity is None
            else sims.soft_affine_mixed_contact_capacity
        )
        if self.soft_contact_capacity <= 0 or self.mixed_contact_capacity <= 0:
            raise ValueError("SoftAffineIPC contact capacities must be positive")
        self.soft_barrier_count = ti.field(ti.i32, shape=())
        self.soft_friction_count = ti.field(ti.i32, shape=())
        self.soft_friction_overflow = ti.field(ti.i32, shape=())
        self.soft_friction_points = ti.Vector.field(2, ti.i32, shape=self.soft_friction_capacity)
        self.soft_friction_normal = ti.Vector.field(3, float, shape=self.soft_friction_capacity)
        self.soft_friction_coeff = ti.field(float, shape=self.soft_friction_capacity)
        self.adjoint_soft_friction_count = ti.field(ti.i32, shape=())
        self.adjoint_soft_friction_points = ti.Vector.field(2, ti.i32, shape=self.soft_friction_capacity)
        self.adjoint_soft_friction_normal = ti.Vector.field(3, float, shape=self.soft_friction_capacity)
        self.adjoint_soft_friction_coeff = ti.field(float, shape=self.soft_friction_capacity)
        self.mixed_friction_count = ti.field(ti.i32, shape=())
        self.mixed_friction_overflow = ti.field(ti.i32, shape=())
        self.mixed_friction_point = ti.field(ti.i32, shape=self.mixed_friction_capacity)
        self.mixed_friction_face = ti.field(ti.i32, shape=self.mixed_friction_capacity)
        self.mixed_friction_bary = ti.Vector.field(3, float, shape=self.mixed_friction_capacity)
        self.mixed_friction_normal = ti.Vector.field(3, float, shape=self.mixed_friction_capacity)
        self.mixed_friction_coeff = ti.field(float, shape=self.mixed_friction_capacity)
        self.adjoint_mixed_friction_count = ti.field(ti.i32, shape=())
        self.adjoint_mixed_friction_point = ti.field(ti.i32, shape=self.mixed_friction_capacity)
        self.adjoint_mixed_friction_face = ti.field(ti.i32, shape=self.mixed_friction_capacity)
        self.adjoint_mixed_friction_bary = ti.Vector.field(3, float, shape=self.mixed_friction_capacity)
        self.adjoint_mixed_friction_normal = ti.Vector.field(3, float, shape=self.mixed_friction_capacity)
        self.adjoint_mixed_friction_coeff = ti.field(float, shape=self.mixed_friction_capacity)
        semi_capacity = 1 << (max(2 * (self.soft_contact_capacity + self.mixed_contact_capacity), 1) - 1).bit_length()
        self.semi_capacity = semi_capacity
        self.semi_state = ti.field(ti.i32, shape=semi_capacity)
        self.semi_key = ti.Vector.field(4, ti.i32, shape=semi_capacity)
        self.semi_multiplier = ti.field(float, shape=semi_capacity)
        self.semi_count = ti.field(ti.i32, shape=())
        self.semi_overflow = ti.field(ti.i32, shape=())
        self.semi_constraint_violation = ti.field(float, shape=())
        self.semi_state.fill(2)

        # Stable GPU contact compaction.  Atomic appends assign a different
        # contact id when CUDA schedules the same broad-phase pairs in a
        # different order, invalidating HashReduction's persistent raw mapping.
        # Count per source surface point / mixed BVH candidate, inclusive-scan
        # entirely on device, then write deterministic contiguous intervals.
        self.soft_contact_prefix_sum = PrefixSumExecutor(max(self.soft_surface_point_num, 1))
        self.mixed_contact_prefix_sum = PrefixSumExecutor(
            max(self.mixed_candidate_capacity, self.soft_surface_point_num, 1)
        )
        self.soft_contact_prefix = ti.field(
            ti.i32,
            shape=max(
                self.soft_surface_point_num,
                self.soft_contact_prefix_sum.get_length(),
                1,
            ),
        )
        self.mixed_contact_prefix = ti.field(
            ti.i32,
            shape=max(
                self.soft_surface_point_num,
                self.mixed_candidate_capacity,
                self.mixed_contact_prefix_sum.get_length(),
                1,
            ),
        )
        self.stable_lagged_contacts = True

        triplets = self._estimate_triplet_capacity(sims)
        reduced_nonzeros = self._estimate_reduced_triplet_capacity(triplets)
        self.hash_triplet = BuildTriplet(
            dim=3,
            max_pairs_num=triplets,
            max_nonzeros=reduced_nonzeros,
            max_active_nodes=max(self.max_dof // 3, 1),
            symmetric=False,
            # Official lagged IPC projects each complete non-inertial local
            # Hessian before block scatter, making the system SPD for PCG. Fully
            # implicit friction retains the exact nonsymmetric Jacobian.
            solver="BiCGSTAB" if self.fully_implicit else "PCG",
            matrix_symmetric=not self.fully_implicit,
            device_reduction=self.cuda_hot_loop,
        )
        self.affine.bind_hash_triplet(self.hash_triplet)
        self._assert_cuda_device_residency()
        self.gravity = vec3f(sims.gravity)
        self.soft_background_damping = float(sims.soft_background_damping)
        if not np.isfinite(self.soft_background_damping) or self.soft_background_damping < 0.0:
            raise ValueError("soft_background_damping must be finite and non-negative")
        self.profile = bool(int(os.environ.get("GT_SOFT_AFFINE_PROFILE", "0")))
        self._configure_fully_implicit_friction(sims)

    def set_timestep(self, timestep):
        self.dt = float(timestep)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError("SoftAffineIPC timestep must be finite and positive")
        self.scale = self.dt * self.dt
        self.dt_device[None] = self.dt
        self.scale_device[None] = self.scale
        self.affine.set_timestep(self.dt)

    def reset_semi_state(self):
        if not self.is_semi:
            return
        self.semi_state.fill(2)
        self.semi_multiplier.fill(0.0)
        self.semi_count[None] = 0
        self.semi_overflow[None] = 0
        self.semi_constraint_violation[None] = 0.0
        self.affine.reset_semi_state()

    @staticmethod
    def _first_parameter(sims, names, default):
        for name in names:
            if hasattr(sims, name):
                return getattr(sims, name)
        return default

    def _configure_fully_implicit_friction(self, sims):
        """Read the paper's Coulomb/Stribeck coefficients for coupled contact.

        A negative static/dynamic override means that the material-pair IPC
        coefficient is used.  This preserves every existing lagged input while
        allowing unequal static and dynamic coefficients in fully implicit mode.
        """
        self.fully_mu_dynamic = float(
            self._first_parameter(
                sims,
                ("affine_dynamic_friction", "affine_friction_dynamic"),
                -1.0,
            )
        )
        self.fully_mu_static = float(
            self._first_parameter(
                sims,
                ("affine_static_friction", "affine_friction_static"),
                -1.0,
            )
        )
        self.fully_mu_viscous = float(
            self._first_parameter(
                sims,
                ("affine_viscous_friction", "affine_friction_viscous"),
                0.0,
            )
        )
        self.fully_stribeck_velocity = float(
            self._first_parameter(
                sims,
                ("affine_stribeck_velocity", "affine_friction_stribeck_velocity"),
                -1.0,
            )
        )
        if self.fully_stribeck_velocity == -1.0:
            self.fully_stribeck_velocity = 10.0 * self.affine.epsv
        elif self.fully_stribeck_velocity < 0.0:
            raise ValueError("SoftAffineIPC stribeck velocity must be non-negative or -1")
        profile = (
            str(
                self._first_parameter(
                    sims,
                    ("affine_friction_profile",),
                    "quadratic",
                )
            )
            .strip()
            .replace("-", "_")
            .lower()
        )
        profiles = {
            "quadratic": 0,
            "paper": 0,
            "ipc": 0,
            "c1": 0,
            "stabilized": 1,
            "stabilised": 1,
            "cinfinity": 1,
            "c_infinity": 1,
        }
        if profile not in profiles:
            raise ValueError("SoftAffineIPC affine_friction_profile must be 'quadratic' " "or 'stabilized'")
        self.fully_profile_id = profiles[profile]
        values = (
            self.fully_mu_dynamic,
            self.fully_mu_static,
            self.fully_mu_viscous,
            self.fully_stribeck_velocity,
            self.affine.epsv,
        )
        if not all(np.isfinite(v) for v in values):
            raise ValueError("SoftAffineIPC friction parameters must be finite")
        if self.fully_mu_viscous < 0.0 or self.affine.epsv <= 0.0:
            raise ValueError("SoftAffineIPC viscous friction must be non-negative and epsv positive")
        if self.fully_implicit:
            if (self.fully_mu_dynamic < 0.0 and self.fully_mu_dynamic != -1.0) or (
                self.fully_mu_static < 0.0 and self.fully_mu_static != -1.0
            ):
                raise ValueError("SoftAffineIPC friction overrides must be non-negative or -1")
            # A negative static override falls back to the resolved dynamic
            # coefficient independently for every material pair.
            unequal_override = self.fully_mu_static >= 0.0 and (
                self.fully_mu_dynamic < 0.0 or self.fully_mu_static != self.fully_mu_dynamic
            )
            if unequal_override and self.fully_stribeck_velocity <= 0.0:
                raise ValueError("SoftAffineIPC stribeck velocity must be positive when mu_s != mu_d")

    def _profile_stage(self, label, start):
        if self.profile:
            ti.sync()
            now = time.time()
            print(f"[SoftAffineIPC] {label}: {now - start:.6f}s", flush=True)
            return now
        return start

    def _contact_triplet_capacity(self):
        shape_nodes = self.soft_shape_nodes
        # Explicit resource bounds.  Exact current contact counts are checked
        # before any barrier/FI scatter, and frozen-friction compaction checks
        # its own count before fill, so these configured bounds are strict
        # without allocating the complete P^2 or P*F topology.
        soft_friction_candidates = max(self.soft_friction_capacity, 0) if self.soft_num > 1 else 0
        mixed_friction_candidates = max(self.mixed_friction_capacity, 0)
        soft_contact_candidates = max(self.soft_contact_capacity, 0) if self.soft_num > 1 else 0
        mixed_contact_candidates = max(self.mixed_contact_capacity, 0)
        soft_block_stencil = 2 * shape_nodes
        mixed_block_stencil = shape_nodes + 3 * 4
        soft_off_diagonal = soft_block_stencil * (soft_block_stencil - 1)
        mixed_off_diagonal = mixed_block_stencil * (mixed_block_stencil - 1)
        if self.fully_implicit:
            # A PP stencil pulls each of its two point sites through every
            # soft shape node.  A mixed PT stencil pulls its point through the
            # soft support and its three triangle vertices through four affine
            # controls each.  The fully implicit residual stores the complete
            # nonsymmetric Jacobian, so capacity must cover every block pair
            # (and both barrier and friction contributions), not the old
            # one-sided ``shape_nodes * 4`` lagged estimate.
            return 2 * soft_contact_candidates * soft_off_diagonal + 2 * mixed_contact_candidates * mixed_off_diagonal
        else:
            # Barrier and frozen friction each scatter one 3x3 block per
            # mapped control pair.  The global upper-block filter only lowers
            # this conservative full-stencil bound.
            return (soft_contact_candidates + soft_friction_candidates) * soft_off_diagonal // 2 + (
                mixed_contact_candidates + mixed_friction_candidates
            ) * mixed_off_diagonal // 2

    def _estimate_triplet_capacity(self, sims):
        shape_nodes = self.soft_shape_nodes
        soft_points = max(self.soft_point_num, 1)
        base = self.affine.max_hash_triplets
        soft_off_diagonal = shape_nodes * max(shape_nodes - 1, 0)
        if not self.fully_implicit:
            soft_off_diagonal //= 2
        base += soft_points * soft_off_diagonal
        base += self._contact_triplet_capacity()
        safety = float(os.environ.get("GT_SOFT_AFFINE_HASH_TRIPLET_SAFETY", "1.0"))
        if not np.isfinite(safety) or safety < 1.0:
            raise ValueError("GT_SOFT_AFFINE_HASH_TRIPLET_SAFETY must be finite and >= 1")
        required = max(8192, int(np.ceil(safety * base)))
        configured = int(sims.soft_affine_hash_triplet_capacity)
        if configured < 0:
            raise ValueError("soft_affine_hash_triplet_capacity must be non-negative")
        # An explicit value is an upward reservation, never permission to
        # allocate less than the required configured-contact bound.
        return max(required, configured)

    def _estimate_reduced_triplet_capacity(self, raw_capacity):
        """Bound unique off-diagonal blocks from grid adjacency, not raw scatters."""
        denominator = 1 if self.fully_implicit else 2
        total_blocks = max(self.max_dof // 3, 1)
        all_pairs = total_blocks * max(total_blocks - 1, 0) // denominator
        affine_controls = max(int(self.affine.control_num), 1)
        affine_pairs = affine_controls * max(affine_controls - 1, 0) // denominator

        shape_nodes = max(int(self.soft_shape_nodes), 1)
        nodes_per_axis = int(round(shape_nodes ** (1.0 / 3.0)))
        if getattr(self.sims, "soft_grid_type_id", 0) == 0 and nodes_per_axis**3 == shape_nodes:
            adjacent_nodes = (2 * nodes_per_axis - 1) ** 3 - 1
            soft_pairs = max(self.soft_grid_num, 1) * adjacent_nodes // denominator
        else:
            soft_pairs = max(self.soft_point_num, 1) * shape_nodes * max(shape_nodes - 1, 0) // denominator
        structural_bound = affine_pairs + soft_pairs + self._contact_triplet_capacity()
        return max(1, min(int(raw_capacity), all_pairs, structural_bound))

    def _raise_hash_triplet_overflow(self, stage):
        if int(self.hash_triplet.overflow[0]) != 0:
            used = int(self.hash_triplet.raw_non_diag_count[0])
            capacity = int(self.hash_triplet.non_diag.max_pairs_num)
            raise RuntimeError(
                "SoftAffineIPC HashTriplet buffer overflow during " f"{stage}: used {used}, capacity {capacity}"
            )

    def _assert_cuda_device_residency(self):
        """Fail before a device solve can use a host sparse/vector path."""
        expected_solver = "BiCGSTAB" if self.fully_implicit else "PCG"
        expected_matrix_symmetric = not self.fully_implicit
        if (
            self.hash_triplet.solver != expected_solver
            or bool(self.hash_triplet.matrix_symmetric) != expected_matrix_symmetric
        ):
            matrix_kind = "full nonsymmetric" if self.fully_implicit else "symmetric"
            raise RuntimeError(
                "SoftAffineIPC device solve requires the " f"{matrix_kind} HashTriplet matrix and {expected_solver}"
            )
        if not self.cuda_hot_loop:
            return
        reducer = self.hash_triplet.non_diag
        if not reducer.device_reduction:
            raise RuntimeError(
                "SoftAffineIPC requires HashTriplet device reduction; host " "sparse reduction is not a runtime backend"
            )
        if reducer.go.__func__ is not reducer.go_with_device_reduction.__func__:
            raise RuntimeError("SoftAffineIPC HashTriplet reducer is bound to a " "host implementation")

    def _reject_cuda_host_vector_path(self, operation):
        if self.cuda_hot_loop:
            raise RuntimeError(
                f"SoftAffineIPC {operation} is a CPU reference path and is "
                "disabled in the runtime backend because it transfers "
                "complete nonlinear vectors through NumPy"
            )

    def begin_step(self):
        self._reject_cuda_host_vector_path("begin_step")
        self.reset_semi_state()
        self.affine_state.begin_step(self.dt)
        # Keep the affine cache initialized for direct operator users.  The
        # coupled driver refreshes all three caches together immediately after
        # the soft grid has been prepared.
        y_flat = self.affine_state.pack()
        self.affine.initialize_contact_damping(y_flat, self.affine_state.hat_y)
        self._prepare_soft_grid()
        self.prefix_sum.run(self.soft_node2dof)
        self.soft_active_nodes = int(self._fill_soft_dof())
        self.total_dof = self.affine_dof + 3 * self.soft_active_nodes
        self._initialize_soft_step()

    def begin_step_device(self):
        """Initialize a step without transferring the affine state to host."""
        self._assert_cuda_device_residency()
        self.reset_semi_state()
        self.affine.device_begin_step(self.dt)
        self._store_affine_step_start_device()
        self._initialize_affine_contact_damping_device()
        self._prepare_soft_grid()
        self.prefix_sum.run(self.soft_node2dof)
        self.soft_active_nodes = int(self._fill_soft_dof())
        self.total_dof = self.affine_dof + 3 * self.soft_active_nodes
        self._initialize_soft_step()

    def _load_affine_step_fields(self, y_flat, tilde_y_flat, hat_y_flat):
        control_num = self.affine.control_num
        self.affine.y.from_numpy(np.ascontiguousarray(np.asarray(y_flat, dtype=np.float64).reshape((control_num, 3))))
        self.affine.tilde_y.from_numpy(
            np.ascontiguousarray(np.asarray(tilde_y_flat, dtype=np.float64).reshape((control_num, 3)))
        )
        self.affine.hat_y.from_numpy(
            np.ascontiguousarray(np.asarray(hat_y_flat, dtype=np.float64).reshape((control_num, 3)))
        )

    @ti.kernel
    def _affine_edge_dispatch_mask(self) -> ti.i32:
        """Pack the two nine-entry EE dispatch tables into one scalar."""
        mask = 0
        for edge_type in range(9):
            if self.affine.edge_type_count[edge_type] > 0:
                mask += 1 << edge_type
            if self.affine.edge_mollifier_type_count[edge_type] > 0:
                mask += 1 << (edge_type + 9)
        return mask

    @ti.kernel
    def _store_affine_step_start_device(self):
        for control in range(self.affine.control_num):
            self.affine_y_step_start[control] = self.affine.y[control]

    @ti.kernel
    def _rollback_step_device(self):
        for control in range(self.affine.control_num):
            self.affine.y[control] = self.affine_y_step_start[control]
        for dof in range(self.max_soft_dof):
            self.soft_disp[dof] = 0.0
            self.soft_disp_base[dof] = 0.0

    def rollback_step_device(self):
        """Restore a failed CUDA step without a host vector upload."""
        self._rollback_step_device()

    def _initialize_affine_contact_damping_device(self):
        """Device-state counterpart of affine.initialize_contact_damping()."""
        affine = self.affine
        affine._clear_contact_damping_matrix()
        affine._reconstruct_vertices()
        if affine.levelset_contact:
            affine.last_candidate_pairs = 0
            affine.friction_contact_count.fill(0)
            affine.friction_contact_overflow.fill(0)
            affine._freeze_levelset_lagged_friction_geometry()
            if not affine.fully_implicit:
                affine._initialize_lagged_wall_friction()
            return
        affine.last_candidate_pairs = affine.neighbor.update(
            affine.x,
            affine.dx,
            affine.faces,
            affine.edges,
            affine.node2body,
            affine.face2body,
            affine.edge2body,
            affine.dhat,
            swept=False,
        )
        affine.friction_contact_count.fill(0)
        affine.friction_contact_overflow.fill(0)
        if not affine.fully_implicit:
            affine._initialize_lagged_mesh_friction(
                affine.neighbor.candidate_count,
                affine.neighbor.candidate_vertex,
                affine.neighbor.candidate_face,
                affine.neighbor.edge_candidate_count,
                affine.neighbor.candidate_edge0,
                affine.neighbor.candidate_edge1,
            )
            affine._initialize_lagged_wall_friction()
            if int(affine.friction_contact_overflow[0]) != 0:
                raise RuntimeError(
                    "Affine lagged-friction contact buffer overflow: "
                    f"used {int(affine.friction_contact_count[0])}, "
                    f"capacity {affine.friction_contact_capacity}"
                )
        if affine.contact_damping_stiffness <= 0.0:
            return
        affine._assemble_body_pair_barrier_hessian(
            affine.neighbor.candidate_count,
            affine.neighbor.candidate_vertex,
            affine.neighbor.candidate_face,
            affine.neighbor.edge_candidate_count,
            affine.neighbor.candidate_edge0,
            affine.neighbor.candidate_edge1,
            1.0,
            MATRIX_CONTACT_DAMPING,
        )
        affine._assemble_wall_contact_damping_hessian()
        affine.finalize_contact_damping_assembly()

    @ti.kernel
    def _build_fully_implicit_velocity_predictor(self):
        for node in range(self.soft_grid_num):
            block = self._soft_block(node)
            if block >= 0:
                for component in ti.static(range(3)):
                    self.soft_direction[3 * block + component] = (
                        self.dt_device[None] * self.scene.soft_grid[node].v[component]
                    )

    @ti.kernel
    def _build_affine_velocity_predictor_device(self):
        for control in range(self.affine.control_num):
            self.affine.direction_y[control] = self.affine.tilde_y[control] - self.affine.y[control]

    def initialize_fully_implicit_velocity_predictor_device(self, sims):
        """Apply the paper predictor without materializing a host vector."""
        self._build_affine_velocity_predictor_device()
        self._build_fully_implicit_velocity_predictor()
        alpha = (
            self.init_step_size_device(
                ccd_type=sims.affine_ccd_type,
                eta=sims.affine_ccd_eta,
                accd_tolerance=sims.affine_accd_tolerance,
                max_iteration=sims.affine_ccd_max_iteration,
            )
            if sims.affine_ccd
            else 1.0
        )
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise RuntimeError("SoftAffineIPC fully implicit predictor produced no feasible " "positive CCD step")
        self.store_coupled_base_device()
        self.set_coupled_trial_device(min(float(alpha), 1.0))

    def initialize_fully_implicit_velocity_predictor(self, sims, y_flat):
        """Initialize Newton with the paper's ``v_{n+1}^0 = v_n`` state."""
        current = np.asarray(y_flat, dtype=np.float64).copy()
        affine_direction = self.affine_state.tilde_y.reshape(-1) - current
        self._build_fully_implicit_velocity_predictor()
        alpha = (
            self.init_step_size(
                current,
                affine_direction,
                ccd_type=sims.affine_ccd_type,
                eta=sims.affine_ccd_eta,
                accd_tolerance=sims.affine_accd_tolerance,
                max_iteration=sims.affine_ccd_max_iteration,
            )
            if sims.affine_ccd
            else 1.0
        )
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise RuntimeError("SoftAffineIPC fully implicit predictor produced no feasible " "positive CCD step")
        alpha = min(float(alpha), 1.0)
        self.soft_disp_base.fill(0.0)
        self._set_soft_trial(alpha)
        return current + alpha * affine_direction

    def refresh_lagged_friction(self, affine_y_flat):
        """Freeze active set, normal force and tangent basis for one inner solve."""
        self._reject_cuda_host_vector_path("refresh_lagged_friction")
        self.affine.initialize_contact_damping(affine_y_flat, self.affine_state.hat_y)
        self._build_mixed_pairs(0)
        self._update_mixed_mesh_candidates(False)
        self._reset_lagged_friction_cache()
        self._initialize_soft_lagged_friction()
        if self.affine.levelset_contact:
            self._freeze_mixed_levelset_friction_geometry()
        else:
            self._initialize_mixed_lagged_friction()
        soft_used = int(self.soft_friction_count[None])
        mixed_used = int(self.mixed_friction_count[None])
        if int(self.soft_friction_overflow[None]) != 0:
            raise RuntimeError(
                "SoftAffineIPC soft-soft lagged-friction buffer overflow: "
                f"used {soft_used}, capacity {self.soft_friction_capacity}"
            )
        if int(self.mixed_friction_overflow[None]) != 0:
            raise RuntimeError(
                "SoftAffineIPC mixed lagged-friction buffer overflow: "
                f"used {mixed_used}, capacity {self.mixed_friction_capacity}"
            )

    def refresh_lagged_friction_device(self):
        """Refresh the frozen IPC contact state from device-resident iterates."""
        self._assert_cuda_device_residency()
        self._initialize_affine_contact_damping_device()
        self._build_mixed_pairs(0)
        self._update_mixed_mesh_candidates(False)
        self._reset_lagged_friction_cache()
        self._initialize_soft_lagged_friction()
        if self.affine.levelset_contact:
            self._freeze_mixed_levelset_friction_geometry()
        else:
            self._initialize_mixed_lagged_friction()
        soft_used = int(self.soft_friction_count[None])
        mixed_used = int(self.mixed_friction_count[None])
        if int(self.soft_friction_overflow[None]) != 0:
            raise RuntimeError(
                "SoftAffineIPC soft-soft lagged-friction buffer overflow: "
                f"used {soft_used}, capacity {self.soft_friction_capacity}"
            )
        if int(self.mixed_friction_overflow[None]) != 0:
            raise RuntimeError(
                "SoftAffineIPC mixed lagged-friction buffer overflow: "
                f"used {mixed_used}, capacity {self.mixed_friction_capacity}"
            )

    @ti.kernel
    def _backup_coupled_lagged_friction_for_adjoint(self):
        soft_count = ti.min(self.soft_friction_count[None], self.soft_friction_capacity)
        mixed_count = ti.min(self.mixed_friction_count[None], self.mixed_friction_capacity)
        self.adjoint_soft_friction_count[None] = soft_count
        self.adjoint_mixed_friction_count[None] = mixed_count
        for contact in range(soft_count):
            self.adjoint_soft_friction_points[contact] = self.soft_friction_points[contact]
            self.adjoint_soft_friction_normal[contact] = self.soft_friction_normal[contact]
            self.adjoint_soft_friction_coeff[contact] = self.soft_friction_coeff[contact]
        for contact in range(mixed_count):
            self.adjoint_mixed_friction_point[contact] = self.mixed_friction_point[contact]
            self.adjoint_mixed_friction_face[contact] = self.mixed_friction_face[contact]
            self.adjoint_mixed_friction_bary[contact] = self.mixed_friction_bary[contact]
            self.adjoint_mixed_friction_normal[contact] = self.mixed_friction_normal[contact]
            self.adjoint_mixed_friction_coeff[contact] = self.mixed_friction_coeff[contact]

    @ti.kernel
    def _restore_coupled_lagged_friction_for_adjoint(self):
        soft_count = self.adjoint_soft_friction_count[None]
        mixed_count = self.adjoint_mixed_friction_count[None]
        self.soft_friction_count[None] = soft_count
        self.soft_friction_overflow[None] = 0
        self.mixed_friction_count[None] = mixed_count
        self.mixed_friction_overflow[None] = 0
        for contact in range(soft_count):
            self.soft_friction_points[contact] = self.adjoint_soft_friction_points[contact]
            self.soft_friction_normal[contact] = self.adjoint_soft_friction_normal[contact]
            self.soft_friction_coeff[contact] = self.adjoint_soft_friction_coeff[contact]
        for contact in range(mixed_count):
            self.mixed_friction_point[contact] = self.adjoint_mixed_friction_point[contact]
            self.mixed_friction_face[contact] = self.adjoint_mixed_friction_face[contact]
            self.mixed_friction_bary[contact] = self.adjoint_mixed_friction_bary[contact]
            self.mixed_friction_normal[contact] = self.adjoint_mixed_friction_normal[contact]
            self.mixed_friction_coeff[contact] = self.adjoint_mixed_friction_coeff[contact]

    def backup_lagged_friction_for_adjoint_device(self):
        self.affine.device_backup_lagged_friction_for_adjoint()
        self._backup_coupled_lagged_friction_for_adjoint()

    def restore_lagged_friction_for_adjoint_device(self):
        self.affine.device_restore_lagged_friction_for_adjoint()
        self._restore_coupled_lagged_friction_for_adjoint()

    def _assemble_affine_self(self, affine_y_flat, need_matrix):
        affine = self.affine
        affine.y.from_numpy(np.ascontiguousarray(affine_y_flat.reshape((affine.control_num, 3)), dtype=np.float64))
        affine.tilde_y.from_numpy(
            np.ascontiguousarray(self.affine_state.tilde_y.reshape((affine.control_num, 3)), dtype=np.float64)
        )
        affine.hat_y.from_numpy(
            np.ascontiguousarray(self.affine_state.hat_y.reshape((affine.control_num, 3)), dtype=np.float64)
        )
        self._assemble_affine_self_fields(bool(need_matrix))
        grad = affine.grad.to_numpy()[: affine.control_num].reshape(-1)
        return float(affine.energy[None]), grad

    def _assemble_affine_self_fields(self, need_matrix, project_spd=None):
        """Assemble the affine subsystem from its current Taichi fields."""
        affine = self.affine
        if project_spd is None:
            project_spd = not self.fully_implicit
        if getattr(affine, "is_semi", False):
            affine.semi_constraint_violation[None] = 0.0
            affine.semi_overflow[None] = 0
        tick = time.time()
        affine._clear_system(bool(need_matrix), MATRIX_HASH_TRIPLET)
        tick = self._profile_stage("affine_clear_system", tick)
        affine._reconstruct_vertices()
        tick = self._profile_stage("affine_reconstruct_vertices", tick)
        affine._assemble_inertia(bool(need_matrix), MATRIX_HASH_TRIPLET)
        tick = self._profile_stage("affine_inertia", tick)
        affine._assemble_body_force_device(bool(need_matrix))
        tick = self._profile_stage("affine_body_force", tick)
        affine._assemble_local_damping(bool(need_matrix), MATRIX_HASH_TRIPLET)
        tick = self._profile_stage("affine_local_damping", tick)
        affine._assemble_rigidity(
            bool(need_matrix),
            MATRIX_HASH_TRIPLET,
            bool(project_spd),
        )
        tick = self._profile_stage("affine_rigidity", tick)
        affine._assemble_joints(bool(need_matrix), MATRIX_HASH_TRIPLET)
        tick = self._profile_stage("affine_joints", tick)
        if affine.levelset_contact:
            affine._assemble_levelset_contacts(bool(need_matrix), MATRIX_HASH_TRIPLET)
            tick = self._profile_stage("affine_levelset_contact", tick)
            if not affine.fully_implicit:
                affine._assemble_levelset_lagged_friction(bool(need_matrix), MATRIX_HASH_TRIPLET)
                tick = self._profile_stage("affine_levelset_friction", tick)
                affine._assemble_lagged_friction(bool(need_matrix), MATRIX_HASH_TRIPLET)
                tick = self._profile_stage("affine_mesh_friction", tick)
            affine.last_candidate_pairs = int(affine.levelset_active_contacts[None])
            if getattr(affine, "is_semi", False):
                affine._assemble_semi_wall_contacts(bool(need_matrix), MATRIX_HASH_TRIPLET)
            else:
                affine._assemble_wall_contacts(bool(need_matrix), MATRIX_HASH_TRIPLET)
            self._profile_stage("affine_wall_contact", tick)
            return
        affine.last_candidate_pairs = affine.neighbor.update(
            affine.x,
            affine.dx,
            affine.faces,
            affine.edges,
            affine.node2body,
            affine.face2body,
            affine.edge2body,
            affine.dhat,
            swept=False,
        )
        tick = self._profile_stage("affine_neighbor", tick)
        if getattr(affine, "is_semi", False):
            affine._assemble_semi_particle_contacts(
                bool(need_matrix),
                affine.neighbor.candidate_count,
                affine.neighbor.candidate_vertex,
                affine.neighbor.candidate_face,
                MATRIX_HASH_TRIPLET,
            )
            affine._assemble_semi_edge_contacts(
                bool(need_matrix),
                affine.neighbor.edge_candidate_count,
                affine.neighbor.candidate_edge0,
                affine.neighbor.candidate_edge1,
                MATRIX_HASH_TRIPLET,
            )
        else:
            affine._assemble_particle_contacts(
                False,
                affine.neighbor.candidate_count,
                affine.neighbor.candidate_vertex,
                affine.neighbor.candidate_face,
                MATRIX_HASH_TRIPLET,
            )
            affine._assemble_edge_contacts(
                False,
                affine.neighbor.edge_candidate_count,
                affine.neighbor.candidate_edge0,
                affine.neighbor.candidate_edge1,
            )
            if need_matrix:
                affine._assemble_body_pair_barrier_hessian(
                    affine.neighbor.candidate_count,
                    affine.neighbor.candidate_vertex,
                    affine.neighbor.candidate_face,
                    affine.neighbor.edge_candidate_count,
                    affine.neighbor.candidate_edge0,
                    affine.neighbor.candidate_edge1,
                    affine.scale,
                    MATRIX_HASH_TRIPLET,
                    project_pd=bool(project_spd),
                )
                tick = self._profile_stage("affine_barrier_hessian", tick)
        tick = self._profile_stage("affine_contact", tick)
        if self.fully_implicit:
            affine._assemble_fully_implicit_friction(bool(need_matrix), MATRIX_HASH_TRIPLET)
        else:
            affine._assemble_lagged_friction(bool(need_matrix), MATRIX_HASH_TRIPLET)
        tick = self._profile_stage("affine_friction", tick)
        if getattr(affine, "is_semi", False):
            affine._assemble_semi_wall_contacts(bool(need_matrix), MATRIX_HASH_TRIPLET)
        else:
            affine._assemble_wall_contacts(bool(need_matrix), MATRIX_HASH_TRIPLET)
        tick = self._profile_stage("affine_wall_contact", tick)
        if affine.contact_damping_stiffness > 0.0:
            affine._assemble_contact_damping(bool(need_matrix), MATRIX_HASH_TRIPLET)
            self._profile_stage("affine_contact_damping", tick)

    @ti.kernel
    def _seed_coupled_residual_from_affine(self):
        self.energy[None] = self.affine.energy[None]
        for dof in range(self.max_dof):
            self.global_grad[dof] = 0.0
        for control in range(self.affine.control_num):
            for component in ti.static(range(3)):
                self.global_grad[3 * control + component] = self.affine.grad[control][component]

    def assemble(self, affine_y_flat, need_matrix=True):
        self._reject_cuda_host_vector_path("assemble")
        if self.is_semi:
            self.semi_constraint_violation[None] = 0.0
            self.semi_overflow[None] = 0
        tick = time.time()
        if need_matrix:
            # Line-search probes only need the energy/residual. Every Taichi
            # assembly kernel below uses a compile-time ``need_matrix`` guard,
            # so keep the last accepted Jacobian intact instead of clearing a
            # potentially large GPU HashTriplet for each rejected trial.
            self.hash_triplet.reset_system()
            tick = self._profile_stage("reset_system", tick)
        energy, affine_grad = self._assemble_affine_self(affine_y_flat, bool(need_matrix))
        tick = self._profile_stage("affine_self_assemble", tick)
        grad = np.zeros(self.max_dof, dtype=np.float64)
        grad[: self.affine_dof] = affine_grad
        self.global_grad.from_numpy(grad)
        self.energy[None] = float(energy)
        tick = self._profile_stage("load_gradient", tick)
        self._assemble_soft_energy_gradient(
            bool(need_matrix),
            bool(not self.fully_implicit),
        )
        tick = self._profile_stage("soft_energy", tick)
        self._build_mixed_pairs(0)
        tick = self._profile_stage("build_pairs", tick)
        self._update_mixed_mesh_candidates(False)
        tick = self._profile_stage("mixed_bvh", tick)
        self._assemble_soft_soft_contact(bool(need_matrix))
        tick = self._profile_stage("soft_soft_contact", tick)
        self._assemble_mixed_contact(bool(need_matrix))
        tick = self._profile_stage("soft_affine_contact", tick)
        matrix_shift = (
            float(self.sims.affine_fully_implicit_jacobian_shift)
            if self.fully_implicit
            else float(self.sims.affine_hessian_shift)
        )
        if need_matrix and matrix_shift > 0.0:
            self._add_hessian_shift(matrix_shift)
            tick = self._profile_stage("hessian_shift", tick)
        if need_matrix:
            # Residual-only line-search probes do not emit triplets.  Every
            # matrix assembly, however, must fail before a truncated Jacobian
            # can reach the linear solver.
            self._raise_hash_triplet_overflow("coupled matrix assembly")
        if self.is_semi and (int(self.semi_overflow[None]) != 0 or int(self.affine.semi_overflow[None]) != 0):
            raise RuntimeError("SoftAffineIPC SemiIPC multiplier hash capacity is too small")
        grad = self.global_grad.to_numpy()[: self.total_dof].copy()
        self._profile_stage("read_gradient", tick)
        return float(self.energy[None]), grad

    def assemble_device(self, need_matrix=True, project_spd=None, solver_shift=True):
        """Assemble at the current device iterate and return scalar energy.

        The coupled residual remains in ``global_grad``.  The assembled full
        nonsymmetric Jacobian remains in ``hash_triplet`` in fully implicit
        mode; no PSD projection, symmetrization or host sparse conversion is
        performed here.  Passing ``project_spd=False, solver_shift=False``
        requests the physical mesh tangent used by the elastic adjoint.
        """
        self._assert_cuda_device_residency()
        if self.is_semi:
            self.semi_constraint_violation[None] = 0.0
            self.semi_overflow[None] = 0
        tick = time.time()
        if need_matrix:
            self.hash_triplet.reset_system()
            tick = self._profile_stage("reset_system", tick)
        if project_spd is None:
            project_spd = not self.fully_implicit
        self._assemble_affine_self_fields(bool(need_matrix), bool(project_spd))
        tick = self._profile_stage("affine_self_assemble", tick)
        self._seed_coupled_residual_from_affine()
        tick = self._profile_stage("seed_residual", tick)
        self._assemble_soft_energy_gradient(bool(need_matrix), bool(project_spd))
        tick = self._profile_stage("soft_energy", tick)
        self._build_mixed_pairs(0)
        tick = self._profile_stage("build_pairs", tick)
        self._update_mixed_mesh_candidates(False)
        tick = self._profile_stage("mixed_bvh", tick)
        self._assemble_soft_soft_contact(bool(need_matrix), bool(project_spd))
        tick = self._profile_stage("soft_soft_contact", tick)
        self._assemble_mixed_contact(bool(need_matrix), bool(project_spd))
        tick = self._profile_stage("soft_affine_contact", tick)
        matrix_shift = (
            float(self.sims.affine_fully_implicit_jacobian_shift)
            if self.fully_implicit
            else float(self.sims.affine_hessian_shift)
        )
        if need_matrix and solver_shift and matrix_shift > 0.0:
            self._add_hessian_shift(matrix_shift)
            tick = self._profile_stage("hessian_shift", tick)
        if need_matrix:
            self._raise_hash_triplet_overflow("coupled matrix assembly")
        if self.is_semi and (int(self.semi_overflow[None]) != 0 or int(self.affine.semi_overflow[None]) != 0):
            raise RuntimeError("SoftAffineIPC SemiIPC multiplier hash capacity is too small")
        self._profile_stage("device_residual_ready", tick)
        return float(self.energy[None])

    def _solve_equilibrium_adjoint(self, loss_gradient, allow_plastic):
        if self.is_semi:
            raise ValueError("Soft-Affine adjoint currently requires BarrierIPC")
        if self.affine.levelset_contact:
            raise ValueError("Soft-Affine adjoint currently requires triangle-mesh coupling")
        if self.fully_implicit:
            raise ValueError(
                "Soft-Affine differentiable simulation supports "
                "friction_mode='lagged' only; fully implicit friction is not implemented"
            )
        is_plastic = bool(getattr(self.soft_material.matProps, "is_finite_strain_plastic", False))
        if is_plastic and not allow_plastic:
            raise ValueError(
                "Soft-Affine elastic adjoint cannot run a plastic material; " "use solve_plastic_equilibrium_adjoint"
            )
        model = getattr(self.soft_material.matProps, "model", None)
        if is_plastic and type(model).__name__ not in {
            "FiniteStrainDruckerPragerModel",
            "FiniteStrainVonMisesModel",
        }:
            raise ValueError(
                "Soft-Affine plastic equilibrium adjoint supports "
                "Drucker-Prager and von Mises; MCC is intentionally excluded"
            )
        if self.affine.wall_num or self.affine.contact_damping_stiffness > 0.0:
            raise ValueError("Soft-Affine adjoint currently excludes walls and contact damping")
        if isinstance(loss_gradient, ti.ScalarField):
            if len(loss_gradient.shape) != 1 or int(loss_gradient.shape[0]) < self.total_dof:
                raise ValueError("loss_gradient field must cover all active coupled degrees of freedom")
            copy_field(self.total_dof, self.adjoint_rhs, loss_gradient)
        else:
            values = np.asarray(loss_gradient, dtype=np.float64).reshape(-1)
            if values.size != self.total_dof or not np.all(np.isfinite(values)):
                raise ValueError("loss_gradient must be finite and match the active coupled degrees of freedom")
            padded = np.zeros(self.max_dof, dtype=np.float64)
            padded[: self.total_dof] = values
            self.adjoint_rhs.from_numpy(padded)
        self.adjoint_solution.fill(0.0)
        self.restore_lagged_friction_for_adjoint_device()
        self.assemble_device(
            need_matrix=True,
            project_spd=False,
            solver_shift=False,
        )
        self.hash_triplet.finalize_taichi_assembly()
        forward_solver = self.hash_triplet.solver
        self.hash_triplet.solver = "PCG" if self.hash_triplet.matrix_symmetric else "BiCGSTAB"
        try:
            result = self.hash_triplet.solve_flat_system(
                self.adjoint_rhs,
                self.adjoint_solution,
                active_nodes=self.total_dof // 3,
                tol=self.sims.affine_linear_tolerance,
                maxiter=self.sims.affine_linear_max_iteration,
                return_solution=False,
                transpose=not self.hash_triplet.matrix_symmetric,
                fallback_to_bicgstab=self.hash_triplet.matrix_symmetric,
            )
        finally:
            self.hash_triplet.solver = forward_solver
        self.last_adjoint_result = result
        if not result["converged"]:
            raise RuntimeError(
                "Soft-Affine equilibrium adjoint solve did not converge: "
                f"residual={result['residual']:.6e}, iterations={result['iterations']}"
            )
        return self.adjoint_solution

    def solve_elastic_adjoint(self, loss_gradient):
        """Solve the exact pre-commit elastic Soft-Affine equilibrium adjoint."""
        return self._solve_equilibrium_adjoint(loss_gradient, allow_plastic=False)

    def solve_plastic_equilibrium_adjoint(self, loss_gradient):
        """Solve one equilibrium adjoint with frozen prior plastic history."""
        if not getattr(self.soft_material.matProps, "is_finite_strain_plastic", False):
            raise ValueError("plastic equilibrium adjoint requires a finite-strain plastic soft material")
        return self._solve_equilibrium_adjoint(loss_gradient, allow_plastic=True)

    def _single_soft_young_modulus(self):
        model = getattr(self.soft_material.matProps, "model", None)
        supported = {
            "NeoHookeanModel",
            "HenckyElasticModel",
            "Gent",
            "Hydrogel",
        }
        young = float(getattr(model, "young", 0.0)) if model is not None else 0.0
        if type(model).__name__ not in supported or not np.isfinite(young) or young <= 0.0:
            # ponytail: table-valued and nonlinearly parameterized material
            # VJPs belong here when multi-material optimization is requested.
            raise ValueError("soft Young-modulus VJP currently requires one supported " "elastic LSMPM material")
        return young

    @ti.kernel
    def _differentiate_elastic_gravity(self):
        self.gravity_vjp[None] = ti.Vector.zero(float, 3)
        scale = self.scale_device[None]
        for body, control in ti.ndrange(self.affine.body_num, 4):
            lumped = 0.0
            for column in range(4):
                lumped += self.affine.mass[body, control, column]
            for component in ti.static(range(3)):
                ti.atomic_add(
                    self.gravity_vjp[None][component],
                    scale * lumped * self.adjoint_solution[12 * body + 3 * control + component],
                )
        for node in range(self.soft_grid_num):
            block = self._soft_block(node)
            if block >= 0:
                mass = self.scene.soft_grid[node].m
                for component in ti.static(range(3)):
                    ti.atomic_add(
                        self.gravity_vjp[None][component],
                        scale * mass * self.adjoint_solution[self.affine_dof + 3 * block + component],
                    )

    @ti.kernel
    def _copy_affine_adjoint(self):
        for dof in range(self.affine_dof):
            self.affine.linear_x[dof] = self.adjoint_solution[dof]

    @ti.kernel
    def _differentiate_soft_young(self, young: float):
        self.soft_young_vjp[None] = 0.0
        for particle in range(self.soft_point_num):
            if int(self.scene.soft_point[particle].active) == 1:
                material_id = self.scene.soft_point[particle].materialID
                deformation_gradient = self._soft_point_trial_F(particle)
                stress_derivative = (
                    self.soft_material.matProps.dPsi_div_dF_at(
                        particle,
                        material_id,
                        deformation_gradient,
                    )
                    * self.scene.soft_point[particle].vol0
                    * self.scale_device[None]
                    / young
                )
                for basis in range(self._soft_support_count(particle)):
                    node = self._soft_support_node(particle, basis)
                    block = self._soft_block(node)
                    if block >= 0:
                        residual_derivative = self._soft_material_node_gradient(
                            stress_derivative,
                            self._soft_support_gradient(particle, basis),
                        )
                        adjoint = ti.Vector.zero(float, 3)
                        for component in ti.static(range(3)):
                            adjoint[component] = self.adjoint_solution[self.affine_dof + 3 * block + component]
                        ti.atomic_add(
                            self.soft_young_vjp[None],
                            -adjoint.dot(residual_derivative),
                        )

    @ti.kernel
    def _differentiate_soft_plastic_input_history_device(self):
        for particle in range(self.soft_point_num):
            self.plastic_inverse_vjp[particle] = ti.Matrix.zero(float, 3, 3)
            self.plastic_equivalent_strain_vjp[particle] = 0.0
            self.plastic_volumetric_strain_vjp[particle] = 0.0
            self.plastic_deformation_vjp[particle] = ti.Matrix.zero(float, 3, 3)
            if int(self.scene.soft_point[particle].active) == 1:
                adjoint_deformation = ti.Matrix.zero(float, 3, 3)
                for basis in range(self._soft_support_count(particle)):
                    node = self._soft_support_node(particle, basis)
                    block = self._soft_block(node)
                    if block >= 0:
                        gradient = self._soft_support_gradient(particle, basis)
                        for row, column in ti.static(ti.ndrange(3, 3)):
                            adjoint_deformation[row, column] += (
                                self.adjoint_solution[self.affine_dof + 3 * block + row] * gradient[column]
                            )

                total_deformation = self._soft_point_trial_F(particle)
                plastic_inverse = self.soft_material.matProps.model.plastic_deformation_inverse[particle]
                elastic_trial = total_deformation @ plastic_inverse
                elastic_stress = self.soft_material.matProps.model.first_piola_stress_at(particle, elastic_trial)
                elastic_tangent = self.soft_material.matProps.model.first_piola_tangent_at(particle, elastic_trial)
                reference_jacobian = 1.0 / plastic_inverse.determinant()
                pulled_stress = elastic_stress @ plastic_inverse.transpose()
                inverse = plastic_inverse.inverse()
                volume_scale = self.scene.soft_point[particle].vol0 * self.scale_device[None]

                history_vjp = ti.Matrix.zero(float, 3, 3)
                for history_row, history_column in ti.static(ti.ndrange(3, 3)):
                    elastic_stress_derivative = ti.Matrix.zero(float, 3, 3)
                    for stress_column, stress_row in ti.static(ti.ndrange(3, 3)):
                        value = 0.0
                        for elastic_row in ti.static(range(3)):
                            value += (
                                elastic_tangent[
                                    3 * stress_column + stress_row,
                                    3 * history_column + elastic_row,
                                ]
                                * total_deformation[elastic_row, history_row]
                            )
                        elastic_stress_derivative[stress_row, stress_column] = value
                    total_stress_derivative = reference_jacobian * (
                        elastic_stress_derivative @ plastic_inverse.transpose()
                        - inverse[history_column, history_row] * pulled_stress
                    )
                    for row in ti.static(range(3)):
                        total_stress_derivative[row, history_row] += (
                            reference_jacobian * elastic_stress[row, history_column]
                        )
                    history_vjp[history_row, history_column] = -volume_scale * contraction(
                        adjoint_deformation, total_stress_derivative
                    )
                self.plastic_inverse_vjp[particle] = history_vjp

                hardening_stress_derivative = (
                    self.soft_material.matProps.model.first_piola_equivalent_plastic_strain_derivative_at(
                        particle, elastic_trial
                    )
                )
                total_hardening_derivative = (
                    reference_jacobian * hardening_stress_derivative @ plastic_inverse.transpose()
                )
                self.plastic_equivalent_strain_vjp[particle] = -volume_scale * contraction(
                    adjoint_deformation, total_hardening_derivative
                )
                self.plastic_deformation_vjp[particle] = (
                    -volume_scale
                    * unflatten_matrix(
                        elastic_tangent.transpose()
                        @ flatten_matrix(reference_jacobian * adjoint_deformation @ plastic_inverse),
                        elastic_trial,
                    )
                    @ plastic_inverse.transpose()
                )

    def _differentiate_soft_plastic_input_history(self):
        """Compatibility wrapper for the history pullback plus parameter VJP."""
        self._differentiate_soft_plastic_input_history_device()
        if hasattr(self, "material_parameter_vjp"):
            self._differentiate_soft_material_parameters()

    @ti.kernel
    def _differentiate_soft_material_parameters(self):
        for particle in range(self.soft_point_num):
            if int(self.scene.soft_point[particle].active) == 1:
                adjoint_deformation = ti.Matrix.zero(float, 3, 3)
                for basis in range(self._soft_support_count(particle)):
                    node = self._soft_support_node(particle, basis)
                    block = self._soft_block(node)
                    if block >= 0:
                        gradient = self._soft_support_gradient(particle, basis)
                        for row, column in ti.static(ti.ndrange(3, 3)):
                            adjoint_deformation[row, column] += (
                                self.adjoint_solution[self.affine_dof + 3 * block + row] * gradient[column]
                            )
                total_deformation = self._soft_point_trial_F(particle)
                parameter_vjp = self.soft_material.matProps.model.total_first_piola_parameter_vjp_at(
                    particle, total_deformation, adjoint_deformation
                )
                volume_scale = self.scene.soft_point[particle].vol0 * self.scale_device[None]
                for parameter_id in ti.static(range(4)):
                    ti.atomic_add(
                        self.material_parameter_vjp[None][parameter_id],
                        -volume_scale * parameter_vjp[parameter_id],
                    )

    @ti.kernel
    def _differentiate_soft_plastic_commit_state_device(self):
        for dof in self.plastic_commit_grid_vjp:
            self.plastic_commit_grid_vjp[dof] = 0.0
        for particle in range(self.soft_point_num):
            self.plastic_commit_deformation_vjp[particle] = ti.Matrix.zero(float, 3, 3)
            self.plastic_commit_inverse_vjp[particle] = ti.Matrix.zero(float, 3, 3)
            self.plastic_commit_equivalent_vjp[particle] = 0.0
            self.plastic_commit_volumetric_vjp[particle] = 0.0
            if int(self.scene.soft_point[particle].active) == 1:
                total_deformation = self._soft_point_trial_F(particle)
                (
                    trial_vjp,
                    plastic_vjp,
                    equivalent_vjp,
                    volumetric_vjp,
                ) = self.soft_material.matProps.model.commit_total_state_vjp(
                    particle,
                    total_deformation,
                    self.plastic_output_deformation_vjp[particle],
                    self.plastic_output_inverse_vjp[particle],
                    self.plastic_output_equivalent_vjp[particle],
                    self.plastic_output_volumetric_vjp[particle],
                )
                self.plastic_commit_deformation_vjp[particle] = trial_vjp
                self.plastic_commit_inverse_vjp[particle] = plastic_vjp
                self.plastic_commit_equivalent_vjp[particle] = equivalent_vjp
                self.plastic_commit_volumetric_vjp[particle] = volumetric_vjp
                for basis in range(self._soft_support_count(particle)):
                    node = self._soft_support_node(particle, basis)
                    block = self._soft_block(node)
                    if block >= 0:
                        gradient = self._soft_support_gradient(particle, basis)
                        for component in ti.static(range(3)):
                            value = 0.0
                            for column in ti.static(range(3)):
                                value += trial_vjp[component, column] * gradient[column]
                            ti.atomic_add(
                                self.plastic_commit_grid_vjp[self.affine_dof + 3 * block + component],
                                value,
                            )

    def _differentiate_soft_plastic_commit_state(self):
        """Compatibility wrapper for accepted-state and parameter pullbacks."""
        self._differentiate_soft_plastic_commit_state_device()
        if hasattr(self, "material_parameter_vjp"):
            self._differentiate_soft_material_commit_parameters()

    @ti.kernel
    def _differentiate_soft_material_commit_parameters(self):
        for particle in range(self.soft_point_num):
            if int(self.scene.soft_point[particle].active) == 1:
                total_deformation = self._soft_point_trial_F(particle)
                parameter_vjp = self.soft_material.matProps.model.commit_total_state_parameter_vjp(
                    particle,
                    total_deformation,
                    self.plastic_output_inverse_vjp[particle],
                    self.plastic_output_equivalent_vjp[particle],
                    self.plastic_output_volumetric_vjp[particle],
                )
                for parameter_id in ti.static(range(4)):
                    ti.atomic_add(
                        self.material_parameter_vjp[None][parameter_id],
                        parameter_vjp[parameter_id],
                    )

    @ti.kernel
    def _combine_soft_plastic_step_vjp(self):
        for particle in range(self.soft_point_num):
            self.plastic_deformation_vjp[particle] += self.plastic_commit_deformation_vjp[particle]
            self.plastic_inverse_vjp[particle] += self.plastic_commit_inverse_vjp[particle]
            self.plastic_equivalent_strain_vjp[particle] += self.plastic_commit_equivalent_vjp[particle]
            self.plastic_volumetric_strain_vjp[particle] += self.plastic_commit_volumetric_vjp[particle]

    @ti.func
    def _soft_point_adjoint(self, particle):
        result = ti.Vector.zero(float, 3)
        for basis in range(self._soft_support_count(particle)):
            node = self._soft_support_node(particle, basis)
            block = self._soft_block(node)
            if block >= 0:
                weight = self._soft_support_shape(particle, basis)
                for component in ti.static(range(3)):
                    result[component] += weight * self.adjoint_solution[self.affine_dof + 3 * block + component]
        return result

    @ti.func
    def _affine_vertex_adjoint(self, vertex):
        body = self.affine.node2body[vertex]
        result = ti.Vector.zero(float, 3)
        for control in range(4):
            weight = self.affine.basis[vertex, control]
            for component in ti.static(range(3)):
                result[component] += weight * self.adjoint_solution[12 * body + 3 * control + component]
        return result

    @ti.kernel
    def _prepare_coupled_trajectory_rhs(self):
        for dof in range(self.max_dof):
            self.trajectory_rhs[dof] = 0.0
        for dof in range(self.affine_dof):
            self.trajectory_rhs[dof] = self.affine.linear_rhs[dof]
        for dof in range(self.total_dof):
            self.trajectory_rhs[dof] += self.plastic_commit_grid_vjp[dof]
        inverse_dt = 1.0 / ti.max(self.dt_device[None], 1.0e-30)
        for particle in range(self.soft_point_num):
            position_vjp = self.soft_output_position_vjp[particle]
            velocity_vjp = self.soft_output_velocity_vjp[particle]
            self.soft_position_vjp[particle] = position_vjp
            self.soft_velocity_vjp[particle] = ti.Vector.zero(float, 3)
            grid_vjp = position_vjp + inverse_dt * velocity_vjp
            for basis in range(self._soft_support_count(particle)):
                node = self._soft_support_node(particle, basis)
                block = self._soft_block(node)
                if block >= 0:
                    weight = self._soft_support_shape(particle, basis)
                    for component in ti.static(range(3)):
                        ti.atomic_add(
                            self.trajectory_rhs[self.affine_dof + 3 * block + component],
                            weight * grid_vjp[component],
                        )

    @ti.kernel
    def _propagate_soft_velocity_input(self):
        dt = self.dt_device[None]
        for particle in range(self.soft_point_num):
            if int(self.scene.soft_point[particle].active) == 1:
                velocity_vjp = ti.Vector.zero(float, 3)
                mass = self.scene.soft_point[particle].m
                for basis in range(self._soft_support_count(particle)):
                    node = self._soft_support_node(particle, basis)
                    block = self._soft_block(node)
                    if block >= 0:
                        weight = self._soft_support_shape(particle, basis)
                        for component in ti.static(range(3)):
                            velocity_vjp[component] += (
                                dt * mass * weight * self.adjoint_solution[self.affine_dof + 3 * block + component]
                            )
                self.soft_velocity_vjp[particle] += velocity_vjp

    @ti.kernel
    def _differentiate_soft_barrier_input_position(self):
        for slot_p in range(self.soft_surface_point_num):
            p = self.scene.soft_surface_point_id[slot_p]
            if int(self.scene.soft_point[p].active) == 1:
                body_p = self._soft_support_body(p)
                x_p = self.scene.soft_point[p].x + self._soft_point_disp(p)
                mat_p = int(self.scene.soft_point[p].materialID)
                for pair_id in range(self.soft_pair_start[body_p], self.soft_pair_end[body_p]):
                    body_q = self.soft_pair[pair_id][1]
                    for slot_q in range(
                        self.scene.soft[body_q].surfacePointStart,
                        self.scene.soft[body_q].surfacePointEnd,
                    ):
                        q = self.scene.soft_surface_point_id[slot_q]
                        if int(self.scene.soft_point[q].active) == 1:
                            delta = x_p - (self.scene.soft_point[q].x + self._soft_point_disp(q))
                            dist2 = delta.dot(delta)
                            mat_q = int(self.scene.soft_point[q].materialID)
                            dhat = self._mat_pair_dhat(mat_p, mat_q)
                            if dist2 < dhat * dhat:
                                area = symmetric_contact_measure(
                                    self._soft_surface_measure(p),
                                    self._soft_surface_measure(q),
                                )
                                _, first, second = self._ipc_barrier_distance2(
                                    dist2,
                                    dhat * dhat,
                                    self._mat_pair_kappa(mat_p, mat_q),
                                )
                                distance_gradient = 2.0 * delta
                                hessian = (
                                    self.scale_device[None]
                                    * area
                                    * (
                                        second * distance_gradient.outer_product(distance_gradient)
                                        + 2.0 * first * ti.Matrix.identity(float, 3)
                                    )
                                )
                                response = hessian @ (self._soft_point_adjoint(p) - self._soft_point_adjoint(q))
                                for component in ti.static(range(3)):
                                    ti.atomic_add(
                                        self.soft_position_vjp[p][component],
                                        -response[component],
                                    )
                                    ti.atomic_add(
                                        self.soft_position_vjp[q][component],
                                        response[component],
                                    )

    @ti.kernel
    def _differentiate_mixed_barrier_input_position(self):
        for candidate in range(self.mixed_bvh.point_triangle_count[None]):
            particle = self.mixed_bvh.point_triangle[candidate][0]
            face = self.mixed_bvh.point_triangle_primitive[candidate][1]
            if int(self.scene.soft_point[particle].active) == 1:
                ab = self.affine.face2body[face]
                point = self.scene.soft_point[particle].x + self._soft_point_disp(particle)
                material = int(self.scene.soft_point[particle].materialID)
                dhat = self._pp_dhat(material, ab)
                tri = self.affine.faces[face]
                a = self.affine.x[tri[0]]
                b = self.affine.x[tri[1]]
                c = self.affine.x[tri[2]]
                dist2, gradient, distance_hessian, unused_type = point_triangle_distance_grad_hess(point, a, b, c)
                if dist2 < dhat * dhat:
                    _, first, second = self._ipc_barrier_distance2(
                        dist2,
                        dhat * dhat,
                        self._pp_kappa(material, ab),
                    )
                    coefficient = self.scale_device[None] * self._mixed_fv_measure(particle)
                    hessian = coefficient * (second * gradient.outer_product(gradient) + first * distance_hessian)
                    local_adjoint = ti.Vector.zero(float, 12)
                    point_adjoint = self._soft_point_adjoint(particle)
                    for component in ti.static(range(3)):
                        local_adjoint[component] = point_adjoint[component]
                    for local_vertex in ti.static(range(3)):
                        vertex_adjoint = self._affine_vertex_adjoint(tri[local_vertex])
                        for component in ti.static(range(3)):
                            local_adjoint[3 * (local_vertex + 1) + component] = vertex_adjoint[component]
                    response = hessian.transpose() @ local_adjoint
                    for component in ti.static(range(3)):
                        ti.atomic_add(
                            self.soft_position_vjp[particle][component],
                            -response[component],
                        )

    @ti.kernel
    def _differentiate_mixed_lagged_friction_reference(self):
        # The soft reference x_n cancels the x_n in x_n + u.  Affine uses
        # absolute y_{n+1}, so only its previous controls remain here.
        for contact in range(self.mixed_friction_count[None]):
            if contact < self.mixed_friction_capacity:
                particle = self.mixed_friction_point[contact]
                face = self.mixed_friction_face[contact]
                bary = self.mixed_friction_bary[contact]
                tri = self.affine.faces[face]
                current_closest = ti.Vector.zero(float, 3)
                reference_closest = ti.Vector.zero(float, 3)
                relative_adjoint = self._soft_point_adjoint(particle)
                for local_vertex in ti.static(range(3)):
                    current_closest += bary[local_vertex] * self.affine.x[tri[local_vertex]]
                    reference_closest += bary[local_vertex] * self.affine.hat_x[tri[local_vertex]]
                    relative_adjoint -= bary[local_vertex] * self._affine_vertex_adjoint(tri[local_vertex])
                relative = self.scene.soft_point[particle].x + self._soft_point_disp(particle) - current_closest
                reference_relative = self.soft_hat_x[particle] - reference_closest
                response = (
                    self.affine._lagged_friction_hessian(
                        relative,
                        reference_relative,
                        self.mixed_friction_normal[contact],
                        self.affine.friction_scale[0] * self.mixed_friction_coeff[contact],
                    )
                    @ relative_adjoint
                )
                body = self.affine.face2body[face]
                for local_vertex in ti.static(range(3)):
                    vertex = tri[local_vertex]
                    for control in range(4):
                        factor = -bary[local_vertex] * self.affine.basis[vertex, control]
                        for component in ti.static(range(3)):
                            ti.atomic_add(
                                self.affine.state_y_vjp[4 * body + control][component],
                                factor * response[component],
                            )

    @ti.kernel
    def _differentiate_coupled_friction_scale_parameter(self):
        for contact in range(self.soft_friction_count[None]):
            if contact < self.soft_friction_capacity:
                p = self.soft_friction_points[contact][0]
                q = self.soft_friction_points[contact][1]
                normal = self.soft_friction_normal[contact]
                projection = ti.Matrix.identity(float, 3) - normal.outer_product(normal)
                relative = (
                    self.scene.soft_point[p].x
                    + self._soft_point_disp(p)
                    - self.scene.soft_point[q].x
                    - self._soft_point_disp(q)
                )
                reference = self.soft_hat_x[p] - self.soft_hat_x[q]
                velocity = projection @ ((relative - reference) / self.dt_device[None])
                force = (
                    self.soft_friction_coeff[contact]
                    * self._friction_f1_div_vbarnorm(velocity.norm(), self.affine.epsv)
                    * (projection @ velocity)
                )
                relative_adjoint = self._soft_point_adjoint(p) - self._soft_point_adjoint(q)
                ti.atomic_add(
                    self.affine.friction_scale_vjp[None],
                    -relative_adjoint.dot(force),
                )

        for contact in range(self.mixed_friction_count[None]):
            if contact < self.mixed_friction_capacity:
                particle = self.mixed_friction_point[contact]
                face = self.mixed_friction_face[contact]
                bary = self.mixed_friction_bary[contact]
                tri = self.affine.faces[face]
                current_closest = ti.Vector.zero(float, 3)
                reference_closest = ti.Vector.zero(float, 3)
                relative_adjoint = self._soft_point_adjoint(particle)
                for local_vertex in ti.static(range(3)):
                    vertex = tri[local_vertex]
                    current_closest += bary[local_vertex] * self.affine.x[vertex]
                    reference_closest += bary[local_vertex] * self.affine.hat_x[vertex]
                    relative_adjoint -= bary[local_vertex] * self._affine_vertex_adjoint(vertex)
                projection = ti.Matrix.identity(float, 3) - self.mixed_friction_normal[contact].outer_product(
                    self.mixed_friction_normal[contact]
                )
                relative = self.scene.soft_point[particle].x + self._soft_point_disp(particle) - current_closest
                reference = self.soft_hat_x[particle] - reference_closest
                velocity = projection @ ((relative - reference) / self.dt_device[None])
                force = (
                    self.mixed_friction_coeff[contact]
                    * self._friction_f1_div_vbarnorm(velocity.norm(), self.affine.epsv)
                    * (projection @ velocity)
                )
                ti.atomic_add(
                    self.affine.friction_scale_vjp[None],
                    -relative_adjoint.dot(force),
                )

    @ti.kernel
    def _carry_coupled_state_vjp(self):
        for particle in range(self.soft_point_num):
            self.soft_output_position_vjp[particle] = self.soft_position_vjp[particle]
            self.soft_output_velocity_vjp[particle] = self.soft_velocity_vjp[particle]
            self.plastic_output_deformation_vjp[particle] = self.plastic_deformation_vjp[particle]
            self.plastic_output_inverse_vjp[particle] = self.plastic_inverse_vjp[particle]
            self.plastic_output_equivalent_vjp[particle] = self.plastic_equivalent_strain_vjp[particle]
            self.plastic_output_volumetric_vjp[particle] = self.plastic_volumetric_strain_vjp[particle]

    def differentiate_elastic_parameters(self, loss_gradient):
        """Return the pre-commit elastic equilibrium VJP for shared gravity."""
        soft_young = self._single_soft_young_modulus()
        adjoint = self.solve_elastic_adjoint(loss_gradient)
        self._differentiate_elastic_gravity()
        self._copy_affine_adjoint()
        self.affine._differentiate_young_parameter()
        self._differentiate_soft_young(soft_young)
        return {
            "gravity": np.asarray(self.gravity_vjp[None], dtype=np.float64),
            "affine_young_modulus": self.affine.young_vjp.to_numpy()[: self.affine.body_num].copy(),
            "soft_young_modulus": float(self.soft_young_vjp[None]),
            "adjoint": adjoint,
        }

    def differentiate_plastic_equilibrium_parameters(self, loss_gradient, reset_material_parameters=True):
        """Return one current-equilibrium VJP with prior history as input."""
        if reset_material_parameters:
            if hasattr(self, "material_parameter_vjp"):
                self.material_parameter_vjp.fill(0.0)
        adjoint = self.solve_plastic_equilibrium_adjoint(loss_gradient)
        self._differentiate_elastic_gravity()
        self._copy_affine_adjoint()
        self.affine._differentiate_young_parameter()
        self._differentiate_soft_plastic_input_history()
        material_values = (
            np.asarray(self.material_parameter_vjp[None], dtype=np.float64)
            if hasattr(self, "material_parameter_vjp")
            else np.zeros(4, dtype=np.float64)
        )
        model_name = type(
            getattr(
                getattr(getattr(self, "soft_material", None), "matProps", None),
                "model",
                None,
            )
        ).__name__
        if model_name == "FiniteStrainDruckerPragerModel":
            material_names = {
                "cohesion": float(material_values[2]),
                "friction_angle_degrees": float(material_values[3]),
            }
        elif model_name == "FiniteStrainVonMisesModel":
            material_names = {
                "yield_stress": float(material_values[2]),
                "hardening_modulus": float(material_values[3]),
            }
        else:
            material_names = {}
        return {
            "gravity": np.asarray(self.gravity_vjp[None], dtype=np.float64),
            "material_parameters": material_values,
            "young_modulus": float(material_values[0]),
            "poisson_ratio": float(material_values[1]),
            **material_names,
            "affine_young_modulus": self.affine.young_vjp.to_numpy()[: self.affine.body_num].copy(),
            "plastic_deformation_inverse": self.plastic_inverse_vjp.to_numpy()[: self.soft_point_num].copy(),
            "equivalent_plastic_strain": self.plastic_equivalent_strain_vjp.to_numpy()[: self.soft_point_num].copy(),
            "volumetric_plastic_strain": self.plastic_volumetric_strain_vjp.to_numpy()[: self.soft_point_num].copy(),
            "deformation_gradient": self.plastic_deformation_vjp.to_numpy()[: self.soft_point_num].copy(),
            "plastic_history": "input_vjp",
            "adjoint": adjoint,
        }

    def _load_soft_plastic_output_state_vjp(self, state_vjp):
        if not isinstance(state_vjp, dict):
            raise TypeError("state_vjp must be a dictionary of per-particle seeds")
        particle_num = self.soft_point_num
        capacity = max(particle_num, 1)

        def load_matrix(name, field):
            values = (
                np.asarray(state_vjp[name], dtype=np.float64)
                if name in state_vjp
                else np.zeros((particle_num, 3, 3), dtype=np.float64)
            )
            if values.shape != (particle_num, 3, 3) or not np.all(np.isfinite(values)):
                raise ValueError(f"state_vjp['{name}'] must be finite with shape " f"({particle_num}, 3, 3)")
            padded = np.zeros((capacity, 3, 3), dtype=np.float64)
            padded[:particle_num] = values
            field.from_numpy(padded)

        def load_scalar(name, field):
            values = (
                np.asarray(state_vjp[name], dtype=np.float64)
                if name in state_vjp
                else np.zeros(particle_num, dtype=np.float64)
            )
            if values.shape != (particle_num,) or not np.all(np.isfinite(values)):
                raise ValueError(f"state_vjp['{name}'] must be finite with shape " f"({particle_num},)")
            padded = np.zeros(capacity, dtype=np.float64)
            padded[:particle_num] = values
            field.from_numpy(padded)

        load_matrix("deformation_gradient", self.plastic_output_deformation_vjp)
        load_matrix("plastic_deformation_inverse", self.plastic_output_inverse_vjp)
        load_scalar("equivalent_plastic_strain", self.plastic_output_equivalent_vjp)
        load_scalar("volumetric_plastic_strain", self.plastic_output_volumetric_vjp)

    def differentiate_plastic_step_parameters(self, loss_gradient, state_vjp):
        """Pull one accepted soft-plastic update through coupled equilibrium."""
        model = getattr(self.soft_material.matProps, "model", None)
        if type(model).__name__ not in {
            "FiniteStrainDruckerPragerModel",
            "FiniteStrainVonMisesModel",
        }:
            raise ValueError(
                "accepted soft-plastic differentiation supports "
                "Drucker-Prager and von Mises; MCC is intentionally excluded"
            )
        self._load_soft_plastic_output_state_vjp(state_vjp)
        if hasattr(self, "material_parameter_vjp"):
            self.material_parameter_vjp.fill(0.0)
        self._differentiate_soft_plastic_commit_state()
        if isinstance(loss_gradient, ti.ScalarField):
            if len(loss_gradient.shape) != 1 or int(loss_gradient.shape[0]) < self.total_dof:
                raise ValueError("loss_gradient field must cover all active coupled degrees of freedom")
            copy_field(self.total_dof, self.plastic_step_rhs, loss_gradient)
        else:
            values = np.asarray(loss_gradient, dtype=np.float64).reshape(-1)
            if values.size != self.total_dof or not np.all(np.isfinite(values)):
                raise ValueError("loss_gradient must be finite and match the active coupled degrees of freedom")
            padded = np.zeros(self.max_dof, dtype=np.float64)
            padded[: self.total_dof] = values
            self.plastic_step_rhs.from_numpy(padded)
        add_field(
            self.total_dof,
            self.plastic_step_rhs,
            self.plastic_step_rhs,
            self.plastic_commit_grid_vjp,
        )
        result = self.differentiate_plastic_equilibrium_parameters(
            self.plastic_step_rhs, reset_material_parameters=False
        )
        self._combine_soft_plastic_step_vjp()
        result.update(
            {
                "deformation_gradient": self.plastic_deformation_vjp.to_numpy()[: self.soft_point_num].copy(),
                "plastic_deformation_inverse": self.plastic_inverse_vjp.to_numpy()[: self.soft_point_num].copy(),
                "equivalent_plastic_strain": self.plastic_equivalent_strain_vjp.to_numpy()[
                    : self.soft_point_num
                ].copy(),
                "volumetric_plastic_strain": self.plastic_volumetric_strain_vjp.to_numpy()[
                    : self.soft_point_num
                ].copy(),
                "plastic_history": "accepted_step_input_vjp",
            }
        )
        return result

    def pullback_coupled_step_device(self):
        """Reverse one accepted DP/VM Soft-Affine step on device fields."""
        model = getattr(self.soft_material.matProps, "model", None)
        if type(model).__name__ not in {
            "FiniteStrainDruckerPragerModel",
            "FiniteStrainVonMisesModel",
        }:
            raise ValueError(
                "coupled trajectory differentiation supports "
                "Drucker-Prager and von Mises; MCC is intentionally excluded"
            )
        if hasattr(self, "material_parameter_vjp"):
            self.material_parameter_vjp.fill(0.0)
        self.affine.device_prepare_step_state_adjoint(0)
        self._differentiate_soft_plastic_commit_state()
        self._prepare_coupled_trajectory_rhs()
        self._solve_equilibrium_adjoint(self.trajectory_rhs, allow_plastic=True)
        self._differentiate_elastic_gravity()
        self._copy_affine_adjoint()
        self.affine._differentiate_young_parameter()
        self.affine._differentiate_joint_target_parameters()
        self.affine._differentiate_joint_damping_parameters()
        self.affine._differentiate_friction_scale_parameter()
        self._differentiate_coupled_friction_scale_parameter()
        self._differentiate_soft_plastic_input_history()
        self.affine.device_propagate_step_state_adjoint()
        self.affine.device_propagate_lagged_friction_state_adjoint()
        self._propagate_soft_velocity_input()
        self._differentiate_soft_barrier_input_position()
        if self.mixed_bvh is not None:
            self._differentiate_mixed_barrier_input_position()
        self._differentiate_mixed_lagged_friction_reference()
        self._combine_soft_plastic_step_vjp()
        self._carry_coupled_state_vjp()

    def solve_direction(self, sims, grad):
        self._reject_cuda_host_vector_path("solve_direction")
        tick = time.time()
        active_blocks = self.total_dof // 3
        rhs = -np.asarray(grad, dtype=np.float64).reshape((active_blocks, 3))
        self.hash_triplet.finalize_taichi_assembly()
        tick = self._profile_stage("finalize_triplet", tick)
        result = self.hash_triplet.solve(
            rhs=rhs,
            active_nodes=active_blocks,
            tol=sims.affine_linear_tolerance,
            maxiter=sims.affine_linear_max_iteration,
            return_solution=True,
        )
        tick = self._profile_stage(
            "bicgstab_solve" if self.fully_implicit else "pcg_solve",
            tick,
        )
        if not result["converged"]:
            raise RuntimeError(
                "SoftAffineIPC HashTriplet "
                f"{self.hash_triplet.solver} did not converge: "
                f"residual={result['residual']:.6e}, "
                f"iterations={result['iterations']}"
            )
        direction = result["x"].reshape(-1)
        affine_dir = direction[: self.affine_dof]
        soft_dir = np.zeros(self.max_soft_dof, dtype=np.float64)
        soft_count = max(self.total_dof - self.affine_dof, 0)
        if soft_count > 0:
            soft_dir[:soft_count] = direction[self.affine_dof : self.total_dof]
        self.soft_direction.from_numpy(soft_dir)
        self._profile_stage("store_direction", tick)
        return affine_dir, soft_dir[:soft_count], direction

    @ti.kernel
    def _load_device_newton_rhs(self, active_dof: ti.i32):
        for block in range(self.max_dof // 3):
            value = ti.Vector.zero(float, 3)
            for component in ti.static(range(3)):
                dof = 3 * block + component
                if dof < active_dof:
                    value[component] = -self.global_grad[dof]
            self.hash_triplet.rhs[block] = value

    @ti.kernel
    def _scale_and_scatter_device_direction(self, active_dof: ti.i32, scale: float):
        self.device_status[None] = 0
        for dof in range(self.max_dof):
            if dof < active_dof:
                block = dof // 3
                component = dof - 3 * block
                value = scale * self.hash_triplet.x[block][component]
                self.hash_triplet.x[block][component] = value
                if value != value or ti.abs(value) > 1.0e15:
                    ti.atomic_max(self.device_status[None], 1)
                if dof < self.affine_dof:
                    control = dof // 3
                    component = dof - 3 * control
                    self.affine.direction_y[control][component] = value
                else:
                    soft_dof = dof - self.affine_dof
                    self.soft_direction[soft_dof] = value
        for soft_dof in range(self.max_soft_dof):
            if soft_dof >= active_dof - self.affine_dof:
                self.soft_direction[soft_dof] = 0.0

    def solve_direction_device(self, sims, clamp_direction=True):
        """Solve ``J p = -R`` without transferring either vector to NumPy."""
        self._assert_cuda_device_residency()
        tick = time.time()
        self.hash_triplet.finalize_taichi_assembly()
        tick = self._profile_stage("finalize_triplet", tick)
        self._load_device_newton_rhs(int(self.total_dof))
        result = self.hash_triplet.solve(
            active_nodes=int(self.total_dof // 3),
            tol=sims.affine_linear_tolerance,
            maxiter=sims.affine_linear_max_iteration,
            return_solution=False,
        )
        tick = self._profile_stage(
            "bicgstab_solve" if self.fully_implicit else "pcg_solve",
            tick,
        )
        if not result["converged"]:
            raise RuntimeError(
                "SoftAffineIPC HashTriplet "
                f"{self.hash_triplet.solver} did not converge: "
                f"residual={result['residual']:.6e}, "
                f"iterations={result['iterations']}"
            )
        direction_norm = float(self.hash_triplet._solution_inf_norm(int(self.total_dof // 3)))
        unclamped_direction_norm = direction_norm
        scale = 1.0
        max_step = float(sims.affine_max_step)
        if clamp_direction and max_step > 0.0 and direction_norm > max_step:
            scale = max_step / direction_norm
            direction_norm = max_step
        self._scale_and_scatter_device_direction(int(self.total_dof), float(scale))
        if int(self.device_status[None]) != 0 or not math.isfinite(direction_norm):
            raise RuntimeError("SoftAffineIPC CUDA Newton correction is non-finite")
        result["unclamped_solution_inf_norm"] = float(unclamped_direction_norm)
        result["solution_inf_norm"] = float(direction_norm)
        self._profile_stage("device_direction_ready", tick)
        return result

    @ti.kernel
    def _device_residual_inf_norm(self, active_dof: ti.i32) -> float:
        self.device_status[None] = 0
        maximum = 0.0
        for dof in range(active_dof):
            value = self.global_grad[dof]
            if value != value or ti.abs(value) > 1.0e15:
                ti.atomic_max(self.device_status[None], 2)
            else:
                ti.atomic_max(maximum, ti.abs(value))
        return maximum

    @ti.kernel
    def _device_residual_squared_norm(self, active_dof: ti.i32) -> float:
        self.device_status[None] = 0
        norm_squared = 0.0
        for dof in range(active_dof):
            value = self.global_grad[dof]
            if value != value or ti.abs(value) > 1.0e15:
                ti.atomic_max(self.device_status[None], 2)
            else:
                norm_squared += value * value
        return norm_squared

    @ti.kernel
    def _device_energy_directional_derivative(self, active_dof: ti.i32) -> float:
        slope = 0.0
        for dof in range(active_dof):
            block = dof // 3
            component = dof - 3 * block
            slope += self.global_grad[dof] * self.hash_triplet.x[block][component]
        return slope

    @ti.kernel
    def _device_residual_merit_directional_derivative(self, active_dof: ti.i32, regularization_shift: float) -> float:
        slope = 0.0
        for dof in range(active_dof):
            block = dof // 3
            component = dof - 3 * block
            physical_jp = (
                self.hash_triplet.Ax[block][component] - regularization_shift * self.hash_triplet.x[block][component]
            )
            slope += self.global_grad[dof] * physical_jp
        return slope

    def residual_inf_norm_device(self):
        value = float(self._device_residual_inf_norm(int(self.total_dof)))
        if int(self.device_status[None]) != 0 or not math.isfinite(value):
            raise RuntimeError("SoftAffineIPC CUDA residual is non-finite")
        return value

    def residual_merit_device(self):
        value = 0.5 * float(self._device_residual_squared_norm(int(self.total_dof)))
        if int(self.device_status[None]) != 0 or not math.isfinite(value):
            return math.inf
        return value

    def energy_directional_derivative_device(self):
        return float(self._device_energy_directional_derivative(int(self.total_dof)))

    def residual_merit_directional_derivative_device(self):
        nnz = int(self.hash_triplet.non_diag.element_pair_num[0])
        self.hash_triplet.matvec(
            int(self.total_dof // 3),
            nnz,
            self.hash_triplet.x,
            self.hash_triplet.Ax,
        )
        shift = float(self.sims.affine_fully_implicit_jacobian_shift)
        return float(self._device_residual_merit_directional_derivative(int(self.total_dof), shift))

    def negate_direction_device(self):
        self._scale_and_scatter_device_direction(int(self.total_dof), -1.0)

    def apply_jacobian(self, direction):
        """Apply the assembled Jacobian to a scalar coupled direction."""
        self._reject_cuda_host_vector_path("apply_jacobian")
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        if direction.size != self.total_dof:
            raise ValueError(
                "SoftAffineIPC Jacobian direction size does not match the " "active coupled degrees of freedom"
            )
        active_blocks = self.total_dof // 3
        self.hash_triplet.x.from_numpy(self.hash_triplet._pad_vector_array(direction.reshape((active_blocks, 3))))
        nnz = int(self.hash_triplet.non_diag.element_pair_num[0])
        self.hash_triplet.matvec(
            active_blocks,
            nnz,
            self.hash_triplet.x,
            self.hash_triplet.Ax,
        )
        product = self.hash_triplet.Ax.to_numpy()[:active_blocks].reshape(-1).copy()
        if self.fully_implicit:
            # The optional diagonal shift regularizes the Newton solve but is
            # not part of the physical residual derivative used by the merit
            # function.  Remove its action after the matrix-vector product.
            shift = float(self.sims.affine_fully_implicit_jacobian_shift)
            if shift > 0.0:
                product -= shift * direction
        return product

    def store_soft_base(self):
        self._copy_soft_disp(self.soft_disp, self.soft_disp_base)

    @ti.kernel
    def _store_coupled_base_device(self):
        for control in range(self.affine.control_num):
            self.affine_y_base[control] = self.affine.y[control]
        for dof in range(self.max_soft_dof):
            self.soft_disp_base[dof] = self.soft_disp[dof]

    @ti.kernel
    def _set_coupled_trial_device(self, alpha: float):
        for control in range(self.affine.control_num):
            self.affine.y[control] = self.affine_y_base[control] + alpha * self.affine.direction_y[control]
        for dof in range(self.max_soft_dof):
            self.soft_disp[dof] = self.soft_disp_base[dof] + alpha * self.soft_direction[dof]

    @ti.kernel
    def _restore_coupled_base_device(self):
        for control in range(self.affine.control_num):
            self.affine.y[control] = self.affine_y_base[control]
        for dof in range(self.max_soft_dof):
            self.soft_disp[dof] = self.soft_disp_base[dof]

    def store_coupled_base_device(self):
        self._store_coupled_base_device()

    def set_coupled_trial_device(self, alpha):
        self._set_coupled_trial_device(float(alpha))

    def restore_coupled_base_device(self):
        self._restore_coupled_base_device()

    def set_soft_trial(self, alpha):
        self._set_soft_trial(float(alpha))

    def restore_soft_base(self):
        self._copy_soft_disp(self.soft_disp_base, self.soft_disp)

    def accept_step(self, affine_y_flat):
        self.affine_state.accept_step(self.affine_state.unpack(affine_y_flat), self.dt)
        self._accept_soft_step()

    def accept_step_device(self):
        """Commit one accepted coupled solution entirely in Taichi fields."""
        self.affine.device_accept_step(self.dt)
        self._accept_soft_step()

    def init_step_size(
        self, y_flat, affine_direction, ccd_type="ccd", eta=0.2, accd_tolerance=1.0e-7, max_iteration=10000
    ):
        self._reject_cuda_host_vector_path("init_step_size")
        affine_alpha = self.affine.init_step_size(
            y_flat,
            affine_direction,
            ccd_type=ccd_type,
            eta=eta,
            accd_tolerance=accd_tolerance,
            max_iteration=max_iteration,
        )
        mixed_alpha = self._mixed_ccd_step_size(
            y_flat,
            affine_direction,
            ccd_type=ccd_type,
            eta=eta,
            accd_tolerance=accd_tolerance,
            max_iteration=max_iteration,
        )
        soft_alpha = self._soft_soft_ccd_step_size(
            ccd_type=ccd_type,
            eta=eta,
            accd_tolerance=accd_tolerance,
            max_iteration=max_iteration,
        )
        return max(0.0, min(1.0, affine_alpha, mixed_alpha, soft_alpha))

    def init_step_size_device(self, ccd_type="ccd", eta=0.2, accd_tolerance=1.0e-7, max_iteration=10000):
        """Coupled IPC CCD using current Taichi iterate/direction fields."""
        self._assert_cuda_device_residency()
        mode, eta, thickness = ccd_mode_parameters(ccd_type, eta, accd_tolerance)
        if mode in ("none", "off"):
            return 1.0
        affine = self.affine
        affine_alpha = affine.init_step_size_device(
            ccd_type=mode,
            eta=eta,
            accd_tolerance=accd_tolerance,
            max_iteration=max_iteration,
        )
        mixed_alpha = self._mixed_ccd_step_size_device(eta, thickness, max_iteration, mode == "accd")
        soft_alpha = self._soft_soft_ccd_step_size(
            ccd_type=mode,
            eta=eta,
            accd_tolerance=accd_tolerance,
            max_iteration=max_iteration,
        )
        return max(0.0, min(1.0, affine_alpha, mixed_alpha, soft_alpha))

    def _mixed_ccd_step_size_device(self, eta, thickness, max_iteration, accd):
        # ``affine.x``/``affine.dx`` were reconstructed by
        # init_step_size_device(), and soft directions already live in their
        # Taichi field after the linear solve.
        self._build_mixed_pairs(1)
        self._update_mixed_mesh_candidates(True)
        self.ccd_alpha[None] = 1.0
        if self.affine.levelset_contact:
            self._compute_mixed_levelset_ccd_alpha(float(eta), min(int(max_iteration), 64))
        elif self.mixed_bvh is not None:
            self._compute_mixed_ccd_alpha(float(eta), float(thickness), int(max_iteration), bool(accd))
        return float(self.ccd_alpha[None])

    def _mixed_ccd_step_size(
        self, y_flat, affine_direction, ccd_type="ccd", eta=0.2, accd_tolerance=1.0e-7, max_iteration=10000
    ):
        mode, eta, thickness = ccd_mode_parameters(ccd_type, eta, accd_tolerance)
        if mode in ("none", "off"):
            return 1.0
        y_np = np.ascontiguousarray(y_flat.reshape((self.affine.control_num, 3)), dtype=np.float64)
        direction_np = np.ascontiguousarray(affine_direction.reshape((self.affine.control_num, 3)), dtype=np.float64)
        self.affine.y.from_numpy(y_np)
        self.affine.direction_y.from_numpy(direction_np)
        self.affine._reconstruct_vertices()
        self.affine._reconstruct_vertex_directions()
        # CCD candidates must cover the full segment from the current
        # iterate to the proposed iterate.  A current-only body AABB misses
        # fast contacts whose endpoints both start outside ``dhat``.
        self._build_mixed_pairs(1)
        self._update_mixed_mesh_candidates(True)
        self.ccd_alpha[None] = 1.0
        if self.affine.levelset_contact:
            self._compute_mixed_levelset_ccd_alpha(float(eta), min(int(max_iteration), 64))
        elif self.mixed_bvh is not None:
            self._compute_mixed_ccd_alpha(float(eta), thickness, int(max_iteration), mode == "accd")
        return float(self.ccd_alpha[None])

    def _soft_soft_ccd_step_size(self, ccd_type="ccd", eta=0.2, accd_tolerance=1.0e-7, max_iteration=10000):
        mode, eta, thickness = ccd_mode_parameters(ccd_type, eta, accd_tolerance)
        if mode in ("none", "off") or self.soft_num <= 1:
            return 1.0
        self._build_mixed_pairs(1)
        self.ccd_alpha[None] = 1.0
        self._compute_soft_soft_ccd_alpha(float(eta), thickness, int(max_iteration), mode == "accd")
        return float(self.ccd_alpha[None])

    def output_snapshot(self):
        # Recorder boundary: synchronize only when the user requests output.
        self.affine.sync_output_state()
        vertices, faces, body_ids, group_ids = self.affine_state.surface_mesh()
        face_body_ids, face_group_ids, face_material_ids = self._surface_cell_data()
        body_vertices = self.affine_state.world_vertices()
        body_num = self.affine_state.body_num
        centers = np.zeros((body_num, 3), dtype=np.float64)
        bbox_min = np.zeros((body_num, 3), dtype=np.float64)
        bbox_max = np.zeros((body_num, 3), dtype=np.float64)
        radius = np.zeros(body_num, dtype=np.float64)
        for body_id, bverts in enumerate(body_vertices):
            centers[body_id] = np.mean(bverts, axis=0)
            bbox_min[body_id] = np.min(bverts, axis=0)
            bbox_max[body_id] = np.max(bverts, axis=0)
            radius[body_id] = np.max(np.linalg.norm(bverts - centers[body_id], axis=1))
        return {
            "body_num": body_num,
            "vertices": vertices,
            "faces": faces,
            "bodyID": body_ids,
            "groupID": group_ids,
            "faceBodyID": face_body_ids,
            "faceGroupID": face_group_ids,
            "faceMaterialID": face_material_ids,
            "bodyGroupID": np.asarray([body["groupID"] for body in self.affine_state.bodies], dtype=np.int32),
            "materialID": np.asarray([body["materialID"] for body in self.affine_state.bodies], dtype=np.int32),
            "y": self.affine_state.y.copy(),
            "v_y": self.affine_state.v_y.copy(),
            "center": centers,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max,
            "radius": radius,
        }

    def _surface_cell_data(self):
        face_body_ids = []
        face_group_ids = []
        face_material_ids = []
        for body_id, body in enumerate(self.affine_state.bodies):
            face_num = int(np.asarray(body["faces"]).shape[0])
            face_body_ids.extend([body_id] * face_num)
            face_group_ids.extend([body["groupID"]] * face_num)
            face_material_ids.extend([body["materialID"]] * face_num)
        return (
            np.asarray(face_body_ids, dtype=np.int32),
            np.asarray(face_group_ids, dtype=np.int32),
            np.asarray(face_material_ids, dtype=np.int32),
        )

    @ti.func
    def _soft_support_body(self, point):
        bodyID = self.scene.soft_point[point].bodyID
        return self.scene.rigid[bodyID].softID

    @ti.func
    def _soft_support_index(self, point):
        support = point
        if ti.static(self.scene.soft_support_shared):
            sb = self._soft_support_body(point)
            support = self.scene.soft[sb].templatePointStart + point - self.scene.soft[sb].startIndex
        return support

    @ti.func
    def _soft_support_count(self, point):
        return self.scene.soft_shape_count[self._soft_support_index(point)]

    @ti.func
    def _soft_support_node(self, point, basis):
        support = self._soft_support_index(point)
        node = self.scene.soft_shape_node[support, basis]
        if ti.static(self.scene.soft_support_shared):
            sb = self._soft_support_body(point)
            node += self.scene.soft[sb].mpmGridStart
        return node

    @ti.func
    def _soft_support_shape(self, point, basis):
        return self.scene.soft_shape[self._soft_support_index(point), basis]

    @ti.func
    def _soft_support_gradient(self, point, basis):
        gradient = self.scene.soft_dshape[self._soft_support_index(point), basis]
        if ti.static(self.scene.soft_support_shared):
            sb = self._soft_support_body(point)
            gradient = self.scene.soft[sb].referenceRotation @ gradient / self.scene.soft[sb].scale
        return gradient

    @ti.func
    def _soft_surface_support_index(self, surface_node):
        bodyID = self.scene.surface[surface_node]
        sb = self.scene.rigid[bodyID].softID
        return (
            sb,
            self.scene.soft[sb].templateSurfaceStart + surface_node - self.scene.soft[sb].startNode,
        )

    @ti.kernel
    def _prepare_soft_grid(self):
        for i in range(self.soft_node2dof.shape[0]):
            self.soft_node2dof[i] = 0
        for i in range(self.soft_grid_num):
            self.scene.soft_grid[i]._grid_reset()
        for p in range(self.soft_point_num):
            if int(self.scene.soft_point[p].active) == 1:
                for n in range(self._soft_support_count(p)):
                    node = self._soft_support_node(p, n)
                    w = self._soft_support_shape(p, n)
                    m = self.scene.soft_point[p].m
                    ti.atomic_add(self.scene.soft_grid[node].m, w * m)
                    ti.atomic_add(self.scene.soft_grid[node].v, w * m * self.scene.soft_point[p].v)
        for node in range(self.soft_grid_num):
            if self.scene.soft_grid[node].m > Threshold:
                if self.soft_node_fixed[node] == 0:
                    self.scene.soft_grid[node].v /= self.scene.soft_grid[node].m
                    self.soft_node2dof[node] = 1
                else:
                    self.scene.soft_grid[node].v = ZEROVEC3f
                    self.scene.soft_grid[node].f = ZEROVEC3f

    @ti.kernel
    def _initialize_soft_fixed_nodes(self):
        for node in range(self.soft_grid_num):
            self.soft_node_fixed[node] = 0
        for constraint in range(self.soft_velocity_constraint_num):
            node = self.scene.soft_velocity_constraint[constraint]
            if 0 <= node and node < self.soft_grid_num:
                self.soft_node_fixed[node] = 1

    @ti.kernel
    def _fill_soft_dof(self) -> ti.i32:
        active = 0
        for node in range(self.soft_grid_num):
            dof = self.soft_node2dof[node]
            if self.scene.soft_grid[node].m > Threshold and self.soft_node_fixed[node] == 0:
                self.soft_dof2node[dof - 1] = node
                ti.atomic_max(active, dof)
            else:
                # The inclusive scan gives inactive nodes the preceding active
                # node's index.  Keep them out of predictor/assembly kernels.
                self.soft_node2dof[node] = -1
        return active

    @ti.kernel
    def _initialize_soft_step(self):
        for i in range(self.max_soft_dof):
            self.soft_disp[i] = 0.0
            self.soft_disp_base[i] = 0.0
            self.soft_direction[i] = 0.0
        for p in range(self.soft_point_num):
            self.soft_hat_x[p] = self.scene.soft_point[p].x

    @ti.kernel
    def _copy_soft_disp(self, source: ti.template(), target: ti.template()):
        for i in range(self.max_soft_dof):
            target[i] = source[i]

    @ti.kernel
    def _set_soft_trial(self, alpha: float):
        for i in range(self.max_soft_dof):
            self.soft_disp[i] = self.soft_disp_base[i] + alpha * self.soft_direction[i]

    @ti.func
    def _soft_block(self, node):
        return self.soft_node2dof[node] - 1

    @ti.func
    def _soft_row(self, node, d):
        return self.affine_dof + 3 * self._soft_block(node) + d

    @ti.func
    def _soft_node_disp(self, node):
        disp = ti.Vector.zero(float, 3)
        block = self._soft_block(node)
        if block >= 0:
            for d in ti.static(range(3)):
                disp[d] = self.soft_disp[3 * block + d]
        return disp

    @ti.func
    def _soft_point_disp(self, p):
        disp = ti.Vector.zero(float, 3)
        for n in range(self._soft_support_count(p)):
            node = self._soft_support_node(p, n)
            disp += self._soft_support_shape(p, n) * self._soft_node_disp(node)
        return disp

    @ti.func
    def _soft_node_direction(self, node):
        disp = ti.Vector.zero(float, 3)
        block = self._soft_block(node)
        if block >= 0:
            for d in ti.static(range(3)):
                disp[d] = self.soft_direction[3 * block + d]
        return disp

    @ti.func
    def _soft_point_direction(self, p):
        disp = ti.Vector.zero(float, 3)
        for n in range(self._soft_support_count(p)):
            node = self._soft_support_node(p, n)
            disp += self._soft_support_shape(p, n) * self._soft_node_direction(node)
        return disp

    @ti.func
    def _soft_point_trial_F(self, p):
        gradu = mat3x3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        for n in range(self._soft_support_count(p)):
            node = self._soft_support_node(p, n)
            gradu += self._soft_node_disp(node).outer_product(self._soft_support_gradient(p, n))
        material_id = self.scene.soft_point[p].materialID
        return self.soft_material.matProps.trial_deformation_gradient(p, material_id, self.scene.soft_point[p].F, gradu)

    @ti.func
    def _add_grad(self, row, value):
        if row >= 0 and row < self.total_dof:
            ti.atomic_add(self.global_grad[row], value)

    @ti.func
    def _add_matrix_block(self, block_i, block_j, block):
        if block_i >= 0 and 3 * block_i < self.total_dof and block_j >= 0 and 3 * block_j < self.total_dof:
            if ti.static(self.fully_implicit):
                self.hash_triplet.add_block_entry(block_i, block_j, block)
            elif block_i <= block_j:
                self.hash_triplet.add_block_entry(block_i, block_j, block)

    @ti.func
    def _local_stiffness(self, dF_dx1, dF_dx2, d2Psi_dF2):
        H = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(3)):
            for t in ti.static(range(3)):
                pair = 0
                while pair < 9:
                    j = pair // 3
                    n = pair - 3 * j
                    H[i, t] += d2Psi_dF2[j * 3 + i, n * 3 + t] * dF_dx1[j] * dF_dx2[n]
                    pair += 1
        return H

    @ti.func
    def _soft_material_node_gradient(self, stress, shape_gradient):
        gradient = ti.Vector.zero(float, 3)
        gradient[0] = stress[0] * shape_gradient[0] + stress[3] * shape_gradient[1] + stress[6] * shape_gradient[2]
        gradient[1] = stress[1] * shape_gradient[0] + stress[4] * shape_gradient[1] + stress[7] * shape_gradient[2]
        gradient[2] = stress[2] * shape_gradient[0] + stress[5] * shape_gradient[1] + stress[8] * shape_gradient[2]
        return gradient

    @ti.kernel
    def _assemble_soft_energy_gradient(
        self,
        need_matrix: ti.template(),
        project_spd: ti.template(),
    ):
        dt = ti.max(self.dt_device[None], 1.0e-30)
        scale = self.scale_device[None]
        damp = self.soft_background_damping
        for node in range(self.soft_grid_num):
            block = self._soft_block(node)
            if block >= 0:
                mass = self.scene.soft_grid[node].m
                old_v = self.scene.soft_grid[node].v
                disp = self._soft_node_disp(node)
                coeff = mass + damp * mass * dt
                linear = mass * old_v * dt + mass * self.gravity * scale
                self.energy[None] += 0.5 * coeff * disp.dot(disp) - linear.dot(disp)
                for d in ti.static(range(3)):
                    row = self._soft_row(node, d)
                    self._add_grad(row, coeff * disp[d] - linear[d])
                if ti.static(need_matrix):
                    self._add_matrix_block(
                        self._soft_row(node, 0) // 3,
                        self._soft_row(node, 0) // 3,
                        coeff * ti.Matrix.identity(float, 3),
                    )

        for p in range(self.soft_point_num):
            if int(self.scene.soft_point[p].active) == 1:
                F = self._soft_point_trial_F(p)
                vol = self.scene.soft_point[p].vol0
                material_id = self.scene.soft_point[p].materialID
                self.energy[None] += self.soft_material.matProps.Psi_at(p, material_id, F) * vol * scale
                dPsi_dF = self.soft_material.matProps.dPsi_div_dF_at(p, material_id, F) * vol * scale
                d2Psi_dF2 = ti.Matrix.zero(float, 9, 9)
                if ti.static(need_matrix):
                    d2Psi_dF2 = self.soft_material.matProps.d2Psi_div_d2F_at(p, material_id, F) * vol * scale
                    # ipc-sim uses projectSPD=true for elasticity in the
                    # lagged minimization.  The fully implicit residual
                    # Jacobian deliberately retains the exact negative
                    # material curvature.
                    if ti.static(project_spd):
                        d2Psi_dF2 = psd_project_nd(d2Psi_dF2)
                for a in range(self._soft_support_count(p)):
                    node_a = self._soft_support_node(p, a)
                    grad_a = self._soft_support_gradient(p, a)
                    block_a = self._soft_block(node_a)
                    if block_a >= 0:
                        dPsi_dx = self._soft_material_node_gradient(dPsi_dF, grad_a)
                        for d in ti.static(range(3)):
                            self._add_grad(self._soft_row(node_a, d), dPsi_dx[d])
                        if ti.static(need_matrix):
                            for b in range(self._soft_support_count(p)):
                                node_b = self._soft_support_node(p, b)
                                block_b = self._soft_block(node_b)
                                if block_b >= 0:
                                    H = self._local_stiffness(
                                        grad_a,
                                        self._soft_support_gradient(p, b),
                                        d2Psi_dF2,
                                    )
                                    self._add_matrix_block(
                                        self._soft_row(node_a, 0) // 3,
                                        self._soft_row(node_b, 0) // 3,
                                        H,
                                    )

    @ti.func
    def _aabb_overlap(self, amin, amax, bmin, bmax):
        overlap = 1
        for d in ti.static(range(3)):
            if amax[d] < bmin[d] or bmax[d] < amin[d]:
                overlap = 0
        return overlap == 1

    @ti.kernel
    def _build_mixed_bvh_positions(self, swept: ti.i32):
        for point in range(self.soft_point_num):
            current = self.scene.soft_point[point].x + self._soft_point_disp(point)
            self.mixed_bvh_position[point] = current
            self.mixed_bvh_end_position[point] = current
            if swept != 0:
                self.mixed_bvh_end_position[point] += self._soft_point_direction(point)
        for vertex in range(self.affine.vertex_num):
            node = self.soft_point_num + vertex
            self.mixed_bvh_position[node] = self.affine.x[vertex]
            self.mixed_bvh_end_position[node] = self.affine.x[vertex]
            if swept != 0:
                self.mixed_bvh_end_position[node] += self.affine.dx[vertex]

    def _update_mixed_mesh_candidates(self, swept=False):
        if self.mixed_bvh is None:
            return 0
        self._build_mixed_bvh_positions(int(bool(swept)))
        try:
            count, _ = self.mixed_bvh.rebuild(
                self.mixed_bvh_position,
                self.affine.dhat,
                self.mixed_bvh_end_position if swept else None,
            )
        except RuntimeError as error:
            if "point-triangle broad-phase capacity" in str(error):
                raise RuntimeError("SoftAffineIPC max_point_triangle_pairs is too small: " f"{error}") from None
            raise
        return count

    @ti.kernel
    def _build_mixed_pairs(self, swept: ti.i32):
        expand = self.affine.dhat
        for sb in range(self.soft_num):
            self.soft_min[sb] = vec3f(1.0e30, 1.0e30, 1.0e30)
            self.soft_max[sb] = vec3f(-1.0e30, -1.0e30, -1.0e30)
            self.soft_center_disp[sb] = ZEROVEC3f
            self.soft_center_direction[sb] = ZEROVEC3f
            self.soft_body_mass[sb] = 0.0

        for p in range(self.soft_point_num):
            if int(self.scene.soft_point[p].active) == 1:
                sb = self._soft_support_body(p)
                mass = self.scene.soft_point[p].m
                point_disp = self._soft_point_disp(p)
                point_direction = self._soft_point_direction(p)
                ti.atomic_add(self.soft_center_disp[sb], mass * point_disp)
                ti.atomic_add(self.soft_center_direction[sb], mass * point_direction)
                ti.atomic_add(self.soft_body_mass[sb], mass)
                pos0 = self.scene.soft_point[p].x + point_disp
                pos1 = pos0
                if swept != 0:
                    pos1 += point_direction
                r = (
                    0.5
                    * ti.pow(
                        ti.max(self.scene.soft_point[p].vol0, 1.0e-30),
                        1.0 / 3.0,
                    )
                    + expand
                )
                for d in ti.static(range(3)):
                    ti.atomic_min(self.soft_min[sb][d], ti.min(pos0[d], pos1[d]) - r)
                    ti.atomic_max(self.soft_max[sb][d], ti.max(pos0[d], pos1[d]) + r)

        for sb in range(self.soft_num):
            if self.soft_body_mass[sb] > Threshold:
                self.soft_center_disp[sb] /= self.soft_body_mass[sb]
                self.soft_center_direction[sb] /= self.soft_body_mass[sb]

        self.mixed_pair_num[None] = 0
        if ti.static(self.affine.levelset_contact):
            for ab in range(self.affine.body_num):
                self.affine_min[ab] = vec3f(1.0e30, 1.0e30, 1.0e30)
                self.affine_max[ab] = vec3f(-1.0e30, -1.0e30, -1.0e30)
            for v in range(self.affine.vertex_num):
                ab = self.affine.node2body[v]
                pos0 = self.affine.x[v]
                pos1 = pos0
                if swept != 0:
                    pos1 += self.affine.dx[v]
                for d in ti.static(range(3)):
                    ti.atomic_min(
                        self.affine_min[ab][d],
                        ti.min(pos0[d], pos1[d]) - expand,
                    )
                    ti.atomic_max(
                        self.affine_max[ab][d],
                        ti.max(pos0[d], pos1[d]) + expand,
                    )

            # Level-set bodies have no surface triangles for the mixed BVH.
            ti.loop_config(serialize=True)
            for sb in range(self.soft_num):
                self.mixed_pair_start[sb] = self.mixed_pair_num[None]
                for ab in range(self.affine.body_num):
                    if self._aabb_overlap(
                        self.soft_min[sb], self.soft_max[sb], self.affine_min[ab], self.affine_max[ab]
                    ):
                        idx = self.mixed_pair_num[None]
                        self.mixed_pair_num[None] = idx + 1
                        if idx < self.mixed_pair.shape[0]:
                            self.mixed_pair[idx] = ti.Vector([sb, ab])
                self.mixed_pair_end[sb] = self.mixed_pair_num[None]

        self.soft_pair_num[None] = 0
        ti.loop_config(serialize=True)
        for source in range(self.soft_num):
            self.soft_pair_start[source] = self.soft_pair_num[None]
            for target in range(source + 1, self.soft_num):
                if self._aabb_overlap(
                    self.soft_min[source], self.soft_max[source], self.soft_min[target], self.soft_max[target]
                ):
                    idx = self.soft_pair_num[None]
                    self.soft_pair_num[None] = idx + 1
                    if idx < self.soft_pair.shape[0]:
                        self.soft_pair[idx] = ti.Vector([source, target])
            self.soft_pair_end[source] = self.soft_pair_num[None]

    @ti.func
    def _mat_pair_dhat(self, mat_i, mat_j):
        mi = self.affine._material_id(mat_i)
        mj = self.affine._material_id(mat_j)
        return self.affine.pp_dhat[mi, mj]

    @ti.func
    def _mat_pair_kappa(self, mat_i, mat_j):
        mi = self.affine._material_id(mat_i)
        mj = self.affine._material_id(mat_j)
        return self.affine.pp_kappa[mi, mj]

    @ti.func
    def _mat_pair_penalty(self, mat_i, mat_j):
        return self._mat_pair_kappa(mat_i, mat_j) * ti.static(self.affine.semi_penalty_scale)

    @ti.func
    def _mat_pair_mu(self, mat_i, mat_j):
        mi = self.affine._material_id(mat_i)
        mj = self.affine._material_id(mat_j)
        mu = self.affine.pp_mu[mi, mj]
        if ti.static(not self.fully_implicit and self.fully_mu_dynamic >= 0.0):
            mu = self.fully_mu_dynamic
        if mu < 0.0:
            mu = 0.0
        return mu

    @ti.func
    def _pp_dhat(self, soft_mat, affine_body):
        return self._mat_pair_dhat(soft_mat, self.affine.body_material[affine_body])

    @ti.func
    def _pp_kappa(self, soft_mat, affine_body):
        return self._mat_pair_kappa(soft_mat, self.affine.body_material[affine_body])

    @ti.func
    def _pp_penalty(self, soft_mat, affine_body):
        return self._mat_pair_penalty(soft_mat, self.affine.body_material[affine_body])

    @ti.func
    def _pp_mu(self, soft_mat, affine_body):
        return self._mat_pair_mu(soft_mat, self.affine.body_material[affine_body])

    @ti.func
    def _ipc_barrier_distance2(self, dist2, active_gap2, kappa):
        return self.affine._ipc_barrier_distance2(dist2, active_gap2, kappa)

    @ti.func
    def _ipc_barrier_gap(self, gap, active_gap, kappa):
        return self.affine._ipc_barrier_gap(gap, active_gap, kappa)

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
        energy, first, second = semi_ipc_terms(gap, multiplier, penalty)
        ti.atomic_max(self.semi_constraint_violation[None], ti.max(-gap, 0.0))
        return energy, first, second

    @ti.kernel
    def _update_semi_multipliers(self):
        self.semi_constraint_violation[None] = 0.0
        offset = ti.Vector([0.25, 0.25, 0.25])
        for slot in range(self.semi_capacity):
            if self.semi_state[slot] == 0:
                key = self.semi_key[slot]
                gap = 0.0
                penalty = self.affine.semi_penalty
                valid = 1
                if key[2] == 0:
                    p, q = key[0], key[1]
                    x_p = self.scene.soft_point[p].x + self._soft_point_disp(p)
                    x_q = self.scene.soft_point[q].x + self._soft_point_disp(q)
                    mat_p = int(self.scene.soft_point[p].materialID)
                    mat_q = int(self.scene.soft_point[q].materialID)
                    gap = (x_p - x_q).norm() - self._mat_pair_dhat(mat_p, mat_q)
                    penalty = self._mat_pair_penalty(mat_p, mat_q)
                elif key[2] == 1:
                    point_id, face_id = key[0], key[1]
                    affine_body = self.affine.face2body[face_id]
                    point = self.scene.soft_point[point_id].x + self._soft_point_disp(point_id)
                    face = self.affine.faces[face_id]
                    closest, unused_barycentric = self._closest_point_triangle(
                        point,
                        self.affine.x[face[0]],
                        self.affine.x[face[1]],
                        self.affine.x[face[2]],
                    )
                    material = int(self.scene.soft_point[point_id].materialID)
                    gap = (point - closest).norm() - self._pp_dhat(material, affine_body)
                    penalty = self._pp_penalty(material, affine_body)
                else:
                    point_id, affine_body = key[0], key[1]
                    point = self.scene.soft_point[point_id].x + self._soft_point_disp(point_id)
                    target_A = self.affine._affine_levelset_matrix(affine_body)
                    if ti.abs(target_A.determinant()) > 1.0e-12:
                        material_point = target_A.inverse() @ (point - self.affine.y[affine_body * 4])
                        scale = self.affine.body_scale[affine_body]
                        coordinate = (material_point - offset) / scale
                        phi, unused_gradient, unused_hessian, inside = self.affine._sample_affine_levelset(
                            affine_body, coordinate
                        )
                        material = int(self.scene.soft_point[point_id].materialID)
                        if inside:
                            gap = scale * phi - self._pp_dhat(material, affine_body)
                        else:
                            valid = 0
                        penalty = self._pp_penalty(material, affine_body)
                    else:
                        valid = 0
                if valid != 0:
                    self.semi_multiplier[slot] = semi_ipc_update_multiplier(gap, self.semi_multiplier[slot], penalty)
                    ti.atomic_max(self.semi_constraint_violation[None], ti.max(-gap, 0.0))
                else:
                    self.semi_multiplier[slot] = 0.0

    def accept_semi_update_device(self):
        if not self.is_semi:
            return
        self.affine.accept_semi_update_device()
        self._update_semi_multipliers()

    def semi_contact_converged(self):
        return (
            not self.is_semi
            or max(
                float(self.semi_constraint_violation[None]),
                float(self.affine.semi_constraint_violation[None]),
            )
            <= self.affine.semi_constraint_tolerance
        )

    @ti.func
    def _soft_levelset_distance_local(self, target_sb, local):
        bodyID = self.scene.soft[target_sb].bodyID
        return self.scene.box[bodyID].distance(local, self.scene.rigid_grid)

    @ti.func
    def _soft_levelset_distance_query(self, target_sb, query):
        bodyID = self.scene.soft[target_sb].bodyID
        rotate = SetToRotate(self.scene.rigid[bodyID].q)
        local = rotate.transpose() @ (query - self.scene.rigid[bodyID]._get_position())
        return self._soft_levelset_distance_local(target_sb, local)

    @ti.func
    def _soft_levelset_sample_gradient_hessian_local(self, bodyID, point):
        body_box = self.scene.box[bodyID]
        indices = body_box.closet_corner(point)
        x_ind, y_ind, z_ind = indices[0], indices[1], indices[2]
        spacing = ti.max(body_box.grid_space, 1.0e-12)
        x_red = (point[0] - (body_box.xmin[0] + x_ind * spacing)) / spacing
        y_red = (point[1] - (body_box.xmin[1] + y_ind * spacing)) / spacing
        z_red = (point[2] - (body_box.xmin[2] + z_ind * spacing)) / spacing
        x_red = ti.min(ti.max(x_red, 0.0), 1.0)
        y_red = ti.min(ti.max(y_red, 0.0), 1.0)
        z_red = ti.min(ti.max(z_red, 0.0), 1.0)

        phi = 0.0
        grad_reduced = ti.Vector.zero(float, 3)
        hess_reduced = ti.Matrix.zero(float, 3, 3)
        for i in ti.static(range(2)):
            wx = (1.0 - x_red) if i == 0 else x_red
            dwx = -1.0 if i == 0 else 1.0
            for j in ti.static(range(2)):
                wy = (1.0 - y_red) if j == 0 else y_red
                dwy = -1.0 if j == 0 else 1.0
                for k in ti.static(range(2)):
                    wz = (1.0 - z_red) if k == 0 else z_red
                    dwz = -1.0 if k == 0 else 1.0
                    idx = linearize3D(x_ind + i, y_ind + j, z_ind + k, body_box.gnum) + body_box.startGrid
                    ls_val = self.scene.rigid_grid[idx].distance_field * body_box.scale
                    phi += ls_val * wx * wy * wz
                    grad_reduced[0] += ls_val * dwx * wy * wz
                    grad_reduced[1] += ls_val * wx * dwy * wz
                    grad_reduced[2] += ls_val * wx * wy * dwz
                    hess_reduced[0, 1] += ls_val * dwx * dwy * wz
                    hess_reduced[0, 2] += ls_val * dwx * wy * dwz
                    hess_reduced[1, 2] += ls_val * wx * dwy * dwz
        hess_reduced[1, 0] = hess_reduced[0, 1]
        hess_reduced[2, 0] = hess_reduced[0, 2]
        hess_reduced[2, 1] = hess_reduced[1, 2]
        grad_local = grad_reduced / spacing
        hess_local = hess_reduced / (spacing * spacing)
        return phi, grad_local, hess_local

    @ti.func
    def _soft_levelset_distance_gradient_hessian(self, target_sb, query):
        bodyID = self.scene.soft[target_sb].bodyID
        rotate = SetToRotate(self.scene.rigid[bodyID].q)
        local = rotate.transpose() @ (query - self.scene.rigid[bodyID]._get_position())
        phi, grad_local, hess_local = self._soft_levelset_sample_gradient_hessian_local(bodyID, local)
        grad = rotate @ grad_local
        hess = rotate @ hess_local @ rotate.transpose()
        return phi, grad, hess

    @ti.func
    def _soft_point_row_weight(self, p, basis_id):
        row_base = -1
        weight = 0.0
        if basis_id < self._soft_support_count(p):
            node = self._soft_support_node(p, basis_id)
            block = self._soft_block(node)
            if block >= 0:
                row_base = self.affine_dof + 3 * block
                weight = self._soft_support_shape(p, basis_id)
        return row_base, weight

    @ti.func
    def _scatter_soft_point_gradient(self, p, grad):
        for n in range(self._soft_support_count(p)):
            row_base, weight = self._soft_point_row_weight(p, n)
            if row_base >= 0 and weight != 0.0:
                for d in ti.static(range(3)):
                    self._add_grad(row_base + d, weight * grad[d])

    @ti.func
    def _scatter_soft_point_hessian(self, p, hess):
        for bi in range(self._soft_support_count(p)):
            row_base, wi = self._soft_point_row_weight(p, bi)
            if row_base >= 0 and wi != 0.0:
                for bj in range(self._soft_support_count(p)):
                    col_base, wj = self._soft_point_row_weight(p, bj)
                    if col_base >= 0 and wj != 0.0:
                        self._add_matrix_block(
                            row_base // 3,
                            col_base // 3,
                            wi * wj * hess,
                        )

    @ti.func
    def _soft_surface_measure(self, p):
        measure = self.scene.soft_point[p].surface_weight
        if measure <= 0.0:
            measure = ti.pow(
                ti.max(self.scene.soft_point[p].vol0, 1.0e-30),
                2.0 / 3.0,
            )
        return measure

    @ti.func
    def _mixed_fv_measure(self, p):
        # Use one quarter of the colliding vertex's lumped surface measure.
        # The barrier width is a
        # constitutive activation distance and is not part of this measure.
        return 0.25 * self._soft_surface_measure(p)

    @ti.func
    def _soft_pair_local_row_weight(self, p, q, local_site, basis_id):
        row_base = -1
        weight = 0.0
        point = p
        if local_site == 1:
            point = q
        if basis_id < self._soft_support_count(point):
            node = self._soft_support_node(point, basis_id)
            block = self._soft_block(node)
            if block >= 0:
                row_base = self.affine_dof + 3 * block
                weight = self._soft_support_shape(point, basis_id)
        return row_base, weight

    @ti.func
    def _scatter_soft_pair_local_gradient(self, p, q, local_gradient):
        for local_site in ti.static(range(2)):
            point = p
            if ti.static(local_site == 1):
                point = q
            for basis_id in range(self._soft_support_count(point)):
                row_base, weight = self._soft_pair_local_row_weight(p, q, local_site, basis_id)
                if row_base >= 0 and weight != 0.0:
                    for component in ti.static(range(3)):
                        self._add_grad(
                            row_base + component,
                            weight * local_gradient[3 * local_site + component],
                        )

    @ti.func
    def _scatter_soft_pair_local_hessian(self, p, q, local_hessian):
        for site_i in range(2):
            point_i = p
            if site_i == 1:
                point_i = q
            for basis_i in range(self._soft_support_count(point_i)):
                row_base, weight_i = self._soft_pair_local_row_weight(p, q, site_i, basis_i)
                if row_base >= 0 and weight_i != 0.0:
                    for site_j in range(2):
                        point_j = p
                        if site_j == 1:
                            point_j = q
                        for basis_j in range(self._soft_support_count(point_j)):
                            col_base, weight_j = self._soft_pair_local_row_weight(p, q, site_j, basis_j)
                            if col_base >= 0 and weight_j != 0.0:
                                block = ti.Matrix.zero(float, 3, 3)
                                for row, column in ti.static(ti.ndrange(3, 3)):
                                    block[row, column] = (
                                        weight_i
                                        * weight_j
                                        * local_hessian[
                                            3 * site_i + row,
                                            3 * site_j + column,
                                        ]
                                    )
                                self._add_matrix_block(row_base // 3, col_base // 3, block)

    @ti.func
    def _assemble_soft_pair_friction(self, p, q, rel, hat_rel, normal, coeff, need_matrix: ti.template()):
        coeff *= self.affine.friction_scale[0]
        if coeff > 0.0:
            P = ti.Matrix.identity(float, 3) - normal.outer_product(normal)
            vbar = P @ ((rel - hat_rel) / ti.max(self.dt_device[None], 1.0e-30))
            vbarnorm = vbar.norm()
            ti.atomic_add(
                self.energy[None],
                coeff * self._friction_f0(vbarnorm, self.affine.epsv, self.dt_device[None]),
            )
            f1 = self._friction_f1_div_vbarnorm(vbarnorm, self.affine.epsv)
            relative_gradient = coeff * f1 * (P @ vbar)
            weights = point_point_stencil_weights()
            local_gradient = pullback_relative_gradient4(weights, relative_gradient)
            self._scatter_soft_pair_local_gradient(p, q, local_gradient)
            if ti.static(need_matrix):
                f_hess = self._friction_hess_term(vbarnorm, self.affine.epsv)
                inner = coeff * f1 * ti.Matrix.identity(float, 3)
                if vbarnorm > 1.0e-30:
                    inner += coeff * f_hess / vbarnorm * vbar.outer_product(vbar)
                if ti.static(not self.fully_implicit):
                    inner = psd_project_nd(inner)
                relative_hessian = P @ inner @ P.transpose() / ti.max(self.dt_device[None], 1.0e-30)
                local_hessian = pullback_relative_hessian4(weights, relative_hessian)
                self._scatter_soft_pair_local_hessian(p, q, local_hessian)

    @ti.kernel
    def _reset_lagged_friction_cache(self):
        self.soft_friction_count[None] = 0
        self.soft_friction_overflow[None] = 0
        self.mixed_friction_count[None] = 0
        self.mixed_friction_overflow[None] = 0

    @ti.kernel
    def _serial_scan_soft_contact_prefix(self):
        ti.loop_config(serialize=True)
        for slot in range(1, self.soft_surface_point_num):
            self.soft_contact_prefix[slot] += self.soft_contact_prefix[slot - 1]

    @ti.kernel
    def _serial_scan_mixed_contact_prefix(self):
        ti.loop_config(serialize=True)
        for candidate in range(1, self.mixed_candidate_capacity):
            self.mixed_contact_prefix[candidate] += self.mixed_contact_prefix[candidate - 1]

    def _scan_soft_contact_counts(self):
        if self.soft_surface_point_num <= 1:
            return
        if self.soft_surface_point_num <= 2:
            self._serial_scan_soft_contact_prefix()
        else:
            self.soft_contact_prefix_sum.run(self.soft_contact_prefix)

    def _scan_mixed_contact_counts(self):
        if self.mixed_candidate_capacity <= 1:
            return
        if self.mixed_candidate_capacity <= 2:
            self._serial_scan_mixed_contact_prefix()
        else:
            self.mixed_contact_prefix_sum.run(self.mixed_contact_prefix)

    @ti.kernel
    def _count_soft_lagged_friction_contacts(self):
        for index in self.soft_contact_prefix:
            self.soft_contact_prefix[index] = 0
        for slot_p in range(self.soft_surface_point_num):
            count = 0
            p = self.scene.soft_surface_point_id[slot_p]
            if int(self.scene.soft_point[p].active) == 1:
                body_p = self._soft_support_body(p)
                x_p = self.scene.soft_point[p].x + self._soft_point_disp(p)
                mat_p = int(self.scene.soft_point[p].materialID)
                for pair_id in range(self.soft_pair_start[body_p], self.soft_pair_end[body_p]):
                    body_q = self.soft_pair[pair_id][1]
                    for slot_q in range(
                        self.scene.soft[body_q].surfacePointStart, self.scene.soft[body_q].surfacePointEnd
                    ):
                        q = self.scene.soft_surface_point_id[slot_q]
                        if int(self.scene.soft_point[q].active) == 1:
                            x_q = self.scene.soft_point[q].x + self._soft_point_disp(q)
                            mat_q = int(self.scene.soft_point[q].materialID)
                            dhat = self._mat_pair_dhat(mat_p, mat_q)
                            delta = x_p - x_q
                            if self._mat_pair_mu(mat_p, mat_q) > 0.0 and delta.dot(delta) < dhat * dhat:
                                count += 1
            self.soft_contact_prefix[slot_p] = count

    @ti.kernel
    def _finalize_soft_lagged_friction_count(self):
        required = 0
        if ti.static(self.soft_surface_point_num > 0):
            required = self.soft_contact_prefix[self.soft_surface_point_num - 1]
        self.soft_friction_count[None] = required
        self.soft_friction_overflow[None] = ti.cast(required > self.soft_friction_capacity, ti.i32)

    @ti.kernel
    def _write_soft_lagged_friction_contacts(self):
        for slot_p in range(self.soft_surface_point_num):
            output = 0
            if slot_p > 0:
                output = self.soft_contact_prefix[slot_p - 1]
            local_index = 0
            p = self.scene.soft_surface_point_id[slot_p]
            if int(self.scene.soft_point[p].active) == 1:
                body_p = self._soft_support_body(p)
                x_p = self.scene.soft_point[p].x + self._soft_point_disp(p)
                mat_p = int(self.scene.soft_point[p].materialID)
                for pair_id in range(self.soft_pair_start[body_p], self.soft_pair_end[body_p]):
                    body_q = self.soft_pair[pair_id][1]
                    for slot_q in range(
                        self.scene.soft[body_q].surfacePointStart, self.scene.soft[body_q].surfacePointEnd
                    ):
                        q = self.scene.soft_surface_point_id[slot_q]
                        if int(self.scene.soft_point[q].active) != 1:
                            continue
                        x_q = self.scene.soft_point[q].x + self._soft_point_disp(q)
                        mat_q = int(self.scene.soft_point[q].materialID)
                        dhat = self._mat_pair_dhat(mat_p, mat_q)
                        delta = x_p - x_q
                        dist2 = delta.dot(delta)
                        mu = self._mat_pair_mu(mat_p, mat_q)
                        if mu > 0.0 and dist2 < dhat * dhat:
                            index = output + local_index
                            local_index += 1
                            if index < self.soft_friction_capacity:
                                area = symmetric_contact_measure(
                                    self._soft_surface_measure(p),
                                    self._soft_surface_measure(q),
                                )
                                dist = ti.sqrt(ti.max(dist2, 1.0e-30))
                                normal_force = 0.0
                                if ti.static(self.is_semi):
                                    _, first, _ = self._semi_terms(
                                        ti.Vector([ti.min(p, q), ti.max(p, q), 0, -1]),
                                        dist - dhat,
                                        self._mat_pair_penalty(mat_p, mat_q),
                                    )
                                    normal_force = ti.max(-first, 0.0)
                                else:
                                    _, db, _ = self._ipc_barrier_distance2(
                                        dist2,
                                        dhat * dhat,
                                        self._mat_pair_kappa(mat_p, mat_q),
                                    )
                                    normal_force = ti.max(-2.0 * db * dist, 0.0)
                                coeff = mu * normal_force * area * self.scale_device[None]
                                fallback = self.soft_hat_x[p] - self.soft_hat_x[q]
                                self.soft_friction_points[index] = ti.Vector([p, q])
                                self.soft_friction_normal[index] = normalized_contact_direction(delta, fallback)
                                self.soft_friction_coeff[index] = coeff

    def _initialize_soft_lagged_friction(self):
        if not self.stable_lagged_contacts:
            self._initialize_soft_lagged_friction_atomic()
            return
        self._count_soft_lagged_friction_contacts()
        self._scan_soft_contact_counts()
        self._finalize_soft_lagged_friction_count()
        required = int(self.soft_friction_count[None])
        if int(self.soft_friction_overflow[None]) != 0:
            raise RuntimeError(
                "SoftAffineIPC soft-soft lagged-friction buffer overflow: "
                f"required {required}, capacity "
                f"{self.soft_friction_capacity}"
            )
        self._write_soft_lagged_friction_contacts()

    @ti.kernel
    def _initialize_soft_lagged_friction_atomic(self):
        for slot_p in range(self.soft_surface_point_num):
            p = self.scene.soft_surface_point_id[slot_p]
            if int(self.scene.soft_point[p].active) == 1:
                body_p = self._soft_support_body(p)
                x_p = self.scene.soft_point[p].x + self._soft_point_disp(p)
                mat_p = int(self.scene.soft_point[p].materialID)
                for pair_id in range(self.soft_pair_start[body_p], self.soft_pair_end[body_p]):
                    body_q = self.soft_pair[pair_id][1]
                    for slot_q in range(
                        self.scene.soft[body_q].surfacePointStart,
                        self.scene.soft[body_q].surfacePointEnd,
                    ):
                        q = self.scene.soft_surface_point_id[slot_q]
                        if int(self.scene.soft_point[q].active) == 1:
                            x_q = self.scene.soft_point[q].x + self._soft_point_disp(q)
                            mat_q = int(self.scene.soft_point[q].materialID)
                            dhat = self._mat_pair_dhat(mat_p, mat_q)
                            delta = x_p - x_q
                            dist2 = delta.dot(delta)
                            mu = self._mat_pair_mu(mat_p, mat_q)
                            if mu > 0.0 and dist2 < dhat * dhat:
                                _, db, _ = self._ipc_barrier_distance2(
                                    dist2,
                                    dhat * dhat,
                                    self._mat_pair_kappa(mat_p, mat_q),
                                )
                                area = symmetric_contact_measure(
                                    self._soft_surface_measure(p),
                                    self._soft_surface_measure(q),
                                )
                                dist = ti.sqrt(ti.max(dist2, 1.0e-30))
                                normal_force = ti.max(-2.0 * db * dist, 0.0)
                                coeff = mu * normal_force * area * self.scale_device[None]
                                fallback = self.soft_hat_x[p] - self.soft_hat_x[q]
                                normal = normalized_contact_direction(delta, fallback)
                                index = ti.atomic_add(self.soft_friction_count[None], 1)
                                if index < self.soft_friction_capacity:
                                    self.soft_friction_points[index] = ti.Vector([p, q])
                                    self.soft_friction_normal[index] = normal
                                    self.soft_friction_coeff[index] = coeff
                                else:
                                    self.soft_friction_overflow[None] = 1

    @ti.kernel
    def _count_soft_barrier_contacts(self):
        self.soft_barrier_count[None] = 0
        for slot_p in range(self.soft_surface_point_num):
            p = self.scene.soft_surface_point_id[slot_p]
            if int(self.scene.soft_point[p].active) == 1:
                body_p = self._soft_support_body(p)
                x_p = self.scene.soft_point[p].x + self._soft_point_disp(p)
                mat_p = int(self.scene.soft_point[p].materialID)
                for pair_id in range(self.soft_pair_start[body_p], self.soft_pair_end[body_p]):
                    body_q = self.soft_pair[pair_id][1]
                    for slot_q in range(
                        self.scene.soft[body_q].surfacePointStart,
                        self.scene.soft[body_q].surfacePointEnd,
                    ):
                        q = self.scene.soft_surface_point_id[slot_q]
                        if int(self.scene.soft_point[q].active) == 1:
                            x_q = self.scene.soft_point[q].x + self._soft_point_disp(q)
                            mat_q = int(self.scene.soft_point[q].materialID)
                            dhat = self._mat_pair_dhat(mat_p, mat_q)
                            delta = x_p - x_q
                            if delta.dot(delta) < dhat * dhat:
                                ti.atomic_add(self.soft_barrier_count[None], 1)

    def _validate_soft_barrier_contact_capacity(self):
        self._count_soft_barrier_contacts()
        required = int(self.soft_barrier_count[None])
        if required > self.soft_contact_capacity:
            raise RuntimeError(
                "SoftAffineIPC soft-soft current-contact buffer overflow: "
                f"required {required}, capacity {self.soft_contact_capacity}"
            )

    def _assemble_soft_soft_barrier(self, need_matrix, project_spd=None):
        if project_spd is None:
            project_spd = not self.fully_implicit
        if bool(need_matrix) and self.stable_lagged_contacts:
            self._validate_soft_barrier_contact_capacity()
        self._assemble_soft_soft_barrier_kernel(bool(need_matrix), bool(project_spd))

    @ti.kernel
    def _assemble_soft_soft_barrier_kernel(
        self,
        need_matrix: ti.template(),
        project_spd: ti.template(),
    ):
        for slot_p in range(self.soft_surface_point_num):
            p = self.scene.soft_surface_point_id[slot_p]
            if int(self.scene.soft_point[p].active) == 1:
                body_p = self._soft_support_body(p)
                x_p = self.scene.soft_point[p].x + self._soft_point_disp(p)
                mat_p = int(self.scene.soft_point[p].materialID)
                for pair_id in range(self.soft_pair_start[body_p], self.soft_pair_end[body_p]):
                    body_q = self.soft_pair[pair_id][1]
                    for slot_q in range(
                        self.scene.soft[body_q].surfacePointStart,
                        self.scene.soft[body_q].surfacePointEnd,
                    ):
                        q = self.scene.soft_surface_point_id[slot_q]
                        if int(self.scene.soft_point[q].active) == 1:
                            x_q = self.scene.soft_point[q].x + self._soft_point_disp(q)
                            mat_q = int(self.scene.soft_point[q].materialID)
                            dhat = self._mat_pair_dhat(mat_p, mat_q)
                            delta = x_p - x_q
                            dist2 = delta.dot(delta)
                            if dist2 < dhat * dhat:
                                area = symmetric_contact_measure(
                                    self._soft_surface_measure(p),
                                    self._soft_surface_measure(q),
                                )
                                base_coeff = self.scale_device[None] * area
                                weights = point_point_stencil_weights()
                                if ti.static(self.is_semi):
                                    distance = ti.sqrt(ti.max(dist2, 1.0e-30))
                                    energy, first, second = self._semi_terms(
                                        ti.Vector([ti.min(p, q), ti.max(p, q), 0, -1]),
                                        distance - dhat,
                                        self._mat_pair_penalty(mat_p, mat_q),
                                    )
                                    ti.atomic_add(self.energy[None], base_coeff * energy)
                                    normal = delta / distance
                                    local_gradient = pullback_relative_gradient4(weights, base_coeff * first * normal)
                                    self._scatter_soft_pair_local_gradient(p, q, local_gradient)
                                    if ti.static(need_matrix):
                                        local_hessian = pullback_relative_hessian4(
                                            weights,
                                            base_coeff * second * normal.outer_product(normal),
                                        )
                                        self._scatter_soft_pair_local_hessian(p, q, local_hessian)
                                else:
                                    energy, db, ddb = self._ipc_barrier_distance2(
                                        dist2,
                                        dhat * dhat,
                                        self._mat_pair_kappa(mat_p, mat_q),
                                    )
                                    ti.atomic_add(self.energy[None], base_coeff * energy)
                                    relative_gradient_d = 2.0 * delta
                                    relative_gradient = base_coeff * db * relative_gradient_d
                                    local_gradient = pullback_relative_gradient4(weights, relative_gradient)
                                    self._scatter_soft_pair_local_gradient(p, q, local_gradient)
                                    if ti.static(need_matrix):
                                        relative_hessian_d = 2.0 * ti.Matrix.identity(float, 3)
                                        relative_hessian = base_coeff * (
                                            ddb * relative_gradient_d.outer_product(relative_gradient_d)
                                            + db * relative_hessian_d
                                        )
                                        if ti.static(project_spd):
                                            relative_hessian = psd_project_nd(relative_hessian)
                                        local_hessian = pullback_relative_hessian4(weights, relative_hessian)
                                        self._scatter_soft_pair_local_hessian(p, q, local_hessian)

    @ti.kernel
    def _assemble_soft_lagged_friction(self, need_matrix: ti.template()):
        for contact in range(self.soft_friction_count[None]):
            if contact < self.soft_friction_capacity:
                p = self.soft_friction_points[contact][0]
                q = self.soft_friction_points[contact][1]
                rel = (
                    self.scene.soft_point[p].x
                    + self._soft_point_disp(p)
                    - self.scene.soft_point[q].x
                    - self._soft_point_disp(q)
                )
                hat_rel = self.soft_hat_x[p] - self.soft_hat_x[q]
                self._assemble_soft_pair_friction(
                    p,
                    q,
                    rel,
                    hat_rel,
                    self.soft_friction_normal[contact],
                    self.soft_friction_coeff[contact],
                    need_matrix,
                )

    @ti.kernel
    def _assemble_soft_fully_implicit_friction(self, need_matrix: ti.template()):
        for slot_p in range(self.soft_surface_point_num):
            p = self.scene.soft_surface_point_id[slot_p]
            if int(self.scene.soft_point[p].active) == 1:
                body_p = self._soft_support_body(p)
                x_p = self.scene.soft_point[p].x + self._soft_point_disp(p)
                mat_p = int(self.scene.soft_point[p].materialID)
                for pair_id in range(self.soft_pair_start[body_p], self.soft_pair_end[body_p]):
                    body_q = self.soft_pair[pair_id][1]
                    for slot_q in range(
                        self.scene.soft[body_q].surfacePointStart,
                        self.scene.soft[body_q].surfacePointEnd,
                    ):
                        q = self.scene.soft_surface_point_id[slot_q]
                        if int(self.scene.soft_point[q].active) == 1:
                            x_q = self.scene.soft_point[q].x + self._soft_point_disp(q)
                            mat_q = int(self.scene.soft_point[q].materialID)
                            delta = x_p - x_q
                            distance2 = delta.dot(delta)
                            dhat = self._mat_pair_dhat(mat_p, mat_q)
                            if distance2 < dhat * dhat:
                                _, db, ddb = self._ipc_barrier_distance2(
                                    distance2,
                                    dhat * dhat,
                                    self._mat_pair_kappa(mat_p, mat_q),
                                )
                                positions = ti.Vector.zero(float, 12)
                                hats = ti.Vector.zero(float, 12)
                                distance_gradient = ti.Vector.zero(float, 12)
                                for component in ti.static(range(3)):
                                    positions[component] = x_p[component]
                                    positions[3 + component] = x_q[component]
                                    hats[component] = self.soft_hat_x[p][component]
                                    hats[3 + component] = self.soft_hat_x[q][component]
                                    distance_gradient[component] = 2.0 * delta[component]
                                    distance_gradient[3 + component] = -2.0 * delta[component]
                                area = symmetric_contact_measure(
                                    self._soft_surface_measure(p),
                                    self._soft_surface_measure(q),
                                )
                                if ti.static(need_matrix):
                                    distance_hessian = ti.Matrix.zero(float, 12, 12)
                                    for component in ti.static(range(3)):
                                        distance_hessian[component, component] = 2.0
                                        distance_hessian[component, 3 + component] = -2.0
                                        distance_hessian[3 + component, component] = -2.0
                                        distance_hessian[3 + component, 3 + component] = 2.0
                                    local_residual, local_jacobian = self._fully_implicit_local_friction(
                                        positions,
                                        hats,
                                        distance2,
                                        distance_gradient,
                                        distance_hessian,
                                        db,
                                        ddb,
                                        self._mat_pair_mu(mat_p, mat_q),
                                        area,
                                        2,
                                        True,
                                    )
                                    self._scatter_soft_pair_local_gradient(p, q, local_residual)
                                    self._scatter_soft_pair_local_hessian(p, q, local_jacobian)
                                else:
                                    local_residual = self._fully_implicit_local_friction_residual(
                                        positions,
                                        hats,
                                        distance2,
                                        distance_gradient,
                                        db,
                                        self._mat_pair_mu(mat_p, mat_q),
                                        area,
                                        2,
                                    )
                                    self._scatter_soft_pair_local_gradient(p, q, local_residual)

    def _assemble_soft_soft_contact(self, need_matrix, project_spd=None):
        if self.soft_num < 2:
            return
        self._assemble_soft_soft_barrier(bool(need_matrix), project_spd)
        if self.fully_implicit:
            self._assemble_soft_fully_implicit_friction(bool(need_matrix))
        else:
            self._assemble_soft_lagged_friction(bool(need_matrix))

    @ti.kernel
    def _compute_soft_soft_ccd_alpha(self, eta: float, thickness: float, max_iteration: ti.i32, accd: ti.template()):
        for slot_p in range(self.soft_surface_point_num):
            p = self.scene.soft_surface_point_id[slot_p]
            if int(self.scene.soft_point[p].active) == 1:
                body_p = self._soft_support_body(p)
                x_p = self.scene.soft_point[p].x + self._soft_point_disp(p)
                dx_p = self._soft_point_direction(p)
                for pair_id in range(self.soft_pair_start[body_p], self.soft_pair_end[body_p]):
                    body_q = self.soft_pair[pair_id][1]
                    for slot_q in range(
                        self.scene.soft[body_q].surfacePointStart,
                        self.scene.soft[body_q].surfacePointEnd,
                    ):
                        q = self.scene.soft_surface_point_id[slot_q]
                        if int(self.scene.soft_point[q].active) == 1:
                            x_q = self.scene.soft_point[q].x + self._soft_point_disp(q)
                            dx_q = self._soft_point_direction(q)
                            p_end, q_end = x_p + dx_p, x_q + dx_q
                            if aabb_overlap_with_clearance(
                                ti.min(x_p, p_end),
                                ti.max(x_p, p_end),
                                ti.min(x_q, q_end),
                                ti.max(x_q, q_end),
                                thickness,
                            ):
                                alpha = 1.0
                                if ti.static(accd):
                                    alpha = point_point_accd(x_p, x_q, dx_p, dx_q, eta, thickness, max_iteration)
                                else:
                                    alpha = point_point_ccd(x_p, x_q, dx_p, dx_q, eta, max_iteration)
                                ti.atomic_min(
                                    self.ccd_alpha[None],
                                    ti.max(0.0, ti.min(1.0, alpha)),
                                )

    @ti.func
    def _point_triangle_tangent(self, p, t0, t1, t2, dtype):
        _, barycentric, normal = point_triangle_contact_frame(p, t0, t1, t2, dtype)
        return barycentric[1], barycentric[2], normal

    @ti.func
    def _closest_point_triangle(self, p, a, b, c):
        return closest_point_triangle(p, a, b, c)

    @ti.func
    def _point_triangle_distance_grad_compact(self, p, a, b, c):
        closest, bary = self._closest_point_triangle(p, a, b, c)
        delta = p - closest
        dist2 = delta.dot(delta)
        grad_d = ti.Vector.zero(float, 12)
        for d in ti.static(range(3)):
            grad_d[d] = 2.0 * delta[d]
            grad_d[3 + d] = -2.0 * bary[0] * delta[d]
            grad_d[6 + d] = -2.0 * bary[1] * delta[d]
            grad_d[9 + d] = -2.0 * bary[2] * delta[d]
        normal = delta.normalized(1.0e-12)
        if delta.norm() <= 1.0e-12:
            normal = (b - a).cross(c - a).normalized(1.0e-12)
        return dist2, grad_d, bary, closest, normal

    @ti.func
    def _scatter_contact_gradient(self, p, face, local_grad):
        for n in range(self._soft_support_count(p)):
            node = self._soft_support_node(p, n)
            block = self._soft_block(node)
            if block >= 0:
                w = self._soft_support_shape(p, n)
                for d in ti.static(range(3)):
                    self._add_grad(self._soft_row(node, d), w * local_grad[d])
        ab = self.affine.face2body[face]
        tri = self.affine.faces[face]
        for lv in ti.static(range(3)):
            vertex = tri[lv]
            for a in range(4):
                w = self.affine.basis[vertex, a]
                control = ab * 4 + a
                for d in ti.static(range(3)):
                    self._add_grad(control * 3 + d, w * local_grad[(lv + 1) * 3 + d])

    @ti.func
    def _local_row_weight(self, p, face, local_vertex, basis_id):
        row_base = -1
        weight = 0.0
        if local_vertex == 0:
            if basis_id < self._soft_support_count(p):
                node = self._soft_support_node(p, basis_id)
                block = self._soft_block(node)
                if block >= 0:
                    row_base = self.affine_dof + 3 * block
                    weight = self._soft_support_shape(p, basis_id)
        else:
            tri = self.affine.faces[face]
            ab = self.affine.face2body[face]
            vertex = tri[local_vertex - 1]
            if basis_id < 4:
                row_base = (ab * 4 + basis_id) * 3
                weight = self.affine.basis[vertex, basis_id]
        return row_base, weight

    @ti.func
    def _scatter_contact_hessian(self, p, face, local_hess):
        for li in range(4):
            for lj in range(4):
                block = ti.Matrix.zero(float, 3, 3)
                for di, dj in ti.static(ti.ndrange(3, 3)):
                    block[di, dj] = local_hess[li * 3 + di, lj * 3 + dj]
                self._scatter_contact_hessian_block(p, face, li, lj, block)

    @ti.func
    def _scatter_contact_hessian_block(self, p, face, li, lj, block):
        max_bi = self._soft_support_count(p)
        if li > 0:
            max_bi = 4
        for bi in range(max_bi):
            row_base, wi = self._local_row_weight(p, face, li, bi)
            if row_base >= 0 and wi != 0.0:
                max_bj = self._soft_support_count(p)
                if lj > 0:
                    max_bj = 4
                for bj in range(max_bj):
                    col_base, wj = self._local_row_weight(p, face, lj, bj)
                    if col_base >= 0 and wj != 0.0:
                        self._add_matrix_block(row_base // 3, col_base // 3, wi * wj * block)

    @ti.func
    def _scatter_contact_outer_hessian(self, p, face, local_vec, coeff):
        for li in range(4):
            max_bi = self._soft_support_count(p)
            if li > 0:
                max_bi = 4
            for bi in range(max_bi):
                row_base, wi = self._local_row_weight(p, face, li, bi)
                if row_base >= 0 and wi != 0.0:
                    for lj in range(4):
                        max_bj = self._soft_support_count(p)
                        if lj > 0:
                            max_bj = 4
                        for bj in range(max_bj):
                            col_base, wj = self._local_row_weight(p, face, lj, bj)
                            if col_base >= 0 and wj != 0.0:
                                weight = coeff * wi * wj
                                vector_i = ti.Vector([local_vec[li * 3 + d] for d in range(3)])
                                vector_j = ti.Vector([local_vec[lj * 3 + d] for d in range(3)])
                                self._add_matrix_block(
                                    row_base // 3,
                                    col_base // 3,
                                    weight * vector_i.outer_product(vector_j),
                                )

    @ti.func
    def _scatter_contact_weighted_hessian(self, p, face, weights, hess):
        for li in range(4):
            max_bi = self._soft_support_count(p)
            if li > 0:
                max_bi = 4
            for bi in range(max_bi):
                row_base, wi = self._local_row_weight(p, face, li, bi)
                if row_base >= 0 and wi != 0.0 and weights[li] != 0.0:
                    for lj in range(4):
                        max_bj = self._soft_support_count(p)
                        if lj > 0:
                            max_bj = 4
                        for bj in range(max_bj):
                            col_base, wj = self._local_row_weight(p, face, lj, bj)
                            if col_base >= 0 and wj != 0.0 and weights[lj] != 0.0:
                                coeff = wi * wj * weights[li] * weights[lj]
                                self._add_matrix_block(
                                    row_base // 3,
                                    col_base // 3,
                                    coeff * hess,
                                )

    @ti.func
    def _friction_f0(self, vbarnorm, epsv, hat_h):
        return ipc_friction_f0(vbarnorm, epsv, hat_h)

    @ti.func
    def _friction_f1_div_vbarnorm(self, vbarnorm, epsv):
        return ipc_friction_f1_over_speed(vbarnorm, epsv)

    @ti.func
    def _friction_hess_term(self, vbarnorm, epsv):
        return ipc_friction_hessian_term(vbarnorm, epsv)

    @ti.func
    def _resolved_fully_implicit_friction(self, pair_mu):
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
        site_count: ti.template(),
    ):
        """Exact paper friction residual without constructing a 12x12 matrix."""
        local_residual = ti.Vector.zero(float, 12)
        distance = ti.sqrt(ti.max(distance2, 1.0e-30))
        normal_force = area * ti.max(-2.0 * barrier_gradient * distance, 0.0)
        if normal_force > 0.0 and (
            pair_mu > 0.0
            or ti.static(self.fully_mu_dynamic > 0.0)
            or ti.static(self.fully_mu_static > 0.0)
            or ti.static(self.fully_mu_viscous > 0.0)
        ):
            normal = ti.Vector.zero(float, 3)
            for component in ti.static(range(3)):
                normal[component] = distance_gradient[component] / (2.0 * distance)

            weights = ti.Vector.zero(float, 4)
            for weight_site in range(site_count):
                numerator = 0.0
                for component in ti.static(range(3)):
                    numerator += distance_gradient[3 * weight_site + component] * normal[component]
                weights[weight_site] = numerator / (2.0 * distance)

            inverse_dt = 1.0 / self.dt_device[None]
            relative_velocity = ti.Vector.zero(float, 3)
            velocity_site = 0
            while velocity_site < site_count:
                for component in ti.static(range(3)):
                    velocity = (
                        positions[3 * velocity_site + component] - hat_positions[3 * velocity_site + component]
                    ) * inverse_dt
                    relative_velocity[component] += weights[velocity_site] * velocity
                velocity_site += 1

            tangent = ti.Matrix.identity(float, 3) - normal.outer_product(normal)
            tangential_velocity = tangent @ relative_velocity
            speed = tangential_velocity.norm()
            profile = ipc_fully_implicit_profile_over_speed(speed, self.affine.epsv, self.fully_profile_id)
            mu_dynamic, mu_static = self._resolved_fully_implicit_friction(pair_mu)
            mu_difference = mu_static - mu_dynamic
            falloff = 0.0
            if mu_difference != 0.0:
                falloff = ipc_stribeck_falloff(speed, self.fully_stribeck_velocity)
            effective_mu = mu_dynamic + mu_difference * falloff
            radial_factor = normal_force * effective_mu * profile + self.fully_mu_viscous
            resistance = self.scale_device[None] * radial_factor * tangential_velocity
            for residual_site in range(site_count):
                for component in ti.static(range(3)):
                    local_residual[3 * residual_site + component] = weights[residual_site] * resistance[component]
        return local_residual

    @ti.func
    def _fully_implicit_local_friction(
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
        site_count: ti.template(),
        need_matrix: ti.template(),
    ):
        """Exact local residual/Jacobian of the paper's friction map.

        ``distance_gradient`` and ``distance_hessian`` are the derivatives of
        squared closest-feature distance with respect to the local point
        coordinates.  They determine, by exact chain rule, the current normal,
        closest-point/contact-map weights and all their configuration
        derivatives.  Consequently this includes the paper's geometry terms
        A--D, velocity term E and normal-multiplier term F_lambda without a
        finite-difference approximation.
        """
        local_residual = ti.Vector.zero(float, 12)
        local_jacobian = ti.Matrix.zero(float, 12, 12)
        distance = ti.sqrt(ti.max(distance2, 1.0e-30))
        # Keep the contact measure inside the Coulomb multiplier.  The
        # constitutive scalar law is eta(v, lambda) =
        # v * (lambda * mu(v) * p(v) + mu_v), so the viscous coefficient is a
        # contact coefficient and must not acquire an additional area factor.
        normal_force = area * ti.max(-2.0 * barrier_gradient * distance, 0.0)
        if normal_force > 0.0 and (
            pair_mu > 0.0
            or ti.static(self.fully_mu_dynamic > 0.0)
            or ti.static(self.fully_mu_static > 0.0)
            or ti.static(self.fully_mu_viscous > 0.0)
        ):
            normal = ti.Vector.zero(float, 3)
            for component in ti.static(range(3)):
                normal[component] = distance_gradient[component] / (2.0 * distance)

            weights = ti.Vector.zero(float, 4)
            for weight_site in range(site_count):
                numerator = 0.0
                for component in ti.static(range(3)):
                    numerator += distance_gradient[3 * weight_site + component] * normal[component]
                weights[weight_site] = numerator / (2.0 * distance)

            inverse_dt = 1.0 / self.dt_device[None]
            site_velocity = ti.Matrix.zero(float, 4, 3)
            relative_velocity = ti.Vector.zero(float, 3)
            velocity_site = 0
            while velocity_site < site_count:
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
            profile = ipc_fully_implicit_profile_over_speed(speed, self.affine.epsv, self.fully_profile_id)
            profile_derivative = ipc_fully_implicit_profile_over_speed_derivative(
                speed, self.affine.epsv, self.fully_profile_id
            )
            mu_dynamic, mu_static = self._resolved_fully_implicit_friction(pair_mu)
            mu_difference = mu_static - mu_dynamic
            falloff = 0.0
            falloff_derivative = 0.0
            if mu_difference != 0.0:
                falloff = ipc_stribeck_falloff(speed, self.fully_stribeck_velocity)
                falloff_derivative = ipc_stribeck_falloff_derivative(speed, self.fully_stribeck_velocity)
            effective_mu = mu_dynamic + mu_difference * falloff
            radial_factor = normal_force * effective_mu * profile + self.fully_mu_viscous
            scale = self.scale_device[None]
            resistance = scale * radial_factor * tangential_velocity
            for residual_site in range(site_count):
                for component in ti.static(range(3)):
                    local_residual[3 * residual_site + component] = weights[residual_site] * resistance[component]

            if ti.static(need_matrix):
                distance_derivative = ti.Vector.zero(float, 12)
                normal_derivative = ti.Matrix.zero(float, 3, 12)
                weight_derivative = ti.Matrix.zero(float, 4, 12)
                for column in range(3 * site_count):
                    distance_derivative[column] = distance_gradient[column] / (2.0 * distance)
                    for component in ti.static(range(3)):
                        normal_derivative[component, column] = (
                            distance_hessian[component, column] / (2.0 * distance)
                            - normal[component] * distance_derivative[column] / distance
                        )
                for derivative_site in range(site_count):
                    for column in range(3 * site_count):
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
                for column in range(3 * site_count):
                    column_site = column // 3
                    column_component = column % 3
                    for component in ti.static(range(3)):
                        velocity_weight_site = 0
                        while velocity_weight_site < site_count:
                            relative_velocity_derivative[component, column] += (
                                weight_derivative[velocity_weight_site, column]
                                * site_velocity[velocity_weight_site, component]
                            )
                            velocity_weight_site += 1
                        if component == column_component:
                            relative_velocity_derivative[component, column] += weights[column_site] * inverse_dt

                tangential_derivative = ti.Matrix.zero(float, 3, 12)
                speed_derivative = ti.Vector.zero(float, 12)
                for column in range(3 * site_count):
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

                for column in range(3 * site_count):
                    normal_force_derivative = (
                        -2.0
                        * area
                        * (
                            barrier_hessian * distance_gradient[column] * distance
                            + barrier_gradient * distance_derivative[column]
                        )
                    )
                    mu_derivative = mu_difference * falloff_derivative * speed_derivative[column]
                    radial_derivative = normal_force_derivative * effective_mu * profile + normal_force * (
                        mu_derivative * profile + effective_mu * profile_derivative * speed_derivative[column]
                    )
                    resistance_derivative = ti.Vector.zero(float, 3)
                    for component in ti.static(range(3)):
                        resistance_derivative[component] = scale * (
                            radial_factor * tangential_derivative[component, column]
                            + radial_derivative * tangential_velocity[component]
                        )
                    for jacobian_site in range(site_count):
                        for component in ti.static(range(3)):
                            local_jacobian[3 * jacobian_site + component, column] = (
                                weight_derivative[jacobian_site, column] * resistance[component]
                                + weights[jacobian_site] * resistance_derivative[component]
                            )
        return local_residual, local_jacobian

    @ti.func
    def _assemble_mixed_friction(self, p, face, bary, rel, hat_rel, normal, coeff, need_matrix: ti.template()):
        coeff *= self.affine.friction_scale[0]
        if coeff > 0.0:
            P = ti.Matrix.identity(float, 3) - normal.outer_product(normal)
            vbar = P @ ((rel - hat_rel) / ti.max(self.dt_device[None], 1.0e-30))
            vbarnorm = vbar.norm()
            ti.atomic_add(
                self.energy[None],
                coeff * self._friction_f0(vbarnorm, self.affine.epsv, self.dt_device[None]),
            )
            f1 = self._friction_f1_div_vbarnorm(vbarnorm, self.affine.epsv)
            grad_f = coeff * f1 * (P @ vbar)
            local_grad = ti.Vector.zero(float, 12)
            for d in ti.static(range(3)):
                local_grad[d] = grad_f[d]
                local_grad[3 + d] = -bary[0] * grad_f[d]
                local_grad[6 + d] = -bary[1] * grad_f[d]
                local_grad[9 + d] = -bary[2] * grad_f[d]
            self._scatter_contact_gradient(p, face, local_grad)
            if ti.static(need_matrix):
                f_hess = self._friction_hess_term(vbarnorm, self.affine.epsv)
                inner = coeff * f1 * ti.Matrix.identity(float, 3)
                if vbarnorm > 1.0e-30:
                    inner += coeff * f_hess / vbarnorm * vbar.outer_product(vbar)
                if ti.static(not self.fully_implicit):
                    inner = psd_project_nd(inner)
                hess = P @ inner @ P.transpose() / ti.max(self.dt_device[None], 1.0e-30)
                weights = ti.Vector([1.0, -bary[0], -bary[1], -bary[2]])
                self._scatter_contact_weighted_hessian(p, face, weights, hess)

    @ti.kernel
    def _count_mixed_lagged_friction_contacts(self):
        for index in self.mixed_contact_prefix:
            self.mixed_contact_prefix[index] = 0
        for candidate in range(self.mixed_bvh.point_triangle_count[None]):
            p = self.mixed_bvh.point_triangle[candidate][0]
            face = self.mixed_bvh.point_triangle_primitive[candidate][1]
            if int(self.scene.soft_point[p].active) == 1:
                soft_mat = int(self.scene.soft_point[p].materialID)
                point = self.scene.soft_point[p].x + self._soft_point_disp(p)
                ab = self.affine.face2body[face]
                dhat = self._pp_dhat(soft_mat, ab)
                tri = self.affine.faces[face]
                closest, _ = self._closest_point_triangle(
                    point,
                    self.affine.x[tri[0]],
                    self.affine.x[tri[1]],
                    self.affine.x[tri[2]],
                )
                delta = point - closest
                if self._pp_mu(soft_mat, ab) > 0.0 and delta.dot(delta) < dhat * dhat:
                    self.mixed_contact_prefix[candidate] = 1

    @ti.kernel
    def _finalize_mixed_lagged_friction_count(self):
        required = 0
        candidate_count = self.mixed_bvh.point_triangle_count[None]
        if candidate_count > 0:
            required = self.mixed_contact_prefix[candidate_count - 1]
        self.mixed_friction_count[None] = required
        self.mixed_friction_overflow[None] = ti.cast(required > self.mixed_friction_capacity, ti.i32)

    @ti.kernel
    def _write_mixed_lagged_friction_contacts(self):
        for candidate in range(self.mixed_bvh.point_triangle_count[None]):
            active = self.mixed_contact_prefix[candidate]
            if candidate > 0:
                active -= self.mixed_contact_prefix[candidate - 1]
            if active != 0:
                index = self.mixed_contact_prefix[candidate] - 1
                p = self.mixed_bvh.point_triangle[candidate][0]
                face = self.mixed_bvh.point_triangle_primitive[candidate][1]
                if index < self.mixed_friction_capacity:
                    ab = self.affine.face2body[face]
                    soft_mat = int(self.scene.soft_point[p].materialID)
                    point = self.scene.soft_point[p].x + self._soft_point_disp(p)
                    area = self._mixed_fv_measure(p)
                    dhat = self._pp_dhat(soft_mat, ab)
                    mu = self._pp_mu(soft_mat, ab)
                    tri = self.affine.faces[face]
                    a = self.affine.x[tri[0]]
                    b = self.affine.x[tri[1]]
                    c = self.affine.x[tri[2]]
                    closest, _ = self._closest_point_triangle(point, a, b, c)
                    delta = point - closest
                    dist2 = delta.dot(delta)
                    contact_type = point_triangle_distance_type(point, a, b, c)
                    _, bary, normal = point_triangle_contact_frame(point, a, b, c, contact_type)
                    dist = ti.sqrt(ti.max(dist2, 1.0e-30))
                    normal_force = 0.0
                    if ti.static(self.is_semi):
                        _, first, _ = self._semi_terms(
                            ti.Vector([p, face, 1, -1]),
                            dist - dhat,
                            self._pp_penalty(soft_mat, ab),
                        )
                        normal_force = ti.max(-first, 0.0)
                    else:
                        _, db, _ = self._ipc_barrier_distance2(
                            dist2,
                            dhat * dhat,
                            self._pp_kappa(soft_mat, ab),
                        )
                        normal_force = ti.max(-2.0 * db * dist, 0.0)
                    self.mixed_friction_point[index] = p
                    self.mixed_friction_face[index] = face
                    self.mixed_friction_bary[index] = bary
                    self.mixed_friction_normal[index] = normal
                    self.mixed_friction_coeff[index] = mu * normal_force * area * self.scale_device[None]

    def _initialize_mixed_lagged_friction(self):
        if self.mixed_bvh is None:
            return
        self._count_mixed_lagged_friction_contacts()
        self._scan_mixed_contact_counts()
        self._finalize_mixed_lagged_friction_count()
        required = int(self.mixed_friction_count[None])
        if int(self.mixed_friction_overflow[None]) != 0:
            raise RuntimeError(
                "SoftAffineIPC mixed lagged-friction buffer overflow: "
                f"required {required}, capacity "
                f"{self.mixed_friction_capacity}"
            )
        self._write_mixed_lagged_friction_contacts()

    @ti.kernel
    def _assemble_mixed_lagged_friction(self, need_matrix: ti.template()):
        for contact in range(self.mixed_friction_count[None]):
            if contact < self.mixed_friction_capacity:
                p = self.mixed_friction_point[contact]
                face = self.mixed_friction_face[contact]
                bary = self.mixed_friction_bary[contact]
                tri = self.affine.faces[face]
                point = self.scene.soft_point[p].x + self._soft_point_disp(p)
                closest = (
                    bary[0] * self.affine.x[tri[0]] + bary[1] * self.affine.x[tri[1]] + bary[2] * self.affine.x[tri[2]]
                )
                hat_closest = (
                    bary[0] * self.affine.hat_x[tri[0]]
                    + bary[1] * self.affine.hat_x[tri[1]]
                    + bary[2] * self.affine.hat_x[tri[2]]
                )
                self._assemble_mixed_friction(
                    p,
                    face,
                    bary,
                    point - closest,
                    self.soft_hat_x[p] - hat_closest,
                    self.mixed_friction_normal[contact],
                    self.mixed_friction_coeff[contact],
                    need_matrix,
                )

    @ti.kernel
    def _assemble_mixed_fully_implicit_friction(self, need_matrix: ti.template()):
        """Assemble all current PT features in one dynamic Taichi kernel.

        Keeping feature classification dynamic avoids compiling seven copies
        of the large 12x12 geometry/friction chain for vertex, edge and face
        cases.  The conservative barrier remains split by feature type.
        """
        for candidate in range(self.mixed_bvh.point_triangle_count[None]):
            p = self.mixed_bvh.point_triangle[candidate][0]
            face = self.mixed_bvh.point_triangle_primitive[candidate][1]
            if int(self.scene.soft_point[p].active) != 1:
                continue
            ab = self.affine.face2body[face]
            soft_mat = int(self.scene.soft_point[p].materialID)
            point = self.scene.soft_point[p].x + self._soft_point_disp(p)
            area = self._mixed_fv_measure(p)
            dhat = self._pp_dhat(soft_mat, ab)
            kappa = self._pp_kappa(soft_mat, ab)
            tri = self.affine.faces[face]
            a = self.affine.x[tri[0]]
            b = self.affine.x[tri[1]]
            c = self.affine.x[tri[2]]
            if ti.static(need_matrix):
                distance2, distance_gradient, distance_hessian, unused_type = point_triangle_distance_grad_hess(
                    point, a, b, c
                )
                if distance2 > 0.0 and distance2 < dhat * dhat:
                    unused_energy, db, ddb = self._ipc_barrier_distance2(distance2, dhat * dhat, kappa)
                    positions = ti.Vector.zero(float, 12)
                    hats = ti.Vector.zero(float, 12)
                    for component in ti.static(range(3)):
                        positions[component] = point[component]
                        hats[component] = self.soft_hat_x[p][component]
                        positions[3 + component] = a[component]
                        positions[6 + component] = b[component]
                        positions[9 + component] = c[component]
                        hats[3 + component] = self.affine.hat_x[tri[0]][component]
                        hats[6 + component] = self.affine.hat_x[tri[1]][component]
                        hats[9 + component] = self.affine.hat_x[tri[2]][component]
                    local_residual, local_jacobian = self._fully_implicit_local_friction(
                        positions,
                        hats,
                        distance2,
                        distance_gradient,
                        distance_hessian,
                        db,
                        ddb,
                        self._pp_mu(soft_mat, ab),
                        area,
                        4,
                        True,
                    )
                    self._scatter_contact_gradient(p, face, local_residual)
                    self._scatter_contact_hessian(p, face, local_jacobian)
            else:
                distance2, distance_gradient, _ = point_triangle_distance_grad(point, a, b, c)
                if distance2 > 0.0 and distance2 < dhat * dhat:
                    unused_energy, db, unused_ddb = self._ipc_barrier_distance2(distance2, dhat * dhat, kappa)
                    positions = ti.Vector.zero(float, 12)
                    hats = ti.Vector.zero(float, 12)
                    for component in ti.static(range(3)):
                        positions[component] = point[component]
                        hats[component] = self.soft_hat_x[p][component]
                        positions[3 + component] = a[component]
                        positions[6 + component] = b[component]
                        positions[9 + component] = c[component]
                        hats[3 + component] = self.affine.hat_x[tri[0]][component]
                        hats[6 + component] = self.affine.hat_x[tri[1]][component]
                        hats[9 + component] = self.affine.hat_x[tri[2]][component]
                    local_residual = self._fully_implicit_local_friction_residual(
                        positions,
                        hats,
                        distance2,
                        distance_gradient,
                        db,
                        self._pp_mu(soft_mat, ab),
                        area,
                        4,
                    )
                    self._scatter_contact_gradient(p, face, local_residual)

    def _assemble_mixed_contact(self, need_matrix, project_spd=None):
        if project_spd is None:
            project_spd = not self.fully_implicit
        if self.affine.levelset_contact:
            if int(self.mixed_pair_num[None]) <= 0:
                return
            self._update_mixed_levelset_transforms()
            self._assemble_mixed_levelset_contact(bool(need_matrix))
            if not self.fully_implicit:
                self._assemble_mixed_levelset_lagged_friction(bool(need_matrix))
            return
        if self.mixed_bvh is None or int(self.mixed_bvh.point_triangle_count[None]) <= 0:
            return
        self._count_mixed_contact_types()
        if bool(need_matrix) and self.stable_lagged_contacts:
            required = int(self._mixed_active_contact_count())
            if required > self.mixed_contact_capacity:
                raise RuntimeError(
                    "SoftAffineIPC mixed current-contact buffer overflow: "
                    f"required {required}, capacity "
                    f"{self.mixed_contact_capacity}"
                )
        contact_mask = int(self._mixed_contact_dispatch_mask())
        for contact_type in range(7):
            if contact_mask & (1 << contact_type):
                self._assemble_mixed_contact_type(
                    contact_type,
                    bool(need_matrix),
                    bool(project_spd),
                )
        if self.fully_implicit:
            self._assemble_mixed_fully_implicit_friction(bool(need_matrix))
        else:
            self._assemble_mixed_lagged_friction(bool(need_matrix))

    @ti.kernel
    def _update_mixed_levelset_transforms(self):
        for point in range(self.soft_point_num):
            self.mixed_levelset_current_point[point] = self.scene.soft_point[point].x + self._soft_point_disp(point)
        for body in range(self.affine.body_num):
            self.mixed_levelset_inverse_valid[body] = 0
            self.mixed_levelset_friction_inverse_valid[body] = 0
            if self.affine.body_contact_type[body] == 1:
                current_A = self.affine._affine_levelset_matrix(body)
                if ti.abs(current_A.determinant()) > 1.0e-12:
                    current_inverse = current_A.inverse()
                    for row, column in ti.static(ti.ndrange(3, 3)):
                        self.mixed_levelset_inverse[body, row, column] = current_inverse[row, column]
                    self.mixed_levelset_inverse_valid[body] = 1
                if ti.static(not self.fully_implicit):
                    frozen_y0 = self.affine.levelset_friction_y[body * 4]
                    frozen_A = ti.Matrix.cols(
                        [
                            self.affine.levelset_friction_y[body * 4 + 1] - frozen_y0,
                            self.affine.levelset_friction_y[body * 4 + 2] - frozen_y0,
                            self.affine.levelset_friction_y[body * 4 + 3] - frozen_y0,
                        ]
                    )
                    if ti.abs(frozen_A.determinant()) > 1.0e-12:
                        frozen_inverse = frozen_A.inverse()
                        for row, column in ti.static(ti.ndrange(3, 3)):
                            self.mixed_levelset_friction_inverse[body, row, column] = frozen_inverse[row, column]
                        self.mixed_levelset_friction_inverse_valid[body] = 1

    @ti.func
    def _mixed_levelset_support_count(self, point, local_dof):
        count = 1
        if local_dof < 3:
            count = self._soft_support_count(point)
        return count

    @ti.func
    def _mixed_levelset_row_weight(self, point, local_dof, basis_id):
        row = -1
        weight = 0.0
        if local_dof < 3:
            row_base, weight = self._soft_point_row_weight(point, basis_id)
            if row_base >= 0:
                row = row_base + local_dof
        elif basis_id == 0:
            affine_local = local_dof - 3
            control = affine_local // 3
            component = affine_local - 3 * control
            row = control * 3 + component
            weight = 1.0
        return row, weight

    @ti.func
    def _mixed_levelset_jacobian_column(self, inverse_A, component, weight):
        return weight * ti.Vector(
            [
                inverse_A[0, component],
                inverse_A[1, component],
                inverse_A[2, component],
            ]
        )

    @ti.kernel
    def _locate_mixed_levelset_cells(
        self,
        pair_id: ti.template(),
        material: ti.template(),
        active: ti.template(),
    ):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair == pair_id:
                slot = flat_id - current_pair * self.soft_surface_point_num
                soft_body = self.mixed_pair[current_pair][0]
                affine_body = self.mixed_pair[current_pair][1]
                if (
                    slot >= self.scene.soft[soft_body].surfacePointStart
                    and slot < self.scene.soft[soft_body].surfacePointEnd
                ):
                    point_id = self.scene.soft_surface_point_id[slot]
                    if active[point_id] == 1:
                        point = (material[point_id] - ti.Vector([0.25, 0.25, 0.25])) / self.affine.body_scale[
                            affine_body
                        ]
                        linear = self.affine.levelset_template_to_grid[affine_body]
                        grid_point = linear @ point + self.affine.levelset_template_to_grid_offset[affine_body]
                        origin = self.affine.levelset_grid_origin[affine_body]
                        spacing = ti.max(self.affine.levelset_grid_spacing[affine_body], 1.0e-12)
                        shape = self.affine.levelset_grid_shape[affine_body]
                        reduced = (grid_point - origin) / spacing
                        inside = (
                            reduced[0] >= 0.0
                            and reduced[1] >= 0.0
                            and reduced[2] >= 0.0
                            and reduced[0] <= shape[0] - 1
                            and reduced[1] <= shape[1] - 1
                            and reduced[2] <= shape[2] - 1
                        )
                        for d in ti.static(range(3)):
                            base = ti.min(
                                shape[d] - 2,
                                ti.max(0, ti.floor(reduced[d], ti.i32)),
                            )
                            self.mixed_levelset_cell_base[point_id, d] = base
                            self.mixed_levelset_cell_fraction[point_id, d] = ti.min(1.0, ti.max(0.0, reduced[d] - base))
                        active[point_id] = int(inside)

    @ti.kernel
    def _interpolate_mixed_levelset_cells(
        self,
        pair_id: ti.template(),
        active: ti.template(),
        phi_target: ti.template(),
        gradient_target: ti.template(),
    ):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair == pair_id:
                slot = flat_id - current_pair * self.soft_surface_point_num
                soft_body = self.mixed_pair[current_pair][0]
                affine_body = self.mixed_pair[current_pair][1]
                if (
                    slot >= self.scene.soft[soft_body].surfacePointStart
                    and slot < self.scene.soft[soft_body].surfacePointEnd
                ):
                    point_id = self.scene.soft_surface_point_id[slot]
                    if active[point_id] == 1:
                        shape = self.affine.levelset_grid_shape[affine_body]
                        base0 = self.mixed_levelset_cell_base[point_id, 0]
                        base1 = self.mixed_levelset_cell_base[point_id, 1]
                        base2 = self.mixed_levelset_cell_base[point_id, 2]
                        fraction0 = self.mixed_levelset_cell_fraction[point_id, 0]
                        fraction1 = self.mixed_levelset_cell_fraction[point_id, 1]
                        fraction2 = self.mixed_levelset_cell_fraction[point_id, 2]
                        phi = 0.0
                        gradient = ti.Vector.zero(float, 3)
                        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
                            wx = (1.0 - fraction0) if i == 0 else fraction0
                            wy = (1.0 - fraction1) if j == 0 else fraction1
                            wz = (1.0 - fraction2) if k == 0 else fraction2
                            dwx = -1.0 if i == 0 else 1.0
                            dwy = -1.0 if j == 0 else 1.0
                            dwz = -1.0 if k == 0 else 1.0
                            node = (
                                base0
                                + i
                                + (base1 + j) * shape[0]
                                + (base2 + k) * shape[0] * shape[1]
                                + self.affine.levelset_grid_start[affine_body]
                            )
                            nodal = self.affine.levelset_grid_value[node]
                            phi += nodal * wx * wy * wz
                            gradient[0] += nodal * dwx * wy * wz
                            gradient[1] += nodal * wx * dwy * wz
                            gradient[2] += nodal * wx * wy * dwz
                        gradient /= ti.max(self.affine.levelset_grid_spacing[affine_body], 1.0e-12)
                        linear = self.affine.levelset_template_to_grid[affine_body]
                        phi_target[point_id] = phi
                        gradient_target[point_id] = linear.transpose() @ gradient

    @ti.kernel
    def _freeze_mixed_levelset_friction_geometry(self):
        for point in range(self.soft_point_num):
            self.mixed_levelset_frozen_point[point] = self.scene.soft_point[point].x + self._soft_point_disp(point)

    @ti.func
    def _scatter_mixed_levelset_friction_gradient(self, point_id, affine_body, target_weight, local_gradient):
        self._scatter_soft_point_gradient(point_id, local_gradient)
        for control, component in ti.ndrange(4, 3):
            self._add_grad(
                affine_body * 12 + control * 3 + component,
                target_weight[control] * local_gradient[component],
            )

    @ti.func
    def _scatter_mixed_levelset_friction_hessian(
        self,
        point_id,
        affine_body,
        target_weight,
        local_hessian,
    ):
        for site_i, site_j in ti.ndrange(5, 5):
            local_i = 0 if site_i == 0 else 3 * site_i
            local_j = 0 if site_j == 0 else 3 * site_j
            coefficient_i = 1.0 if site_i == 0 else target_weight[site_i - 1]
            coefficient_j = 1.0 if site_j == 0 else target_weight[site_j - 1]
            for basis_i in range(self._mixed_levelset_support_count(point_id, local_i)):
                row, weight_i = self._mixed_levelset_row_weight(point_id, local_i, basis_i)
                if site_i > 0 and row >= 0:
                    row += affine_body * 12
                if row >= 0 and weight_i != 0.0:
                    for basis_j in range(self._mixed_levelset_support_count(point_id, local_j)):
                        column, weight_j = self._mixed_levelset_row_weight(point_id, local_j, basis_j)
                        if site_j > 0 and column >= 0:
                            column += affine_body * 12
                        if column >= 0 and weight_j != 0.0:
                            self._add_matrix_block(
                                row // 3,
                                column // 3,
                                weight_i * weight_j * coefficient_i * coefficient_j * local_hessian,
                            )

    def _assemble_mixed_levelset_lagged_friction(self, need_matrix):
        self.mixed_friction_count[None] = 0
        self.mixed_friction_overflow[None] = 0
        for pair_id in range(int(self.mixed_pair_num[None])):
            self._clear_mixed_levelset_friction_pair(pair_id)
            self._prepare_mixed_levelset_friction_material_pair(pair_id)
            self._locate_mixed_levelset_cells(
                pair_id,
                self.mixed_levelset_friction_material,
                self.mixed_levelset_friction_active,
            )
            self._interpolate_mixed_levelset_cells(
                pair_id,
                self.mixed_levelset_friction_active,
                self.mixed_levelset_friction_phi,
                self.mixed_levelset_friction_phi_gradient,
            )
            self._finalize_mixed_levelset_friction_pair(pair_id)
            required = int(self.mixed_friction_count[None])
            if required > self.mixed_friction_capacity:
                self.mixed_friction_overflow[None] = 1
                raise RuntimeError(
                    "SoftAffineIPC mixed Level Set lagged-friction buffer overflow: "
                    f"required {required}, capacity {self.mixed_friction_capacity}"
                )
            self._assemble_mixed_levelset_lagged_friction_residual(pair_id)
            if bool(need_matrix):
                self._assemble_mixed_levelset_lagged_friction_hessian(pair_id)

    @ti.kernel
    def _clear_mixed_levelset_friction_pair(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair != pair_id:
                continue
            slot = flat_id - current_pair * self.soft_surface_point_num
            soft_body = self.mixed_pair[current_pair][0]
            if (
                slot >= self.scene.soft[soft_body].surfacePointStart
                and slot < self.scene.soft[soft_body].surfacePointEnd
            ):
                self.mixed_levelset_friction_active[self.scene.soft_surface_point_id[slot]] = 0

    @ti.kernel
    def _prepare_mixed_levelset_friction_material_pair(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair != pair_id:
                continue
            slot = flat_id - current_pair * self.soft_surface_point_num
            soft_body = self.mixed_pair[current_pair][0]
            affine_body = self.mixed_pair[current_pair][1]
            if (
                slot < self.scene.soft[soft_body].surfacePointStart
                or slot >= self.scene.soft[soft_body].surfacePointEnd
            ):
                continue
            point_id = self.scene.soft_surface_point_id[slot]
            if (
                self.mixed_levelset_friction_inverse_valid[affine_body] == 1
                and int(self.scene.soft_point[point_id].active) == 1
            ):
                relative = self.mixed_levelset_frozen_point[point_id] - self.affine.levelset_friction_y[affine_body * 4]
                material0 = (
                    self.mixed_levelset_friction_inverse[affine_body, 0, 0] * relative[0]
                    + self.mixed_levelset_friction_inverse[affine_body, 0, 1] * relative[1]
                    + self.mixed_levelset_friction_inverse[affine_body, 0, 2] * relative[2]
                )
                material1 = (
                    self.mixed_levelset_friction_inverse[affine_body, 1, 0] * relative[0]
                    + self.mixed_levelset_friction_inverse[affine_body, 1, 1] * relative[1]
                    + self.mixed_levelset_friction_inverse[affine_body, 1, 2] * relative[2]
                )
                material2 = (
                    self.mixed_levelset_friction_inverse[affine_body, 2, 0] * relative[0]
                    + self.mixed_levelset_friction_inverse[affine_body, 2, 1] * relative[1]
                    + self.mixed_levelset_friction_inverse[affine_body, 2, 2] * relative[2]
                )
                self.mixed_levelset_friction_material[point_id] = ti.Vector([material0, material1, material2])
                self.mixed_levelset_friction_active[point_id] = 1

    @ti.kernel
    def _finalize_mixed_levelset_friction_pair(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair != pair_id:
                continue
            slot = flat_id - current_pair * self.soft_surface_point_num
            soft_body = self.mixed_pair[current_pair][0]
            affine_body = self.mixed_pair[current_pair][1]
            if (
                slot < self.scene.soft[soft_body].surfacePointStart
                or slot >= self.scene.soft[soft_body].surfacePointEnd
            ):
                continue
            point_id = self.scene.soft_surface_point_id[slot]
            if self.mixed_levelset_friction_active[point_id] == 1:
                scale = self.affine.body_scale[affine_body]
                gap = scale * self.mixed_levelset_friction_phi[point_id]
                soft_material = int(self.scene.soft_point[point_id].materialID)
                dhat = self._pp_dhat(soft_material, affine_body)
                mu = self._pp_mu(soft_material, affine_body)
                accepted = 0
                if mu > 0.0 and ((0.0 < gap < dhat) or ti.static(self.is_semi)):
                    barrier_gradient = 0.0
                    if ti.static(self.is_semi):
                        unused_energy, barrier_gradient, unused_second = self._semi_terms(
                            ti.Vector([point_id, affine_body, 2, -1]),
                            gap - dhat,
                            self._pp_penalty(soft_material, affine_body),
                        )
                    else:
                        unused_energy, barrier_gradient, unused_second = self._ipc_barrier_gap(
                            gap,
                            dhat,
                            self._pp_kappa(soft_material, affine_body),
                        )
                    phi_gradient = self.mixed_levelset_friction_phi_gradient[point_id]
                    normal0 = (
                        self.mixed_levelset_friction_inverse[affine_body, 0, 0] * phi_gradient[0]
                        + self.mixed_levelset_friction_inverse[affine_body, 1, 0] * phi_gradient[1]
                        + self.mixed_levelset_friction_inverse[affine_body, 2, 0] * phi_gradient[2]
                    )
                    normal1 = (
                        self.mixed_levelset_friction_inverse[affine_body, 0, 1] * phi_gradient[0]
                        + self.mixed_levelset_friction_inverse[affine_body, 1, 1] * phi_gradient[1]
                        + self.mixed_levelset_friction_inverse[affine_body, 2, 1] * phi_gradient[2]
                    )
                    normal2 = (
                        self.mixed_levelset_friction_inverse[affine_body, 0, 2] * phi_gradient[0]
                        + self.mixed_levelset_friction_inverse[affine_body, 1, 2] * phi_gradient[1]
                        + self.mixed_levelset_friction_inverse[affine_body, 2, 2] * phi_gradient[2]
                    )
                    normal = ti.Vector([normal0, normal1, normal2])
                    normal_norm = normal.norm()
                    coefficient = (
                        self.scale_device[None]
                        * self._soft_surface_measure(point_id)
                        * mu
                        * ti.max(-barrier_gradient, 0.0)
                    )
                    if normal_norm > 1.0e-14 and coefficient > 0.0:
                        self.mixed_levelset_friction_unit_normal[point_id] = normal / normal_norm
                        self.mixed_levelset_friction_coefficient[point_id] = coefficient
                        accepted = 1
                self.mixed_levelset_friction_active[point_id] = accepted
                if accepted == 1:
                    ti.atomic_add(self.mixed_friction_count[None], 1)

    @ti.func
    def _mixed_levelset_friction_kinematics(self, point_id, affine_body):
        material = self.mixed_levelset_friction_material[point_id]
        unit_normal = self.mixed_levelset_friction_unit_normal[point_id]
        projector = ti.Matrix.identity(float, 3) - unit_normal.outer_product(unit_normal)
        target_weight = ti.Vector(
            [
                -(1.0 - material[0] - material[1] - material[2]),
                -material[0],
                -material[1],
                -material[2],
            ]
        )
        relative = self.mixed_levelset_current_point[point_id]
        reference_relative = self.soft_hat_x[point_id]
        for control in ti.static(range(4)):
            relative += target_weight[control] * self.affine.y[affine_body * 4 + control]
            reference_relative += target_weight[control] * self.affine.hat_y[affine_body * 4 + control]
        tangential_velocity = projector @ ((relative - reference_relative) / self.dt_device[None])
        return target_weight, projector, tangential_velocity

    @ti.kernel
    def _assemble_mixed_levelset_lagged_friction_residual(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair != pair_id:
                continue
            slot = flat_id - current_pair * self.soft_surface_point_num
            soft_body = self.mixed_pair[current_pair][0]
            affine_body = self.mixed_pair[current_pair][1]
            if (
                slot < self.scene.soft[soft_body].surfacePointStart
                or slot >= self.scene.soft[soft_body].surfacePointEnd
            ):
                continue
            point_id = self.scene.soft_surface_point_id[slot]
            if self.mixed_levelset_friction_active[point_id] == 1:
                target_weight, projector, tangential_velocity = self._mixed_levelset_friction_kinematics(
                    point_id, affine_body
                )
                speed = tangential_velocity.norm()
                coefficient = self.mixed_levelset_friction_coefficient[point_id]
                ti.atomic_add(
                    self.energy[None],
                    coefficient * self._friction_f0(speed, self.affine.epsv, self.dt_device[None]),
                )
                f1_over_speed = self._friction_f1_div_vbarnorm(speed, self.affine.epsv)
                local_gradient = coefficient * f1_over_speed * (projector @ tangential_velocity)
                self._scatter_mixed_levelset_friction_gradient(
                    point_id,
                    affine_body,
                    target_weight,
                    local_gradient,
                )

    @ti.kernel
    def _assemble_mixed_levelset_lagged_friction_hessian(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair != pair_id:
                continue
            slot = flat_id - current_pair * self.soft_surface_point_num
            soft_body = self.mixed_pair[current_pair][0]
            affine_body = self.mixed_pair[current_pair][1]
            if (
                slot < self.scene.soft[soft_body].surfacePointStart
                or slot >= self.scene.soft[soft_body].surfacePointEnd
            ):
                continue
            point_id = self.scene.soft_surface_point_id[slot]
            if self.mixed_levelset_friction_active[point_id] == 1:
                target_weight, projector, tangential_velocity = self._mixed_levelset_friction_kinematics(
                    point_id, affine_body
                )
                speed = tangential_velocity.norm()
                coefficient = self.mixed_levelset_friction_coefficient[point_id]
                f1_over_speed = self._friction_f1_div_vbarnorm(speed, self.affine.epsv)
                inner = coefficient * f1_over_speed * ti.Matrix.identity(float, 3)
                if speed > 1.0e-30:
                    inner += (
                        coefficient
                        * self._friction_hess_term(speed, self.affine.epsv)
                        / speed
                        * tangential_velocity.outer_product(tangential_velocity)
                    )
                local_hessian = (
                    projector @ psd_project_nd(inner) @ projector.transpose() / ti.max(self.dt_device[None], 1.0e-30)
                )
                self._scatter_mixed_levelset_friction_hessian(
                    point_id,
                    affine_body,
                    target_weight,
                    local_hessian,
                )

    def _assemble_mixed_levelset_contact(self, need_matrix):
        if self.fully_implicit:
            self._assemble_mixed_levelset_contact_fully_implicit(bool(need_matrix))
            return
        self.mixed_levelset_contact_count[None] = 0
        for pair_id in range(int(self.mixed_pair_num[None])):
            self._clear_mixed_levelset_contact_pair(pair_id)
            self._prepare_mixed_levelset_contact_material_pair(pair_id)
            self._locate_mixed_levelset_cells(
                pair_id,
                self.mixed_levelset_contact_material,
                self.mixed_levelset_contact_active,
            )
            self._interpolate_mixed_levelset_cells(
                pair_id,
                self.mixed_levelset_contact_active,
                self.mixed_levelset_contact_phi,
                self.mixed_levelset_contact_phi_gradient,
            )
            self._prepare_mixed_levelset_contact_barrier_pair(pair_id)
            self._evaluate_mixed_levelset_contact_barrier_pair(pair_id)
            self._finalize_mixed_levelset_contact_normal_pair(pair_id)
            required = int(self.mixed_levelset_contact_count[None])
            if bool(need_matrix) and required > self.mixed_contact_capacity:
                raise RuntimeError(
                    "SoftAffineIPC mixed Level Set current-contact buffer overflow: "
                    f"required {required}, capacity {self.mixed_contact_capacity}"
                )
            self._assemble_mixed_levelset_contact_residual(pair_id)
            if bool(need_matrix):
                self._assemble_mixed_levelset_contact_hessian(pair_id)

    @ti.kernel
    def _clear_mixed_levelset_contact_pair(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair != pair_id:
                continue
            slot = flat_id - current_pair * self.soft_surface_point_num
            soft_body = self.mixed_pair[current_pair][0]
            if (
                slot >= self.scene.soft[soft_body].surfacePointStart
                and slot < self.scene.soft[soft_body].surfacePointEnd
            ):
                self.mixed_levelset_contact_active[self.scene.soft_surface_point_id[slot]] = 0

    @ti.kernel
    def _prepare_mixed_levelset_contact_material_pair(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair != pair_id:
                continue
            slot = flat_id - current_pair * self.soft_surface_point_num
            soft_body = self.mixed_pair[current_pair][0]
            affine_body = self.mixed_pair[current_pair][1]
            if (
                slot < self.scene.soft[soft_body].surfacePointStart
                or slot >= self.scene.soft[soft_body].surfacePointEnd
            ):
                continue
            point_id = self.scene.soft_surface_point_id[slot]
            if self.mixed_levelset_inverse_valid[affine_body] == 1 and int(self.scene.soft_point[point_id].active) == 1:
                relative = self.mixed_levelset_current_point[point_id] - self.affine.y[affine_body * 4]
                material0 = (
                    self.mixed_levelset_inverse[affine_body, 0, 0] * relative[0]
                    + self.mixed_levelset_inverse[affine_body, 0, 1] * relative[1]
                    + self.mixed_levelset_inverse[affine_body, 0, 2] * relative[2]
                )
                material1 = (
                    self.mixed_levelset_inverse[affine_body, 1, 0] * relative[0]
                    + self.mixed_levelset_inverse[affine_body, 1, 1] * relative[1]
                    + self.mixed_levelset_inverse[affine_body, 1, 2] * relative[2]
                )
                material2 = (
                    self.mixed_levelset_inverse[affine_body, 2, 0] * relative[0]
                    + self.mixed_levelset_inverse[affine_body, 2, 1] * relative[1]
                    + self.mixed_levelset_inverse[affine_body, 2, 2] * relative[2]
                )
                self.mixed_levelset_contact_material[point_id] = ti.Vector([material0, material1, material2])
                self.mixed_levelset_contact_active[point_id] = 1

    @ti.kernel
    def _prepare_mixed_levelset_contact_barrier_pair(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair == pair_id:
                slot = flat_id - current_pair * self.soft_surface_point_num
                soft_body = self.mixed_pair[current_pair][0]
                affine_body = self.mixed_pair[current_pair][1]
                if (
                    slot >= self.scene.soft[soft_body].surfacePointStart
                    and slot < self.scene.soft[soft_body].surfacePointEnd
                ):
                    point_id = self.scene.soft_surface_point_id[slot]
                    if self.mixed_levelset_contact_active[point_id] == 1:
                        gap = self.affine.body_scale[affine_body] * self.mixed_levelset_contact_phi[point_id]
                        soft_material = int(self.scene.soft_point[point_id].materialID)
                        dhat = self._pp_dhat(soft_material, affine_body)
                        self.mixed_levelset_contact_phi[point_id] = gap
                        self.mixed_levelset_contact_barrier[point_id] = ti.Vector(
                            [dhat, self._pp_kappa(soft_material, affine_body), 0.0]
                        )
                        active = gap < dhat
                        if ti.static(self.is_semi):
                            key = ti.Vector([point_id, affine_body, 2, -1])
                            multiplier_slot = semi_ipc_find(
                                self.semi_state,
                                self.semi_key,
                                key,
                                ti.static(self.semi_capacity),
                            )
                            multiplier = 0.0
                            if multiplier_slot >= 0:
                                multiplier = self.semi_multiplier[multiplier_slot]
                            active = multiplier - self._pp_penalty(soft_material, affine_body) * (gap - dhat) >= 0.0
                        self.mixed_levelset_contact_active[point_id] = int(active)

    @ti.kernel
    def _evaluate_mixed_levelset_contact_barrier_pair(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair == pair_id:
                slot = flat_id - current_pair * self.soft_surface_point_num
                soft_body = self.mixed_pair[current_pair][0]
                if (
                    slot >= self.scene.soft[soft_body].surfacePointStart
                    and slot < self.scene.soft[soft_body].surfacePointEnd
                ):
                    point_id = self.scene.soft_surface_point_id[slot]
                    if self.mixed_levelset_contact_active[point_id] == 1:
                        inputs = self.mixed_levelset_contact_barrier[point_id]
                        energy = 0.0
                        dphi = 0.0
                        ddphi = 0.0
                        if ti.static(self.is_semi):
                            affine_body = self.mixed_pair[current_pair][1]
                            soft_material = int(self.scene.soft_point[point_id].materialID)
                            energy, dphi, ddphi = self._semi_terms(
                                ti.Vector([point_id, affine_body, 2, -1]),
                                self.mixed_levelset_contact_phi[point_id] - inputs[0],
                                self._pp_penalty(soft_material, affine_body),
                            )
                        else:
                            energy, dphi, ddphi = self._ipc_barrier_gap(
                                self.mixed_levelset_contact_phi[point_id],
                                inputs[0],
                                inputs[1],
                            )
                        self.mixed_levelset_contact_barrier[point_id] = ti.Vector([energy, dphi, ddphi])
                        self.mixed_levelset_contact_coefficient[point_id] = self.scale_device[
                            None
                        ] * self._soft_surface_measure(point_id)

    @ti.kernel
    def _finalize_mixed_levelset_contact_normal_pair(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair != pair_id:
                continue
            slot = flat_id - current_pair * self.soft_surface_point_num
            soft_body = self.mixed_pair[current_pair][0]
            affine_body = self.mixed_pair[current_pair][1]
            if (
                slot < self.scene.soft[soft_body].surfacePointStart
                or slot >= self.scene.soft[soft_body].surfacePointEnd
            ):
                continue
            point_id = self.scene.soft_surface_point_id[slot]
            if self.mixed_levelset_contact_active[point_id] == 1:
                phi_gradient = self.mixed_levelset_contact_phi_gradient[point_id]
                normal0 = (
                    self.mixed_levelset_inverse[affine_body, 0, 0] * phi_gradient[0]
                    + self.mixed_levelset_inverse[affine_body, 1, 0] * phi_gradient[1]
                    + self.mixed_levelset_inverse[affine_body, 2, 0] * phi_gradient[2]
                )
                normal1 = (
                    self.mixed_levelset_inverse[affine_body, 0, 1] * phi_gradient[0]
                    + self.mixed_levelset_inverse[affine_body, 1, 1] * phi_gradient[1]
                    + self.mixed_levelset_inverse[affine_body, 2, 1] * phi_gradient[2]
                )
                normal2 = (
                    self.mixed_levelset_inverse[affine_body, 0, 2] * phi_gradient[0]
                    + self.mixed_levelset_inverse[affine_body, 1, 2] * phi_gradient[1]
                    + self.mixed_levelset_inverse[affine_body, 2, 2] * phi_gradient[2]
                )
                self.mixed_levelset_contact_world_normal[point_id] = ti.Vector([normal0, normal1, normal2])
                ti.atomic_add(self.mixed_levelset_contact_count[None], 1)

    @ti.kernel
    def _assemble_mixed_levelset_contact_residual(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair != pair_id:
                continue
            slot = flat_id - current_pair * self.soft_surface_point_num
            soft_body = self.mixed_pair[current_pair][0]
            affine_body = self.mixed_pair[current_pair][1]
            if (
                slot < self.scene.soft[soft_body].surfacePointStart
                or slot >= self.scene.soft[soft_body].surfacePointEnd
            ):
                continue
            point_id = self.scene.soft_surface_point_id[slot]
            if self.mixed_levelset_contact_active[point_id] == 1:
                material = self.mixed_levelset_contact_material[point_id]
                barrier = self.mixed_levelset_contact_barrier[point_id]
                coefficient = self.mixed_levelset_contact_coefficient[point_id]
                world_normal = self.mixed_levelset_contact_world_normal[point_id]
                target_weight = ti.Vector(
                    [
                        -(1.0 - material[0] - material[1] - material[2]),
                        -material[0],
                        -material[1],
                        -material[2],
                    ]
                )
                ti.atomic_add(self.energy[None], coefficient * barrier[0])
                contact_gradient = coefficient * barrier[1] * world_normal
                self._scatter_soft_point_gradient(point_id, contact_gradient)
                for control, component in ti.ndrange(4, 3):
                    self._add_grad(
                        affine_body * 12 + control * 3 + component,
                        target_weight[control] * contact_gradient[component],
                    )

    @ti.kernel
    def _assemble_mixed_levelset_contact_hessian(self, pair_id: ti.template()):
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            current_pair = flat_id // self.soft_surface_point_num
            if current_pair != pair_id:
                continue
            slot = flat_id - current_pair * self.soft_surface_point_num
            soft_body = self.mixed_pair[current_pair][0]
            affine_body = self.mixed_pair[current_pair][1]
            if (
                slot < self.scene.soft[soft_body].surfacePointStart
                or slot >= self.scene.soft[soft_body].surfacePointEnd
            ):
                continue
            point_id = self.scene.soft_surface_point_id[slot]
            if self.mixed_levelset_contact_active[point_id] == 1:
                material = self.mixed_levelset_contact_material[point_id]
                target_weight = ti.Vector(
                    [
                        -(1.0 - material[0] - material[1] - material[2]),
                        -material[0],
                        -material[1],
                        -material[2],
                    ]
                )
                world_normal = self.mixed_levelset_contact_world_normal[point_id]
                local_hessian = (
                    self.mixed_levelset_contact_coefficient[point_id]
                    * self.mixed_levelset_contact_barrier[point_id][2]
                    * world_normal.outer_product(world_normal)
                )
                self._scatter_mixed_levelset_friction_hessian(
                    point_id,
                    affine_body,
                    target_weight,
                    local_hessian,
                )

    @ti.kernel
    def _assemble_mixed_levelset_contact_fully_implicit(self, need_matrix: ti.template()):
        """Soft surface point against an affine-body implicit SDF."""
        offset = ti.Vector([0.25, 0.25, 0.25])
        for flat_id in range(self.mixed_pair_num[None] * self.soft_surface_point_num):
            pair_id = flat_id // self.soft_surface_point_num
            slot = flat_id - pair_id * self.soft_surface_point_num
            sb = self.mixed_pair[pair_id][0]
            ab = self.mixed_pair[pair_id][1]
            if (
                self.mixed_levelset_inverse_valid[ab] == 1
                and slot >= self.scene.soft[sb].surfacePointStart
                and slot < self.scene.soft[sb].surfacePointEnd
            ):
                point_id = self.scene.soft_surface_point_id[slot]
                if int(self.scene.soft_point[point_id].active) == 1:
                    point = self.mixed_levelset_current_point[point_id]
                    relative_material = point - self.affine.y[ab * 4]
                    inv00 = self.mixed_levelset_inverse[ab, 0, 0]
                    inv01 = self.mixed_levelset_inverse[ab, 0, 1]
                    inv02 = self.mixed_levelset_inverse[ab, 0, 2]
                    inv10 = self.mixed_levelset_inverse[ab, 1, 0]
                    inv11 = self.mixed_levelset_inverse[ab, 1, 1]
                    inv12 = self.mixed_levelset_inverse[ab, 1, 2]
                    inv20 = self.mixed_levelset_inverse[ab, 2, 0]
                    inv21 = self.mixed_levelset_inverse[ab, 2, 1]
                    inv22 = self.mixed_levelset_inverse[ab, 2, 2]
                    material0 = (
                        inv00 * relative_material[0] + inv01 * relative_material[1] + inv02 * relative_material[2]
                    )
                    material1 = (
                        inv10 * relative_material[0] + inv11 * relative_material[1] + inv12 * relative_material[2]
                    )
                    material2 = (
                        inv20 * relative_material[0] + inv21 * relative_material[1] + inv22 * relative_material[2]
                    )
                    material = ti.Vector([material0, material1, material2])
                    target_scale = self.affine.body_scale[ab]
                    coordinate = (material - offset) / target_scale
                    phi, phi_gradient, phi_hessian, inside = self.affine._sample_affine_levelset(ab, coordinate)
                    if inside:
                        gap = target_scale * phi
                        soft_material = int(self.scene.soft_point[point_id].materialID)
                        dhat = self._pp_dhat(soft_material, ab)
                        if gap < dhat:
                            kappa = self._pp_kappa(soft_material, ab)
                            energy, dphi, ddphi = self._ipc_barrier_gap(gap, dhat, kappa)
                            coefficient = self.scale_device[None] * self._soft_surface_measure(point_id)
                            ti.atomic_add(self.energy[None], coefficient * energy)
                            target_weight = ti.Vector(
                                [
                                    -(1.0 - material[0] - material[1] - material[2]),
                                    -material[0],
                                    -material[1],
                                    -material[2],
                                ]
                            )
                            normal0 = inv00 * phi_gradient[0] + inv10 * phi_gradient[1] + inv20 * phi_gradient[2]
                            normal1 = inv01 * phi_gradient[0] + inv11 * phi_gradient[1] + inv21 * phi_gradient[2]
                            normal2 = inv02 * phi_gradient[0] + inv12 * phi_gradient[1] + inv22 * phi_gradient[2]
                            world_normal = ti.Vector([normal0, normal1, normal2])
                            contact_gradient = coefficient * dphi * world_normal
                            self._scatter_soft_point_gradient(point_id, contact_gradient)
                            for control, component in ti.ndrange(4, 3):
                                self._add_grad(
                                    ab * 12 + control * 3 + component,
                                    target_weight[control] * contact_gradient[component],
                                )
                            if ti.static(need_matrix):
                                if ti.static(not self.fully_implicit):
                                    local_hessian = coefficient * ddphi * world_normal.outer_product(world_normal)
                                    self._scatter_mixed_levelset_friction_hessian(
                                        point_id,
                                        ab,
                                        target_weight,
                                        local_hessian,
                                    )
                                else:
                                    inverse_A = ti.Matrix(
                                        [
                                            [inv00, inv01, inv02],
                                            [inv10, inv11, inv12],
                                            [inv20, inv21, inv22],
                                        ]
                                    )
                                    hessian_q = phi_hessian / target_scale
                                    for site_i, site_j in ti.ndrange(5, 5):
                                        local_base_i = 0 if site_i == 0 else 3 * site_i
                                        local_base_j = 0 if site_j == 0 else 3 * site_j
                                        block = ti.Matrix.zero(float, 3, 3)
                                        for component_i, component_j in ti.static(ti.ndrange(3, 3)):
                                            weight_i = 1.0 if site_i == 0 else target_weight[site_i - 1]
                                            weight_j = 1.0 if site_j == 0 else target_weight[site_j - 1]
                                            jacobian_i = self._mixed_levelset_jacobian_column(
                                                inverse_A, component_i, weight_i
                                            )
                                            jacobian_j = self._mixed_levelset_jacobian_column(
                                                inverse_A, component_j, weight_j
                                            )
                                            gap_gradient_i = phi_gradient.dot(jacobian_i)
                                            gap_gradient_j = phi_gradient.dot(jacobian_j)
                                            gap_hessian = jacobian_i.dot(hessian_q @ jacobian_j)
                                            if site_i > 0:
                                                gap_hessian -= world_normal[
                                                    component_i
                                                ] * self.affine._affine_levelset_coefficient_dot(site_i - 1, jacobian_j)
                                            if site_j > 0:
                                                gap_hessian -= world_normal[
                                                    component_j
                                                ] * self.affine._affine_levelset_coefficient_dot(site_j - 1, jacobian_i)
                                            block[component_i, component_j] = coefficient * (
                                                ddphi * gap_gradient_i * gap_gradient_j + dphi * gap_hessian
                                            )
                                        for basis_i in range(
                                            self._mixed_levelset_support_count(point_id, local_base_i)
                                        ):
                                            row, weight_i = self._mixed_levelset_row_weight(
                                                point_id, local_base_i, basis_i
                                            )
                                            if site_i > 0 and row >= 0:
                                                row += ab * 12
                                            if row >= 0 and weight_i != 0.0:
                                                for basis_j in range(
                                                    self._mixed_levelset_support_count(point_id, local_base_j)
                                                ):
                                                    column, weight_j = self._mixed_levelset_row_weight(
                                                        point_id, local_base_j, basis_j
                                                    )
                                                    if site_j > 0 and column >= 0:
                                                        column += ab * 12
                                                    if column >= 0 and weight_j != 0.0:
                                                        self._add_matrix_block(
                                                            row // 3,
                                                            column // 3,
                                                            weight_i * weight_j * block,
                                                        )

    @ti.kernel
    def _count_mixed_contact_types(self):
        for i in range(7):
            self.mixed_contact_type_count[i] = 0
        for candidate in range(self.mixed_bvh.point_triangle_count[None]):
            p = self.mixed_bvh.point_triangle[candidate][0]
            face = self.mixed_bvh.point_triangle_primitive[candidate][1]
            if int(self.scene.soft_point[p].active) != 1:
                continue
            ab = self.affine.face2body[face]
            soft_mat = int(self.scene.soft_point[p].materialID)
            point = self.scene.soft_point[p].x + self._soft_point_disp(p)
            dhat = self._pp_dhat(soft_mat, ab)
            tri = self.affine.faces[face]
            a = self.affine.x[tri[0]]
            b = self.affine.x[tri[1]]
            c = self.affine.x[tri[2]]
            closest, bary = self._closest_point_triangle(point, a, b, c)
            dist2 = (point - closest).dot(point - closest)
            if dist2 < dhat * dhat:
                dtype = point_triangle_distance_type(point, a, b, c)
                if dtype >= 0 and dtype < 7:
                    ti.atomic_add(self.mixed_contact_type_count[dtype], 1)

    @ti.kernel
    def _mixed_active_contact_count(self) -> ti.i32:
        count = 0
        for contact_type in range(7):
            count += self.mixed_contact_type_count[contact_type]
        return count

    @ti.kernel
    def _mixed_contact_dispatch_mask(self) -> ti.i32:
        mask = 0
        for contact_type in range(7):
            if self.mixed_contact_type_count[contact_type] > 0:
                mask += 1 << contact_type
        return mask

    @ti.kernel
    def _assemble_mixed_contact_type(
        self,
        contact_type: ti.template(),
        need_matrix: ti.template(),
        project_spd: ti.template(),
    ):
        for candidate in range(self.mixed_bvh.point_triangle_count[None]):
            p = self.mixed_bvh.point_triangle[candidate][0]
            face = self.mixed_bvh.point_triangle_primitive[candidate][1]
            if int(self.scene.soft_point[p].active) != 1:
                continue
            ab = self.affine.face2body[face]
            soft_mat = int(self.scene.soft_point[p].materialID)
            point = self.scene.soft_point[p].x + self._soft_point_disp(p)
            area = self._mixed_fv_measure(p)
            dhat = self._pp_dhat(soft_mat, ab)
            kappa = self._pp_kappa(soft_mat, ab)
            tri = self.affine.faces[face]
            a = self.affine.x[tri[0]]
            b = self.affine.x[tri[1]]
            c = self.affine.x[tri[2]]
            dtype = point_triangle_distance_type(point, a, b, c)
            if dtype != contact_type:
                continue
            active_gap2 = dhat * dhat
            if ti.static(need_matrix):
                dist2, grad_d, hess_d = point_triangle_distance_grad_hess_by_type(point, a, b, c, contact_type)
                if dist2 < active_gap2:
                    base_coeff = self.scale_device[None] * area
                    energy = 0.0
                    local_grad = ti.Vector.zero(float, 12)
                    local_hessian = ti.Matrix.zero(float, 12, 12)
                    if ti.static(self.is_semi):
                        distance = ti.sqrt(ti.max(dist2, 1.0e-30))
                        energy, first, second = self._semi_terms(
                            ti.Vector([p, face, 1, -1]),
                            distance - dhat,
                            self._pp_penalty(soft_mat, ab),
                        )
                        gap_gradient = grad_d / (2.0 * distance)
                        local_grad = base_coeff * first * gap_gradient
                        local_hessian = base_coeff * second * gap_gradient.outer_product(gap_gradient)
                    else:
                        energy, db, ddb = self._ipc_barrier_distance2(dist2, active_gap2, kappa)
                        local_grad = base_coeff * db * grad_d
                        local_hessian = base_coeff * (ddb * grad_d.outer_product(grad_d) + db * hess_d)
                        if ti.static(project_spd):
                            local_hessian = psd_project_nd(local_hessian)
                    ti.atomic_add(self.energy[None], base_coeff * energy)
                    self._scatter_contact_gradient(p, face, local_grad)
                    self._scatter_contact_hessian(p, face, local_hessian)
            else:
                dist2, grad_d = point_triangle_distance_grad_by_type(point, a, b, c, contact_type)
                if dist2 < active_gap2:
                    base_coeff = self.scale_device[None] * area
                    energy = 0.0
                    local_grad = ti.Vector.zero(float, 12)
                    if ti.static(self.is_semi):
                        distance = ti.sqrt(ti.max(dist2, 1.0e-30))
                        energy, first, unused_second = self._semi_terms(
                            ti.Vector([p, face, 1, -1]),
                            distance - dhat,
                            self._pp_penalty(soft_mat, ab),
                        )
                        local_grad = base_coeff * first * grad_d / (2.0 * distance)
                    else:
                        energy, db, unused_second = self._ipc_barrier_distance2(dist2, active_gap2, kappa)
                        local_grad = base_coeff * db * grad_d
                    ti.atomic_add(self.energy[None], base_coeff * energy)
                    self._scatter_contact_gradient(p, face, local_grad)

    @ti.kernel
    def _compute_mixed_levelset_ccd_alpha(self, eta: float, max_iteration: ti.i32):
        for pair_id, slot in ti.ndrange(self.mixed_pair_num[None], self.soft_surface_point_num):
            sb = self.mixed_pair[pair_id][0]
            ab = self.mixed_pair[pair_id][1]
            if (
                self.affine.body_contact_type[ab] == 1
                and slot >= self.scene.soft[sb].surfacePointStart
                and slot < self.scene.soft[sb].surfacePointEnd
            ):
                point_id = self.scene.soft_surface_point_id[slot]
                if int(self.scene.soft_point[point_id].active) == 1:
                    point = self.scene.soft_point[point_id].x + self._soft_point_disp(point_id)
                    direction = self._soft_point_direction(point_id)
                    gap0, inside0 = self.affine._affine_levelset_world_gap_at_step(point, direction, ab, 0.0)
                    soft_material = int(self.scene.soft_point[point_id].materialID)
                    dhat = self._pp_dhat(soft_material, ab)
                    if inside0 and gap0 <= 0.0:
                        if ti.static(not self.is_semi):
                            ti.atomic_min(self.ccd_alpha[None], 0.0)
                    else:
                        reference_gap = gap0 if inside0 else dhat
                        safe_gap = ti.max(
                            1.0e-12,
                            (1.0 - 0.5 * eta) * reference_gap,
                        )
                        path_bound = direction.norm()
                        for control in range(4):
                            path_bound += 0.25 * self.affine.direction_y[ab * 4 + control].norm()
                        cell_size = ti.max(
                            1.0e-12,
                            self.affine.body_scale[ab] * self.affine.levelset_grid_spacing[ab],
                        )
                        segment_count = ti.min(
                            64,
                            ti.max(
                                8,
                                int(ti.ceil(4.0 * path_bound / cell_size)),
                            ),
                        )
                        lower = 0.0
                        upper = 1.0
                        bracketed = False
                        for segment in range(1, segment_count + 1):
                            sample_alpha = float(segment) / float(segment_count)
                            (
                                sample_gap,
                                sample_inside,
                            ) = self.affine._affine_levelset_world_gap_at_step(
                                point,
                                direction,
                                ab,
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
                                    (
                                        middle_gap,
                                        middle_inside,
                                    ) = self.affine._affine_levelset_world_gap_at_step(
                                        point,
                                        direction,
                                        ab,
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
                                        (1.0 - 0.5 * eta) * lower,
                                    ),
                                ),
                            )

    @ti.kernel
    def _compute_mixed_ccd_alpha(self, eta: float, thickness: float, max_iteration: ti.i32, accd: ti.template()):
        for candidate in range(self.mixed_bvh.point_triangle_count[None]):
            p = self.mixed_bvh.point_triangle[candidate][0]
            face = self.mixed_bvh.point_triangle_primitive[candidate][1]
            if int(self.scene.soft_point[p].active) != 1:
                continue
            point = self.scene.soft_point[p].x + self._soft_point_disp(p)
            dpoint = self._soft_point_direction(p)
            tri = self.affine.faces[face]
            alpha = 1.0
            if ti.static(accd):
                alpha = point_triangle_accd(
                    point,
                    self.affine.x[tri[0]],
                    self.affine.x[tri[1]],
                    self.affine.x[tri[2]],
                    dpoint,
                    self.affine.dx[tri[0]],
                    self.affine.dx[tri[1]],
                    self.affine.dx[tri[2]],
                    eta,
                    thickness,
                    max_iteration,
                )
            else:
                alpha = point_triangle_ccd(
                    point,
                    self.affine.x[tri[0]],
                    self.affine.x[tri[1]],
                    self.affine.x[tri[2]],
                    dpoint,
                    self.affine.dx[tri[0]],
                    self.affine.dx[tri[1]],
                    self.affine.dx[tri[2]],
                    eta,
                    max_iteration,
                )
            ti.atomic_min(self.ccd_alpha[None], ti.max(0.0, ti.min(1.0, alpha)))

    @ti.kernel
    def _add_hessian_shift(self, shift: float):
        for block in range(self.total_dof // 3):
            self._add_matrix_block(block, block, shift * ti.Matrix.identity(float, 3))

    @ti.kernel
    def _accept_soft_step(self):
        dt = ti.max(self.dt_device[None], 1.0e-30)
        for node in range(self.soft_grid_num):
            block = self._soft_block(node)
            if block >= 0:
                disp = self._soft_node_disp(node)
                self.scene.soft_grid[node].v = disp / dt
                self.scene.soft_grid[node].f = ZEROVEC3f

        for p in range(self.soft_point_num):
            if int(self.scene.soft_point[p].active) == 1:
                disp = self._soft_point_disp(p)
                vel = ti.Vector.zero(float, 3)
                gradu = mat3x3([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
                for n in range(self._soft_support_count(p)):
                    node = self._soft_support_node(p, n)
                    v = self._soft_node_disp(node) / dt
                    vel += self._soft_support_shape(p, n) * v
                    gradu += self._soft_node_disp(node).outer_product(self._soft_support_gradient(p, n))
                self.scene.soft_point[p].x += disp
                self.scene.soft_point[p].v = vel
                material_id = self.scene.soft_point[p].materialID
                trial_F = self.soft_material.matProps.trial_deformation_gradient(
                    p,
                    material_id,
                    self.scene.soft_point[p].F,
                    gradu,
                )
                committed_F, stress = self.soft_material.matProps.commit_soft_particle_state(
                    p,
                    material_id,
                    trial_F,
                    self.soft_material.stateVars,
                )
                self.scene.soft_point[p].F = committed_F
                self.scene.soft_point[p].stress = stress

        for sb in range(self.soft_num):
            bodyID = self.scene.soft[sb].bodyID
            self.scene.soft[sb].previous_center = self.scene.soft[sb].mass_center
            self.scene.soft[sb].mass_center = ZEROVEC3f
            self.scene.soft[sb].v = ZEROVEC3f
            self.soft_body_mass[sb] = 0.0
            self.scene.box[bodyID].shape_min = vec3f(1.0e30, 1.0e30, 1.0e30)
            self.scene.box[bodyID].shape_max = vec3f(-1.0e30, -1.0e30, -1.0e30)
            self.scene.box[bodyID].shape_radius = 0.0

        for p in range(self.soft_point_num):
            if int(self.scene.soft_point[p].active) == 1:
                sb = self._soft_support_body(p)
                mass = self.scene.soft_point[p].m
                ti.atomic_add(
                    self.scene.soft[sb].mass_center,
                    mass * self.scene.soft_point[p].x,
                )
                ti.atomic_add(
                    self.scene.soft[sb].v,
                    mass * self.scene.soft_point[p].v,
                )
                ti.atomic_add(self.soft_body_mass[sb], mass)

        for sb in range(self.soft_num):
            bodyID = self.scene.soft[sb].bodyID
            if self.soft_body_mass[sb] > Threshold:
                self.scene.soft[sb].mass_center /= self.soft_body_mass[sb]
                self.scene.soft[sb].v /= self.soft_body_mass[sb]
            self.scene.rigid[bodyID].mass_center = self.scene.soft[sb].mass_center
            self.scene.rigid[bodyID].v = self.scene.soft[sb].v

        for p in range(self.soft_point_num):
            if int(self.scene.soft_point[p].active) == 1:
                bodyID = self.scene.soft_point[p].bodyID
                sb = self.scene.rigid[bodyID].softID
                inv_rotate = SetToRotate(self.scene.rigid[bodyID].q).transpose()
                local_x = inv_rotate @ (self.scene.soft_point[p].x - self.scene.soft[sb].mass_center)
                point_padding = 0.5 * ti.pow(self.scene.soft_point[p].vol0, 1.0 / 3.0)
                for d in ti.static(range(3)):
                    ti.atomic_min(
                        self.scene.box[bodyID].shape_min[d],
                        local_x[d] - point_padding,
                    )
                    ti.atomic_max(
                        self.scene.box[bodyID].shape_max[d],
                        local_x[d] + point_padding,
                    )
                ti.atomic_max(
                    self.scene.box[bodyID].shape_radius,
                    local_x.norm() + ti.sqrt(3.0) * point_padding,
                )

        for ns in range(self.surface_num):
            bodyID = self.scene.surface[ns]
            if int(self.scene.rigid[bodyID].is_soft) == 1:
                sb = self.scene.rigid[bodyID].softID
                support = self.scene.soft[sb].templateSurfaceStart + ns - self.scene.soft[sb].startNode
                local_node = self.scene.soft[sb].global_node_to_local(ns)
                rotate_matrix = SetToRotate(self.scene.rigid[bodyID].q)
                scale = self.scene.box[bodyID].scale
                old_global = self.scene.soft[sb].previous_center + rotate_matrix @ (
                    scale * self.scene.vertice[local_node].x
                )
                node_disp = ZEROVEC3f
                for n in range(self.scene.surface_shape_count[support]):
                    node = self.scene.soft[sb].mpmGridStart + self.scene.surface_shape_node[support, n]
                    node_disp += self.scene.surface_shape[support, n] * self._soft_node_disp(node)
                new_global = old_global + node_disp
                new_local = rotate_matrix.transpose() @ (new_global - self.scene.soft[sb].mass_center) / scale
                self.scene.vertice[local_node]._update_kinematics(new_local, node_disp / dt)
                new_local_scaled = scale * new_local
                for d in ti.static(range(3)):
                    ti.atomic_min(self.scene.box[bodyID].shape_min[d], new_local_scaled[d])
                    ti.atomic_max(self.scene.box[bodyID].shape_max[d], new_local_scaled[d])
                ti.atomic_max(
                    self.scene.box[bodyID].shape_radius,
                    new_local_scaled.norm(),
                )

        for sb in range(self.soft_num):
            bodyID = self.scene.soft[sb].bodyID
            self.scene.particle[bodyID]._follow_deformed_shape(
                self.scene.soft[sb].mass_center,
                SetToRotate(self.scene.rigid[bodyID].q),
                self.scene.box[bodyID].shape_min,
                self.scene.box[bodyID].shape_max,
                self.scene.box[bodyID].shape_radius,
                self.scene.box[bodyID].grid_space,
            )


__all__ = [
    "SoftAffineIPCOperator",
    "_validate_soft_affine_lagged_friction_configuration",
    "soft_affine_friction_capabilities",
    "write_soft_affine_surface_vtu",
]
