"""Level-set contact primitives for affine bodies.

The production affine-body solver stores one body by four three-dimensional
control points.  A material point with barycentric coordinates ``b`` is

``x(X, y) = sum_a b_a(X) y_a``.

This module supplies the host form of the level-set contact formulation used
by the Taichi assemblers. It intentionally keeps
the geometry calculus independent of the nonlinear solver so that finite
difference unit tests, initial-state optimization, and coupled MPM/affine
contact all share one oracle.

The signed-distance convention is the one used by ``src.sdf``: negative
inside and positive outside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from src.physics_model.contact_model.ipc.IPC import (
    ipc_barrier_distance_terms_py,
)

_AFFINE_OFFSET = np.full(3, 0.25, dtype=np.float64)


def _as_float_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(array)


def _as_positive_scalar(value, name):
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


@dataclass(frozen=True)
class TrilinearLevelSet:
    """A scalar level-set field on an x-fast regular Cartesian grid.

    ``template_to_grid`` maps an affine template coordinate to the coordinate
    frame in which the Cartesian SDF grid is axis aligned.  This permits the
    affine template preprocessing step to center/rotate its visualization and
    quadrature mesh without resampling the signed-distance field.
    """

    origin: np.ndarray
    spacing: float
    shape: np.ndarray
    values: np.ndarray
    template_to_grid: np.ndarray = None
    template_to_grid_offset: np.ndarray = None

    def __post_init__(self):
        origin = _as_float_array(self.origin, (3,), "origin")
        spacing = _as_positive_scalar(self.spacing, "spacing")
        shape = np.asarray(self.shape, dtype=np.int32)
        if shape.shape != (3,) or np.any(shape < 2):
            raise ValueError("shape must contain three grid sizes >= 2")
        values = np.asarray(self.values, dtype=np.float64).reshape(-1)
        expected = int(np.prod(shape, dtype=np.int64))
        if values.size != expected:
            raise ValueError(f"values contains {values.size} nodes, expected {expected}")
        if not np.all(np.isfinite(values)):
            raise ValueError("values must be finite")
        linear = (
            np.eye(3, dtype=np.float64)
            if self.template_to_grid is None
            else _as_float_array(self.template_to_grid, (3, 3), "template_to_grid")
        )
        offset = (
            np.zeros(3, dtype=np.float64)
            if self.template_to_grid_offset is None
            else _as_float_array(
                self.template_to_grid_offset,
                (3,),
                "template_to_grid_offset",
            )
        )
        determinant = float(np.linalg.det(linear))
        if abs(determinant) <= 1.0e-12:
            raise ValueError("template_to_grid must be nonsingular")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "shape", np.ascontiguousarray(shape, dtype=np.int32))
        object.__setattr__(self, "values", np.ascontiguousarray(values, dtype=np.float64))
        object.__setattr__(self, "template_to_grid", linear)
        object.__setattr__(self, "template_to_grid_offset", offset)

    @property
    def upper(self):
        return self.origin + self.spacing * (self.shape - 1)

    def grid_coordinates(self, template_point):
        point = _as_float_array(template_point, (3,), "template_point")
        return self.template_to_grid @ point + self.template_to_grid_offset

    def contains(self, template_point, margin=0.0):
        point = self.grid_coordinates(template_point)
        margin = float(margin)
        return bool(np.all(point >= self.origin + margin) and np.all(point <= self.upper - margin))

    def _node(self, i, j, k):
        nx, ny, _ = self.shape
        return self.values[int(i + j * nx + k * nx * ny)]

    def sample(self, template_point, *, clamp=False):
        """Return ``(phi, gradient, Hessian, inside_grid)``.

        Derivatives are with respect to the *template* coordinate.  They are
        exact derivatives of the piecewise-trilinear interpolant away from
        cell boundaries.  Pure second derivatives vanish inside a cell while
        the three mixed derivatives are generally nonzero.
        """

        grid_point = self.grid_coordinates(template_point)
        reduced = (grid_point - self.origin) / self.spacing
        inside = bool(np.all(reduced >= 0.0) and np.all(reduced <= self.shape - 1))
        if not inside and not clamp:
            return (
                np.inf,
                np.zeros(3, dtype=np.float64),
                np.zeros((3, 3), dtype=np.float64),
                False,
            )

        base = np.floor(reduced).astype(np.int64)
        base = np.minimum(self.shape - 2, np.maximum(0, base))
        fraction = np.clip(reduced - base, 0.0, 1.0)

        value = 0.0
        gradient_reduced = np.zeros(3, dtype=np.float64)
        hessian_reduced = np.zeros((3, 3), dtype=np.float64)
        for i in range(2):
            wx = 1.0 - fraction[0] if i == 0 else fraction[0]
            dwx = -1.0 if i == 0 else 1.0
            for j in range(2):
                wy = 1.0 - fraction[1] if j == 0 else fraction[1]
                dwy = -1.0 if j == 0 else 1.0
                for k in range(2):
                    wz = 1.0 - fraction[2] if k == 0 else fraction[2]
                    dwz = -1.0 if k == 0 else 1.0
                    nodal = self._node(base[0] + i, base[1] + j, base[2] + k)
                    value += nodal * wx * wy * wz
                    gradient_reduced += nodal * np.array(
                        [dwx * wy * wz, wx * dwy * wz, wx * wy * dwz],
                        dtype=np.float64,
                    )
                    hessian_reduced[0, 1] += nodal * dwx * dwy * wz
                    hessian_reduced[0, 2] += nodal * dwx * wy * dwz
                    hessian_reduced[1, 2] += nodal * wx * dwy * dwz
        hessian_reduced[1, 0] = hessian_reduced[0, 1]
        hessian_reduced[2, 0] = hessian_reduced[0, 2]
        hessian_reduced[2, 1] = hessian_reduced[1, 2]

        gradient_grid = gradient_reduced / self.spacing
        hessian_grid = hessian_reduced / (self.spacing * self.spacing)
        linear = self.template_to_grid
        gradient_template = linear.T @ gradient_grid
        hessian_template = linear.T @ hessian_grid @ linear
        return value, gradient_template, hessian_template, inside

    @classmethod
    def from_object_grid(
        cls,
        grid,
        *,
        template_to_grid=None,
        template_to_grid_offset=None,
    ):
        return cls(
            origin=np.asarray(grid.start_point, dtype=np.float64),
            spacing=float(grid.grid_space),
            shape=np.asarray(grid.gnum, dtype=np.int32),
            values=np.asarray(grid.distance_field, dtype=np.float64),
            template_to_grid=template_to_grid,
            template_to_grid_offset=template_to_grid_offset,
        )


@dataclass
class LevelSetGap:
    """One oriented source-quadrature/target-level-set gap stencil."""

    gap: float
    gradient: np.ndarray
    hessian: np.ndarray
    source_point: np.ndarray
    target_coordinate: np.ndarray
    inside_grid: bool


def affine_basis(reference_point):
    point = _as_float_array(reference_point, (3,), "reference_point")
    material = point + _AFFINE_OFFSET
    return np.array(
        [
            1.0 - material[0] - material[1] - material[2],
            material[0],
            material[1],
            material[2],
        ],
        dtype=np.float64,
    )


def affine_matrix(controls):
    controls = _as_float_array(controls, (4, 3), "controls")
    return np.column_stack(
        (
            controls[1] - controls[0],
            controls[2] - controls[0],
            controls[3] - controls[0],
        )
    )


def affine_levelset_gap(
    source_controls,
    target_controls,
    source_basis,
    target_scale,
    target_levelset,
):
    """Evaluate a signed level-set gap and its exact 24-DOF derivatives.

    The local DOF order is ``source[4,3], target[4,3]``.  Let ``A`` be the
    target affine matrix and ``r=A^-1(x-y0)``.  The target template coordinate
    is ``X=(r-c)/s`` and the physical implicit gap is

    ``g(y) = s phi(X)``.

    For rigid motion and a signed-distance ``phi`` this is the exact Euclidean
    signed distance.  For a general affine deformation it is the standard
    material level-set pullback; the optional normalized-distance
    approximation is deliberately not used because its derivative introduces
    grid-resolution-dependent third derivatives.
    """

    source_controls = _as_float_array(source_controls, (4, 3), "source_controls")
    target_controls = _as_float_array(target_controls, (4, 3), "target_controls")
    source_basis = _as_float_array(source_basis, (4,), "source_basis")
    target_scale = _as_positive_scalar(target_scale, "target_scale")
    if not isinstance(target_levelset, TrilinearLevelSet):
        raise TypeError("target_levelset must be a TrilinearLevelSet")

    source_point = source_basis @ source_controls
    target_A = affine_matrix(target_controls)
    determinant = float(np.linalg.det(target_A))
    if abs(determinant) <= 1.0e-12:
        raise ValueError("target affine map is singular")
    inverse_A = np.linalg.inv(target_A)
    material = inverse_A @ (source_point - target_controls[0])
    target_coordinate = (material - _AFFINE_OFFSET) / target_scale
    phi, phi_gradient, phi_hessian, inside = target_levelset.sample(target_coordinate)
    if not inside:
        return LevelSetGap(
            gap=np.inf,
            gradient=np.zeros(24, dtype=np.float64),
            hessian=np.zeros((24, 24), dtype=np.float64),
            source_point=source_point,
            target_coordinate=target_coordinate,
            inside_grid=False,
        )

    gap = target_scale * phi
    # q = material - c has physical template units.  Since
    # g=s*phi(q/s), dg/dq=grad(phi) and d2g/dq2=H(phi)/s.
    normal_q = phi_gradient
    hessian_q = phi_hessian / target_scale

    site_weights = np.concatenate(
        (
            source_basis,
            -np.array(
                [
                    1.0 - material.sum(),
                    material[0],
                    material[1],
                    material[2],
                ],
                dtype=np.float64,
            ),
        )
    )
    q_jacobian = np.zeros((3, 24), dtype=np.float64)
    for site in range(8):
        for component in range(3):
            dof = 3 * site + component
            q_jacobian[:, dof] = inverse_A[:, component] * site_weights[site]

    gradient = q_jacobian.T @ normal_q
    hessian = q_jacobian.T @ hessian_q @ q_jacobian

    # The inverse affine map is nonlinear in the target controls.  From
    # A q_dot = x_dot - T_dot, its mixed second variation is
    # q_uv=-A^-1(A_u q_v + A_v q_u).
    control_A_coefficients = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    world_normal = inverse_A.T @ normal_q
    for local_i in range(24):
        site_i = local_i // 3
        component_i = local_i % 3
        target_i = site_i >= 4
        coeff_i = control_A_coefficients[site_i - 4] if target_i else None
        for local_j in range(local_i, 24):
            site_j = local_j // 3
            component_j = local_j % 3
            correction = 0.0
            if target_i:
                correction -= world_normal[component_i] * float(coeff_i @ q_jacobian[:, local_j])
            if site_j >= 4:
                coeff_j = control_A_coefficients[site_j - 4]
                correction -= world_normal[component_j] * float(coeff_j @ q_jacobian[:, local_i])
            hessian[local_i, local_j] += correction
            if local_i != local_j:
                hessian[local_j, local_i] += correction

    return LevelSetGap(
        gap=float(gap),
        gradient=np.ascontiguousarray(gradient),
        hessian=np.ascontiguousarray(0.5 * (hessian + hessian.T)),
        source_point=source_point,
        target_coordinate=target_coordinate,
        inside_grid=True,
    )


def point_affine_levelset_gap(
    source_point,
    target_controls,
    target_scale,
    target_levelset,
):
    """Evaluate the MPM-point/affine-SDF gap and exact 15-DOF derivatives.

    The local DOF order is ``source_point[3], target[4,3]``.  This is the
    reference operator used to verify the soft-particle/affine-body coupling.
    The production MPM assembler subsequently pulls the first three rows and
    columns back to grid displacement DOFs through the surface-point shape
    functions.
    """

    source_point = _as_float_array(source_point, (3,), "source_point")
    source_controls = np.zeros((4, 3), dtype=np.float64)
    source_controls[0] = source_point
    full = affine_levelset_gap(
        source_controls,
        target_controls,
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        target_scale,
        target_levelset,
    )
    indices = np.concatenate((np.arange(3, dtype=np.int64), np.arange(12, 24, dtype=np.int64)))
    return LevelSetGap(
        gap=full.gap,
        gradient=np.ascontiguousarray(full.gradient[indices]),
        hessian=np.ascontiguousarray(full.hessian[np.ix_(indices, indices)]),
        source_point=full.source_point,
        target_coordinate=full.target_coordinate,
        inside_grid=full.inside_grid,
    )


def continued_ipc_barrier_distance_terms(
    distance,
    dhat,
    *,
    kappa=1.0,
    continuation_distance=None,
):
    """A finite C2 continuation of the normal IPC barrier into overlap.

    Standard IPC is retained for ``distance >= delta``.  Below the positive
    continuation point ``delta`` its second-order Taylor polynomial is used.
    The extension is finite for negative gaps, has a restoring normal force,
    and matches value/gradient/Hessian at ``delta``.  It is intended only for
    constructing a strictly feasible initial state; dynamics always use the
    singular, non-continued IPC barrier.
    """

    dhat = _as_positive_scalar(dhat, "dhat")
    kappa = _as_positive_scalar(kappa, "kappa")
    if continuation_distance is None:
        continuation_distance = 0.1 * dhat
    delta = _as_positive_scalar(continuation_distance, "continuation_distance")
    if delta >= dhat:
        raise ValueError("continuation_distance must be smaller than dhat")
    distance = float(distance)
    if not np.isfinite(distance):
        raise ValueError("distance must be finite")
    if distance >= delta:
        return ipc_barrier_distance_terms_py(distance, dhat, kappa=kappa)
    value_delta, gradient_delta, hessian_delta = ipc_barrier_distance_terms_py(delta, dhat, kappa=kappa)
    increment = distance - delta
    return (
        value_delta + gradient_delta * increment + 0.5 * hessian_delta * increment * increment,
        gradient_delta + hessian_delta * increment,
        hessian_delta,
    )


@dataclass
class AffineLevelSetBody:
    controls: np.ndarray
    scale: float
    levelset: TrilinearLevelSet
    surface_basis: np.ndarray
    surface_weight: np.ndarray
    body_id: int = -1

    def __post_init__(self):
        self.controls = _as_float_array(self.controls, (4, 3), "controls")
        self.scale = _as_positive_scalar(self.scale, "scale")
        if not isinstance(self.levelset, TrilinearLevelSet):
            raise TypeError("levelset must be a TrilinearLevelSet")
        basis = np.asarray(self.surface_basis, dtype=np.float64)
        if basis.ndim != 2 or basis.shape[1] != 4:
            raise ValueError("surface_basis must have shape (point_count, 4)")
        weights = np.asarray(self.surface_weight, dtype=np.float64).reshape(-1)
        if weights.size != basis.shape[0]:
            raise ValueError("surface_weight must contain one value per surface point")
        if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("surface basis/weights must be finite and positive")
        self.surface_basis = np.ascontiguousarray(basis)
        self.surface_weight = np.ascontiguousarray(weights)


@dataclass
class LevelSetIPCAssembly:
    energy: float
    gradient: np.ndarray
    hessian: np.ndarray
    minimum_gap: float
    active_contacts: int
    sampled_contacts: int
    feasible: bool


def _project_psd(matrix):
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


class AffineLevelSetIPC:
    """Symmetric surface-quadrature IPC for level-set affine bodies."""

    def __init__(
        self,
        bodies: Iterable[AffineLevelSetBody],
        *,
        dhat,
        kappa,
        time_scale=1.0,
        hessian_mode="exact",
        excluded_body_pairs=(),
    ):
        self.bodies = list(bodies)
        if not self.bodies:
            raise ValueError("at least one affine level-set body is required")
        self.dhat = _as_positive_scalar(dhat, "dhat")
        self.kappa = _as_positive_scalar(kappa, "kappa")
        self.time_scale = _as_positive_scalar(time_scale, "time_scale")
        mode = str(hessian_mode).strip().replace("-", "_").lower()
        aliases = {
            "exact": "exact",
            "projected": "projected",
            "psd": "projected",
            "gauss_newton": "gauss_newton",
            "gaussnewton": "gauss_newton",
        }
        if mode not in aliases:
            raise ValueError("hessian_mode must be exact, projected, or gauss_newton")
        self.hessian_mode = aliases[mode]
        self.body_num = len(self.bodies)
        self.dof = 12 * self.body_num
        self.excluded_body_pairs = {tuple(sorted((int(body_a), int(body_b)))) for body_a, body_b in excluded_body_pairs}

    def pack_controls(self):
        return np.concatenate([body.controls.reshape(-1) for body in self.bodies])

    def set_controls(self, packed):
        values = _as_float_array(packed, (self.dof,), "packed controls")
        for body_id, body in enumerate(self.bodies):
            body.controls = values[12 * body_id : 12 * (body_id + 1)].reshape(4, 3).copy()

    def _local_indices(self, source, target):
        return np.concatenate(
            (
                np.arange(12 * source, 12 * (source + 1)),
                np.arange(12 * target, 12 * (target + 1)),
            )
        )

    def assemble(
        self,
        controls=None,
        *,
        need_hessian=True,
        continued=False,
        continuation_distance=None,
        kappa=None,
    ):
        if controls is not None:
            self.set_controls(controls)
        barrier_kappa = self.kappa if kappa is None else _as_positive_scalar(kappa, "kappa")
        energy = 0.0
        gradient = np.zeros(self.dof, dtype=np.float64)
        hessian = np.zeros((self.dof, self.dof), dtype=np.float64)
        minimum_gap = np.inf
        active_contacts = 0
        sampled_contacts = 0
        feasible = True

        for source_id, source in enumerate(self.bodies):
            for target_id, target in enumerate(self.bodies):
                if source_id == target_id or tuple(sorted((source_id, target_id))) in self.excluded_body_pairs:
                    continue
                indices = self._local_indices(source_id, target_id)
                for basis, weight in zip(source.surface_basis, source.surface_weight):
                    gap_data = affine_levelset_gap(
                        source.controls,
                        target.controls,
                        basis,
                        target.scale,
                        target.levelset,
                    )
                    if not gap_data.inside_grid:
                        continue
                    sampled_contacts += 1
                    gap = float(gap_data.gap)
                    minimum_gap = min(minimum_gap, gap)
                    feasible = feasible and gap > 0.0
                    coefficient = 0.5 * self.time_scale * float(weight)
                    if continued:
                        value, first, second = continued_ipc_barrier_distance_terms(
                            gap,
                            self.dhat,
                            kappa=barrier_kappa,
                            continuation_distance=continuation_distance,
                        )
                    else:
                        value, first, second = ipc_barrier_distance_terms_py(
                            gap,
                            self.dhat,
                            kappa=barrier_kappa,
                        )
                    if not np.isfinite(value):
                        return LevelSetIPCAssembly(
                            energy=np.inf,
                            gradient=gradient,
                            hessian=hessian,
                            minimum_gap=minimum_gap,
                            active_contacts=active_contacts,
                            sampled_contacts=sampled_contacts,
                            feasible=False,
                        )
                    if value == 0.0 and first == 0.0 and second == 0.0:
                        continue
                    active_contacts += 1
                    local_gradient = first * gap_data.gradient
                    exact_hessian = second * np.outer(gap_data.gradient, gap_data.gradient) + first * gap_data.hessian
                    if self.hessian_mode == "projected":
                        local_hessian = _project_psd(exact_hessian)
                    elif self.hessian_mode == "gauss_newton":
                        local_hessian = second * np.outer(gap_data.gradient, gap_data.gradient)
                    else:
                        local_hessian = exact_hessian
                    energy += coefficient * value
                    gradient[indices] += coefficient * local_gradient
                    if need_hessian:
                        hessian[np.ix_(indices, indices)] += coefficient * local_hessian

        return LevelSetIPCAssembly(
            energy=float(energy),
            gradient=gradient,
            hessian=0.5 * (hessian + hessian.T),
            minimum_gap=float(minimum_gap),
            active_contacts=active_contacts,
            sampled_contacts=sampled_contacts,
            feasible=bool(feasible),
        )


@dataclass
class NonpenetrationResult:
    controls: np.ndarray
    translations: np.ndarray
    initial_minimum_gap: float
    minimum_gap: float
    iterations: int
    continuation_stages: int
    success: bool
    message: str
    objective: float


class AdjointIPCNonpenetration:
    """Differentiable translation solve for strictly feasible IPC starts.

    The inner problem is

    ``min_t 1/2 ||t-a||_W^2 + E_IPC^C2(y0 + P t)``,

    where ``P`` translates all four controls of a body equally.  The
    continuation makes the energy finite for overlap.  Barrier stiffness is
    increased until every sampled oriented gap is positive.  At the
    equilibrium, an outer objective can be differentiated with one adjoint
    solve of the translation Hessian.
    """

    def __init__(
        self,
        contact: AffineLevelSetIPC,
        *,
        anchor_stiffness=1.0,
        continuation_ratio=0.1,
        feasibility_tolerance=1.0e-8,
        maximum_stages=8,
        stiffness_growth=10.0,
    ):
        required = ("body_num", "dof", "pack_controls", "set_controls", "assemble")
        if not all(hasattr(contact, name) for name in required):
            raise TypeError("contact does not implement the affine IPC interface")
        self.contact = contact
        self.anchor_stiffness = _as_positive_scalar(anchor_stiffness, "anchor_stiffness")
        self.continuation_ratio = float(continuation_ratio)
        if not np.isfinite(self.continuation_ratio) or self.continuation_ratio <= 0.0 or self.continuation_ratio >= 1.0:
            raise ValueError("continuation_ratio must lie in (0, 1)")
        self.feasibility_tolerance = _as_positive_scalar(feasibility_tolerance, "feasibility_tolerance")
        self.maximum_stages = int(maximum_stages)
        if self.maximum_stages <= 0:
            raise ValueError("maximum_stages must be positive")
        self.stiffness_growth = _as_positive_scalar(stiffness_growth, "stiffness_growth")
        if self.stiffness_growth <= 1.0:
            raise ValueError("stiffness_growth must be larger than one")
        self.initial_controls = self.contact.pack_controls().copy()
        self.anchor = np.zeros(3 * self.contact.body_num, dtype=np.float64)
        self.translation_hessian = None
        self.free_translation_mask = None
        self.last_translations = None
        self.last_result = None

    def _translated_controls(self, translations):
        translations = _as_float_array(
            translations,
            (3 * self.contact.body_num,),
            "translations",
        ).reshape(self.contact.body_num, 3)
        controls = self.initial_controls.reshape(self.contact.body_num, 4, 3).copy()
        controls += translations[:, None, :]
        return controls.reshape(-1)

    def _pullback_translation(self, assembly):
        body_num = self.contact.body_num
        gradient = assembly.gradient.reshape(body_num, 4, 3).sum(axis=1)
        hessian = np.zeros((3 * body_num, 3 * body_num), dtype=np.float64)
        full = assembly.hessian.reshape(body_num, 4, 3, body_num, 4, 3)
        for i in range(body_num):
            for j in range(body_num):
                hessian[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = full[i, :, :, j, :, :].sum(axis=(0, 2))
        return gradient.reshape(-1), 0.5 * (hessian + hessian.T)

    def _objective(self, translations, kappa, need_hessian=False):
        controls = self._translated_controls(translations)
        assembly = self.contact.assemble(
            controls,
            need_hessian=need_hessian,
            continued=True,
            continuation_distance=(self.continuation_ratio * self.contact.dhat),
            kappa=kappa,
        )
        displacement = np.asarray(translations) - self.anchor
        value = 0.5 * self.anchor_stiffness * float(displacement @ displacement) + assembly.energy
        contact_gradient, contact_hessian = self._pullback_translation(assembly)
        gradient = self.anchor_stiffness * displacement + contact_gradient
        if not need_hessian:
            return value, gradient, assembly
        hessian = self.anchor_stiffness * np.eye(3 * self.contact.body_num) + contact_hessian
        return value, gradient, hessian, assembly

    def solve(self, *, anchor=None, bounds=None, maximum_iterations=300):
        # Explicit host-side continuation utility; keep SciPy lazy and out of
        # production Taichi backend initialization.
        from scipy.optimize import minimize

        if anchor is not None:
            self.anchor = _as_float_array(
                anchor,
                (3 * self.contact.body_num,),
                "anchor",
            )
        translations = self.anchor.copy()
        scipy_bounds = None
        if bounds is not None:
            bounds_array = _as_float_array(
                bounds,
                (3 * self.contact.body_num, 2),
                "translation bounds",
            )
            if np.any(bounds_array[:, 0] > bounds_array[:, 1]):
                raise ValueError("translation lower bounds exceed upper bounds")
            scipy_bounds = [tuple(row) for row in bounds_array]
            translations = np.minimum(
                np.maximum(translations, bounds_array[:, 0]),
                bounds_array[:, 1],
            )
        initial = self.contact.assemble(
            self._translated_controls(translations),
            need_hessian=False,
            continued=True,
            continuation_distance=(self.continuation_ratio * self.contact.dhat),
        )
        initial_gap = initial.minimum_gap
        total_iterations = 0
        final_opt = None
        final_assembly = initial
        used_stages = 0
        for stage in range(self.maximum_stages):
            used_stages = stage + 1
            stage_kappa = self.contact.kappa * (self.stiffness_growth**stage)

            def fun(value):
                objective, gradient, _ = self._objective(value, stage_kappa, need_hessian=False)
                return objective, gradient

            final_opt = minimize(
                fun,
                translations,
                method="L-BFGS-B",
                jac=True,
                bounds=scipy_bounds,
                options={
                    "maxiter": int(maximum_iterations),
                    "ftol": 1.0e-14,
                    "gtol": 1.0e-10,
                    "maxls": 80,
                },
            )
            translations = np.asarray(final_opt.x, dtype=np.float64)
            total_iterations += int(getattr(final_opt, "nit", 0))
            final_assembly = self.contact.assemble(
                self._translated_controls(translations),
                need_hessian=True,
                continued=True,
                continuation_distance=(self.continuation_ratio * self.contact.dhat),
                kappa=stage_kappa,
            )
            if final_assembly.minimum_gap >= self.feasibility_tolerance:
                break

        controls = self._translated_controls(translations)
        self.contact.set_controls(controls)
        _, _, translation_hessian, final_assembly = self._objective(
            translations,
            self.contact.kappa * (self.stiffness_growth ** (used_stages - 1)),
            need_hessian=True,
        )
        self.translation_hessian = translation_hessian
        self.free_translation_mask = np.ones(3 * self.contact.body_num, dtype=bool)
        if scipy_bounds is not None:
            tolerance = 1.0e-8 * max(1.0, float(np.linalg.norm(translations, ord=np.inf)))
            for dof, (lower, upper) in enumerate(scipy_bounds):
                if translations[dof] <= lower + tolerance or translations[dof] >= upper - tolerance:
                    self.free_translation_mask[dof] = False
        self.last_translations = translations.copy()
        success = bool(final_assembly.minimum_gap >= self.feasibility_tolerance)
        message = (
            "strictly feasible affine IPC state constructed"
            if success
            else ("continuation stages exhausted before the requested " "positive-gap tolerance")
        )
        self.last_result = NonpenetrationResult(
            controls=controls.reshape(self.contact.body_num, 4, 3),
            translations=translations.reshape(self.contact.body_num, 3),
            initial_minimum_gap=float(initial_gap),
            minimum_gap=float(final_assembly.minimum_gap),
            iterations=total_iterations,
            continuation_stages=used_stages,
            success=success,
            message=message,
            objective=float(getattr(final_opt, "fun", np.nan) if final_opt is not None else np.nan),
        )
        return self.last_result

    def adjoint_anchor_gradient(
        self,
        loss_translation_gradient,
        *,
        direct_anchor_gradient=None,
    ):
        """Differentiate an outer loss through the converged push-out solve."""

        if self.translation_hessian is None or self.last_result is None:
            raise RuntimeError("solve() must succeed before the adjoint call")
        loss_gradient = _as_float_array(
            loss_translation_gradient,
            (3 * self.contact.body_num,),
            "loss_translation_gradient",
        )
        direct = (
            np.zeros_like(loss_gradient)
            if direct_anchor_gradient is None
            else _as_float_array(
                direct_anchor_gradient,
                loss_gradient.shape,
                "direct_anchor_gradient",
            )
        )
        free = self.free_translation_mask
        if free is None:
            free = np.ones_like(loss_gradient, dtype=bool)
        adjoint = np.zeros_like(loss_gradient)
        if np.any(free):
            reduced_hessian = self.translation_hessian[np.ix_(free, free)]
            adjoint[free] = np.linalg.solve(reduced_hessian.T, loss_gradient[free])
        # R(t,a)=W(t-a)+grad E_c(t)=0, so
        # dL/da = partial_a L - lambda^T partial_a R
        #       = partial_a L + W lambda.
        return direct + self.anchor_stiffness * adjoint


__all__ = [
    "AdjointIPCNonpenetration",
    "AffineLevelSetBody",
    "AffineLevelSetIPC",
    "LevelSetGap",
    "LevelSetIPCAssembly",
    "NonpenetrationResult",
    "TrilinearLevelSet",
    "affine_basis",
    "affine_levelset_gap",
    "affine_matrix",
    "continued_ipc_barrier_distance_terms",
    "point_affine_levelset_gap",
]
