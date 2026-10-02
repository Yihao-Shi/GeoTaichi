"""Unit checks for Taichi fully-implicit friction blocks."""

import numpy as np
import pytest
import taichi as ti

from tests.helpers.igampm_fully_implicit_friction_reference import (
    point_nurbs_friction_residual_jacobian,
)
from src.igampm.engines.FullyImplicitFriction import (
    curve_geometry_jacobian,
    curve_negative_force_jacobian_block,
    curve_parameter_jacobian,
    curve_relative_velocity_jacobian,
    fully_implicit_resistance_state,
    resistance_input_jacobian,
    surface_geometry_jacobian,
    surface_negative_force_jacobian_block,
    surface_parameter_jacobian,
    surface_relative_velocity_jacobian,
)
from src.nurbs.NurbsBasis import (
    NurbsBasis2ndDers1d,
    NurbsBasis2ndDers2d,
    NurbsBasisInterpolations2ndDers1d,
    NurbsBasisInterpolations2ndDers2d,
)
from src.physics_model.contact_model.ipc import (
    ipc_barrier_distance_terms_py,
)
from src.physics_model.contact_model.ipc.NurbsContact import (
    closest_curve_point_py,
)


def _friction_parameters():
    return {
        "area": 0.7,
        "dhat": 0.4,
        "dmin": 0.0,
        "kappa": 3.0,
        "use_physical_barrier": False,
        "mu_dynamic": 0.35,
        "mu_static": 0.62,
        "mu_viscous": 0.03,
        "stribeck_velocity": 0.45,
        "epsv": 0.025,
        "friction_profile": "quadratic",
    }


def _barrier_state(distance, parameters):
    _, gradient, hessian = ipc_barrier_distance_terms_py(
        distance,
        parameters["dhat"],
        parameters["dmin"],
        parameters["kappa"],
        parameters["use_physical_barrier"],
    )
    return -parameters["area"] * gradient, hessian


def _flatten_blocks(blocks):
    blocks = np.asarray(blocks, dtype=np.float64)
    block_count, _, dimension, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(
        (block_count * dimension, block_count * dimension)
    )


def _refine_curve_parameter(knots, weights, control, point, seed):
    parameter = float(seed)
    for _ in range(30):
        position, tangent, curvature = NurbsBasisInterpolations2ndDers1d(
            parameter, 2, knots, control, weights
        )
        residual = position - point
        tangent = tangent[0]
        curvature = curvature[0]
        gradient = float(np.dot(residual, tangent))
        hessian = float(np.dot(tangent, tangent) + np.dot(residual, curvature))
        update = gradient / hessian
        parameter = float(np.clip(parameter - update, 0.0, 1.0))
        if abs(update) < 1.0e-14:
            break
    return parameter


def _refine_surface_parameter(knots, weights, control, point, seed):
    parameter = np.asarray(seed, dtype=np.float64).copy()
    for _ in range(40):
        position, tangent, curvature = NurbsBasisInterpolations2ndDers2d(
            parameter[0], parameter[1], 2, 2,
            knots, knots, control, weights,
        )
        residual = position - point
        tangent_u, tangent_v = tangent
        curvature_uu, curvature_vv, curvature_uv = curvature
        gradient = np.asarray([
            np.dot(residual, tangent_u),
            np.dot(residual, tangent_v),
        ])
        hessian = np.asarray([
            [
                np.dot(tangent_u, tangent_u)
                + np.dot(residual, curvature_uu),
                np.dot(tangent_u, tangent_v)
                + np.dot(residual, curvature_uv),
            ],
            [
                np.dot(tangent_v, tangent_u)
                + np.dot(residual, curvature_uv),
                np.dot(tangent_v, tangent_v)
                + np.dot(residual, curvature_vv),
            ],
        ])
        update = np.linalg.solve(hessian, gradient)
        parameter = np.clip(parameter - update, 0.0, 1.0)
        if np.linalg.norm(update, ord=np.inf) < 1.0e-14:
            break
    return parameter


@ti.data_oriented
class _CurveBlockHarness:
    def __init__(
        self,
        *,
        residual,
        tangent,
        curvature,
        relative_velocity,
        surface_velocity_derivative,
        direct_weight,
        relative_weight,
        shape_derivative,
        is_control,
        endpoint_velocity_scale,
        distance,
        normal_force,
        area,
        barrier_hessian,
        parameters,
    ):
        self.block_count = int(np.asarray(direct_weight).size)
        self.residual = ti.Vector.field(2, ti.f64, shape=())
        self.tangent = ti.Vector.field(2, ti.f64, shape=())
        self.curvature = ti.Vector.field(2, ti.f64, shape=())
        self.relative_velocity = ti.Vector.field(2, ti.f64, shape=())
        self.surface_velocity_derivative = ti.Vector.field(
            2, ti.f64, shape=())
        self.direct_weight = ti.field(ti.f64, shape=self.block_count)
        self.relative_weight = ti.field(ti.f64, shape=self.block_count)
        self.shape_derivative = ti.field(ti.f64, shape=self.block_count)
        self.is_control = ti.field(ti.i32, shape=self.block_count)
        self.force = ti.Vector.field(2, ti.f64, shape=self.block_count)
        self.parameter_jacobian = ti.Vector.field(
            2, ti.f64, shape=self.block_count)
        self.normal_jacobian = ti.Matrix.field(
            2, 2, ti.f64, shape=self.block_count)
        self.negative_jacobian = ti.Matrix.field(
            2, 2, ti.f64, shape=(self.block_count, self.block_count))
        self.projector = ti.Matrix.field(2, 2, ti.f64, shape=())
        self.valid = ti.field(ti.i32, shape=self.block_count)

        self.residual[None] = residual
        self.tangent[None] = tangent
        self.curvature[None] = curvature
        self.relative_velocity[None] = relative_velocity
        self.surface_velocity_derivative[None] = surface_velocity_derivative
        self.direct_weight.from_numpy(np.asarray(direct_weight, dtype=np.float64))
        self.relative_weight.from_numpy(np.asarray(relative_weight, dtype=np.float64))
        self.shape_derivative.from_numpy(np.asarray(
            shape_derivative, dtype=np.float64))
        self.is_control.from_numpy(np.asarray(is_control, dtype=np.int32))
        self.endpoint_velocity_scale = float(endpoint_velocity_scale)
        self.distance = float(distance)
        self.normal_force = float(normal_force)
        self.area = float(area)
        self.barrier_hessian = float(barrier_hessian)
        self.mu_dynamic = float(parameters["mu_dynamic"])
        self.mu_static = float(parameters["mu_static"])
        self.mu_viscous = float(parameters["mu_viscous"])
        self.stribeck_velocity = float(parameters["stribeck_velocity"])
        self.epsv = float(parameters["epsv"])

    @ti.kernel
    def evaluate(self):
        normal = self.residual[None] / self.distance
        (
            projector,
            _,
            resistance,
            velocity_jacobian,
            factor_per_normal_force,
        ) = fully_implicit_resistance_state(
            self.relative_velocity[None],
            normal,
            self.normal_force,
            self.mu_dynamic,
            self.mu_static,
            self.mu_viscous,
            self.stribeck_velocity,
            self.epsv,
            0,
        )
        self.projector[None] = projector
        for output_block in range(self.block_count):
            self.force[output_block] = (
                -self.relative_weight[output_block] * resistance
            )
        for input_block in range(self.block_count):
            parameter_jacobian, valid = curve_parameter_jacobian(
                self.residual[None],
                self.tangent[None],
                self.curvature[None],
                self.direct_weight[input_block],
                self.shape_derivative[input_block],
                self.is_control[input_block],
                1,
            )
            self.valid[input_block] = valid
            self.parameter_jacobian[input_block] = parameter_jacobian
            residual_geometry_jacobian = curve_geometry_jacobian(
                self.direct_weight[input_block],
                self.tangent[None],
                parameter_jacobian,
            )
            self.normal_jacobian[input_block] = (
                projector @ residual_geometry_jacobian / self.distance
            )
            relative_velocity_jacobian = curve_relative_velocity_jacobian(
                self.relative_weight[input_block],
                self.endpoint_velocity_scale,
                self.surface_velocity_derivative[None],
                parameter_jacobian,
            )
            resistance_jacobian = resistance_input_jacobian(
                residual_geometry_jacobian,
                relative_velocity_jacobian,
                self.relative_velocity[None],
                normal,
                self.distance,
                projector,
                projector @ self.relative_velocity[None],
                velocity_jacobian,
                factor_per_normal_force,
                self.area,
                self.barrier_hessian,
            )
            for output_block in range(self.block_count):
                self.negative_jacobian[output_block, input_block] = (
                    curve_negative_force_jacobian_block(
                        self.relative_weight[output_block],
                        self.is_control[output_block],
                        self.shape_derivative[output_block],
                        parameter_jacobian,
                        resistance,
                        resistance_jacobian,
                    )
                )


@ti.data_oriented
class _SurfaceBlockHarness:
    def __init__(
        self,
        *,
        residual,
        tangents,
        curvatures,
        relative_velocity,
        surface_velocity_derivatives,
        direct_weight,
        relative_weight,
        shape_derivatives,
        is_control,
        endpoint_velocity_scale,
        distance,
        normal_force,
        area,
        barrier_hessian,
        parameters,
    ):
        self.block_count = int(np.asarray(direct_weight).size)
        self.residual = ti.Vector.field(3, ti.f64, shape=())
        self.tangent_u = ti.Vector.field(3, ti.f64, shape=())
        self.tangent_v = ti.Vector.field(3, ti.f64, shape=())
        self.curvature_uu = ti.Vector.field(3, ti.f64, shape=())
        self.curvature_vv = ti.Vector.field(3, ti.f64, shape=())
        self.curvature_uv = ti.Vector.field(3, ti.f64, shape=())
        self.relative_velocity = ti.Vector.field(3, ti.f64, shape=())
        self.surface_velocity_derivative_u = ti.Vector.field(
            3, ti.f64, shape=())
        self.surface_velocity_derivative_v = ti.Vector.field(
            3, ti.f64, shape=())
        self.direct_weight = ti.field(ti.f64, shape=self.block_count)
        self.relative_weight = ti.field(ti.f64, shape=self.block_count)
        self.shape_derivative_u = ti.field(ti.f64, shape=self.block_count)
        self.shape_derivative_v = ti.field(ti.f64, shape=self.block_count)
        self.is_control = ti.field(ti.i32, shape=self.block_count)
        self.force = ti.Vector.field(3, ti.f64, shape=self.block_count)
        self.parameter_jacobian = ti.Matrix.field(
            2, 3, ti.f64, shape=self.block_count)
        self.normal_jacobian = ti.Matrix.field(
            3, 3, ti.f64, shape=self.block_count)
        self.negative_jacobian = ti.Matrix.field(
            3, 3, ti.f64, shape=(self.block_count, self.block_count))
        self.projector = ti.Matrix.field(3, 3, ti.f64, shape=())
        self.valid = ti.field(ti.i32, shape=self.block_count)

        self.residual[None] = residual
        self.tangent_u[None] = tangents[0]
        self.tangent_v[None] = tangents[1]
        self.curvature_uu[None] = curvatures[0]
        self.curvature_vv[None] = curvatures[1]
        self.curvature_uv[None] = curvatures[2]
        self.relative_velocity[None] = relative_velocity
        self.surface_velocity_derivative_u[None] = (
            surface_velocity_derivatives[0])
        self.surface_velocity_derivative_v[None] = (
            surface_velocity_derivatives[1])
        self.direct_weight.from_numpy(np.asarray(direct_weight, dtype=np.float64))
        self.relative_weight.from_numpy(np.asarray(relative_weight, dtype=np.float64))
        self.shape_derivative_u.from_numpy(np.asarray(
            shape_derivatives[:, 0], dtype=np.float64))
        self.shape_derivative_v.from_numpy(np.asarray(
            shape_derivatives[:, 1], dtype=np.float64))
        self.is_control.from_numpy(np.asarray(is_control, dtype=np.int32))
        self.endpoint_velocity_scale = float(endpoint_velocity_scale)
        self.distance = float(distance)
        self.normal_force = float(normal_force)
        self.area = float(area)
        self.barrier_hessian = float(barrier_hessian)
        self.mu_dynamic = float(parameters["mu_dynamic"])
        self.mu_static = float(parameters["mu_static"])
        self.mu_viscous = float(parameters["mu_viscous"])
        self.stribeck_velocity = float(parameters["stribeck_velocity"])
        self.epsv = float(parameters["epsv"])

    @ti.kernel
    def evaluate(self):
        normal = self.residual[None] / self.distance
        (
            projector,
            _,
            resistance,
            velocity_jacobian,
            factor_per_normal_force,
        ) = fully_implicit_resistance_state(
            self.relative_velocity[None],
            normal,
            self.normal_force,
            self.mu_dynamic,
            self.mu_static,
            self.mu_viscous,
            self.stribeck_velocity,
            self.epsv,
            0,
        )
        self.projector[None] = projector
        for output_block in range(self.block_count):
            self.force[output_block] = (
                -self.relative_weight[output_block] * resistance
            )
        for input_block in range(self.block_count):
            parameter_jacobian, valid = surface_parameter_jacobian(
                self.residual[None],
                self.tangent_u[None],
                self.tangent_v[None],
                self.curvature_uu[None],
                self.curvature_vv[None],
                self.curvature_uv[None],
                self.direct_weight[input_block],
                self.shape_derivative_u[input_block],
                self.shape_derivative_v[input_block],
                self.is_control[input_block],
                1,
                1,
            )
            self.valid[input_block] = valid
            self.parameter_jacobian[input_block] = parameter_jacobian
            residual_geometry_jacobian = surface_geometry_jacobian(
                self.direct_weight[input_block],
                self.tangent_u[None],
                self.tangent_v[None],
                parameter_jacobian,
            )
            self.normal_jacobian[input_block] = (
                projector @ residual_geometry_jacobian / self.distance
            )
            relative_velocity_jacobian = surface_relative_velocity_jacobian(
                self.relative_weight[input_block],
                self.endpoint_velocity_scale,
                self.surface_velocity_derivative_u[None],
                self.surface_velocity_derivative_v[None],
                parameter_jacobian,
            )
            resistance_jacobian = resistance_input_jacobian(
                residual_geometry_jacobian,
                relative_velocity_jacobian,
                self.relative_velocity[None],
                normal,
                self.distance,
                projector,
                projector @ self.relative_velocity[None],
                velocity_jacobian,
                factor_per_normal_force,
                self.area,
                self.barrier_hessian,
            )
            for output_block in range(self.block_count):
                self.negative_jacobian[output_block, input_block] = (
                    surface_negative_force_jacobian_block(
                        self.relative_weight[output_block],
                        self.is_control[output_block],
                        self.shape_derivative_u[output_block],
                        self.shape_derivative_v[output_block],
                        parameter_jacobian,
                        resistance,
                        resistance_jacobian,
                    )
                )


def _assert_complete_match(harness, force, matrix, diagnostics, parameters):
    harness.evaluate()
    device_force = harness.force.to_numpy().reshape(-1)
    device_matrix = _flatten_blocks(harness.negative_jacobian.to_numpy())
    np.testing.assert_allclose(device_force, force, rtol=2.0e-12, atol=2.0e-12)
    np.testing.assert_allclose(device_matrix, matrix, rtol=2.0e-11, atol=3.0e-12)
    assert np.all(harness.valid.to_numpy() == 1)

    block_parameters = harness.parameter_jacobian.to_numpy()
    if block_parameters.ndim == 2:
        device_parameter_jacobian = block_parameters.reshape((1, -1))
    else:
        device_parameter_jacobian = block_parameters.transpose(
            1, 0, 2).reshape((block_parameters.shape[1], -1))
    np.testing.assert_allclose(
        device_parameter_jacobian,
        diagnostics["parameter_jacobian"],
        rtol=2.0e-11,
        atol=3.0e-12,
    )
    block_normals = harness.normal_jacobian.to_numpy()
    device_normal_jacobian = block_normals.transpose(1, 0, 2).reshape(
        (block_normals.shape[1], -1)
    )
    np.testing.assert_allclose(
        device_normal_jacobian,
        diagnostics["normal_jacobian"],
        rtol=2.0e-11,
        atol=3.0e-12,
    )
    normal = np.asarray(harness.residual[None]) / diagnostics["distance"]
    np.testing.assert_allclose(
        harness.projector.to_numpy(),
        np.eye(normal.size) - np.outer(normal, normal),
        rtol=0.0,
        atol=2.0e-14,
    )

    assert diagnostics["normal_force"] > 0.0
    assert 0.0 < diagnostics["speed"] < parameters["stribeck_velocity"]
    assert parameters["mu_static"] != parameters["mu_dynamic"]
    assert np.linalg.norm(diagnostics["parameter_jacobian"]) > 1.0e-4
    assert np.linalg.norm(diagnostics["normal_jacobian"]) > 1.0e-4
    assert np.linalg.norm(matrix - matrix.T, ord=np.inf) > 1.0e-5


def test_curve_block_helpers_compile_and_match_complete_numpy_oracle(
    taichi_runtime,
):
    parameters = _friction_parameters()
    knots = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    weights = np.asarray([1.0, 0.83, 1.12])
    reference_control = np.asarray([
        [-0.8, 0.02], [0.0, 0.31], [0.9, -0.04],
    ])
    reference_point = np.asarray([0.18, -0.105])
    displacement = np.asarray([
        0.006, -0.004, -0.003, 0.005,
        0.004, -0.002, 0.008, 0.003,
    ])
    current_control = reference_control + displacement[:6].reshape((3, 2))
    point = reference_point + displacement[6:]
    seed, _ = closest_curve_point_py(
        2, knots, current_control, weights, point)
    parameter = _refine_curve_parameter(
        knots, weights, current_control, point, seed)
    shape, derivative, second = NurbsBasis2ndDers1d(
        parameter, 2, knots, weights)

    velocity_scale = 7.25
    velocity_offset = np.asarray([
        -0.04, 0.03, 0.02, -0.01,
        0.06, 0.025, 0.17, -0.035,
    ])
    endpoint_velocity = velocity_offset + velocity_scale * displacement
    control_dofs = np.arange(6, dtype=np.int64).reshape((3, 2))
    point_dofs = np.asarray([[6, 7]], dtype=np.int64)
    force, matrix, diagnostics = point_nurbs_friction_residual_jacobian(
        control_positions=current_control,
        point_position=point,
        shape_values=shape,
        shape_first_derivatives=derivative[:, None],
        shape_second_derivatives=second[:, None, None],
        control_dofs=control_dofs,
        point_dofs=point_dofs,
        point_weights=[1.0],
        endpoint_velocity=endpoint_velocity,
        endpoint_velocity_displacement_scale=np.full(8, velocity_scale),
        free_parameters=[True],
        **parameters,
    )

    residual = shape @ current_control - point
    distance = np.linalg.norm(residual)
    normal_force, barrier_hessian = _barrier_state(distance, parameters)
    control_velocity = endpoint_velocity[:6].reshape((3, 2))
    relative_weight = np.r_[-shape, 1.0]
    harness = _CurveBlockHarness(
        residual=residual,
        tangent=derivative @ current_control,
        curvature=second @ current_control,
        relative_velocity=(
            endpoint_velocity[6:] - shape @ control_velocity),
        surface_velocity_derivative=derivative @ control_velocity,
        direct_weight=-relative_weight,
        relative_weight=relative_weight,
        shape_derivative=np.r_[derivative, 0.0],
        is_control=np.r_[np.ones(3, dtype=np.int32), 0],
        endpoint_velocity_scale=velocity_scale,
        distance=distance,
        normal_force=normal_force,
        area=parameters["area"],
        barrier_hessian=barrier_hessian,
        parameters=parameters,
    )
    _assert_complete_match(harness, force, matrix, diagnostics, parameters)


def test_surface_block_helpers_compile_and_match_complete_numpy_oracle(
    taichi_runtime,
):
    parameters = _friction_parameters()
    knots = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    controls = []
    weights = []
    for v in np.linspace(-0.7, 0.7, 3):
        for u in np.linspace(-0.8, 0.8, 3):
            controls.append([
                u, v, 0.16 * (1.0 - u * u) + 0.05 * u * v,
            ])
            weights.append(1.0 + 0.08 * u - 0.04 * v)
    reference_control = np.asarray(controls)
    weights = np.asarray(weights)
    reference_point = np.asarray([0.16, -0.11, -0.105])
    rng = np.random.default_rng(91827)
    displacement = rng.normal(scale=1.5e-3, size=30)
    current_control = reference_control + displacement[:27].reshape((9, 3))
    point = reference_point + displacement[27:]
    parameter = _refine_surface_parameter(
        knots, weights, current_control, point, [0.6, 0.42])
    (
        shape,
        derivative_u,
        derivative_v,
        second_uu,
        second_vv,
        second_uv,
    ) = NurbsBasis2ndDers2d(
        parameter[0], parameter[1], 2, 2, knots, knots, weights)
    first = np.column_stack([derivative_u, derivative_v])
    second = np.empty((9, 2, 2), dtype=np.float64)
    second[:, 0, 0] = second_uu
    second[:, 1, 1] = second_vv
    second[:, 0, 1] = second_uv
    second[:, 1, 0] = second_uv

    velocity_scale = 6.75
    velocity_offset = rng.normal(scale=0.035, size=30)
    velocity_offset[27:] += np.asarray([0.11, -0.065, 0.028])
    endpoint_velocity = velocity_offset + velocity_scale * displacement
    control_dofs = np.arange(27, dtype=np.int64).reshape((9, 3))
    point_dofs = np.asarray([[27, 28, 29]], dtype=np.int64)
    force, matrix, diagnostics = point_nurbs_friction_residual_jacobian(
        control_positions=current_control,
        point_position=point,
        shape_values=shape,
        shape_first_derivatives=first,
        shape_second_derivatives=second,
        control_dofs=control_dofs,
        point_dofs=point_dofs,
        point_weights=[1.0],
        endpoint_velocity=endpoint_velocity,
        endpoint_velocity_displacement_scale=np.full(30, velocity_scale),
        free_parameters=[True, True],
        **parameters,
    )

    residual = shape @ current_control - point
    distance = np.linalg.norm(residual)
    normal_force, barrier_hessian = _barrier_state(distance, parameters)
    control_velocity = endpoint_velocity[:27].reshape((9, 3))
    relative_weight = np.r_[-shape, 1.0]
    harness = _SurfaceBlockHarness(
        residual=residual,
        tangents=(
            derivative_u @ current_control,
            derivative_v @ current_control,
        ),
        curvatures=(
            second_uu @ current_control,
            second_vv @ current_control,
            second_uv @ current_control,
        ),
        relative_velocity=(
            endpoint_velocity[27:] - shape @ control_velocity),
        surface_velocity_derivatives=(
            derivative_u @ control_velocity,
            derivative_v @ control_velocity,
        ),
        direct_weight=-relative_weight,
        relative_weight=relative_weight,
        shape_derivatives=np.vstack([
            first, np.zeros((1, 2), dtype=np.float64),
        ]),
        is_control=np.r_[np.ones(9, dtype=np.int32), 0],
        endpoint_velocity_scale=velocity_scale,
        distance=distance,
        normal_force=normal_force,
        area=parameters["area"],
        barrier_hessian=barrier_hessian,
        parameters=parameters,
    )
    _assert_complete_match(harness, force, matrix, diagnostics, parameters)
