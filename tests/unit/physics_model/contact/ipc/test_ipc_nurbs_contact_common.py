from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact, pytest.mark.usefixtures("taichi_runtime")]

from src.nurbs.core.NurbsGeometry import NurbsBasisFunction1d, NurbsBasisFunction2d
from src.physics_model.contact_model.ipc.ContactAssembly import psd_project_nd
from src.physics_model.contact_model.ipc.NurbsContact import (
    PointNurbsDerivative,
    closest_curve_point_py,
    eval_curve_py,
    get_distance_to_curve_fixed_dim,
    get_distance_to_curve_moving_fixed_dim,
    get_distance_to_surface_fixed_dim,
    get_distance_to_surface_moving_fixed_dim,
)


@ti.data_oriented
class _CurveDistanceHarness:
    def __init__(self):
        self.basis = NurbsBasisFunction1d(2, dimension=2)
        self.knots = ti.field(ti.f64, shape=6)
        self.ctrlpts = ti.Vector.field(2, ti.f64, shape=3)
        self.control_directions = ti.Vector.field(2, ti.f64, shape=3)
        self.weights = ti.field(ti.f64, shape=3)
        self.parameter = ti.field(ti.f64, shape=())
        self.distance = ti.field(ti.f64, shape=())
        self.residual = ti.Vector.field(2, ti.f64, shape=())
        self.knots.from_numpy(np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64))
        self.ctrlpts.from_numpy(np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float64))
        self.weights.from_numpy(np.ones(3, dtype=np.float64))

    @ti.kernel
    def query(self, point: ti.types.vector(2, ti.f64)):
        parameter, distance, residual = get_distance_to_curve_fixed_dim(
            0,
            0,
            6,
            self.knots,
            self.ctrlpts,
            self.weights,
            point,
            self.basis,
        )
        self.parameter[None] = parameter
        self.distance[None] = distance
        self.residual[None] = residual

    @ti.kernel
    def query_moving(self, point: ti.types.vector(2, ti.f64), alpha: ti.f64):
        parameter, distance, residual = get_distance_to_curve_moving_fixed_dim(
            0,
            0,
            6,
            self.knots,
            self.ctrlpts,
            self.control_directions,
            alpha,
            self.weights,
            point,
            self.basis,
        )
        self.parameter[None] = parameter
        self.distance[None] = distance
        self.residual[None] = residual


@ti.data_oriented
class _DerivativeProjectionHarness:
    def __init__(self):
        self.derivative2 = PointNurbsDerivative(dimension=2)
        self.derivative3 = PointNurbsDerivative(dimension=3)
        self.point_gradient2 = ti.Vector.field(2, ti.f64, shape=())
        self.ctrl_gradient2 = ti.Vector.field(2, ti.f64, shape=())
        self.point_hessian2 = ti.Matrix.field(2, 2, ti.f64, shape=())
        self.ctrl_hessian2 = ti.Matrix.field(2, 2, ti.f64, shape=())
        self.cross_hessian2 = ti.Matrix.field(2, 2, ti.f64, shape=())
        self.velocity_point2 = ti.Matrix.field(2, 2, ti.f64, shape=())
        self.velocity_ctrl2 = ti.Matrix.field(2, 2, ti.f64, shape=())

        self.point_gradient3 = ti.Vector.field(3, ti.f64, shape=())
        self.ctrl_gradient3 = ti.Vector.field(3, ti.f64, shape=())
        self.point_hessian3 = ti.Matrix.field(3, 3, ti.f64, shape=())
        self.ctrl_hessian3 = ti.Matrix.field(3, 3, ti.f64, shape=())
        self.cross_hessian3 = ti.Matrix.field(3, 3, ti.f64, shape=())
        self.projected2 = ti.Matrix.field(2, 2, ti.f64, shape=())
        self.projected3 = ti.Matrix.field(3, 3, ti.f64, shape=())

    @ti.kernel
    def evaluate_curve_derivatives(self):
        pointer = ti.Vector([0.0, -0.25])
        tangent = ti.Vector([1.0, 0.0])
        curvature = ti.Vector([0.0, 0.0])
        self.point_gradient2[None] = self.derivative2.Ddistance2_div_Dpoint(pointer)
        self.ctrl_gradient2[None] = self.derivative2.Ddistance2_div_Dctrlpt(pointer, 0.25)
        self.point_hessian2[None] = self.derivative2.D2distance2_div_D2point_Curve(pointer, tangent, curvature)
        self.ctrl_hessian2[None] = self.derivative2.D2distance2_div_D2ctrlpt_Curve(
            pointer, tangent, curvature, 0.25, 0.0, 0.25, 0.0
        )
        self.cross_hessian2[None] = self.derivative2.D2distance2_div_DctrlptDpoint_Curve(
            pointer, tangent, curvature, 0.25, 0.0
        )
        self.velocity_point2[None] = self.derivative2.Dvelocity_div_Dpoint()
        self.velocity_ctrl2[None] = self.derivative2.Dvelocity_div_Dctrlpt(0.25)

    @ti.kernel
    def evaluate_surface_derivatives(self):
        pointer = ti.Vector([0.0, 0.0, -0.3])
        tangent_u = ti.Vector([1.0, 0.0, 0.0])
        tangent_v = ti.Vector([0.0, 1.0, 0.0])
        zero = ti.Vector([0.0, 0.0, 0.0])
        self.point_gradient3[None] = self.derivative3.Ddistance2_div_Dpoint(pointer)
        self.ctrl_gradient3[None] = self.derivative3.Ddistance2_div_Dctrlpt(pointer, 0.25)
        self.point_hessian3[None] = self.derivative3.D2distance2_div_D2point_Surface(
            pointer, tangent_u, tangent_v, zero, zero, zero
        )
        self.ctrl_hessian3[None] = self.derivative3.D2distance2_div_D2ctrlpt_Surface(
            pointer,
            tangent_u,
            tangent_v,
            zero,
            zero,
            zero,
            0.25,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
        )
        self.cross_hessian3[None] = self.derivative3.D2distance2_div_DctrlptDpoint_Surface(
            pointer,
            tangent_u,
            tangent_v,
            zero,
            zero,
            zero,
            0.25,
            0.0,
            0.0,
        )

    @ti.kernel
    def evaluate_active_boundary_derivatives(self, free_u: ti.i32, free_v: ti.i32):
        pointer2 = ti.Vector([0.0, -0.25])
        tangent2 = ti.Vector([1.0, 0.0])
        zero2 = ti.Vector([0.0, 0.0])
        self.point_hessian2[None] = self.derivative2.D2distance2_div_D2point_CurveActive(
            pointer2, tangent2, zero2, free_u
        )
        self.ctrl_hessian2[None] = self.derivative2.D2distance2_div_D2ctrlpt_CurveActive(
            pointer2, tangent2, zero2, 0.25, 0.0, 0.25, 0.0, free_u
        )
        self.cross_hessian2[None] = self.derivative2.D2distance2_div_DctrlptDpoint_CurveActive(
            pointer2, tangent2, zero2, 0.25, 0.0, free_u
        )

        pointer3 = ti.Vector([0.0, 0.0, -0.3])
        tangent_u = ti.Vector([1.0, 0.0, 0.0])
        tangent_v = ti.Vector([0.0, 1.0, 0.0])
        zero3 = ti.Vector([0.0, 0.0, 0.0])
        self.point_hessian3[None] = self.derivative3.D2distance2_div_D2point_SurfaceActive(
            pointer3,
            tangent_u,
            tangent_v,
            zero3,
            zero3,
            zero3,
            free_u,
            free_v,
        )

    @ti.kernel
    def evaluate_psd_projection(self):
        matrix2 = ti.Matrix([[1.0, 3.0], [-1.0, -2.0]])
        matrix3 = ti.Matrix([[-1.0, 2.0, 0.0], [0.0, 1.0, 0.5], [0.0, -0.5, 3.0]])
        self.projected2[None] = psd_project_nd(matrix2)
        self.projected3[None] = psd_project_nd(matrix3)


@ti.data_oriented
class _SurfaceDistanceHarness:
    def __init__(self):
        self.basis = NurbsBasisFunction2d(2, 2, dimension=3)
        self.knots_u = ti.field(ti.f64, shape=6)
        self.knots_v = ti.field(ti.f64, shape=6)
        self.ctrlpts = ti.Vector.field(3, ti.f64, shape=9)
        self.control_directions = ti.Vector.field(3, ti.f64, shape=9)
        self.weights = ti.field(ti.f64, shape=9)
        self.parameter = ti.Vector.field(2, ti.f64, shape=())
        self.distance = ti.field(ti.f64, shape=())
        self.residual = ti.Vector.field(3, ti.f64, shape=())
        self.shape = ti.field(ti.f64, shape=9)
        self.derivative_u = ti.field(ti.f64, shape=9)
        self.derivative_v = ti.field(ti.f64, shape=9)
        knots = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        self.knots_u.from_numpy(knots)
        self.knots_v.from_numpy(knots)
        ctrlpts = np.asarray(
            [[i / 2.0, j / 2.0, 0.0] for j in range(3) for i in range(3)],
            dtype=np.float64,
        )
        self.ctrlpts.from_numpy(ctrlpts)
        self.weights.from_numpy(np.asarray([1.0, 1.4, 0.8, 1.1, 2.0, 0.9, 0.7, 1.3, 1.6]))

    @ti.kernel
    def query(self, point: ti.types.vector(3, ti.f64)):
        u, v, distance, residual = get_distance_to_surface_fixed_dim(
            0,
            0,
            0,
            6,
            6,
            self.knots_u,
            self.knots_v,
            self.ctrlpts,
            self.weights,
            point,
            self.basis,
        )
        self.parameter[None] = ti.Vector([u, v])
        self.distance[None] = distance
        self.residual[None] = residual

    @ti.kernel
    def query_moving(self, point: ti.types.vector(3, ti.f64), alpha: ti.f64):
        u, v, distance, residual = get_distance_to_surface_moving_fixed_dim(
            0,
            0,
            0,
            6,
            6,
            self.knots_u,
            self.knots_v,
            self.ctrlpts,
            self.control_directions,
            alpha,
            self.weights,
            point,
            self.basis,
        )
        self.parameter[None] = ti.Vector([u, v])
        self.distance[None] = distance
        self.residual[None] = residual

    @ti.kernel
    def evaluate_hessian_basis(self, u: ti.f64, v: ti.f64):
        _, _, shape, derivative_u, derivative_v, _, _, _, _, _ = self.basis.NurbsBasisHessian(
            0,
            0,
            0,
            6,
            6,
            u,
            v,
            self.knots_u,
            self.knots_v,
            self.ctrlpts,
            self.weights,
        )
        for i in ti.static(range(9)):
            self.shape[i] = shape[i]
            self.derivative_u[i] = derivative_u[i]
            self.derivative_v[i] = derivative_v[i]

    @ti.kernel
    def evaluate_shape(self, u: ti.f64, v: ti.f64):
        _, _, shape = self.basis.NurbsBasisShape(
            0,
            0,
            0,
            6,
            6,
            u,
            v,
            self.knots_u,
            self.knots_v,
            self.weights,
        )
        for i in ti.static(range(9)):
            self.shape[i] = shape[i]


@ti.data_oriented
class _MultiSpanClosestHarness:
    def __init__(self):
        self.curve_basis = NurbsBasisFunction1d(2, dimension=2)
        self.surface_basis = NurbsBasisFunction2d(2, 2, dimension=3)
        self.knots_u = ti.field(ti.f64, shape=8)
        self.knots_v = ti.field(ti.f64, shape=6)
        self.curve_ctrlpts = ti.Vector.field(2, ti.f64, shape=5)
        self.surface_ctrlpts = ti.Vector.field(3, ti.f64, shape=15)
        self.curve_weights = ti.field(ti.f64, shape=5)
        self.surface_weights = ti.field(ti.f64, shape=15)
        self.curve_parameter = ti.field(ti.f64, shape=())
        self.curve_distance = ti.field(ti.f64, shape=())
        self.curve_residual = ti.Vector.field(2, ti.f64, shape=())
        self.surface_parameter = ti.Vector.field(2, ti.f64, shape=())
        self.surface_distance = ti.field(ti.f64, shape=())
        self.surface_residual = ti.Vector.field(3, ti.f64, shape=())

        knots_u = np.asarray(
            [0.0, 0.0, 0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0, 1.0, 1.0],
            dtype=np.float64,
        )
        knots_v = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        curve_ctrlpts = np.asarray(
            [[-2.0, 0.0], [-1.0, 2.0], [0.0, 0.0], [1.0, 2.0], [2.0, 0.0]],
            dtype=np.float64,
        )
        surface_ctrlpts = np.asarray(
            [[curve_ctrlpts[i, 0], j / 2.0, curve_ctrlpts[i, 1]] for j in range(3) for i in range(5)],
            dtype=np.float64,
        )
        self.knots_u.from_numpy(knots_u)
        self.knots_v.from_numpy(knots_v)
        self.curve_ctrlpts.from_numpy(curve_ctrlpts)
        self.surface_ctrlpts.from_numpy(surface_ctrlpts)
        self.curve_weights.from_numpy(np.ones(5, dtype=np.float64))
        self.surface_weights.from_numpy(np.ones(15, dtype=np.float64))

    @ti.kernel
    def query(
        self,
        curve_point: ti.types.vector(2, ti.f64),
        surface_point: ti.types.vector(3, ti.f64),
    ):
        curve_parameter, curve_distance, curve_residual = get_distance_to_curve_fixed_dim(
            0,
            0,
            8,
            self.knots_u,
            self.curve_ctrlpts,
            self.curve_weights,
            curve_point,
            self.curve_basis,
        )
        self.curve_parameter[None] = curve_parameter
        self.curve_distance[None] = curve_distance
        self.curve_residual[None] = curve_residual

        u, v, surface_distance, surface_residual = get_distance_to_surface_fixed_dim(
            0,
            0,
            0,
            8,
            6,
            self.knots_u,
            self.knots_v,
            self.surface_ctrlpts,
            self.surface_weights,
            surface_point,
            self.surface_basis,
        )
        self.surface_parameter[None] = ti.Vector([u, v])
        self.surface_distance[None] = surface_distance
        self.surface_residual[None] = surface_residual


@ti.data_oriented
class _CachedSurfaceDistanceHarness:
    def __init__(self, controls, weights, knots_u, knots_v, degrees, points):
        from src.igampm.contact.ContactSurface import CouplingContactSurface

        count_u = len(knots_u) - degrees[0] - 1
        count_v = len(knots_v) - degrees[1] - 1
        primitive = SimpleNamespace(
            num_ctrlpts_u=count_u,
            num_ctrlpts_v=count_v,
            control_points=controls,
            weights=weights,
            gather_boundary_ctrlpts=lambda: (
                [len(controls)],
                np.arange(len(controls)),
                [(knots_u, knots_v)],
                [degrees],
            ),
        )
        iga = SimpleNamespace(
            patch=SimpleNamespace(
                primitive=SimpleNamespace(body={"surface": {"primitive": primitive}}), prefix_total_num_ctrlpts=[0]
            )
        )
        self.surface = CouplingContactSurface(iga)
        self.surface.update_surface_bounds()
        self.surface.update_span_bounds()
        self.count_u, self.count_v = len(knots_u), len(knots_v)
        self.points = ti.Vector.field(3, ti.f64, shape=len(points))
        self.points.from_numpy(points)
        self.seeds = ti.Vector.field(2, ti.f64, shape=len(points))
        self.results = ti.Vector.field(6, ti.f64, shape=len(points))

    @ti.kernel
    def query(self, cached: ti.template(), seed: ti.types.vector(2, ti.f64)):
        for i in self.points:
            u, v, distance = 0.0, 0.0, 0.0
            residual = ti.Vector.zero(ti.f64, 3)
            if ti.static(cached):
                # Match the coupling engine: a vector loaded from a Taichi
                # field must be recognized as an extra seed, not just Matrix.
                self.seeds[i] = seed
                u, v, distance, residual = get_distance_to_surface_fixed_dim(
                    0,
                    0,
                    0,
                    self.count_u,
                    self.count_v,
                    self.surface.knot_vector_u,
                    self.surface.knot_vector_v,
                    self.surface.control_points_hat,
                    self.surface.weights,
                    self.points[i],
                    self.surface.basis[0],
                    self.surface,
                    0,
                    self.seeds[i],
                )
            else:
                u, v, distance, residual = get_distance_to_surface_fixed_dim(
                    0,
                    0,
                    0,
                    self.count_u,
                    self.count_v,
                    self.surface.knot_vector_u,
                    self.surface.knot_vector_v,
                    self.surface.control_points_hat,
                    self.surface.weights,
                    self.points[i],
                    self.surface.basis[0],
                )
            self.results[i] = ti.Vector([u, v, distance, residual[0], residual[1], residual[2]])


@pytest.mark.isolated_dimension(3)
@pytest.mark.parametrize("waves", [False, True], ids=["repeated-knot-rational", "15-waves"])
def test_cached_surface_projection_preserves_multistart_after_deformation(taichi_runtime, waves):
    if waves:
        degrees = (3, 1)
        knots_u = np.r_[np.zeros(3), np.linspace(0, 1, 121), np.ones(3)]
        knots_v = np.r_[0.0, np.linspace(0, 1, 13), 1.0]
    else:
        degrees = (2, 2)
        knots_u = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0])
        knots_v = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    count_u, count_v = len(knots_u) - degrees[0] - 1, len(knots_v) - degrees[1] - 1
    u, v = np.meshgrid(np.linspace(0, 1, count_u), np.linspace(0, 1, count_v))
    controls = np.c_[0.06 * np.sin(2 * np.pi * (15 if waves else 2) * u.ravel()), 3 * u.ravel(), 3 * v.ravel()]
    weights = 1 + 0.2 * np.cos(np.arange(len(controls)))
    points = np.array(
        [[0.075, y, z] for y, z in [(-0.1, 0), (0.1, 0.5), (0.8, 1.2), (1.51, 2.4), (2.8, 3), (3.1, 3.1)]]
    )
    harness = _CachedSurfaceDistanceHarness(controls, weights, knots_u, knots_v, degrees, points)
    for deformed in (False, True):
        if deformed:
            controls[:, 0] += 0.015 * np.cos(3 * controls[:, 1]) * controls[:, 2]
            harness.surface.control_points_hat.from_numpy(controls)
            harness.surface.update_surface_bounds()
            harness.surface.update_span_bounds()
        harness.query(False, np.array([0.0, 0.0]))
        reference = harness.results.to_numpy()
        # Deliberately poor, out-of-domain and invalid hints must not suppress
        # another wave, boundary minimum, or the uncached Greville seed.
        for seed in ([0.99, 0.01], [-2.0, 3.0], [np.nan, np.nan]):
            harness.query(True, np.asarray(seed, dtype=np.float64))
            actual = harness.results.to_numpy()
            assert np.isfinite(actual).all()
            assert np.all((actual[:, :2] >= 0) & (actual[:, :2] <= 1))
            np.testing.assert_allclose(np.linalg.norm(actual[:, 3:], axis=1), actual[:, 2], rtol=1e-8, atol=2e-9)
            if np.isfinite(seed).all():
                # An extra seed can find a better basin near a knot boundary;
                # it must not worsen the original multistart distance.
                assert np.all(actual[:, 2] <= reference[:, 2] + 2e-9 + 1e-8 * reference[:, 2])
            else:
                np.testing.assert_allclose(actual[:, 2:], reference[:, 2:], rtol=1e-8, atol=2e-9)
                np.testing.assert_allclose(actual[:, :2], reference[:, :2], rtol=0, atol=2e-8)


def test_surface_projection_attempts_tensor_hint_seeds(monkeypatch):
    from src.physics_model.contact_model.ipc import NurbsContact

    harness = _SurfaceDistanceHarness()
    attempts = ti.field(ti.i32, shape=())
    seeds = ti.Vector.field(2, ti.f64, shape=3)

    @ti.func
    def record_candidate(
        start_knot_u,
        start_knot_v,
        start_ctrlpt,
        num_knot_u,
        num_knot_v,
        knot_vector_u,
        knot_vector_v,
        ctrlpts,
        control_directions,
        alpha,
        weight,
        point,
        seed_u,
        seed_v,
        lower_u,
        upper_u,
        lower_v,
        upper_v,
        basis,
    ):
        index = ti.atomic_add(attempts[None], 1)
        seeds[index] = ti.Vector([seed_u, seed_v])
        return seed_u, seed_v, 1.0, ti.Vector.zero(ti.f64, point.n), 1

    monkeypatch.setattr(NurbsContact, "_surface_projected_newton_candidate", record_candidate)

    @ti.kernel
    def query(hint: ti.types.vector(2, ti.f64), kind: ti.template()):
        seed = hint
        if ti.static(kind == "field"):
            seed = harness.parameter[None]
        elif ti.static(kind == "literal"):
            seed = ti.Vector([0.23, 0.67])
        get_distance_to_surface_fixed_dim(
            0,
            0,
            0,
            6,
            6,
            harness.knots_u,
            harness.knots_v,
            harness.ctrlpts,
            harness.weights,
            ti.Vector([0.37, 0.62, 0.2]),
            harness.basis,
            initial_parameter=seed,
        )

    # Without a hint, only the span midpoint and Greville seed run.
    harness.query(np.asarray([0.37, 0.62, 0.2]))
    assert attempts[None] == 2
    hint = np.asarray([0.23, 0.67])
    for kind in ("argument", "field", "literal"):
        attempts[None] = 0
        harness.parameter[None] = hint
        query(hint, kind)
        assert attempts[None] == 3, kind
        np.testing.assert_allclose(np.asarray(seeds[0]), hint, rtol=0, atol=1e-14)
    for invalid in ([np.nan, np.nan], [np.inf, 0.67]):
        attempts[None] = 0
        query(np.asarray(invalid), "argument")
        assert attempts[None] == 2


def test_cpu_nurbs_curve_closest_point():
    knots = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    ctrlpts = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    weights = np.asarray([1.0, 2.0, 1.0])
    point = np.asarray([0.37, 0.25])

    parameter, distance = closest_curve_point_py(2, knots, ctrlpts, weights, point)
    closest = eval_curve_py(2, knots, ctrlpts, weights, parameter)
    assert np.allclose(closest, [point[0], 0.0], atol=1.0e-6)
    assert np.isclose(distance, point[1], atol=1.0e-10)


def test_taichi_curve_distance_matches_cpu_geometry():
    harness = _CurveDistanceHarness()
    point = np.asarray([0.37, 0.25], dtype=np.float64)
    harness.query(point)

    parameter = float(harness.parameter[None])
    distance = float(harness.distance[None])
    residual = np.asarray(harness.residual[None])
    cpu_point = eval_curve_py(
        2,
        np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]]),
        np.ones(3),
        parameter,
    )
    assert np.isclose(parameter, point[0], atol=1.0e-10)
    assert np.isclose(distance, point[1], atol=1.0e-10)
    assert np.allclose(residual, cpu_point - point, atol=1.0e-10)


def test_moving_curve_query_matches_an_explicit_trial_geometry():
    harness = _CurveDistanceHarness()
    point = np.asarray([0.43, 0.31], dtype=np.float64)
    alpha = 0.37
    control_points = harness.ctrlpts.to_numpy()
    directions = np.asarray([[0.08, 0.15], [-0.04, 0.22], [0.11, 0.06]], dtype=np.float64)
    harness.control_directions.from_numpy(directions)
    harness.query_moving(point, alpha)
    moving_result = (
        float(harness.parameter[None]),
        float(harness.distance[None]),
        np.asarray(harness.residual[None]),
    )

    harness.ctrlpts.from_numpy(control_points + alpha * directions)
    harness.query(point)
    assert np.isclose(moving_result[0], harness.parameter[None], atol=1.0e-12)
    assert np.isclose(moving_result[1], harness.distance[None], atol=1.0e-12)
    np.testing.assert_allclose(moving_result[2], np.asarray(harness.residual[None]), atol=1.0e-12)


@pytest.mark.parametrize(
    "point, expected_parameter",
    [
        (np.asarray([-0.2, 0.3], dtype=np.float64), 0.0),
        (np.asarray([1.4, -0.2], dtype=np.float64), 1.0),
    ],
)
def test_cpu_and_taichi_curve_closest_points_select_exact_endpoints(point, expected_parameter):
    knots = np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    ctrlpts = np.asarray([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    weights = np.ones(3)
    cpu_parameter, cpu_distance = closest_curve_point_py(2, knots, ctrlpts, weights, point)
    assert cpu_parameter == expected_parameter

    harness = _CurveDistanceHarness()
    harness.query(point)
    parameter = float(harness.parameter[None])
    distance = float(harness.distance[None])
    residual = np.asarray(harness.residual[None])
    expected_residual = ctrlpts[int(expected_parameter * 2)] - point
    assert parameter == expected_parameter
    assert np.isclose(distance, cpu_distance, atol=1.0e-12)
    np.testing.assert_allclose(residual, expected_residual, atol=1.0e-12)


def test_curve_closest_point_handles_a_degenerate_zero_tangent():
    harness = _CurveDistanceHarness()
    constant = np.asarray([0.25, -0.5], dtype=np.float64)
    ctrlpts = np.repeat(constant[None, :], 3, axis=0)
    harness.ctrlpts.from_numpy(ctrlpts)
    point = np.asarray([1.25, 1.0], dtype=np.float64)
    harness.query(point)

    parameter = float(harness.parameter[None])
    distance = float(harness.distance[None])
    residual = np.asarray(harness.residual[None])
    assert 0.0 <= parameter <= 1.0
    assert np.isfinite(distance)
    np.testing.assert_allclose(residual, constant - point, atol=1.0e-12)
    assert np.isclose(distance, np.linalg.norm(constant - point), atol=1.0e-12)

    cpu_parameter, cpu_distance = closest_curve_point_py(
        2,
        np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        ctrlpts,
        np.ones(3),
        point,
    )
    assert 0.0 <= cpu_parameter <= 1.0
    assert np.isclose(cpu_distance, distance, atol=1.0e-12)


def test_multispan_curve_and_surface_choose_the_global_of_two_local_minima():
    harness = _MultiSpanClosestHarness()
    curve_point = np.asarray([0.1, 1.0], dtype=np.float64)
    surface_point = np.asarray([0.1, 0.42, 1.0], dtype=np.float64)
    harness.query(curve_point, surface_point)

    knots = harness.knots_u.to_numpy()
    ctrlpts = harness.curve_ctrlpts.to_numpy()
    weights = harness.curve_weights.to_numpy()
    cpu_parameter, cpu_distance = closest_curve_point_py(2, knots, ctrlpts, weights, curve_point)
    samples = np.linspace(0.0, 1.0, 20001)
    sampled_distance2 = np.asarray(
        [np.sum((eval_curve_py(2, knots, ctrlpts, weights, parameter) - curve_point) ** 2) for parameter in samples]
    )
    local_minima = (
        np.flatnonzero(
            (sampled_distance2[1:-1] <= sampled_distance2[:-2]) & (sampled_distance2[1:-1] <= sampled_distance2[2:])
        )
        + 1
    )
    assert local_minima.size >= 2
    assert cpu_parameter > 0.5

    taichi_parameter = float(harness.curve_parameter[None])
    taichi_distance = float(harness.curve_distance[None])
    assert np.isclose(taichi_parameter, cpu_parameter, atol=2.0e-8)
    assert np.isclose(taichi_distance, cpu_distance, atol=2.0e-9)

    surface_parameter = np.asarray(harness.surface_parameter[None])
    surface_distance = float(harness.surface_distance[None])
    surface_residual = np.asarray(harness.surface_residual[None])
    assert np.isclose(surface_parameter[0], cpu_parameter, atol=2.0e-8)
    assert np.isclose(surface_parameter[1], surface_point[1], atol=2.0e-8)
    assert np.isclose(surface_distance, cpu_distance, atol=2.0e-9)
    assert np.isclose(surface_residual[1], 0.0, atol=2.0e-9)


def test_cpu_curve_closest_rejects_nonfinite_geometry():
    with pytest.raises(ValueError, match="finite"):
        closest_curve_point_py(
            2,
            np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            np.asarray([[0.0, 0.0], [np.nan, 0.0], [1.0, 0.0]]),
            np.ones(3),
            np.asarray([0.5, 0.2]),
        )


def test_taichi_surface_closest_point_and_rational_basis_derivatives():
    harness = _SurfaceDistanceHarness()
    harness.query(np.asarray([0.37, 0.62, 0.2], dtype=np.float64))
    parameter = np.asarray(harness.parameter[None])
    assert np.all((0.0 <= parameter) & (parameter <= 1.0))
    assert np.isclose(float(harness.distance[None]), 0.2, atol=1.0e-9)
    assert np.allclose(harness.residual[None], [0.0, 0.0, -0.2], atol=1.0e-8)

    u, v = 0.37, 0.61
    step = 1.0e-6
    harness.evaluate_hessian_basis(u, v)
    derivative_u = harness.derivative_u.to_numpy()
    derivative_v = harness.derivative_v.to_numpy()
    harness.evaluate_shape(u + step, v)
    shape_u_plus = harness.shape.to_numpy()
    harness.evaluate_shape(u - step, v)
    shape_u_minus = harness.shape.to_numpy()
    harness.evaluate_shape(u, v + step)
    shape_v_plus = harness.shape.to_numpy()
    harness.evaluate_shape(u, v - step)
    shape_v_minus = harness.shape.to_numpy()
    np.testing.assert_allclose(
        derivative_u,
        (shape_u_plus - shape_u_minus) / (2.0 * step),
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    np.testing.assert_allclose(
        derivative_v,
        (shape_v_plus - shape_v_minus) / (2.0 * step),
        rtol=2.0e-6,
        atol=2.0e-8,
    )
    assert np.isclose(derivative_u.sum(), 0.0, atol=1.0e-10)
    assert np.isclose(derivative_v.sum(), 0.0, atol=1.0e-10)


def test_surface_closest_point_recovers_when_newton_step_projects_to_zero():
    harness = _SurfaceDistanceHarness()
    tangent_u = np.asarray([1.0, 0.0, 0.0])
    tangent_v = np.asarray([0.995, 0.1, 0.0])
    control_points = np.asarray([0.5 * i * tangent_u + 0.5 * j * tangent_v for j in range(3) for i in range(3)])
    harness.ctrlpts.from_numpy(control_points)
    harness.weights.fill(1.0)
    point = -0.5 * tangent_u + 0.8 * tangent_v + np.asarray([0.0, 0.0, 0.2])

    harness.query(point)

    expected_v = float(np.dot(tangent_v, point) / np.dot(tangent_v, tangent_v))
    parameter = np.asarray(harness.parameter[None])
    assert np.isfinite(float(harness.distance[None]))
    np.testing.assert_allclose(parameter, [0.0, expected_v], atol=1.0e-9)


def test_moving_surface_query_matches_an_explicit_trial_geometry():
    harness = _SurfaceDistanceHarness()
    point = np.asarray([0.37, 0.62, 0.43], dtype=np.float64)
    alpha = 0.41
    control_points = harness.ctrlpts.to_numpy()
    directions = np.asarray(
        [[0.03 * i, -0.02 * j, 0.08 + 0.025 * i * j] for j in range(3) for i in range(3)],
        dtype=np.float64,
    )
    harness.control_directions.from_numpy(directions)
    harness.query_moving(point, alpha)
    moving_result = (
        np.asarray(harness.parameter[None]),
        float(harness.distance[None]),
        np.asarray(harness.residual[None]),
    )

    harness.ctrlpts.from_numpy(control_points + alpha * directions)
    harness.query(point)
    np.testing.assert_allclose(moving_result[0], np.asarray(harness.parameter[None]), atol=1.0e-11)
    assert np.isclose(moving_result[1], harness.distance[None], atol=1.0e-11)
    np.testing.assert_allclose(moving_result[2], np.asarray(harness.residual[None]), atol=1.0e-11)


def test_curve_and_surface_distance_derivatives_smoke():
    harness = _DerivativeProjectionHarness()
    harness.evaluate_curve_derivatives()
    harness.evaluate_surface_derivatives()

    assert np.allclose(harness.point_gradient2[None], [0.0, 0.5])
    assert np.allclose(harness.ctrl_gradient2[None], [0.0, -0.125])
    assert np.allclose(harness.point_hessian2[None], np.diag([0.0, 2.0]))
    assert np.allclose(harness.ctrl_hessian2[None], np.diag([0.0, 0.125]))
    assert np.allclose(harness.cross_hessian2[None], np.diag([0.0, -0.5]))
    assert np.allclose(harness.velocity_point2[None], np.eye(2))
    assert np.allclose(harness.velocity_ctrl2[None], -0.25 * np.eye(2))

    assert np.allclose(harness.point_gradient3[None], [0.0, 0.0, 0.6])
    assert np.allclose(harness.ctrl_gradient3[None], [0.0, 0.0, -0.15])
    assert np.allclose(harness.point_hessian3[None], np.diag([0.0, 0.0, 2.0]))
    assert np.allclose(harness.ctrl_hessian3[None], np.diag([0.0, 0.0, 0.125]))
    assert np.allclose(harness.cross_hessian3[None], np.diag([0.0, 0.0, -0.5]))


def test_boundary_active_set_uses_curve_and_endpoint_hessians():
    harness = _DerivativeProjectionHarness()

    harness.evaluate_active_boundary_derivatives(0, 0)
    np.testing.assert_allclose(harness.point_hessian2[None], 2.0 * np.eye(2))
    np.testing.assert_allclose(harness.ctrl_hessian2[None], 0.125 * np.eye(2))
    np.testing.assert_allclose(harness.cross_hessian2[None], -0.5 * np.eye(2))
    np.testing.assert_allclose(harness.point_hessian3[None], 2.0 * np.eye(3))

    harness.evaluate_active_boundary_derivatives(1, 0)
    np.testing.assert_allclose(harness.point_hessian3[None], np.diag([0.0, 2.0, 2.0]))

    harness.evaluate_active_boundary_derivatives(1, 1)
    np.testing.assert_allclose(harness.point_hessian3[None], np.diag([0.0, 0.0, 2.0]))


def test_psd_projection_is_dimension_generic():
    harness = _DerivativeProjectionHarness()
    harness.evaluate_psd_projection()
    for projected in (
        np.asarray(harness.projected2[None]),
        np.asarray(harness.projected3[None]),
    ):
        assert np.allclose(projected, projected.T, atol=1.0e-12)
        assert np.min(np.linalg.eigvalsh(projected)) >= -1.0e-12


@pytest.mark.isolated_dimension(3)
def test_surface_projection_uses_field_parameter_seed(taichi_runtime, monkeypatch):
    from src.physics_model.contact_model.ipc import NurbsContact

    checked_seed = ti.field(ti.i32, shape=())
    original_finite = NurbsContact._isfinite_vector

    @ti.func
    def trace_finite(vector):
        if ti.static(vector.n == 2):
            if vector[0] == -2.0 and vector[1] == 3.0:
                ti.atomic_add(checked_seed[None], 1)
        return original_finite(vector)

    monkeypatch.setattr(NurbsContact, "_isfinite_vector", trace_finite)
    controls = np.array([[0.0, u, v] for v in (0.0, 1.0) for u in (0.0, 1.0)])
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    harness = _CachedSurfaceDistanceHarness(controls, np.ones(4), knots, knots, (1, 1), np.array([[0.1, 0.4, 0.4]]))
    harness.query(True, np.array([-2.0, 3.0]))
    # On this plane, Newton gradients and clamped parameters cannot equal
    # the out-of-domain hint. A check records the field seed's own dispatch.
    assert checked_seed[None] > 0
    np.testing.assert_allclose(harness.results.to_numpy()[0, :3], [0.4, 0.4, 0.1], atol=1e-12)
