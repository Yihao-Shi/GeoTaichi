"""Deterministic curve and surface fitting checks for the IGA layer."""

from src.nurbs.Utilities import compute_params_curve
from src.nurbs.Utilities import compute_params_surface
from src.nurbs.Utilities import global_approximate_bspline_curve
from src.nurbs.Utilities import global_approximate_bspline_surface
from src.nurbs.Utilities import global_interpolate_bspline_curve
from src.nurbs.Utilities import global_interpolate_bspline_surface

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from src.nurbs import Utilities
from src.nurbs.NurbsVolume import NurbsVolume
from src.nurbs.MIPS import MIPS
from src.nurbs.SurfacePoints import SurfacePoints
from src.nurbs.element.NurbsSurfaceElement import NurbsSurfaceElement


def test_global_curve_interpolation_reproduces_every_data_point():
    points = np.asarray([[0.0, 0.0], [3.0, 4.0], [-1.0, 4.0], [-4.0, 0.0], [-5.0, -1.0]])
    parameters = compute_params_curve(points)
    curve = global_interpolate_bspline_curve(points=points, degree=3)

    interpolated = np.asarray([curve.single_point(parameter) for parameter in parameters])
    np.testing.assert_allclose(interpolated, points, rtol=0.0, atol=5.0e-12)


def test_global_curve_approximation_preserves_fixed_endpoints():
    points = np.asarray([[0.0, 0.0], [3.0, 4.0], [-1.0, 4.0], [-4.0, 0.0], [-5.0, -1.0]])
    curve = global_approximate_bspline_curve(points=points, degree=3)
    distances = curve.distance(points)

    assert curve.control_points.shape == (4, 2)
    np.testing.assert_allclose(curve.single_point(0.0), points[0], atol=2.0e-13)
    np.testing.assert_allclose(curve.single_point(1.0), points[-1], atol=2.0e-12)
    assert np.all(np.isfinite(distances))
    assert distances[0] < 2.0e-12
    assert distances[-1] < 2.0e-12


def _bilinear_data_grid():
    return np.asarray([[u, v, 0.2 * u * v] for v in (0.0, 1.0, 2.0, 4.0) for u in (0.0, 1.0, 2.0, 3.0)])


def test_global_surface_interpolation_reproduces_every_data_point():
    points = _bilinear_data_grid()
    reshaped = points.reshape((4, 4, 3))
    parameters_u, parameters_v = compute_params_surface(reshaped, 4, 4)
    surface = global_interpolate_bspline_surface(
        degree_u=2,
        degree_v=2,
        num_datapt_u=4,
        num_datapt_v=4,
        points=points,
    )

    interpolated = np.asarray(
        [surface.single_point(parameter_u, parameter_v) for parameter_v in parameters_v for parameter_u in parameters_u]
    )
    np.testing.assert_allclose(interpolated, points, rtol=0.0, atol=5.0e-12)


def test_global_surface_approximation_uses_consistent_u_v_dimensions():
    parameters_u = (0.0, 0.5, 1.5, 2.5, 3.0)
    parameters_v = (0.0, 1.0, 2.0, 4.0)
    points = np.asarray([[u, v, 0.2 * u * v] for v in parameters_v for u in parameters_u])
    surface = global_approximate_bspline_surface(
        degree_u=2,
        degree_v=2,
        num_datapt_u=5,
        num_datapt_v=4,
        points=points,
    )

    assert surface.control_points.shape == (12, 3)
    np.testing.assert_allclose(surface.single_point(0.0, 0.0), points[0], rtol=0.0, atol=2.0e-13)
    np.testing.assert_allclose(surface.single_point(1.0, 1.0), points[-1], rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(surface.distance(points), 0.0, rtol=0.0, atol=3.0e-12)


def _rectangle_boundary():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 0.5], [2.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.5]])
    return points, points[[0, 2, 4, 6]]


@pytest.mark.parametrize("clockwise", [False, True])
def test_four_corner_input_splits_a_closed_boundary_without_a_gui(monkeypatch, clockwise):
    points, corners = _rectangle_boundary()
    monkeypatch.setattr(Utilities, "select_boundary_corners", lambda *_: pytest.fail("Unexpected GUI"))
    if clockwise:
        points = points[::-1]
    points = np.vstack((points, points[0]))
    surface = Utilities.fit_surface_from_boundary(points, corner_points=corners[[0, 2, 1, 3]])
    for u, v in ((0.0, 0.0), (0.2, 0.7), (0.75, 0.1), (1.0, 1.0)):
        np.testing.assert_allclose(surface.single_point(u, v), [2 * u, v], rtol=0, atol=2e-12)
    assert Utilities.validate_spline_jacobian(surface) == pytest.approx(2.0, abs=2e-12)


def test_missing_corners_open_the_picker_and_snap_to_boundary_vertices(monkeypatch):
    points, corners = _rectangle_boundary()
    calls = []
    monkeypatch.setattr(Utilities.plt, "get_backend", lambda: "TkAgg")

    def pick(count, timeout):
        calls.append((count, timeout))
        return corners + 1e-3

    monkeypatch.setattr(Utilities.plt, "ginput", pick)
    figures_before = Utilities.plt.get_fignums()
    surface = Utilities.fit_surface_from_boundary(points)
    assert calls == [(4, 0)]
    assert Utilities.plt.get_fignums() == figures_before
    np.testing.assert_allclose(surface.single_point(0.0, 0.0), corners[0], atol=1e-13)


def test_cancelled_or_repeated_corner_selection_is_rejected(monkeypatch):
    points, corners = _rectangle_boundary()
    monkeypatch.setattr(Utilities.plt, "get_backend", lambda: "TkAgg")
    monkeypatch.setattr(Utilities.plt, "ginput", lambda *_args, **_kwargs: corners[:3])
    with pytest.raises(ValueError, match="cancelled"):
        Utilities.fit_surface_from_boundary(points)
    with pytest.raises(ValueError, match="distinct"):
        Utilities.fit_surface_from_boundary(points, corners[[0, 1, 1, 3]])


def _cubic_surface(x_coordinates=(0.0, 0.2, 0.8, 1.0)):
    points = np.array([[x, y] for y in (0.0, 0.5, 1.0) for x in x_coordinates])
    return NurbsVolume.nurbs_surface(3, 2, [0.0] * 4 + [1.0] * 4, [0.0] * 3 + [1.0] * 3, points)


def test_folded_control_net_is_rejected_even_when_the_spline_jacobian_is_positive():
    surface = _cubic_surface((0.0, 0.8, 0.2, 1.0))
    assert np.diff(surface.control_points[:4, 0]).min() < 0.0
    # x'(u)=3[.8(1-u)^2-1.2u(1-u)+.8u^2], whose minimum is .3.
    element = NurbsSurfaceElement(_cubic_surface())
    assert np.min(element.jacobian_determinants(surface.control_points)) == pytest.approx(0.3, abs=2e-12)
    with pytest.raises(ValueError, match="Control net"):
        Utilities.validate_spline_jacobian(surface)
    folded = _cubic_surface((0.0, 1.0, -0.8, 1.0))
    assert np.min(element.jacobian_determinants(folded.control_points)) < 0.0


def test_iga_gradient_and_control_point_derivatives_use_the_rational_basis():
    surface = _cubic_surface()
    surface.weights = np.linspace(0.7, 1.4, surface.num_ctrlpts)
    element = NurbsSurfaceElement(surface, energy=MIPS())
    gradient = element.compute_deformation_gradient(surface.control_points)
    from itertools import product

    gauss_u = 0.5 * (np.polynomial.legendre.leggauss(4)[0] + 1)
    gauss_v = 0.5 * (np.polynomial.legendre.leggauss(3)[0] + 1)
    step = 1e-6
    expected = []
    for u, v in product(gauss_u, gauss_v):
        du = (surface.single_point(u + step, v) - surface.single_point(u - step, v)) / (2 * step)
        dv = (surface.single_point(u, v + step) - surface.single_point(u, v - step)) / (2 * step)
        expected.append(np.column_stack((du, dv)))
    np.testing.assert_allclose(gradient, expected, rtol=2e-8, atol=2e-9)
    direction = np.random.default_rng(42).normal(size=surface.control_points.shape)
    derivatives = element.compute_ddeformation_dx()
    for index, conn in enumerate(element._sample_connectivity):
        predicted = derivatives[index] @ direction[conn].ravel()
        np.testing.assert_allclose(
            predicted, element.compute_deformation_gradient(direction)[index].ravel(order="F"), atol=2e-14
        )


def test_iga_step_cannot_cross_a_zero_jacobian_and_return_to_positive():
    surface = _cubic_surface()
    element = NurbsSurfaceElement(surface)
    change = -2 * surface.control_points
    step = element.compute_step_size(change, element.compute_deformation_gradient(surface.control_points))
    assert 0.0 < step < 0.5
    element.validate(surface.control_points + step * change)


def test_control_net_alone_can_limit_the_fitting_step():
    surface = _cubic_surface()
    element = NurbsSurfaceElement(surface)
    target = _cubic_surface((0.0, 0.8, 0.2, 1.0)).control_points
    assert np.min(element.jacobian_determinants(target)) > 0.0
    change = target - surface.control_points
    step = element.compute_step_size(change, element.compute_deformation_gradient(surface.control_points))
    assert 0.0 < step < 0.5
    element.validate(surface.control_points + step * change)


def test_iga_energy_gradient_and_hessian_match_directional_differences():
    surface = _cubic_surface()
    element = NurbsSurfaceElement(surface, energy=MIPS())
    points = surface.control_points
    direction = np.random.default_rng(7).normal(size=points.shape)
    derivative = element.compute_ddeformation_dx()
    current = element.compute_deformation_gradient(points)
    step = 1e-6
    plus = element.compute_deformation_gradient(points + step * direction)
    minus = element.compute_deformation_gradient(points - step * direction)
    gradient = element.surface_energy_gradient(current, derivative)
    expected = (element.surface_energy(plus) - element.surface_energy(minus)) / (2 * step)
    assert gradient @ direction.ravel() == pytest.approx(expected, rel=2e-7, abs=2e-8)
    hessian = element.surface_energy_hessian(current, derivative)
    expected = (
        element.surface_energy_gradient(plus, derivative) - element.surface_energy_gradient(minus, derivative)
    ) / (2 * step)
    np.testing.assert_allclose(hessian @ direction.ravel(), expected, rtol=2e-6, atol=2e-7)


def test_curved_boundary_fitting_and_extrusion_keep_both_orientation_constraints():
    u = np.linspace(0.0, 1.0, 31)
    bottom = np.column_stack((u, np.zeros_like(u)))
    right = np.column_stack((np.ones_like(u), u))
    top = np.column_stack((u, 1.0 + 0.1 * np.sin(np.pi * u)))
    left = np.column_stack((np.zeros_like(u), u))
    boundary = np.concatenate((bottom[:-1], right[:-1], top[:0:-1], left[:0:-1]))
    corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    surface = Utilities.fit_surface_from_boundary(boundary, corners, degree=(3, 2), num_ctrlpts=(6, 4))
    errors = [
        minimize_scalar(
            lambda parameter: np.sum((surface.single_point(parameter, 1.0) - point) ** 2),
            bounds=(0.0, 1.0),
            method="bounded",
        ).fun
        for point in top
    ]
    assert np.sqrt(max(errors)) < 1e-3
    assert Utilities.validate_spline_jacobian(surface) > 0.0
    volume = Utilities.extrude_surface(surface, [0.0, 0.0, 0.2], degree=2)
    assert Utilities.validate_spline_jacobian(volume) > 0.0


@pytest.mark.parametrize("degree", [1, 2])
def test_extrusion_preserves_the_rational_surface_exactly(degree, monkeypatch):
    from src.iga import config
    from src.iga.generator.Primitives import Primitives

    surface = _cubic_surface()
    surface.weights = np.linspace(0.7, 1.4, surface.num_ctrlpts)
    original = surface.control_points.copy()
    vector = np.array([0.2, -0.1, 2.0])
    volume = Utilities.extrude_surface(surface, vector, degree=degree)
    for u, v, w in ((0.1, 0.2, 0.0), (0.3, 0.7, 0.4), (0.8, 0.9, 1.0)):
        expected = np.r_[surface.single_point(u, v), 0.0] + w * vector
        np.testing.assert_allclose(volume.single_point(u, v, w), expected, rtol=0, atol=3e-12)
        np.testing.assert_allclose(volume.derivative(u, v, w)[2], vector, rtol=0, atol=3e-12)
    np.testing.assert_array_equal(volume.weights, np.tile(surface.weights, degree + 1))
    np.testing.assert_array_equal(surface.control_points, original)
    monkeypatch.setattr(config, "DIM", 3)
    primitives = Primitives()
    primitives.append(volume)
    primitives.finialize()
    assert primitives.num_ctrlpts == volume.num_ctrlpts
    assert primitives.num_element == list(volume.num_element_list)


@pytest.mark.parametrize("vector", [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, np.nan]])
def test_extrusion_rejects_degenerate_or_inverted_volumes(vector):
    with pytest.raises(ValueError):
        Utilities.extrude_surface(_cubic_surface(), vector)


@pytest.mark.parametrize(
    "name",
    [
        "bspline_surface_fitting_gauss_newton2d",
        "bspline_surface_fitting_newton_raphson2d",
        "bspline_surface_fitting_newton_raphson_four_points2d",
    ],
)
def test_existing_fitters_finish_with_positive_iga_jacobians(name):
    surface = _cubic_surface()
    points, corners = _rectangle_boundary()
    points = points.copy()
    corners = corners.copy()
    points[:, 0] *= 0.5
    corners[:, 0] *= 0.5
    fitting = SurfacePoints()
    fitting.setup(boundary=points, penalty=0.0)
    extra = {}
    if "four_points" in name:
        extra["boundary_categories"] = list(Utilities.split_boundary_points(None, points, corners).values())
    assert getattr(Utilities, name)(fitting, surface, tol=1.0, **extra) is surface


@pytest.mark.parametrize("projection_failure", [False, True])
def test_rejected_fitting_step_restores_the_surface_and_boundary_curves(monkeypatch, projection_failure):
    points, corners = _rectangle_boundary()
    surface = Utilities.fit_surface_from_boundary(points, corners)
    original = surface.control_points.copy()
    fitting = SurfacePoints()
    fitting.setup(boundary=points + [0.1, 0.0], penalty=0.0)
    change = np.tile([-0.1, 0.0], (surface.num_ctrlpts, 1)).ravel()
    monkeypatch.setattr(Utilities, "checked_scipy_krylov", lambda *_args, **_kwargs: change)
    failure_type, message = Utilities.NonlinearSolveError, "descent step"
    if projection_failure:
        original_inversion = surface.boundary_inversion
        calls = 0

        def fail_trial_projection(points):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError("trial projection failed")
            return original_inversion(points)

        monkeypatch.setattr(surface, "boundary_inversion", fail_trial_projection)
        failure_type, message = RuntimeError, "trial projection failed"
    with pytest.raises(failure_type, match=message):
        Utilities.bspline_surface_fitting_gauss_newton2d(fitting, surface, tol=0.01, iteration=1)
    np.testing.assert_array_equal(surface.control_points, original)
    np.testing.assert_array_equal(surface.boundary[0].control_points, original[: surface.num_ctrlpts_u])


@pytest.mark.parametrize(
    "points",
    [
        np.array([[0.0, 0.0], [2.0, 2.0], [0.0, 3.0], [3.0, 0.0]]),
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, np.nan], [0.0, 1.0]]),
    ],
)
def test_invalid_boundary_is_rejected_before_selecting_corners(points, monkeypatch):
    monkeypatch.setattr(Utilities, "select_boundary_corners", lambda *_: pytest.fail("Unexpected GUI"))
    with pytest.raises(ValueError):
        Utilities.fit_surface_from_boundary(points)
