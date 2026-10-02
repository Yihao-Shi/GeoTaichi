import numpy as np
import pytest
import taichi as ti

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]

from src.physics_model.contact_model.ipc.ContactMeasure import (
    lumped_boundary_vertex_measures,
    segment_length,
)
from src.physics_model.contact_model.ipc.ContactDistance import (
    edge_edge_distance_type,
    point_edge_distance_type,
)
from src.physics_model.contact_model.ipc.ContactGeometry import (
    edge_edge_distance_type_ipc,
    ipc_ee_type_to_legacy_py,
    legacy_ee_type_to_ipc_py,
)
from src.physics_model.contact_model.ipc.ContactMollifier import (
    edge_edge_mollifier,
    edge_edge_mollifier_grad_hess,
)


_positions = None
_gradient = None
_hessian = None


@pytest.fixture(scope="module", autouse=True)
def _initialize_taichi():
    global _positions, _gradient, _hessian
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, offline_cache=False)
    _positions = ti.Vector.field(3, ti.f64, shape=4)
    _gradient = ti.field(ti.f64, shape=12)
    _hessian = ti.field(ti.f64, shape=(12, 12))
    yield
    ti.reset()


@ti.kernel
def _evaluate(eps_x: ti.f64) -> ti.f64:
    value = edge_edge_mollifier(_positions[0], _positions[1], _positions[2], _positions[3], eps_x)
    gradient, hessian = edge_edge_mollifier_grad_hess(_positions[0], _positions[1], _positions[2], _positions[3], eps_x)
    for i in range(12):
        _gradient[i] = gradient[i]
        for j in range(12):
            _hessian[i, j] = hessian[i, j]
    return value


@ti.kernel
def _evaluate_ee_types() -> ti.types.vector(2, ti.i32):
    return ti.Vector(
        [
            edge_edge_distance_type_ipc(_positions[0], _positions[1], _positions[2], _positions[3]),
            edge_edge_distance_type(_positions[0], _positions[1], _positions[2], _positions[3]),
        ]
    )


@ti.kernel
def _evaluate_point_edge_type() -> ti.i32:
    return point_edge_distance_type(_positions[0], _positions[1], _positions[2])


def _value(flat, eps_x):
    _positions.from_numpy(np.asarray(flat, dtype=np.float64).reshape(4, 3))
    return float(_evaluate(eps_x))


def test_edge_edge_mollifier_gradient_and_hessian_match_finite_difference():
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.1, 0.2, 0.0], [1.1, 0.2, 0.025]],
        dtype=np.float64,
    ).reshape(-1)
    eps_x = 1.0e-3
    _value(positions, eps_x)
    gradient = _gradient.to_numpy()
    hessian = _hessian.to_numpy()
    step = 2.0e-6
    fd_gradient = np.zeros(12)
    fd_hessian = np.zeros((12, 12))
    for j in range(12):
        direction = np.zeros(12)
        direction[j] = step
        fd_gradient[j] = (_value(positions + direction, eps_x) - _value(positions - direction, eps_x)) / (2.0 * step)
        _value(positions + direction, eps_x)
        gradient_plus = _gradient.to_numpy()
        _value(positions - direction, eps_x)
        gradient_minus = _gradient.to_numpy()
        fd_hessian[:, j] = (gradient_plus - gradient_minus) / (2.0 * step)

    np.testing.assert_allclose(gradient, fd_gradient, rtol=2.0e-6, atol=2.0e-8)
    np.testing.assert_allclose(hessian, fd_hessian, rtol=5.0e-5, atol=2.0e-6)
    np.testing.assert_allclose(hessian, hessian.T, rtol=0.0, atol=1.0e-12)


def test_degenerate_rest_edge_has_finite_unmollified_value():
    positions = np.zeros((4, 3), dtype=np.float64)
    assert _value(positions, 0.0) == 1.0
    np.testing.assert_array_equal(_gradient.to_numpy(), np.zeros(12))
    np.testing.assert_array_equal(_hessian.to_numpy(), np.zeros((12, 12)))


def test_2d_boundary_segment_lumping():
    vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]])
    edges = np.array([[0, 1], [1, 2]])
    assert segment_length(vertices[[0, 1]]) == pytest.approx(2.0)
    np.testing.assert_allclose(lumped_boundary_vertex_measures(vertices, edges), [1.0, 2.5, 1.5])


def test_ee_feature_enum_translation_round_trip():
    for legacy_type in range(9):
        official_type = legacy_ee_type_to_ipc_py(legacy_type)
        assert ipc_ee_type_to_legacy_py(official_type) == legacy_type


def test_official_ee_classifier_handles_degenerate_and_parallel_regressions():
    cases = [
        # Both edges are points -> official EA0_EB0.
        (np.zeros((4, 3)), 0),
        # A is a point, B is a segment -> official EA0_EB.
        (np.array([[0, 0, 0], [0, 0, 0], [-1, 1, 0], [1, 1, 0.0]]), 6),
        # B is a point, A is a segment -> official EA_EB0.
        (np.array([[-1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0.0]]), 4),
        # Degenerate coordinates that previously triggered a crash.
        (
            np.array(
                [
                    [-0.81818181276321411, 0.073941159961546266, 0.090909108519554152],
                    [-0.81818181276321411, 0.073941161500775773, 0.272727280855178830],
                    [-0.81818181276321411, 0.073941163540152718, 0.454545468091964780],
                    [-0.81818181276321411, 0.073941167300585323, 0.636363625526428220],
                ]
            ),
            2,
        ),
    ]
    for positions, expected_official in cases:
        _positions.from_numpy(np.asarray(positions, dtype=np.float64))
        official_type, legacy_type = map(int, _evaluate_ee_types())
        assert official_type == expected_official
        assert legacy_ee_type_to_ipc_py(legacy_type) == expected_official


def test_official_point_edge_classifier_uses_endpoint_for_zero_length_edge():
    _positions.from_numpy(np.zeros((4, 3), dtype=np.float64))
    assert int(_evaluate_point_edge_type()) == 0
