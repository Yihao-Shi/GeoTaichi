"""Quaternion rotation contracts extracted from the legacy print-only probe."""

import numpy as np
import pytest
import taichi as ti

from src.utils.Quaternion import rodrigues_rotation_matrix, set_from_two_vec, set_to_rotate

pytestmark = [pytest.mark.cpu, pytest.mark.geometry]


@ti.kernel
def _rotation_between(
    source: ti.types.vector(3, ti.f64),
    target: ti.types.vector(3, ti.f64),
    rotation: ti.template(),
):
    rotation[None] = set_to_rotate(set_from_two_vec(source, target))


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ([0.0, 0.0, 1.0], [0.0, 0.0, 1.0]),
        ([0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        ([1.0, 2.0, -3.0], [-2.0, 1.0, 4.0]),
    ],
    ids=("parallel", "quarter-turn", "antiparallel", "oblique"),
)
def test_rotation_from_two_vectors_is_proper_and_maps_direction(taichi_runtime, source, target):
    rotation = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())
    _rotation_between(source, target, rotation)
    matrix = rotation.to_numpy()
    source_unit = np.asarray(source) / np.linalg.norm(source)
    target_unit = np.asarray(target) / np.linalg.norm(target)

    np.testing.assert_allclose(matrix @ source_unit, target_unit, rtol=0.0, atol=3.0e-7)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), rtol=0.0, atol=3.0e-7)
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=3.0e-7)


@pytest.mark.parametrize("dtype,tolerance", [(ti.f32, 5e-7), (ti.f64, 1e-13)])
def test_rodrigues_preserves_requested_precision_in_host_and_device(taichi_runtime, dtype, tolerance):
    source = np.array([1.0, 2.0, 3.0])
    source /= np.linalg.norm(source)
    target = np.array([-2.0, 1.0, 4.0])
    target /= np.linalg.norm(target)
    result = ti.Matrix.field(3, 3, dtype=dtype, shape=())

    @ti.kernel
    def rotate(first: ti.types.vector(3, ti.f64), second: ti.types.vector(3, ti.f64)):
        result[None] = rodrigues_rotation_matrix(first, second, dtype)

    rotate(source, target)
    matrix = result.to_numpy()
    host = rodrigues_rotation_matrix(ti.Vector(source), ti.Vector(target), dtype).to_numpy()
    np.testing.assert_allclose(matrix, host, atol=tolerance)
    np.testing.assert_allclose(matrix @ source, target, atol=tolerance)
    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=tolerance)
