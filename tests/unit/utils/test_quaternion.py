"""Quaternion rotation contracts extracted from the legacy print-only probe."""

import numpy as np
import pytest
import taichi as ti

from src.utils.Quaternion import SetFromTwoVec, SetToRotate


pytestmark = [pytest.mark.cpu, pytest.mark.geometry]


@ti.kernel
def _rotation_between(
    source: ti.types.vector(3, ti.f64),
    target: ti.types.vector(3, ti.f64),
    rotation: ti.template(),
):
    rotation[None] = SetToRotate(SetFromTwoVec(source, target))


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
def test_rotation_from_two_vectors_is_proper_and_maps_direction(
    taichi_runtime, source, target
):
    rotation = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())
    _rotation_between(source, target, rotation)
    matrix = rotation.to_numpy()
    source_unit = np.asarray(source) / np.linalg.norm(source)
    target_unit = np.asarray(target) / np.linalg.norm(target)

    np.testing.assert_allclose(
        matrix @ source_unit, target_unit, rtol=0.0, atol=3.0e-7
    )
    np.testing.assert_allclose(
        matrix.T @ matrix, np.eye(3), rtol=0.0, atol=3.0e-7
    )
    assert np.linalg.det(matrix) == pytest.approx(1.0, abs=3.0e-7)
