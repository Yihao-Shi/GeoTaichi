"""Compile-time shape selection must retain every existing runtime basis value."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import _mac_weight_1d


@ti.kernel
def compare(shape: ti.template(), runtime_shape: ti.template(), values: ti.types.ndarray()):
    for i, boundary in ti.ndrange(values.shape[0], 5):
        xp = (i - 50) * 0.004
        values[i, boundary, 0] = _mac_weight_1d(xp, 0.0, 10.0, 0.025, boundary, shape)
        values[i, boundary, 1] = _mac_weight_1d(xp, 0.0, 10.0, 0.025, boundary, runtime_shape[None])


@pytest.mark.parametrize("shape", [0, 1, 2, 3])
def test_specialized_shape_matches_runtime_basis_at_all_boundaries(taichi_runtime, shape):
    runtime = ti.field(ti.i32, shape=())
    runtime[None] = shape
    values = np.zeros((101, 5, 2))
    compare(shape, runtime, values)
    np.testing.assert_allclose(values[:, :, 0], values[:, :, 1], rtol=1e-13, atol=1e-13)
