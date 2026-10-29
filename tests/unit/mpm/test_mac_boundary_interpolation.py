"""Roundoff at a domain node must not remove its B-spline endpoint weight."""

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.TwoPhaseDoubleLayerKernel import _mac_weight_1d


@pytest.mark.parametrize("boundary", [1, 4])
def test_quadratic_mac_endpoint_weight_survives_roundoff(taichi_runtime, boundary):
    result = ti.field(ti.f64, shape=())

    @ti.kernel
    def sample(position: ti.f64):
        result[None] = _mac_weight_1d(position, 0.2, 200.0, 0.0025, boundary, 2)

    for position in [0.2, np.nextafter(0.2, -np.inf), np.nextafter(0.2, np.inf)]:
        sample(position)
        assert result[None] == pytest.approx(1.0, abs=1e-13)
    sample(0.2 + (-1.0 if boundary == 1 else 1.0) * 0.0001)
    assert result[None] == 0.0
