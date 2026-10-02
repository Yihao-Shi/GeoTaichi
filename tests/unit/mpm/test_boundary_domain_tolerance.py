from types import SimpleNamespace

import numpy as np
import pytest

from src.mpm.boundaries.BoundaryConstraint import BoundaryConstraints


def test_boundary_domain_accepts_decimal_endpoint_after_float32_rounding():
    boundary = BoundaryConstraints()
    simulation = SimpleNamespace(dimension=2, domain=np.asarray([0.4, 0.12], dtype=np.float32))

    boundary.check_boundary_domain(simulation, [0.0, 0.0], [0.4, 0.12])

    with pytest.raises(RuntimeError, match="EndPoint"):
        boundary.check_boundary_domain(simulation, [0.0, 0.0], [0.4, 0.121])
