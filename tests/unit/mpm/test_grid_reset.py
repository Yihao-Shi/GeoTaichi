"""Sub-cutoff deposits belong to one assembly, never to the next timestep."""

import numpy as np
import pytest

from src.mpm.engines.EngineKernel import grid_internal_force_reset, grid_mass_reset, grid_reset, tlgrid_reset
from src.mpm.structs.GridNode import ContactNodes2D, ImplicitNodes2D

pytestmark = pytest.mark.isolated_dimension(2)


@pytest.mark.parametrize("reset", [grid_reset, tlgrid_reset, grid_mass_reset, grid_internal_force_reset])
def test_reset_includes_mass_below_and_at_solve_cutoff(taichi_runtime, reset):
    cutoff = 1e-7
    cls = ImplicitNodes2D if reset is grid_internal_force_reset else ContactNodes2D
    node = cls.field(shape=(3, 1))
    masses = np.array([cutoff / 100, cutoff, cutoff * 10])
    force = node.int_force if reset is grid_internal_force_reset else node.force
    for i, mass in enumerate(masses):
        node[i, 0].m = mass
        force[i, 0] = [2, -3]
        if reset is not grid_internal_force_reset:
            node[i, 0].momentum = [4, 5]
            node[i, 0].contact_force = [6, 7]
            node[i, 0].grad_domain = [8, 9]
    reset(cutoff, node)
    if reset in (grid_reset, grid_mass_reset):
        np.testing.assert_array_equal(node.m.to_numpy(), 0)
    else:
        np.testing.assert_allclose(node.m.to_numpy().ravel(), masses, rtol=1e-6)
    if reset is not grid_mass_reset:
        np.testing.assert_array_equal(force.to_numpy(), 0)
    if reset in (grid_reset, tlgrid_reset):
        for field in (node.momentum, node.contact_force, node.grad_domain):
            np.testing.assert_array_equal(field.to_numpy(), 0)
