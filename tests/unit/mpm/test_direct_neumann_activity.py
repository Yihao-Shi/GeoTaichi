"""Nodal loads must not transfer to another node when their node empties."""

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.serial]

dimensions = [
    pytest.param(2, marks=pytest.mark.isolated_dimension(2)),
    pytest.param(3, marks=pytest.mark.isolated_dimension(3)),
]


@pytest.mark.parametrize("dimension", dimensions)
@pytest.mark.parametrize("solver", ["implicit", "explicit"])
def test_direct_neumann_uses_only_mass_active_nodes(taichi_runtime, dimension, solver):
    ti = taichi_runtime
    from src.mpm.boundaries.BoundaryCondition import NeumannBoundary
    from src.mpm.engines.direct.ExplicitMPM import ExplicitMPM
    from src.mpm.engines.direct.ImplicitMPM import ImplicitMPM
    from src.utils.PrefixSum import PrefixSumExecutor

    node_count = 7
    dof_count = dimension * node_count
    engine_type = ImplicitMPM if solver == "implicit" else ExplicitMPM
    engine = object.__new__(engine_type)
    engine.val_lim = 1.0e-12
    grid_type = ti.types.struct(m=ti.f64, a=ti.types.vector(dimension, ti.f64))
    engine.grid = grid_type.field(shape=node_count)
    loads = np.arange(1, dof_count + 1, dtype=np.float64).reshape(node_count, dimension)
    loads[:, -1] *= -1
    # Include a zero load on a node that will be empty (0/0 in the old explicit kernel).
    loads[0, 0] = 0.0
    engine.neumann = NeumannBoundary()
    engine.neumann.append([list(range(dof_count))], loads.ravel().tolist())
    engine.neumann.finalize()

    if solver == "implicit":
        engine.node2dof = ti.field(ti.i32, shape=node_count)
        engine.rhs = ti.field(ti.f64, shape=dof_count)
        engine.energy = ti.field(ti.f64, shape=())
        displacement = ti.field(ti.f64, shape=dof_count)
        trial_displacement = np.arange(1, dof_count + 1, dtype=np.float64)
        displacement.from_numpy(trial_displacement)
        scan = PrefixSumExecutor(node_count)

    mixed_mass = np.array([0.0, 2.0, 0.0, engine.val_lim, 0.5 * engine.val_lim, 4.0, 0.0])
    # Exercise loss of support and reactivation without rebuilding the boundary list.
    for mass in (np.ones(node_count), mixed_mass, np.zeros(node_count), mixed_mass):
        engine.grid.m.from_numpy(mass)
        active = mass > engine.val_lim
        if solver == "implicit":
            engine.find_active_node()
            scan.run(engine.node2dof)
            engine.rhs.fill(3.0)
            engine.energy[None] = 5.0
            engine.apply_neumann()
            engine.get_neumann_energy(displacement)
            expected_load = np.zeros(dof_count)
            expected_load[: dimension * active.sum()] = loads[active].ravel()
            np.testing.assert_array_equal(engine.rhs.to_numpy(), 3.0 + expected_load)
            assert engine.energy[None] == 5.0 - np.dot(expected_load, trial_displacement)
        else:
            engine.grid.a.fill(3.0)
            engine.apply_neumann()
            expected_acceleration = np.full((node_count, dimension), 3.0)
            expected_acceleration[active] += loads[active] / mass[active, None]
            np.testing.assert_array_equal(engine.grid.a.to_numpy(), expected_acceleration)
