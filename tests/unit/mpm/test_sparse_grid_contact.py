"""Contact boundaries use current compact IDs and safely skip inactive blocks."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

DIMS = [pytest.param(dim, marks=pytest.mark.isolated_dimension(dim)) for dim in (2, 3)]


@pytest.mark.parametrize("dimension", DIMS)
@pytest.mark.parametrize("kind", ["velocity", "reflection"])
def test_contact_boundaries_follow_block_map(taichi_runtime, dimension, kind):
    from src.mpm.boundaries.BoundaryCore import apply_contact_velocity_constraint, apply_contact_reflection_constraint
    from src.mpm.structs.GridNode import ContactNodes, ContactNodes2D
    from src.mpm.sparse_grid.BlockSparseGrid import BlockSparseGrid

    constraints = ti.Struct.field(dict(node=int, level=int, dirs=int, signs=int), shape=3)
    gnum = ti.Vector([9] * dimension)
    physical = sum(4 * 9**axis for axis in range(dimension))
    block = sum(2 * 5**axis for axis in range(dimension))
    grid = BlockSparseGrid(
        SimpleNamespace(dimension=dimension, sparse_grid_block_size=2, sparse_grid_max_blocks=2),
        SimpleNamespace(gnum=[9] * dimension, gridSum=9**dimension),
        2,
    )
    nodes = (ContactNodes2D if dimension == 2 else ContactNodes).field(shape=(grid.node_capacity, 2))
    constraints[0].node, constraints[0].level = physical, 1
    constraints[0].dirs, constraints[0].signs = 0, 1
    constraints[1].node, constraints[1].level = 0, 1  # Inactive block, unrelated compact slot 0 stays untouched.
    constraints[2].node, constraints[2].level = -1, 1
    apply = apply_contact_velocity_constraint if kind == "velocity" else apply_contact_reflection_constraint
    for compact_block in (0, 1):
        grid.block_map.fill(-1)
        grid.block_map[block] = compact_block
        nodes.contact_force.fill(3.0)
        nodes.momentum.fill(1.0)
        apply(1e-12, 3, constraints, nodes, gnum, grid)
        expected = np.full((grid.node_capacity, 2, dimension), 3.0)
        expected[compact_block * grid.block_volume, 1, 0] = 0.0
        np.testing.assert_array_equal(nodes.contact_force.to_numpy(), expected)
    if kind == "reflection":
        nodes.contact_force.fill(3.0)
        nodes.momentum.fill(-1.0)
        apply(1e-12, 3, constraints, nodes, gnum, grid)
        np.testing.assert_array_equal(nodes.contact_force.to_numpy(), 3.0)


@pytest.mark.isolated_dimension(2)
@pytest.mark.parametrize("contact", ["MPMContact", "GeoContact", "DEMContact"])
@pytest.mark.parametrize(
    "solver,material,axisymmetric,shape,stabilize",
    [
        ("Implicit", "Solid", False, "GIMP", None),
        ("Explicit", "Fluid", False, "GIMP", None),
        ("Explicit", "Solid", True, "Linear", None),
        ("Explicit", "Solid", True, "GIMP", "F-Bar Method"),
        ("Implicit", "Solid", True, "GIMP", None),
    ],
)
def test_blockscan_contact_keeps_unsupported_guards(
    taichi_runtime, contact, solver, material, axisymmetric, shape, stabilize
):
    from src.mpm.SceneManager import myScene
    from src.mpm.Simulation import Simulation

    sims = Simulation()
    sims.dimension, sims.sparse_grid, sims.contact_detection = 2, True, contact
    sims.solver_type, sims.material_type, sims.is_2DAxisy = solver, material, axisymmetric
    sims.shape_function, sims.stabilize = shape, stabilize
    with pytest.raises(RuntimeError, match="[Bb]lockScan"):
        myScene().activate_element(sims, {"ElementType": "Q4N2D"})
