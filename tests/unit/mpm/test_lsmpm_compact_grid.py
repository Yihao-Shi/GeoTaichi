from types import SimpleNamespace

import numpy as np

from src.mpm.soft_particle.GridTopology import (
    SoftMechanicalGrid,
    _linear_support,
    select_soft_grid_topology,
    verlet_padding_cell_count,
)


class _Grid:
    def __init__(self, gnum=(11, 11, 11), spacing=1.0):
        self.gnum = np.asarray(gnum, dtype=np.int32)
        self.gridSum = int(np.prod(self.gnum))
        self.grid_space = float(spacing)
        self.start_point = -0.5 * (self.gnum - 1) * self.grid_space

    def minBox(self):
        return self.start_point


def _template(vertices):
    grid = _Grid()
    return SimpleNamespace(
        objects=SimpleNamespace(
            grid=grid,
            mesh=SimpleNamespace(vertices=np.asarray(vertices, dtype=np.float64)),
        )
    )


def _mechanical_grid(grid):
    return SoftMechanicalGrid(
        origin=np.ascontiguousarray(grid.minBox(), dtype=np.float64),
        shape=np.ascontiguousarray(grid.gnum, dtype=np.int32),
        spacing=float(grid.grid_space),
    )


def _logical_ijk(logical_id, gnum):
    return np.asarray(
        (
            logical_id % gnum[0],
            (logical_id % (gnum[0] * gnum[1])) // gnum[0],
            logical_id // (gnum[0] * gnum[1]),
        ),
        dtype=np.int32,
    )


def test_compact_grid_crops_the_extent_expanded_logical_grid():
    vertices = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    material_points = np.asarray(
        [[-0.25, -0.25, -0.25], [0.25, 0.25, 0.25]]
    )
    template = _template(vertices)
    padding_cells = 2

    topology = select_soft_grid_topology(
        template,
        material_points,
        shape_function_type=0,
        storage="Compact",
        padding_cells=padding_cells,
        mechanical_grid=_mechanical_grid(template.objects.grid),
    )

    grid = template.objects.grid
    supported = set()
    for point in np.vstack((material_points, vertices)):
        supported.update(
            _linear_support(
                point, grid.minBox(), grid.grid_space, grid.gnum
            )
        )
    supported_ijk = np.asarray(
        [_logical_ijk(node, grid.gnum) for node in supported]
    )
    expected_origin = np.maximum(
        0, supported_ijk.min(axis=0) - padding_cells
    )
    expected_upper = np.minimum(
        grid.gnum, supported_ijk.max(axis=0) + 1 + padding_cells
    )

    np.testing.assert_array_equal(topology.compact_origin, expected_origin)
    np.testing.assert_array_equal(
        topology.compact_shape, expected_upper - expected_origin
    )
    assert topology.compact_count < topology.logical_count

    compact_id = 0
    for k in range(int(expected_origin[2]), int(expected_upper[2])):
        for j in range(int(expected_origin[1]), int(expected_upper[1])):
            for i in range(int(expected_origin[0]), int(expected_upper[0])):
                ijk = np.asarray([i, j, k], dtype=np.int32)
                logical_id = int(
                    i
                    + j * grid.gnum[0]
                    + k * grid.gnum[0] * grid.gnum[1]
                )
                assert logical_id in range(topology.logical_count)
                local = ijk - topology.compact_origin
                direct_id = int(
                    local[0]
                    + local[1] * topology.compact_shape[0]
                    + local[2]
                    * topology.compact_shape[0]
                    * topology.compact_shape[1]
                )
                assert direct_id == compact_id
                compact_id += 1
    assert compact_id == topology.compact_count

    for logical_id in supported:
        ijk = _logical_ijk(int(logical_id), grid.gnum)
        local = ijk - topology.compact_origin
        assert np.all(local >= 0)
        assert np.all(local < topology.compact_shape)


def test_dense_uses_the_full_box_but_retains_true_support_count():
    template = _template([[0.0, 0.0, 0.0]])
    topology = select_soft_grid_topology(
        template,
        np.asarray([[0.0, 0.0, 0.0]]),
        shape_function_type=0,
        storage="Dense",
        padding_cells=0,
        mechanical_grid=_mechanical_grid(template.objects.grid),
    )

    assert topology.storage == "Dense"
    assert topology.compact_count == topology.logical_count
    assert topology.support_count < topology.logical_count
    np.testing.assert_array_equal(topology.compact_origin, [0, 0, 0])
    np.testing.assert_array_equal(
        topology.compact_shape, template.objects.grid.gnum
    )


def test_full_crop_is_not_an_automatic_dense_fallback():
    template = _template([[0.0, 0.0, 0.0]])
    topology = select_soft_grid_topology(
        template,
        np.asarray([[0.0, 0.0, 0.0]]),
        shape_function_type=0,
        storage="Compact",
        padding_cells=100,
        mechanical_grid=_mechanical_grid(template.objects.grid),
    )

    assert topology.storage == "Compact"
    assert topology.compact_count == topology.logical_count
    np.testing.assert_array_equal(topology.compact_origin, [0, 0, 0])
    np.testing.assert_array_equal(topology.compact_shape, [11, 11, 11])


def test_verlet_padding_ignores_only_roundoff_above_an_integer_ratio():
    assert verlet_padding_cell_count(0.2 * (1.0 + 2.0e-16), 0.2) == 1
    assert verlet_padding_cell_count(0.2 * (1.0 + 1.0e-10), 0.2) == 2
    assert verlet_padding_cell_count(0.2 * (1.0 + 2.0e-16), 0.025) == 8


def test_topology_keeps_extent_and_verlet_padding_metadata_separate():
    template = _template([[0.0, 0.0, 0.0]])
    topology = select_soft_grid_topology(
        template,
        np.asarray([[0.0, 0.0, 0.0]]),
        shape_function_type=0,
        storage="Compact",
        padding_cells=4,
        mechanical_grid=_mechanical_grid(template.objects.grid),
        levelset_extent_cells=4,
        verlet_padding_cells=1,
    )

    assert topology.padding_cells == 4
    assert topology.levelset_extent_cells == 4
    assert topology.verlet_padding_cells == 1
