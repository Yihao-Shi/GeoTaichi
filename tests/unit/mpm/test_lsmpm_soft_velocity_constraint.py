from types import SimpleNamespace

import numpy as np
import pytest

from src.mpm.generator.BodyGenerator import compact_soft_grid_nodes_in_region


def _support():
    return SimpleNamespace(
        grid_origin=np.asarray([-0.8, -0.4, -0.3]),
        grid_space=0.1,
        grid_base_space=0.1,
    )


def _topology():
    return SimpleNamespace(
        compact_origin=np.asarray([1, 1, 1], dtype=np.int32),
        compact_shape=np.asarray([15, 7, 6], dtype=np.int32),
    )


def test_region_is_preprocessed_to_compact_local_node_ids():
    nodes = compact_soft_grid_nodes_in_region(
        {
            "StartPoint": [-0.61, -0.21, -0.06],
            "EndPoint": [-0.49, 0.21, 0.06],
        },
        _support(),
        _topology(),
    )

    assert nodes.dtype == np.int32
    assert nodes.size == 10
    assert np.unique(nodes).size == nodes.size
    shape = _topology().compact_shape
    i = nodes % shape[0]
    j = (nodes % (shape[0] * shape[1])) // shape[0]
    k = nodes // (shape[0] * shape[1])
    np.testing.assert_array_equal(np.unique(i), [1, 2])
    np.testing.assert_array_equal(np.unique(j), [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(np.unique(k), [2])


def test_nonzero_prescribed_velocity_is_rejected():
    with pytest.raises(ValueError, match="zero velocity"):
        compact_soft_grid_nodes_in_region(
            {
                "StartPoint": [-0.6, -0.2, -0.05],
                "EndPoint": [-0.5, 0.2, 0.05],
                "Velocity": [0.0, 0.0, 1.0],
            },
            _support(),
            _topology(),
        )
