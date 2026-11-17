"""Index helpers keep the x-fast layout used by linked-cell searches."""

import numpy as np
import pytest
import taichi as ti

from src.utils.ScalarFunction import linearize, vectorize_id

pytestmark = pytest.mark.cpu


@pytest.mark.parametrize("dimensions", [(3, 4), (3, 4, 5)])
def test_cell_indices_match_numpy_and_round_trip_with_keyword_arguments(taichi_runtime, dimensions):
    count = int(np.prod(dimensions))
    coordinates = ti.Vector.field(len(dimensions), ti.i32, shape=count)
    linear_indices = ti.field(ti.i32, shape=count)

    @ti.kernel
    def round_trip():
        for index in linear_indices:
            coordinate = ti.Vector(vectorize_id(index=index, count_vec=ti.Vector(dimensions)))
            coordinates[index] = coordinate
            linear_indices[index] = linearize(coordinate, ti.Vector(dimensions))

    round_trip()

    expected = np.stack(np.unravel_index(np.arange(count), dimensions, order="F"), axis=1)
    np.testing.assert_array_equal(coordinates.to_numpy(), expected)
    np.testing.assert_array_equal(linear_indices.to_numpy(), np.arange(count))
