"""Deterministic key/value contracts for the production odd-even sort."""

import numpy as np
import pytest
import taichi as ti

from src.utils.sorting.ParallelSort import (
    parallel_sort,
    parallel_sort_with_two_values,
    parallel_sort_with_value,
)


pytestmark = [pytest.mark.cpu]


def test_parallel_sort_handles_non_power_of_two_range(taichi_runtime):
    original = np.asarray([90, 80, 7, -3, 12, 5, 0, 11, -8, 70], dtype=np.int32)
    keys = ti.field(ti.i32, shape=original.size)
    keys.from_numpy(original)

    parallel_sort(keys, start=2, length=7)

    expected = original.copy()
    expected[2:9] = np.sort(expected[2:9])
    np.testing.assert_array_equal(keys.to_numpy(), expected)


def test_parallel_sort_keeps_values_attached_to_unique_keys(taichi_runtime):
    original_keys = np.asarray([9, -2, 7, 0, 3, -8, 5], dtype=np.int32)
    original_values = np.asarray([91.0, 22.0, 73.0, 4.0, 35.0, 86.0, 57.0])
    keys = ti.field(ti.i32, shape=original_keys.size)
    values = ti.field(ti.f64, shape=original_values.size)
    keys.from_numpy(original_keys)
    values.from_numpy(original_values)

    parallel_sort_with_value(keys, values, start=0, length=original_keys.size)

    order = np.argsort(original_keys)
    np.testing.assert_array_equal(keys.to_numpy(), original_keys[order])
    np.testing.assert_allclose(
        values.to_numpy(), original_values[order], rtol=0.0, atol=0.0
    )


def test_parallel_sort_keeps_two_value_arrays_attached(taichi_runtime):
    original_keys = np.asarray([4.0, -1.0, 8.0, 2.0, -5.0, 3.0])
    first_values = np.arange(original_keys.size, dtype=np.int32) + 10
    second_values = -first_values
    keys = ti.field(ti.f64, shape=original_keys.size)
    first = ti.field(ti.i32, shape=original_keys.size)
    second = ti.field(ti.i32, shape=original_keys.size)
    keys.from_numpy(original_keys)
    first.from_numpy(first_values)
    second.from_numpy(second_values)

    parallel_sort_with_two_values(
        keys, first, second, start=0, length=original_keys.size
    )

    order = np.argsort(original_keys)
    np.testing.assert_allclose(
        keys.to_numpy(), original_keys[order], rtol=0.0, atol=0.0
    )
    np.testing.assert_array_equal(first.to_numpy(), first_values[order])
    np.testing.assert_array_equal(second.to_numpy(), second_values[order])
