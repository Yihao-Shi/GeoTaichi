"""Inclusive prefix-sum contracts for CPU kernels and boundary sizes."""

import numpy as np
import pytest
import taichi as ti

from src.utils.PrefixSum import PrefixSumExecutor, serial


pytestmark = [pytest.mark.cpu]


@pytest.mark.parametrize("length", (1, 2, 3, 7, 31, 65))
@pytest.mark.parametrize("dtype", (ti.i32, ti.i64))
def test_cpu_prefix_sum_matches_numpy(taichi_runtime, length, dtype):
    values = np.arange(1, length + 1, dtype=np.int64)
    executor = PrefixSumExecutor(length, dtype=dtype)
    field = ti.field(dtype=dtype, shape=executor.get_length())
    field.from_numpy(values.astype(np.int32 if dtype == ti.i32 else np.int64))

    executor.run(field)

    np.testing.assert_array_equal(field.to_numpy(), np.cumsum(values))


def test_serial_prefix_sum_matches_numpy(taichi_runtime):
    values = np.asarray([3, -1, 4, 0, 2, -5, 8], dtype=np.int32)
    field = ti.field(dtype=ti.i32, shape=values.size)
    field.from_numpy(values)

    serial(field)

    np.testing.assert_array_equal(field.to_numpy(), np.cumsum(values))
