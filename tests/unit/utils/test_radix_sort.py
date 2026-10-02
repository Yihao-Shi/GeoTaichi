"""Regression tests for the key/value GPU-oriented radix sorter."""

import numpy as np
import pytest
import taichi as ti

from src.utils.sorting.RadixSort import RadixSort


pytestmark = [pytest.mark.cpu]


@pytest.mark.parametrize("dtype", (ti.i32, ti.i64))
def test_general_radix_sort_orders_all_bytes_and_keeps_values(taichi_runtime, dtype):
    keys = np.asarray(
        [65536, 1, 257, 256, 0, 16777217, 65535, 2, 511, 16777216],
        dtype=np.int64,
    )
    values = np.arange(keys.size, dtype=np.int64) + 100
    rows = np.column_stack((keys, values))
    numpy_dtype = np.int32 if dtype == ti.i32 else np.int64
    sorter = RadixSort(keys.size, dtype=dtype, val_col=1)
    sorter.data_in.from_numpy(rows.astype(numpy_dtype))

    sorter.run(keys.size)

    order = np.argsort(keys, kind="stable")
    np.testing.assert_array_equal(
        sorter.data_out.to_numpy(), rows[order].astype(numpy_dtype)
    )
