"""Bit-exact contracts for splitting and merging IEEE-754 doubles."""

import numpy as np
import pytest
import taichi as ti

from src.utils.BitFunction import merge_f64, split_f64


pytestmark = [pytest.mark.cpu]


@ti.kernel
def _split_values(
    values: ti.template(), low_words: ti.template(), high_words: ti.template()
):
    for i in values:
        low_words[i], high_words[i] = split_f64(values[i])


@ti.kernel
def _merge_words(
    low_words: ti.template(),
    high_words: ti.template(),
    values: ti.template(),
):
    for i in values:
        values[i] = merge_f64(low_words[i], high_words[i])


def test_split_f64_matches_ieee754_words(taichi_runtime):
    values_np = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -2.5,
            123.45678901234567,
            np.finfo(np.float64).tiny,
            np.finfo(np.float64).max,
        ],
        dtype=np.float64,
    )
    values = ti.field(ti.f64, shape=values_np.shape)
    low_words = ti.field(ti.u32, shape=values_np.shape)
    high_words = ti.field(ti.u32, shape=values_np.shape)
    values.from_numpy(values_np)

    _split_values(values, low_words, high_words)

    bits = values_np.view(np.uint64)
    np.testing.assert_array_equal(
        low_words.to_numpy(), (bits & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    )
    np.testing.assert_array_equal(
        high_words.to_numpy(), (bits >> np.uint64(32)).astype(np.uint32)
    )


def test_merge_f64_reconstructs_ieee754_words(taichi_runtime):
    expected = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -2.5,
            123.45678901234567,
            np.finfo(np.float64).tiny,
            np.finfo(np.float64).max,
        ],
        dtype=np.float64,
    )
    expected_bits = expected.view(np.uint64)
    low_np = (expected_bits & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    high_np = (expected_bits >> np.uint64(32)).astype(np.uint32)
    low_words = ti.field(ti.u32, shape=low_np.shape)
    high_words = ti.field(ti.u32, shape=high_np.shape)
    values = ti.field(ti.f64, shape=expected.shape)
    low_words.from_numpy(low_np)
    high_words.from_numpy(high_np)

    _merge_words(low_words, high_words, values)

    np.testing.assert_array_equal(values.to_numpy().view(np.uint64), expected_bits)
