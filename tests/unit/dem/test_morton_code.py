import pytest
import taichi as ti

from src.utils.BitFunction import morton3d32


pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.geometry, pytest.mark.cpu]


def _spread_10_bits(value):
    result = 0
    for bit in range(10):
        result |= ((value >> bit) & 1) << (3 * bit)
    return result


def _morton_oracle(x, y, z):
    quantized = [
        min(max(int(value * 1023.0), 0), 1023) for value in (x, y, z)
    ]
    return (
        (_spread_10_bits(quantized[0]) << 2)
        | (_spread_10_bits(quantized[1]) << 1)
        | _spread_10_bits(quantized[2])
    )


@pytest.mark.parametrize(
    ("coordinates", "expected"),
    [
        ((0.0, 0.0, 0.0), 0),
        ((1.0, 1.0, 1.0), 0x3FFFFFFF),
        ((1.0 / 1023.0, 2.0 / 1023.0, 4.0 / 1023.0), None),
        ((-2.0, 0.5, 3.0), None),
    ],
)
def test_morton3d32_matches_bit_interleaving_oracle(
    taichi_runtime, coordinates, expected
):
    result = ti.field(dtype=ti.u32, shape=())

    @ti.kernel
    def encode(x: ti.f64, y: ti.f64, z: ti.f64):
        result[None] = morton3d32(x, y, z)

    encode(*coordinates)
    oracle = _morton_oracle(*coordinates) if expected is None else expected
    assert int(result[None]) == oracle
