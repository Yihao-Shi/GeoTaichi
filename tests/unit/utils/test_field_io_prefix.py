import numpy as np
import taichi as ti
import pytest


from src.utils.FieldIO import field_to_numpy_prefix, field_to_numpy_slice, field_from_numpy_prefix


@ti.kernel
def _init_fields(scalar: ti.template(), vector: ti.template(), matrix: ti.template()):
    for i in scalar:
        scalar[i] = i * 2 + 1
        vector[i] = ti.Vector([i + 0.25, i + 0.5, i + 0.75])
        for j, k in ti.static(ti.ndrange(2, 2)):
            matrix[i][j, k] = 10.0 * i + 2.0 * j + k


def run_case():
    ti.init(arch=ti.cpu, offline_cache=False, log_level=ti.ERROR)
    scalar = ti.field(dtype=ti.i32, shape=16)
    vector = ti.Vector.field(3, dtype=ti.f64, shape=16)
    matrix = ti.Matrix.field(2, 2, dtype=ti.f64, shape=16)
    _init_fields(scalar, vector, matrix)

    assert np.array_equal(field_to_numpy_slice(scalar, 3, 11), scalar.to_numpy()[3:11])
    assert np.allclose(field_to_numpy_slice(vector, 2, 9), vector.to_numpy()[2:9])
    assert np.allclose(field_to_numpy_slice(matrix, 1, 7), matrix.to_numpy()[1:7])
    assert np.array_equal(field_to_numpy_prefix(scalar, 5), scalar.to_numpy()[:5])
    assert field_to_numpy_prefix(vector, 0).shape == (0, 3)


def test_field_to_numpy_slice_handles_scalar_vector_and_matrix_fields():
    run_case()


def test_numpy_prefix_preserves_unused_capacity_and_rejects_invalid_values(taichi_runtime):
    fields = [ti.field(ti.u8, shape=5), ti.Vector.field(3, ti.f64, shape=5), ti.Matrix.field(2, 2, ti.f64, shape=5)]
    for field in fields:
        field.fill(7)
        expected = field.to_numpy()
        values = np.full(expected[:2].shape, 3)
        field_from_numpy_prefix(field, values)
        expected[:2] = values
        np.testing.assert_array_equal(field.to_numpy(), expected)
    for values in [
        np.array([-1]),
        np.array([256]),
        np.array([0.5]),
        np.array([np.nan]),
        np.array([1 + 1j]),
        np.zeros(6),
    ]:
        with pytest.raises(ValueError):
            field_from_numpy_prefix(fields[0], values)
    with pytest.raises(ValueError):
        field_from_numpy_prefix(fields[1], np.zeros((2, 2)))
    with pytest.raises(ValueError):
        field_from_numpy_prefix(fields[1], np.ones((2, 3), dtype=complex))


if __name__ == "__main__":
    test_field_to_numpy_slice_handles_scalar_vector_and_matrix_fields()
