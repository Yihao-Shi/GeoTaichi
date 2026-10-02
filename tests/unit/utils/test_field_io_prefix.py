import numpy as np
import taichi as ti


from src.utils.FieldIO import field_to_numpy_prefix, field_to_numpy_slice


@ti.kernel
def _init_fields(scalar: ti.template(), vector: ti.template(), matrix: ti.template()):
    for i in scalar:
        scalar[i] = i * 2 + 1
        vector[i] = ti.Vector([i + 0.25, i + 0.5, i + 0.75])
        for j, k in ti.static(ti.ndrange(2, 2)):
            matrix[i][j, k] = 10. * i + 2. * j + k


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


if __name__ == "__main__":
    test_field_to_numpy_slice_handles_scalar_vector_and_matrix_fields()
