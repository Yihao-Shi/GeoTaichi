import numpy as np
import taichi as ti

from src.mpm.engines.AssembleMatrixKernel import (  # noqa: E402
    assemble_diagonal_stiffness_matrix,
    assemble_elastic_diagonal_stiffness_matrix,
    assemble_elastic_stiffness_matrix,
    assemble_stiffness_matrix,
)


@ti.kernel
def _evaluate_stiffness(
    idshape_np: ti.types.ndarray(),
    jdshape_np: ti.types.ndarray(),
    stiffness_np: ti.types.ndarray(),
    stress_np: ti.types.ndarray(),
    plastic_full: ti.types.ndarray(),
    plastic_diag: ti.types.ndarray(),
    elastic_full: ti.types.ndarray(),
    elastic_diag: ti.types.ndarray(),
):
    idshape = ti.Vector([idshape_np[0], idshape_np[1], idshape_np[2]])
    jdshape = ti.Vector([jdshape_np[0], jdshape_np[1], jdshape_np[2]])
    stress = ti.Vector([stress_np[0], stress_np[1], stress_np[2], stress_np[3], stress_np[4], stress_np[5]])
    stiffness = ti.Matrix.zero(ti.f64, 6, 6)
    i = 0
    while i < 6:
        j = 0
        while j < 6:
            stiffness[i, j] = stiffness_np[i, j]
            j += 1
        i += 1

    p_full = assemble_stiffness_matrix(idshape, jdshape, stiffness, stress)
    p_diag = assemble_diagonal_stiffness_matrix(idshape, jdshape, stiffness, stress)
    e_full = assemble_elastic_stiffness_matrix(idshape, jdshape, stiffness, stress)
    e_diag = assemble_elastic_diagonal_stiffness_matrix(idshape, jdshape, stiffness, stress)

    for i in ti.static(range(9)):
        plastic_full[i] = p_full[i]
        elastic_full[i] = e_full[i]
    for i in ti.static(range(3)):
        plastic_diag[i] = p_diag[i]
        elastic_diag[i] = e_diag[i]


def _stress_tensor(stress):
    return np.array(
        [
            [stress[0], stress[3], stress[5]],
            [stress[3], stress[1], stress[4]],
            [stress[5], stress[4], stress[2]],
        ],
        dtype=np.float64,
    )


def _sigrot_tensor(stress, displacement_component, grad):
    velocity_gradient = np.zeros((3, 3), dtype=np.float64)
    velocity_gradient[displacement_component, :] = grad
    dw = 0.5 * np.array(
        [
            velocity_gradient[1, 0] - velocity_gradient[0, 1],
            velocity_gradient[2, 1] - velocity_gradient[1, 2],
            velocity_gradient[0, 2] - velocity_gradient[2, 0],
        ],
        dtype=np.float64,
    )

    s0, s1, s2, s3, s4, s5 = stress
    sigrot = np.array(
        [
            2.0 * (-dw[2] * s5 + dw[0] * s3),
            2.0 * (-dw[0] * s3 + dw[1] * s4),
            2.0 * (-dw[1] * s4 + dw[2] * s5),
            -dw[2] * s4 + dw[1] * s5 + dw[0] * (s1 - s0),
            -dw[0] * s5 + dw[2] * s3 + dw[1] * (s2 - s1),
            -dw[1] * s3 + dw[0] * s4 + dw[2] * (s0 - s2),
        ],
        dtype=np.float64,
    )
    return _stress_tensor(sigrot)


def _b_operator(grad):
    gx, gy, gz = grad
    return np.array(
        [
            [gx, 0.0, 0.0, gy, 0.0, gz],
            [0.0, gy, 0.0, gx, gz, 0.0],
            [0.0, 0.0, gz, 0.0, gy, gx],
        ],
        dtype=np.float64,
    )


def _reference_full(idshape, jdshape, stiffness, stress):
    sigma = _stress_tensor(stress)
    material = _b_operator(idshape) @ stiffness @ _b_operator(jdshape).T
    stress_stiffness = np.zeros((3, 3), dtype=np.float64)
    for column in range(3):
        sigrot = _sigrot_tensor(stress, column, jdshape)
        for row in range(3):
            stress_stiffness[row, column] = (
                jdshape[column] * np.dot(idshape, sigma[row, :])
                - np.dot(idshape, sigrot[row, :])
            )
    return (material + stress_stiffness).reshape(9)


def _elastic_stiffness_matrix(bulk, shear):
    stiffness = np.zeros((6, 6), dtype=np.float64)
    a1 = bulk + 4.0 * shear / 3.0
    a2 = bulk - 2.0 * shear / 3.0
    stiffness[:3, :3] = a2
    np.fill_diagonal(stiffness[:3, :3], a1)
    stiffness[3, 3] = shear
    stiffness[4, 4] = shear
    stiffness[5, 5] = shear
    return stiffness


def _run_kernel(idshape, jdshape, stiffness, stress):
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False, log_level=ti.ERROR)
    plastic_full = np.zeros(9, dtype=np.float64)
    plastic_diag = np.zeros(3, dtype=np.float64)
    elastic_full = np.zeros(9, dtype=np.float64)
    elastic_diag = np.zeros(3, dtype=np.float64)
    _evaluate_stiffness(
        idshape.astype(np.float64),
        jdshape.astype(np.float64),
        stiffness.astype(np.float64),
        stress.astype(np.float64),
        plastic_full,
        plastic_diag,
        elastic_full,
        elastic_diag,
    )
    ti.reset()
    return plastic_full, plastic_diag, elastic_full, elastic_diag


def test_3d_plastic_stiffness_includes_stress_tangent_and_matches_diagonal():
    idshape = np.array([0.31, -0.27, 0.19], dtype=np.float64)
    jdshape = np.array([-0.23, 0.41, 0.17], dtype=np.float64)
    stress = np.array([2.2, -1.7, 0.9, 0.35, -0.28, 0.46], dtype=np.float64)
    stiffness = np.array(
        [
            [6.1, 1.2, -0.4, 0.7, -0.3, 0.2],
            [0.8, 5.4, 1.1, -0.5, 0.6, -0.1],
            [-0.2, 0.9, 4.8, 0.3, -0.7, 0.5],
            [0.4, -0.6, 0.2, 2.5, 0.8, -0.4],
            [-0.3, 0.5, -0.8, 0.1, 3.0, 0.9],
            [0.2, -0.1, 0.7, -0.5, 0.4, 2.8],
        ],
        dtype=np.float64,
    )

    plastic_full, plastic_diag, _, _ = _run_kernel(idshape, jdshape, stiffness, stress)
    expected = _reference_full(idshape, jdshape, stiffness, stress)

    np.testing.assert_allclose(plastic_full, expected, rtol=1.0e-7, atol=1.0e-7)
    np.testing.assert_allclose(plastic_diag, expected[[0, 4, 8]], rtol=1.0e-7, atol=1.0e-7)


def test_3d_elastic_stiffness_uses_same_stress_tangent_and_matches_diagonal():
    idshape = np.array([-0.37, 0.22, 0.14], dtype=np.float64)
    jdshape = np.array([0.29, -0.18, 0.33], dtype=np.float64)
    stress = np.array([-1.1, 2.4, 0.7, -0.31, 0.52, -0.26], dtype=np.float64)
    stiffness = _elastic_stiffness_matrix(bulk=7.5, shear=2.25)

    _, _, elastic_full, elastic_diag = _run_kernel(idshape, jdshape, stiffness, stress)
    expected = _reference_full(idshape, jdshape, stiffness, stress)

    np.testing.assert_allclose(elastic_full, expected, rtol=1.0e-7, atol=1.0e-7)
    np.testing.assert_allclose(elastic_diag, expected[[0, 4, 8]], rtol=1.0e-7, atol=1.0e-7)
