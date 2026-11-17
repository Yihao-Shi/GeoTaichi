import numpy as np

from src.mpm.SoftParticleOutput import pk1_to_cauchy, von_mises_stress, vtk_vector


def test_pk1_to_cauchy_uses_current_area_transformation():
    deformation = np.asarray(
        [
            [
                [1.2, 0.0, 0.0],
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 1.1],
            ]
        ],
        dtype=np.float64,
    )
    first_piola = np.asarray(
        [
            [
                [3.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
                [0.0, 0.0, 5.0],
            ]
        ],
        dtype=np.float64,
    )

    cauchy, jacobian = pk1_to_cauchy(first_piola, deformation)

    expected_jacobian = np.linalg.det(deformation[0])
    expected_cauchy = first_piola[0] @ deformation[0].T / expected_jacobian
    np.testing.assert_allclose(jacobian, [expected_jacobian])
    np.testing.assert_allclose(cauchy[0], expected_cauchy)


def test_von_mises_stress_accepts_full_cauchy_tensor():
    uniaxial = np.asarray(
        [
            [
                [12.5, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    hydrostatic = np.asarray(
        [
            [
                [-7.0, 0.0, 0.0],
                [0.0, -7.0, 0.0],
                [0.0, 0.0, -7.0],
            ]
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(von_mises_stress(uniaxial), [12.5])
    np.testing.assert_allclose(von_mises_stress(hydrostatic), [0.0])


def test_vtk_vector_pads_2d_vectors_with_zero_z_component():
    velocity = np.asarray([[1.0, -2.0], [3.0, -4.0]])

    components = vtk_vector(velocity)

    np.testing.assert_array_equal(components[0], velocity[:, 0])
    np.testing.assert_array_equal(components[1], velocity[:, 1])
    np.testing.assert_array_equal(components[2], np.zeros(2))
