import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.ipc, pytest.mark.contact]

from src.physics_model.contact_model.ipc.ContactAssembly import (
    dense_local_triplets,
    project_to_psd_dense,
    pullback_dense,
)


def test_complete_local_hessian_is_projected_before_scatter():
    geometry_gradient = np.asarray([1.0, -2.0])
    geometry_hessian = np.asarray([[2.0, 3.0], [3.0, -4.0]])
    interpolation = np.asarray(
        [[1.0, 0.0, 0.5], [0.0, 2.0, -1.0]], dtype=np.float64
    )
    gradient, local_hessian = pullback_dense(
        geometry_gradient, geometry_hessian, interpolation
    )
    np.testing.assert_allclose(gradient, interpolation.T @ geometry_gradient)

    projected = project_to_psd_dense(local_hessian, method="clamp")
    assert np.linalg.eigvalsh(projected).min() >= -1.0e-12
    np.testing.assert_allclose(projected, projected.T, atol=1.0e-14)

    dofs = np.asarray([7, 2, 11])
    rows, columns, values = dense_local_triplets(dofs, projected)
    reconstructed = np.zeros((12, 12), dtype=np.float64)
    np.add.at(reconstructed, (rows, columns), values)
    np.testing.assert_allclose(
        reconstructed[np.ix_(dofs, dofs)], projected, atol=1.0e-14
    )


def test_psd_projection_policies_match_ipc_semantics():
    matrix = np.asarray([[2.0, 1.0], [1.0, -3.0]])
    eigenvalues = np.linalg.eigvalsh(matrix)
    clamped = project_to_psd_dense(matrix, "clamp")
    absolute = project_to_psd_dense(matrix, "abs")
    unchanged = project_to_psd_dense(matrix, "none")
    np.testing.assert_allclose(
        np.linalg.eigvalsh(clamped), np.maximum(eigenvalues, 0.0), atol=1.0e-14
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(absolute), np.sort(np.abs(eigenvalues)), atol=1.0e-14
    )
    np.testing.assert_allclose(unchanged, matrix)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: project_to_psd_dense(np.zeros((2, 3))), "square"),
        (lambda: project_to_psd_dense(np.eye(2), "bad"), "method"),
        (
            lambda: dense_local_triplets([0, 1], np.eye(3)),
            "shape",
        ),
    ],
)
def test_contact_assembly_rejects_invalid_local_systems(call, match):
    with pytest.raises(ValueError, match=match):
        call()
