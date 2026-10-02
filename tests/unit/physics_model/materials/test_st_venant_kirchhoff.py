"""Shared host/device tests for St. Venant--Kirchhoff elasticity."""

import numpy as np
import pytest
import taichi as ti

from src.physics_model.consititutive_model.finite_strain import (
    StVenantKirchhoffModel,
)


pytestmark = [pytest.mark.materials, pytest.mark.cpu]


def _model():
    return StVenantKirchhoffModel().initialize_from_kwargs(
        density=1100.,
        young_modulus=2.4e5,
        poisson_ratio=0.28,
        thickness=0.02,
    )


@pytest.mark.parametrize(
    "deformation_gradient",
    [
        np.asarray(((1.12, 0.04), (0.02, 0.91))),
        np.asarray(
            (
                (1.08, 0.03, 0.00),
                (0.01, 0.94, 0.02),
                (0.00, 0.01, 1.04),
            )
        ),
    ],
    ids=["plane-stress", "three-dimensional"],
)
def test_stvk_device_response_matches_host_reference(
    taichi_material_cpu, deformation_gradient
):
    model = _model()
    dimension = deformation_gradient.shape[0]
    device_energy = ti.field(dtype=ti.f64, shape=())
    device_pk1 = ti.Matrix.field(
        dimension, dimension, dtype=ti.f64, shape=()
    )
    device_tangent = ti.Matrix.field(
        dimension * dimension,
        dimension * dimension,
        dtype=ti.f64,
        shape=(),
    )

    @ti.kernel
    def evaluate_kernel(
        deformation: ti.types.ndarray(dtype=ti.f64, ndim=2),
    ):
        deformation_matrix = ti.Matrix.zero(
            ti.f64, dimension, dimension
        )
        for row, column in ti.static(
            ti.ndrange(dimension, dimension)
        ):
            deformation_matrix[row, column] = deformation[row, column]
        device_energy[None] = model.Psi(deformation_matrix)
        device_pk1[None] = model.first_piola_stress(
            deformation_matrix
        )
        device_tangent[None] = model.first_piola_tangent(
            deformation_matrix
        )

    evaluate_kernel(
        np.ascontiguousarray(deformation_gradient, dtype=np.float64)
    )
    host_energy, host_pk1, host_tangent = model.evaluate(
        deformation_gradient
    )
    host_matrix = np.empty(
        (dimension * dimension, dimension * dimension),
        dtype=np.float64,
    )
    for material_i in range(dimension):
        for spatial_i in range(dimension):
            row = spatial_i + material_i * dimension
            for material_j in range(dimension):
                for spatial_j in range(dimension):
                    column = spatial_j + material_j * dimension
                    host_matrix[row, column] = host_tangent[
                        spatial_i,
                        material_i,
                        spatial_j,
                        material_j,
                    ]

    assert float(device_energy[None]) == pytest.approx(
        host_energy, rel=2.e-13, abs=2.e-13
    )
    np.testing.assert_allclose(
        device_pk1.to_numpy()[()],
        host_pk1,
        rtol=2.e-13,
        atol=2.e-11,
    )
    np.testing.assert_allclose(
        device_tangent.to_numpy()[()],
        host_matrix,
        rtol=2.e-13,
        atol=2.e-10,
    )


def test_stvk_initialization_uses_shared_finite_strain_fields():
    model = _model()

    assert model.density == 1100.
    assert model.young_modulus == 2.4e5
    assert model.poisson_ratio == 0.28
    assert model.thickness == 0.02
    assert model.mu_ == pytest.approx(2.4e5 / (2. * 1.28))
