"""Analytic oracles for the 3D IGA element-local chain rule."""

import numpy as np
import pytest


pytestmark = [pytest.mark.unit, pytest.mark.cpu, pytest.mark.serial]


@pytest.mark.isolated_dimension(3)
def test_structured_local_pullbacks_match_random_dense_oracles(
    taichi_runtime,
):
    import taichi as ti

    from src.iga.elements.Element import Element

    support_count = 4
    dimension = 3
    deformation_size = dimension * dimension

    # These local operators only depend on the support count.  Avoid building
    # an entire patch so this remains a small, deterministic unit test.
    element = object.__new__(Element)
    element.total_knot_range = support_count

    dNdnat_field = ti.Matrix.field(
        support_count,
        dimension,
        dtype=ti.f64,
        shape=(),
    )
    dnatdX_field = ti.Matrix.field(
        dimension,
        dimension,
        dtype=ti.f64,
        shape=(),
    )
    coordinates_field = ti.Matrix.field(
        support_count,
        dimension,
        dtype=ti.f64,
        shape=(),
    )
    stress_field = ti.Vector.field(
        deformation_size,
        dtype=ti.f64,
        shape=(),
    )
    tangent_field = ti.Matrix.field(
        deformation_size,
        deformation_size,
        dtype=ti.f64,
        shape=(),
    )

    dFdx_field = ti.field(
        dtype=ti.f64,
        shape=(support_count * dimension, deformation_size),
    )
    deformation_gradient_field = ti.field(
        dtype=ti.f64,
        shape=(dimension, dimension),
    )
    structured_deformation_gradient_field = ti.field(
        dtype=ti.f64,
        shape=(dimension, dimension),
    )
    shape_gradients_field = ti.field(
        dtype=ti.f64,
        shape=(support_count, dimension),
    )
    gradient_field = ti.field(
        dtype=ti.f64,
        shape=support_count * dimension,
    )
    structured_gradient_field = ti.field(
        dtype=ti.f64,
        shape=support_count * dimension,
    )
    hessian_field = ti.field(
        dtype=ti.f64,
        shape=(
            support_count * dimension,
            support_count * dimension,
        ),
    )
    structured_hessian_field = ti.field(
        dtype=ti.f64,
        shape=(
            support_count * dimension,
            support_count * dimension,
        ),
    )

    @ti.kernel
    def evaluate():
        shape_gradients = element.compute_shape_gradients(
            dNdnat_field[None],
            dnatdX_field[None],
        )
        dFdx = element.compute_dF_div_dx(
            dNdnat_field[None],
            dnatdX_field[None],
        )
        deformation_gradient = element.compute_deformation_gradient(
            dNdnat_field[None],
            dnatdX_field[None],
            coordinates_field[None],
        )
        structured_deformation_gradient = (
            element.compute_deformation_gradient_from_shape_gradients(
                shape_gradients,
                coordinates_field[None],
            )
        )

        row = 0
        while row < support_count * dimension:
            column = 0
            while column < deformation_size:
                dFdx_field[row, column] = dFdx[row, column]
                column += 1
            row += 1

        support = 0
        while support < support_count:
            material_axis = 0
            while material_axis < dimension:
                shape_gradients_field[support, material_axis] = (
                    shape_gradients[support, material_axis]
                )
                material_axis += 1
            support += 1

        row = 0
        while row < dimension:
            column = 0
            while column < dimension:
                deformation_gradient_field[row, column] = (
                    deformation_gradient[row, column]
                )
                structured_deformation_gradient_field[row, column] = (
                    structured_deformation_gradient[row, column]
                )
                column += 1
            row += 1

        support_i = 0
        while support_i < support_count:
            local_gradient = element.compute_local_gradient(
                support_i,
                stress_field[None],
                dFdx,
            )
            component = 0
            while component < dimension:
                gradient_field[
                    dimension * support_i + component
                ] = local_gradient[component]
                component += 1

            structured_local_gradient = (
                element.compute_local_gradient_from_shape_gradients(
                    support_i,
                    stress_field[None],
                    shape_gradients,
                )
            )
            component = 0
            while component < dimension:
                structured_gradient_field[
                    dimension * support_i + component
                ] = structured_local_gradient[component]
                component += 1

            support_j = 0
            while support_j < support_count:
                local_hessian = element.compute_local_hessian(
                    support_i,
                    support_j,
                    tangent_field[None],
                    dFdx,
                )
                row_component = 0
                while row_component < dimension:
                    column_component = 0
                    while column_component < dimension:
                        hessian_field[
                            dimension * support_i + row_component,
                            dimension * support_j + column_component,
                        ] = local_hessian[
                            row_component,
                            column_component,
                        ]
                        column_component += 1
                    row_component += 1

                structured_local_hessian = (
                    element.compute_local_hessian_from_shape_gradients(
                        support_i,
                        support_j,
                        tangent_field[None],
                        shape_gradients,
                    )
                )
                row_component = 0
                while row_component < dimension:
                    column_component = 0
                    while column_component < dimension:
                        structured_hessian_field[
                            dimension * support_i + row_component,
                            dimension * support_j + column_component,
                        ] = structured_local_hessian[
                            row_component,
                            column_component,
                        ]
                        column_component += 1
                    row_component += 1
                support_j += 1
            support_i += 1

    rng = np.random.default_rng(7319)
    dNdnat = rng.normal(size=(support_count, dimension))
    dnatdX = rng.normal(size=(dimension, dimension))
    dnatdX += 2.0 * np.eye(dimension)
    coordinates = rng.normal(size=(support_count, dimension))
    stress = rng.normal(size=deformation_size)
    tangent = rng.normal(
        size=(deformation_size, deformation_size)
    )

    dNdnat_field.from_numpy(dNdnat)
    dnatdX_field.from_numpy(dnatdX)
    coordinates_field.from_numpy(coordinates)
    stress_field.from_numpy(stress)
    tangent_field.from_numpy(tangent)
    evaluate()

    dNdX = dNdnat @ dnatdX
    expected_dFdx = np.zeros(
        (support_count * dimension, deformation_size),
        dtype=np.float64,
    )
    for support in range(support_count):
        for spatial_component in range(dimension):
            for deformation_column in range(dimension):
                expected_dFdx[
                    dimension * support + spatial_component,
                    spatial_component
                    + dimension * deformation_column,
                ] = dNdX[support, deformation_column]

    expected_deformation_gradient = coordinates.T @ dNdX
    expected_gradient = expected_dFdx @ stress
    expected_hessian = (
        expected_dFdx @ tangent @ expected_dFdx.T
    )

    np.testing.assert_allclose(
        shape_gradients_field.to_numpy(),
        dNdX,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        dFdx_field.to_numpy(),
        expected_dFdx,
        rtol=0.0,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        deformation_gradient_field.to_numpy(),
        expected_deformation_gradient,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        structured_deformation_gradient_field.to_numpy(),
        expected_deformation_gradient,
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    np.testing.assert_allclose(
        gradient_field.to_numpy(),
        expected_gradient,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        structured_gradient_field.to_numpy(),
        expected_gradient,
        rtol=2.0e-14,
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        hessian_field.to_numpy(),
        expected_hessian,
        rtol=5.0e-14,
        atol=5.0e-14,
    )
    np.testing.assert_allclose(
        structured_hessian_field.to_numpy(),
        expected_hessian,
        rtol=5.0e-14,
        atol=5.0e-14,
    )
