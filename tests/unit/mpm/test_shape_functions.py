"""Partition-of-unity and derivative contracts for MPM shape functions."""

import numpy as np
import pytest
import taichi as ti

from src.utils.ShapeFunctions import (
    GShapeBsplineC,
    GShapeBsplineQ,
    GShapeGIMP,
    GShapeLinear,
    ShapeBsplineC,
    ShapeBsplineQ,
    ShapeGIMP,
    ShapeLinear,
)


pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.cpu]


@ti.kernel
def _evaluate_linear(
    particle: ti.f64,
    values: ti.template(),
    gradients: ti.template(),
):
    for node in values:
        coordinate = ti.cast(node, ti.f64)
        values[node] = ShapeLinear(
            particle, coordinate, 1.0, 0.0
        )
        gradients[node] = GShapeLinear(
            particle, coordinate, 1.0, 0.0
        )


@ti.kernel
def _evaluate_gimp(
    particle: ti.f64,
    values: ti.template(),
    gradients: ti.template(),
):
    for node in values:
        coordinate = ti.cast(node, ti.f64)
        values[node] = ShapeGIMP(
            particle, coordinate, 1.0, 0.25
        )
        gradients[node] = GShapeGIMP(
            particle, coordinate, 1.0, 0.25
        )


@ti.kernel
def _evaluate_quadratic_bspline(
    particle: ti.f64,
    values: ti.template(),
    gradients: ti.template(),
):
    for node in values:
        coordinate = ti.cast(node, ti.f64)
        values[node] = ShapeBsplineQ(
            particle, coordinate, 1.0, 0
        )
        gradients[node] = GShapeBsplineQ(
            particle, coordinate, 1.0, 0
        )


@ti.kernel
def _evaluate_cubic_bspline(
    particle: ti.f64,
    values: ti.template(),
    gradients: ti.template(),
):
    for node in values:
        coordinate = ti.cast(node, ti.f64)
        values[node] = ShapeBsplineC(
            particle, coordinate, 1.0, 0
        )
        gradients[node] = GShapeBsplineC(
            particle, coordinate, 1.0, 0
        )


@pytest.mark.parametrize(
    ("evaluator", "node_count"),
    [
        (_evaluate_linear, 6),
        (_evaluate_gimp, 6),
        (_evaluate_quadratic_bspline, 6),
        (_evaluate_cubic_bspline, 6),
    ],
    ids=("linear", "gimp", "quadratic-bspline", "cubic-bspline"),
)
def test_shape_partition_of_unity_and_zero_gradient(
    taichi_runtime, evaluator, node_count
):
    values = ti.field(ti.f64, shape=node_count)
    gradients = ti.field(ti.f64, shape=node_count)

    evaluator(2.37, values, gradients)

    assert np.sum(values.to_numpy()) == pytest.approx(
        1.0, abs=2.0e-14
    )
    assert np.sum(gradients.to_numpy()) == pytest.approx(
        0.0, abs=2.0e-14
    )


@pytest.mark.parametrize(
    ("evaluator", "node_count"),
    [
        (_evaluate_linear, 6),
        (_evaluate_gimp, 6),
        (_evaluate_quadratic_bspline, 6),
        (_evaluate_cubic_bspline, 6),
    ],
    ids=("linear", "gimp", "quadratic-bspline", "cubic-bspline"),
)
def test_shape_gradient_matches_finite_difference(
    taichi_runtime, evaluator, node_count
):
    values = ti.field(ti.f64, shape=node_count)
    gradients = ti.field(ti.f64, shape=node_count)
    step = 1.0e-6

    evaluator(2.37 + step, values, gradients)
    plus = values.to_numpy()
    evaluator(2.37 - step, values, gradients)
    minus = values.to_numpy()
    evaluator(2.37, values, gradients)
    analytical = gradients.to_numpy()

    np.testing.assert_allclose(
        analytical,
        (plus - minus) / (2.0 * step),
        rtol=2.0e-9,
        atol=2.0e-9,
    )
