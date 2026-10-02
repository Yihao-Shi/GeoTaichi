import numpy as np
import pytest
import taichi as ti

from src.dem.structs.BaseStruct import PolySuperEllipsoid


pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.geometry, pytest.mark.cpu]


def test_spherical_special_case_has_exact_implicit_derivatives(taichi_runtime):
    primitive = PolySuperEllipsoid.field(shape=1)
    value = ti.field(dtype=ti.f64, shape=())
    gradient = ti.Vector.field(3, dtype=ti.f64, shape=())
    hessian = ti.Matrix.field(3, 3, dtype=ti.f64, shape=())
    nearest = ti.Vector.field(3, dtype=ti.f64, shape=())
    support = ti.Vector.field(3, dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate():
        primitive[0]._add_template_parameter(
            2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0
        )
        parameters = primitive[0].physical_parameters(1.0)
        point = ti.Vector([1.0, 0.5, 0.25])
        value[None] = primitive[0].fx(
            point[0], point[1], point[2], parameters
        )
        gradient[None] = primitive[0].gradient(
            point[0], point[1], point[2], parameters
        )
        hessian[None] = primitive[0].hessian(
            point[0], point[1], point[2], parameters
        )
        nearest[None] = primitive[0].nearest_point(
            ti.Vector([3.0, 0.0, 0.0]),
            ti.Vector([-1.0, 0.0, 0.0]),
            parameters,
        )
        support[None] = primitive[0].support(
            ti.Vector([1.0, 0.0, 0.0]), parameters
        )

    evaluate()
    point = np.asarray([1.0, 0.5, 0.25])
    assert value[None] == pytest.approx(np.dot(point, point) / 4.0 - 1.0)
    np.testing.assert_allclose(gradient[None], 0.5 * point, atol=1.0e-12)
    np.testing.assert_allclose(hessian[None], 0.5 * np.eye(3), atol=1.0e-12)
    np.testing.assert_allclose(nearest[None], [2.0, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(support[None], [2.0, 0.0, 0.0], atol=1.0e-12)
