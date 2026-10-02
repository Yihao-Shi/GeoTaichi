import math

import numpy as np
import pytest
import taichi as ti

from src.dem.structs.BaseStruct import FacetFamily
from src.utils.GeometryFunction import (
    DistanceFromPointToTriangle2,
    PointProjectionToRectangle,
)


pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.geometry, pytest.mark.cpu]


def test_point_projection_to_axis_aligned_rectangle(taichi_runtime):
    result = ti.Vector.field(3, dtype=ti.f64, shape=())

    @ti.kernel
    def project():
        result[None] = PointProjectionToRectangle(
            ti.Vector([5.0, 0.0, 4.0]),
            ti.Vector([1.0, 2.0, 3.0]),
            ti.Vector([0.0, 0.0, 1.0]),
            2.0,
            1.0,
            0.5,
        )

    project()
    np.testing.assert_allclose(result[None], [3.0, 1.0, 3.5], atol=1.0e-12)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.5, 0.5, 2.0), 2.0),
        ((-1.0, 0.0, 0.0), 1.0),
        ((2.0, 2.0, 0.0), 3.0 / math.sqrt(2.0)),
    ],
)
def test_point_triangle_distance_matches_elementary_geometry(
    taichi_runtime, point, expected
):
    result = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def distance(query: ti.types.vector(3, ti.f64)):
        result[None] = DistanceFromPointToTriangle2(
            query,
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([1.0, 0.0, 0.0]),
            ti.Vector([0.0, 1.0, 0.0]),
        )

    distance(ti.Vector(point))
    assert result[None] == pytest.approx(expected, abs=1.0e-12)


def test_two_triangles_partition_an_interior_sphere_cross_section(taichi_runtime):
    facets = FacetFamily.field(shape=2)
    fractions = ti.Vector.field(2, dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        facets[0].add_wall_geometry(
            wallID=0,
            vertice1=ti.Vector([0.0, 0.0, 1.1]),
            vertice2=ti.Vector([15.0, 0.0, 1.1]),
            vertice3=ti.Vector([15.0, 6.0, 1.1]),
            norm=ti.Vector([0.0, 0.0, 1.0]),
            init_v=ti.Vector([0.0, 0.0, 0.0]),
        )
        facets[1].add_wall_geometry(
            wallID=1,
            vertice1=ti.Vector([0.0, 0.0, 1.1]),
            vertice2=ti.Vector([15.0, 6.0, 1.1]),
            vertice3=ti.Vector([0.0, 6.0, 1.1]),
            norm=ti.Vector([0.0, 0.0, 1.0]),
            init_v=ti.Vector([0.0, 0.0, 0.0]),
        )

    @ti.kernel
    def evaluate():
        center = ti.Vector([7.5, 3.0, 2.6])
        radius = 1.6
        for index in ti.static(range(2)):
            distance = facets[index]._get_norm_distance(center)
            fractions[None][index] = facets[index].processCircleShape(
                center, radius, distance
            )

    initialize()
    evaluate()
    fraction_values = np.asarray(fractions[None])
    np.testing.assert_allclose(fraction_values, [0.5, 0.5], atol=2.0e-6)
    assert float(np.sum(fraction_values)) == pytest.approx(1.0, abs=2.0e-6)
