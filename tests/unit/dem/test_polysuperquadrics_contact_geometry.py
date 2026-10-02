import numpy as np
import pytest
import taichi as ti

from src.dem.contact.ContactKernel import implicit_surface_wall_contact_geometry
from src.dem.contact.ContactKernel import (
    ISparticle_wall_contact_model_type1,
    ISparticle_wall_contact_model_type2,
    implicit_surface_finite_wall_contact_geometry,
    kernel_ISparticle_wall_force_assemble_,
    update_wall_contact_table_,
)
from src.dem.contact.contact_point_root.ErodedGJK import GJKiteration
from src.dem.structs.BaseStruct import (
    ContactTable,
    FacetFamily,
    ImplicitSurfaceParticle,
    PatchFamily,
    PlaneFamily,
    PolySuperEllipsoid,
    PolySuperQuadrics,
    RollingContactTable,
)
from src.physics_model.contact_model.LinearRollingModel import LinearRollingSurfaceProperty
from src.physics_model.contact_model.LinearModel import LinearSurfaceProperty


pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.geometry, pytest.mark.cpu]


def _plane_support_oracle(shape, blockiness, normal):
    """Independent plane-intersection oracle."""
    a, b, c = np.asarray(shape, dtype=float)
    n1, n2 = np.asarray(blockiness, dtype=float)
    nx, ny, nz = np.asarray(normal, dtype=float)

    if abs(nx) < 1.0e-10 and abs(ny) < 1.0e-10:
        return np.asarray([0.0, 0.0, np.sign(nz) * c])

    if abs(nx) > abs(ny):
        alpha = abs((ny * b) / (nx * a)) ** (1.0 / (n2 - 1.0))
        gamma1 = 1.0 + abs(alpha) ** n2
        gamma = gamma1 ** (n1 / n2 - 1.0)
        beta = abs((nz * c) / (nx * a) * gamma) ** (1.0 / (n1 - 1.0))
        base = (gamma1 ** (n1 / n2) + abs(beta) ** n1) ** (-1.0 / n1)
        local = np.asarray([base, alpha * base, beta * base])
    else:
        alpha = abs((nx * a) / (ny * b)) ** (1.0 / (n2 - 1.0))
        gamma1 = 1.0 + abs(alpha) ** n2
        gamma = gamma1 ** (n1 / n2 - 1.0)
        beta = abs((nz * c) / (ny * b) * gamma) ** (1.0 / (n1 - 1.0))
        base = (gamma1 ** (n1 / n2) + abs(beta) ** n1) ** (-1.0 / n1)
        local = np.asarray([alpha * base, base, beta * base])

    return local * np.asarray(shape) * np.sign(normal)


def _closest_point_on_triangle(point, triangle):
    """Ericson closest-feature oracle used for the spherical special case."""
    point = np.asarray(point, dtype=float)
    a, b, c = np.asarray(triangle, dtype=float)
    ab, ac, ap = b - a, c - a, point - a
    d1, d2 = np.dot(ab, ap), np.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return a, 3

    bp = point - b
    d3, d4 = np.dot(ab, bp), np.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return b, 3

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        parameter = d1 / (d1 - d3)
        return a + parameter * ab, 2

    cp = point - c
    d5, d6 = np.dot(ab, cp), np.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return c, 3

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        parameter = d2 / (d2 - d6)
        return a + parameter * ac, 2

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        parameter = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + parameter * (c - b), 2

    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    return a + ab * v + ac * w, 1


def test_polysuperellipsoid_support_matches_liggghts_plane_oracle(taichi_runtime):
    primitive = PolySuperEllipsoid.field(shape=1)
    support = ti.Vector.field(3, dtype=ti.f64, shape=3)

    directions = np.asarray(
        [
            [0.3, -0.7, 0.4],
            [-0.4, 0.2, -0.8],
            [0.0, 0.0, -1.0],
        ]
    )

    @ti.kernel
    def evaluate():
        primitive[0]._add_template_parameter(2.0, 1.5, 1.0, 0.5, 0.8, 2.0, 1.5, 1.0)
        params = primitive[0].physical_parameters(1.0)
        local_directions = ti.Matrix.rows(
            [
                ti.Vector([0.3, -0.7, 0.4]),
                ti.Vector([-0.4, 0.2, -0.8]),
                ti.Vector([0.0, 0.0, -1.0]),
            ]
        )
        for i in ti.static(range(3)):
            support[i] = primitive[0].support(local_directions[i, :], params)

    evaluate()
    expected = np.asarray(
        [_plane_support_oracle([2.0, 1.5, 1.0], [2.0 / 0.8, 2.0 / 0.5], direction) for direction in directions]
    )
    np.testing.assert_allclose(support.to_numpy(), expected, atol=2.0e-10)


def test_oblique_wall_gap_matches_liggghts_maximum_penetration_oracle(taichi_runtime):
    primitive = PolySuperEllipsoid.field(shape=1)
    wall = PlaneFamily.field(shape=1)
    support = ti.Vector.field(3, dtype=ti.f64, shape=())
    projection = ti.Vector.field(3, dtype=ti.f64, shape=())
    gap = ti.field(dtype=ti.f64, shape=())

    angle = 0.4
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    center = np.asarray([0.2, -0.1, 0.7])
    normal = np.asarray([0.2, -0.3, 0.9327379053088815])
    normal /= np.linalg.norm(normal)

    @ti.kernel
    def evaluate():
        primitive[0]._add_template_parameter(2.0, 1.5, 1.0, 0.5, 0.8, 2.0, 1.5, 1.0)
        wall[0]._restart(
            1,
            0,
            0,
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([normal[0], normal[1], normal[2]]),
        )
        rotation_matrix = ti.Matrix(
            [
                [cosine, 0.0, sine],
                [0.0, 1.0, 0.0],
                [-sine, 0.0, cosine],
            ]
        )
        _, support[None], projection[None], gap[None] = implicit_surface_wall_contact_geometry(
            ti.Vector([center[0], center[1], center[2]]),
            1.0,
            rotation_matrix,
            primitive[0],
            wall[0],
        )

    evaluate()

    local_direction = rotation.T @ (-normal)
    local_support = _plane_support_oracle([2.0, 1.5, 1.0], [2.0 / 0.8, 2.0 / 0.5], local_direction)
    expected_support = center + rotation @ local_support
    expected_gap = np.dot(expected_support, normal)
    expected_projection = expected_support - expected_gap * normal

    np.testing.assert_allclose(support[None], expected_support, atol=2.0e-10)
    np.testing.assert_allclose(projection[None], expected_projection, atol=2.0e-10)
    assert gap[None] == pytest.approx(expected_gap, abs=2.0e-10)


def test_polysuperquadrics_support_satisfies_surface_and_normal_condition(taichi_runtime):
    primitive = PolySuperQuadrics.field(shape=1)
    support = ti.Vector.field(3, dtype=ti.f64, shape=3)
    value = ti.field(dtype=ti.f64, shape=3)
    gradient = ti.Vector.field(3, dtype=ti.f64, shape=3)

    @ti.kernel
    def evaluate():
        primitive[0]._add_template_parameter(2.0, 1.5, 1.0, 0.5, 0.8, 1.0, 3.0, 2.5, 4.0)
        params = primitive[0].physical_parameters(1.0)
        directions = ti.Matrix.rows(
            [
                ti.Vector([1.0, 0.0, 0.0]),
                ti.Vector([-1.0, 0.0, 0.0]),
                ti.Vector([0.35, -0.7, 0.4]),
            ]
        )
        for i in ti.static(range(3)):
            point = primitive[0].support(directions[i, :], params)
            support[i] = point
            value[i] = primitive[0].fx(point[0], point[1], point[2], params)
            gradient[i] = primitive[0].gradient(point[0], point[1], point[2], params)

    evaluate()

    np.testing.assert_allclose(support[0], [3.0, 0.0, 0.0], atol=1.0e-10)
    np.testing.assert_allclose(support[1], [-2.0, 0.0, 0.0], atol=1.0e-10)
    np.testing.assert_allclose(value.to_numpy(), 0.0, atol=2.0e-10)

    direction = np.asarray([0.35, -0.7, 0.4])
    grad = gradient[2]
    assert np.dot(grad, direction) > 0.0
    np.testing.assert_allclose(
        np.cross(grad / np.linalg.norm(grad), direction / np.linalg.norm(direction)),
        0.0,
        atol=2.0e-9,
    )


def test_plane_lowest_potential_satisfies_kkt_for_both_shapes(taichi_runtime):
    pse = PolySuperEllipsoid.field(shape=1)
    psq = PolySuperQuadrics.field(shape=1)
    points = ti.Vector.field(3, dtype=ti.f64, shape=2)
    gradients = ti.Vector.field(3, dtype=ti.f64, shape=2)
    direction = np.asarray([0.3, -0.4, 0.8660254037844386])
    offset = -0.7

    @ti.kernel
    def evaluate():
        normal = ti.Vector([direction[0], direction[1], direction[2]])
        pse[0]._add_template_parameter(2.0, 1.5, 1.0, 0.5, 0.8, 2.5, 1.8, 1.2)
        psq[0]._add_template_parameter(2.0, 1.5, 1.0, 0.5, 0.8, 1.0, 2.5, 1.8, 1.2)
        params_pse = pse[0].physical_parameters(1.0)
        params_psq = psq[0].physical_parameters(1.0)
        points[0] = pse[0].plane_minimum(normal, offset, params_pse)
        points[1] = psq[0].plane_minimum(normal, offset, params_psq)
        gradients[0] = pse[0].gradient(*points[0], params_pse)
        gradients[1] = psq[0].gradient(*points[1], params_psq)

    evaluate()

    for point, gradient in zip(points.to_numpy(), gradients.to_numpy()):
        assert np.dot(direction, point) == pytest.approx(offset, abs=2.0e-8)
        normalized_gradient = gradient / np.linalg.norm(gradient)
        assert abs(np.dot(normalized_gradient, direction)) > 1.0 - 2.0e-8


def test_asymmetric_polysuperellipsoid_uses_rad1_on_negative_side(taichi_runtime):
    primitive = PolySuperEllipsoid.field(shape=1)
    support = ti.Vector.field(3, dtype=ti.f64, shape=2)

    @ti.kernel
    def evaluate():
        primitive[0]._add_template_parameter(2.0, 1.0, 1.0, 1.0, 1.0, 3.0, 1.0, 1.0)
        params = primitive[0].physical_parameters(1.0)
        support[0] = primitive[0].support(ti.Vector([-1.0, 0.0, 0.0]), params)
        support[1] = primitive[0].support(ti.Vector([1.0, 0.0, 0.0]), params)

    evaluate()
    np.testing.assert_allclose(support[0], [-2.0, 0.0, 0.0], atol=1.0e-12)
    np.testing.assert_allclose(support[1], [3.0, 0.0, 0.0], atol=1.0e-12)


def test_polysuperquadrics_wall_geometry_preserves_signed_gap(taichi_runtime):
    primitive = PolySuperQuadrics.field(shape=1)
    wall = PlaneFamily.field(shape=1)
    support = ti.Vector.field(3, dtype=ti.f64, shape=2)
    projection = ti.Vector.field(3, dtype=ti.f64, shape=2)
    gap = ti.field(dtype=ti.f64, shape=2)

    @ti.kernel
    def evaluate():
        primitive[0]._add_template_parameter(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0)
        wall[0]._restart(
            1,
            0,
            0,
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([0.0, 0.0, 1.0]),
        )
        rotation = ti.Matrix.identity(float, 3)
        for i in ti.static(range(2)):
            center = ti.Vector([0.0, 0.0, 0.8 + 0.4 * i])
            _, support[i], projection[i], gap[i] = implicit_surface_wall_contact_geometry(
                center, 1.0, rotation, primitive[0], wall[0]
            )

    evaluate()
    np.testing.assert_allclose(support[0], [0.0, 0.0, -0.2], atol=1.0e-10)
    np.testing.assert_allclose(projection[0], [0.0, 0.0, 0.0], atol=1.0e-10)
    assert gap[0] == pytest.approx(-0.2, abs=1.0e-10)
    assert gap[1] == pytest.approx(0.2, abs=1.0e-10)


@pytest.mark.parametrize(
    ("center", "expected_type"),
    [
        ([0.0, 0.5, 0.8], 1),
        ([0.0, -0.6, 0.6], 2),
        ([-1.4, -0.4, 0.6], 3),
        ([2.0, -0.4, 0.5], 3),
    ],
)
def test_finite_triangle_features_match_sphere_triangle_oracle(taichi_runtime, center, expected_type):
    primitive = PolySuperQuadrics.field(shape=1)
    wall = FacetFamily.field(shape=1)
    normal = ti.Vector.field(3, dtype=ti.f64, shape=())
    support = ti.Vector.field(3, dtype=ti.f64, shape=())
    wall_point = ti.Vector.field(3, dtype=ti.f64, shape=())
    gap = ti.field(dtype=ti.f64, shape=())
    feature_type = ti.field(dtype=ti.i32, shape=())

    triangle = np.asarray([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    @ti.kernel
    def evaluate():
        primitive[0]._add_template_parameter(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        wall[0]._restart(
            1,
            0,
            0,
            ti.Vector([-1.0, 0.0, 0.0]),
            ti.Vector([1.0, 0.0, 0.0]),
            ti.Vector([0.0, 2.0, 0.0]),
            ti.Vector([0.0, 0.0, 1.0]),
            ti.Vector([0.0, 0.0, 0.0]),
        )
        (
            normal[None],
            support[None],
            wall_point[None],
            gap[None],
            feature_type[None],
        ) = implicit_surface_finite_wall_contact_geometry(
            ti.Vector([center[0], center[1], center[2]]),
            1.0,
            ti.Matrix.identity(ti.f64, 3),
            primitive[0],
            wall[0],
        )

    evaluate()

    expected_wall_point, oracle_type = _closest_point_on_triangle(center, triangle)
    distance = np.linalg.norm(np.asarray(center) - expected_wall_point)
    expected_normal = (np.asarray(center) - expected_wall_point) / distance
    expected_support = np.asarray(center) - expected_normal

    assert oracle_type == expected_type
    assert feature_type[None] == expected_type
    np.testing.assert_allclose(wall_point[None], expected_wall_point, atol=2.0e-7)
    np.testing.assert_allclose(normal[None], expected_normal, atol=2.0e-7)
    np.testing.assert_allclose(support[None], expected_support, atol=2.0e-7)
    assert gap[None] == pytest.approx(distance - 1.0, abs=2.0e-7)


def test_polysuperellipsoid_finite_edge_contact_matches_sphere_oracle(taichi_runtime):
    primitive = PolySuperEllipsoid.field(shape=1)
    wall = FacetFamily.field(shape=1)
    normal = ti.Vector.field(3, dtype=ti.f64, shape=())
    support = ti.Vector.field(3, dtype=ti.f64, shape=())
    wall_point = ti.Vector.field(3, dtype=ti.f64, shape=())
    gap = ti.field(dtype=ti.f64, shape=())
    feature_type = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def evaluate():
        primitive[0]._add_template_parameter(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        wall[0]._restart(
            1,
            0,
            0,
            ti.Vector([-1.0, 0.0, 0.0]),
            ti.Vector([1.0, 0.0, 0.0]),
            ti.Vector([0.0, 2.0, 0.0]),
            ti.Vector([0.0, 0.0, 1.0]),
            ti.Vector([0.0, 0.0, 0.0]),
        )
        (
            normal[None],
            support[None],
            wall_point[None],
            gap[None],
            feature_type[None],
        ) = implicit_surface_finite_wall_contact_geometry(
            ti.Vector([0.0, -0.6, 0.6]),
            1.0,
            ti.Matrix.identity(ti.f64, 3),
            primitive[0],
            wall[0],
        )

    evaluate()

    distance = np.sqrt(0.6**2 + 0.6**2)
    expected_normal = np.asarray([0.0, -0.6, 0.6]) / distance
    assert feature_type[None] == 2
    np.testing.assert_allclose(wall_point[None], [0.0, 0.0, 0.0], atol=2.0e-7)
    np.testing.assert_allclose(normal[None], expected_normal, atol=2.0e-7)
    np.testing.assert_allclose(support[None], np.asarray([0.0, -0.6, 0.6]) - expected_normal, atol=2.0e-7)
    assert gap[None] == pytest.approx(distance - 1.0, abs=2.0e-7)


def test_finite_patch_offset_is_used_by_feature_geometry(taichi_runtime):
    primitive = PolySuperQuadrics.field(shape=1)
    wall = PatchFamily.field(shape=1)
    support = ti.Vector.field(3, dtype=ti.f64, shape=())
    wall_point = ti.Vector.field(3, dtype=ti.f64, shape=())
    gap = ti.field(dtype=ti.f64, shape=())
    feature_type = ti.field(dtype=ti.i32, shape=())

    @ti.kernel
    def evaluate():
        primitive[0]._add_template_parameter(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        wall[0]._restart(
            1,
            0,
            0,
            ti.Vector([-1.0, 0.0, 0.0]),
            ti.Vector([1.0, 0.0, 0.0]),
            ti.Vector([0.0, 2.0, 0.0]),
            ti.Vector([0.0, 0.0, 1.0]),
            0.1,
        )
        (
            _,
            support[None],
            wall_point[None],
            gap[None],
            feature_type[None],
        ) = implicit_surface_finite_wall_contact_geometry(
            ti.Vector([0.0, 0.5, 0.8]),
            1.0,
            ti.Matrix.identity(ti.f64, 3),
            primitive[0],
            wall[0],
        )

    evaluate()

    assert feature_type[None] == 1
    np.testing.assert_allclose(support[None], [0.0, 0.5, -0.2], atol=2.0e-7)
    np.testing.assert_allclose(wall_point[None], [0.0, 0.5, 0.1], atol=2.0e-7)
    assert gap[None] == pytest.approx(-0.3, abs=2.0e-7)


def test_finite_edge_force_uses_edge_normal(taichi_runtime):
    primitive = PolySuperQuadrics.field(shape=1)
    rigid = ImplicitSurfaceParticle.field(shape=1)
    wall = FacetFamily.field(shape=1)
    contacts = ContactTable.field(shape=1)
    properties = LinearSurfaceProperty.field(shape=1)
    particle_wall = ti.field(dtype=ti.i32, shape=2)
    contact_type = ti.field(dtype=ti.u8, shape=1)
    dt = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        primitive[0]._add_template_parameter(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        rigid[0].materialID = ti.u8(0)
        rigid[0].templateID = ti.u8(0)
        rigid[0].scale = 1.0
        rigid[0].m = 1.0
        rigid[0].equi_r = 1.0
        rigid[0].mass_center = ti.Vector([0.0, -0.6, 0.6])
        rigid[0].q = ti.Vector([0.0, 0.0, 0.0, 1.0])
        wall[0]._restart(
            1,
            0,
            0,
            ti.Vector([-1.0, 0.0, 0.0]),
            ti.Vector([1.0, 0.0, 0.0]),
            ti.Vector([0.0, 2.0, 0.0]),
            ti.Vector([0.0, 0.0, 1.0]),
            ti.Vector([0.0, 0.0, 0.0]),
        )
        contacts[0]._set_id(0, 0)
        properties[0].kn = 100.0
        properties[0].ks = 50.0
        properties[0].ncut = 0.0
        particle_wall[1] = 1
        contact_type[0] = ti.u8(2)
        dt[None] = 1.0e-3

    initialize()
    kernel_ISparticle_wall_force_assemble_(
        1,
        1,
        dt,
        1,
        properties,
        rigid,
        primitive,
        wall,
        contacts,
        particle_wall,
        contact_type,
        ISparticle_wall_contact_model_type1,
    )

    distance = np.sqrt(0.6**2 + 0.6**2)
    expected_normal = np.asarray([0.0, -0.6, 0.6]) / distance
    expected_force = 100.0 * (1.0 - distance) * expected_normal
    np.testing.assert_allclose(contacts[0].cnforce, expected_force, atol=2.0e-5)
    np.testing.assert_allclose(rigid[0].contact_force, expected_force, atol=2.0e-5)


def test_finite_wall_table_deduplicates_shared_feature(taichi_runtime):
    primitive = PolySuperQuadrics.field(shape=1)
    rigid = ImplicitSurfaceParticle.field(shape=1)
    wall = FacetFamily.field(shape=2)
    contacts = ContactTable.field(shape=2)
    particle_wall = ti.field(dtype=ti.i32, shape=2)
    potential_walls = ti.field(dtype=ti.i32, shape=2)
    contact_type = ti.field(dtype=ti.u8, shape=2)

    @ti.kernel
    def initialize():
        primitive[0]._add_template_parameter(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        rigid[0].templateID = ti.u8(0)
        rigid[0].scale = 1.0
        rigid[0].equi_r = 1.0
        rigid[0].mass_center = ti.Vector([0.0, 0.0, 0.8])
        rigid[0].q = ti.Vector([0.0, 0.0, 0.0, 1.0])
        wall[0]._restart(
            1,
            0,
            0,
            ti.Vector([-1.0, 0.0, 0.0]),
            ti.Vector([1.0, 0.0, 0.0]),
            ti.Vector([0.0, 2.0, 0.0]),
            ti.Vector([0.0, 0.0, 1.0]),
            ti.Vector([0.0, 0.0, 0.0]),
        )
        wall[1]._restart(
            1,
            1,
            0,
            ti.Vector([1.0, 0.0, 0.0]),
            ti.Vector([-1.0, 0.0, 0.0]),
            ti.Vector([0.0, -2.0, 0.0]),
            ti.Vector([0.0, 0.0, 1.0]),
            ti.Vector([0.0, 0.0, 0.0]),
        )
        particle_wall[0] = 0
        particle_wall[1] = 2
        potential_walls[0] = 0
        potential_walls[1] = 1

    initialize()
    update_wall_contact_table_(
        2,
        1,
        rigid,
        primitive,
        wall,
        particle_wall,
        potential_walls,
        contacts,
        contact_type,
    )

    assert np.count_nonzero(contact_type.to_numpy()) == 1


def test_eroded_gjk_keeps_touching_and_deep_ball_ball_contacts(taichi_runtime):
    primitive = PolySuperEllipsoid.field(shape=2)
    touching = ti.field(dtype=ti.i32, shape=3)
    point_a = ti.Vector.field(3, dtype=ti.f64, shape=3)
    point_b = ti.Vector.field(3, dtype=ti.f64, shape=3)

    @ti.kernel
    def evaluate():
        for i in ti.static(range(2)):
            primitive[i]._add_template_parameter(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        rotation = ti.Matrix.identity(float, 3)
        distances = ti.Vector([1.9, 1.8, 2.1])
        for i in ti.static(range(3)):
            is_touch, pa, pb, _ = GJKiteration(
                0.05,
                0.05,
                1.0,
                1.0,
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Vector([distances[i], 0.0, 0.0]),
                ti.Vector([0.0, 0.0, 0.0]),
                1.0,
                1.0,
                rotation,
                rotation,
                primitive[0],
                primitive[1],
            )
            touching[i] = is_touch
            point_a[i] = pa
            point_b[i] = pb

    evaluate()
    np.testing.assert_array_equal(touching.to_numpy(), [1, 1, 0])
    assert np.linalg.norm(point_a[0] - point_b[0]) == pytest.approx(0.1, abs=1.0e-9)
    assert np.linalg.norm(point_a[1] - point_b[1]) == pytest.approx(0.2, abs=1.0e-9)


def test_deep_rotated_ball_ball_contact_has_common_normal_witnesses(taichi_runtime):
    primitive = PolySuperEllipsoid.field(shape=2)
    touching = ti.field(dtype=ti.i32, shape=())
    point_a = ti.Vector.field(3, dtype=ti.f64, shape=())
    point_b = ti.Vector.field(3, dtype=ti.f64, shape=())
    separating_axis = ti.Vector.field(3, dtype=ti.f64, shape=())

    @ti.kernel
    def evaluate():
        for i in ti.static(range(2)):
            primitive[i]._add_template_parameter(1.2, 0.9, 0.8, 0.6, 0.8, 1.2, 0.9, 0.8)
        angle1 = 0.35
        angle2 = -0.45
        c1, s1 = ti.cos(angle1), ti.sin(angle1)
        c2, s2 = ti.cos(angle2), ti.sin(angle2)
        rotation1 = ti.Matrix([[c1, 0.0, s1], [0.0, 1.0, 0.0], [-s1, 0.0, c1]])
        rotation2 = ti.Matrix([[c2, -s2, 0.0], [s2, c2, 0.0], [0.0, 0.0, 1.0]])
        is_touch, pa, pb, axis = GJKiteration(
            0.05,
            0.05,
            1.0,
            1.0,
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([1.0, 0.15, 0.05]),
            ti.Vector([0.0, 0.0, 0.0]),
            1.0,
            1.0,
            rotation1,
            rotation2,
            primitive[0],
            primitive[1],
        )
        touching[None] = is_touch
        point_a[None] = pa
        point_b[None] = pb
        separating_axis[None] = axis

    evaluate()
    assert touching[None] == 1
    overlap = point_a[None] - point_b[None]
    axis = -separating_axis[None]
    assert np.linalg.norm(overlap) > 0.1
    assert np.linalg.norm(overlap) < 1.4
    assert np.dot(overlap / np.linalg.norm(overlap), axis / np.linalg.norm(axis)) > 0.999


def test_polysuperquadrics_rolling_wall_kernel_uses_signed_gap(taichi_runtime):
    primitive = PolySuperQuadrics.field(shape=1)
    rigid = ImplicitSurfaceParticle.field(shape=1)
    wall = PlaneFamily.field(shape=1)
    contacts = RollingContactTable.field(shape=1)
    properties = LinearRollingSurfaceProperty.field(shape=1)
    particle_wall = ti.field(dtype=ti.i32, shape=2)
    contact_type = ti.field(dtype=ti.u8, shape=1)
    dt = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        primitive[0]._add_template_parameter(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0)
        rigid[0].materialID = ti.u8(0)
        rigid[0].templateID = ti.u8(0)
        rigid[0].scale = 1.0
        rigid[0].m = 1.0
        rigid[0].equi_r = 1.0
        rigid[0].mass_center = ti.Vector([0.0, 0.0, 0.8])
        rigid[0].v = ti.Vector([0.0, 0.0, 0.0])
        rigid[0].w = ti.Vector([0.0, 0.0, 0.0])
        rigid[0].q = ti.Vector([0.0, 0.0, 0.0, 1.0])
        wall[0]._restart(
            1,
            0,
            0,
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([0.0, 0.0, 1.0]),
        )
        contacts[0]._set_id(0, 0)
        properties[0].kn = 100.0
        properties[0].ks = 50.0
        properties[0].kr = 10.0
        properties[0].kt = 10.0
        properties[0].ncut = 0.0
        particle_wall[0] = 0
        particle_wall[1] = 1
        contact_type[0] = ti.u8(1)
        dt[None] = 1.0e-3

    initialize()
    kernel_ISparticle_wall_force_assemble_(
        1,
        0,
        dt,
        1,
        properties,
        rigid,
        primitive,
        wall,
        contacts,
        particle_wall,
        contact_type,
        ISparticle_wall_contact_model_type2,
    )

    np.testing.assert_allclose(contacts[0].cnforce, [0.0, 0.0, 20.0], atol=1.0e-8)
    np.testing.assert_allclose(rigid[0].contact_force, [0.0, 0.0, 20.0], atol=1.0e-8)


def test_polysuperquadrics_linear_wall_kernel_uses_signed_gap(taichi_runtime):
    primitive = PolySuperQuadrics.field(shape=1)
    rigid = ImplicitSurfaceParticle.field(shape=1)
    wall = PlaneFamily.field(shape=1)
    contacts = ContactTable.field(shape=1)
    properties = LinearSurfaceProperty.field(shape=1)
    particle_wall = ti.field(dtype=ti.i32, shape=2)
    contact_type = ti.field(dtype=ti.u8, shape=1)
    dt = ti.field(dtype=ti.f64, shape=())

    @ti.kernel
    def initialize():
        primitive[0]._add_template_parameter(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0)
        rigid[0].materialID = ti.u8(0)
        rigid[0].templateID = ti.u8(0)
        rigid[0].scale = 1.0
        rigid[0].m = 1.0
        rigid[0].equi_r = 1.0
        rigid[0].mass_center = ti.Vector([0.0, 0.0, 0.8])
        rigid[0].q = ti.Vector([0.0, 0.0, 0.0, 1.0])
        wall[0]._restart(
            1,
            0,
            0,
            ti.Vector([0.0, 0.0, 0.0]),
            ti.Vector([0.0, 0.0, 1.0]),
        )
        contacts[0]._set_id(0, 0)
        properties[0].kn = 100.0
        properties[0].ks = 50.0
        properties[0].ncut = 0.0
        particle_wall[1] = 1
        contact_type[0] = ti.u8(1)
        dt[None] = 1.0e-3

    initialize()
    kernel_ISparticle_wall_force_assemble_(
        1,
        0,
        dt,
        1,
        properties,
        rigid,
        primitive,
        wall,
        contacts,
        particle_wall,
        contact_type,
        ISparticle_wall_contact_model_type1,
    )

    np.testing.assert_allclose(contacts[0].cnforce, [0.0, 0.0, 20.0], atol=1.0e-8)
    np.testing.assert_allclose(rigid[0].contact_force, [0.0, 0.0, 20.0], atol=1.0e-8)
