from types import SimpleNamespace

import numpy as np

from src.mpm.soft_particle.GridTopology import (
    build_soft_mechanical_grid,
    select_soft_grid_topology,
)
from src.mpm.soft_particle.TemplateSupport import (
    HEXAHEDRON,
    TETRAHEDRON,
    build_hexahedral_template_support,
    build_tetrahedral_template_support,
    normalize_soft_grid_type,
)


class _Grid:
    def __init__(self, gnum=(7, 7, 7), spacing=0.5, extent=1):
        self.gnum = np.asarray(gnum, dtype=np.int32)
        self.gridSum = int(np.prod(self.gnum))
        self.grid_space = float(spacing)
        self.extent = int(extent)
        self.start_point = -0.5 * (self.gnum - 1) * self.grid_space

    def minBox(self):
        return self.start_point


class _Sphere:
    def __init__(self):
        self.grid = _Grid()
        self.mesh = SimpleNamespace(
            vertices=np.asarray(
                [
                    [x, y, z]
                    for x in (-1.0, 1.0)
                    for y in (-1.0, 1.0)
                    for z in (-1.0, 1.0)
                ],
                dtype=np.float64,
            )
        )
        self.volume = 4.0 * np.pi / 3.0
        self.eqradius = 1.0

    def __call__(self, points):
        return np.linalg.norm(np.asarray(points), axis=-1) - 1.0


def _template():
    return SimpleNamespace(objects=_Sphere())


def test_soft_grid_type_aliases_are_three_dimensional():
    assert normalize_soft_grid_type("hex") == ("Hexahedron", HEXAHEDRON)
    assert normalize_soft_grid_type("triangle") == (
        "Tetrahedron",
        TETRAHEDRON,
    )


def test_template_frame_factorization_matches_global_support_algebra():
    rng = np.random.default_rng(2718)
    rotation_seed = rng.normal(size=(3, 3))
    rotation, _ = np.linalg.qr(rotation_seed)
    if np.linalg.det(rotation) < 0.0:
        rotation[:, 0] *= -1.0
    scale = 1.73
    stress = rng.normal(size=(3, 3))
    template_gradients = rng.normal(size=(27, 3))
    nodal_velocities = rng.normal(size=(27, 3))

    global_gradients = template_gradients @ rotation.T / scale
    direct_internal = (stress @ global_gradients.T).T
    factored_internal = (
        (stress @ rotation / scale) @ template_gradients.T
    ).T

    direct_f_rate = sum(
        np.outer(velocity, gradient)
        for velocity, gradient in zip(
            nodal_velocities, global_gradients, strict=True
        )
    )
    template_f_rate = sum(
        np.outer(velocity, gradient)
        for velocity, gradient in zip(
            nodal_velocities, template_gradients, strict=True
        )
    )
    factored_f_rate = template_f_rate @ rotation.T / scale

    np.testing.assert_allclose(
        factored_internal, direct_internal, rtol=1.0e-14, atol=1.0e-14
    )
    np.testing.assert_allclose(
        factored_f_rate, direct_f_rate, rtol=1.0e-14, atol=1.0e-14
    )


def test_hexahedral_support_stores_compact_local_node_ids():
    template = _template()
    mechanical_grid = build_soft_mechanical_grid(
        template, spacing=0.5, shape_function_type=0
    )
    points = np.asarray(
        [[-0.25, -0.25, -0.25], [0.25, 0.25, 0.25]],
        dtype=np.float64,
    )
    topology = select_soft_grid_topology(
        template,
        points,
        shape_function_type=0,
        storage="Compact",
        padding_cells=1,
        mechanical_grid=mechanical_grid,
    )
    support = build_hexahedral_template_support(
        template,
        points,
        topology,
        shape_function_type=0,
        mechanical_grid=mechanical_grid,
    )

    assert np.all(support.point_count == 8)
    assert np.min(support.point_node) >= 0
    assert np.max(support.point_node) < topology.compact_count
    np.testing.assert_allclose(np.sum(support.point_shape, axis=1), 1.0)
    np.testing.assert_allclose(
        np.sum(support.point_dshape, axis=1), 0.0, atol=1.0e-14
    )

    local_node = int(support.point_node[1, 3])
    assert 17 + local_node != 113 + local_node
    assert 17 + local_node == 17 + support.point_node[1, 3]
    assert 113 + local_node == 113 + support.point_node[1, 3]


def test_tetrahedral_support_uses_gauss_points_and_independent_grid_resolution():
    template = _template()
    mechanical_grid = build_soft_mechanical_grid(
        template, spacing=0.25, shape_function_type=0
    )
    points, point_volume, topology, support = (
        build_tetrahedral_template_support(
            template,
            mechanical_grid=mechanical_grid,
            storage="Compact",
            padding_cells=2,
            levelset_extent_cells=1,
            verlet_padding_cells=1,
        )
    )

    np.testing.assert_array_equal(support.grid_shape, mechanical_grid.gnum)
    assert support.levelset_node_number == template.objects.grid.gridSum
    assert points.shape[0] == support.material_point_number
    assert np.all(point_volume > 0.0)
    np.testing.assert_allclose(
        np.sum(point_volume), template.objects.volume, rtol=1.0e-13
    )
    assert np.all(support.point_count == 4)
    assert np.min(support.point_node) >= 0
    assert np.max(support.point_node) < topology.compact_count
    np.testing.assert_allclose(np.sum(support.point_shape, axis=1), 1.0)
    np.testing.assert_allclose(
        np.sum(support.point_dshape, axis=1), 0.0, atol=1.0e-13
    )
    np.testing.assert_allclose(
        support.point_shape[:, :4], 0.25, atol=1.0e-13
    )


def test_tetrahedral_contact_region_refinement_is_conforming_and_conservative():
    template = _template()
    reference_volume = 4.25
    mechanical_grid = build_soft_mechanical_grid(
        template,
        spacing=0.5,
        shape_function_type=0,
        refinement={
            "RegionMin": [-0.55, -0.55, -1.05],
            "RegionMax": [0.55, 0.55, -0.05],
            "FineSpacing": 0.25,
        },
    )
    points, point_volume, topology, support = (
        build_tetrahedral_template_support(
            template,
            mechanical_grid=mechanical_grid,
            storage="Compact",
            padding_cells=1,
            levelset_extent_cells=1,
            verlet_padding_cells=1,
            reference_volume=reference_volume,
        )
    )

    assert not mechanical_grid.is_uniform
    assert mechanical_grid.minimum_spacing == 0.25
    assert support.grid_base_space == 0.5
    assert support.grid_space == 0.25
    assert np.ptp(point_volume) > 0.0
    np.testing.assert_allclose(
        np.sum(point_volume), reference_volume, rtol=1.0e-13
    )
    np.testing.assert_allclose(
        np.sum(support.point_shape, axis=1), 1.0, atol=1.0e-14
    )
    np.testing.assert_allclose(
        np.sum(support.point_dshape, axis=1), 0.0, atol=1.0e-13
    )
    assert np.all(support.surface_count == 4)
    np.testing.assert_allclose(
        np.sum(support.surface_shape, axis=1), 1.0, atol=1.0e-13
    )
    assert np.max(support.point_node) < topology.compact_count
    assert points.shape[0] == point_volume.shape[0]


def test_tetrahedral_mechanics_does_not_change_with_sdf_resolution():
    coarse = _template()
    fine = _template()
    fine.objects.grid = _Grid(gnum=(15, 15, 15), spacing=0.25, extent=2)
    mechanical_grid = build_soft_mechanical_grid(
        coarse, spacing=0.25, shape_function_type=0
    )

    coarse_result = build_tetrahedral_template_support(
        coarse,
        mechanical_grid=mechanical_grid,
        storage="Compact",
        padding_cells=1,
        levelset_extent_cells=1,
        verlet_padding_cells=1,
    )
    fine_result = build_tetrahedral_template_support(
        fine,
        mechanical_grid=mechanical_grid,
        storage="Compact",
        padding_cells=1,
        levelset_extent_cells=2,
        verlet_padding_cells=1,
    )

    np.testing.assert_allclose(coarse_result[0], fine_result[0])
    np.testing.assert_array_equal(
        coarse_result[3].point_node, fine_result[3].point_node
    )
    np.testing.assert_allclose(
        coarse_result[3].point_dshape, fine_result[3].point_dshape
    )
    assert (
        coarse_result[3].levelset_node_number
        != fine_result[3].levelset_node_number
    )


def test_hexahedral_mechanics_does_not_change_with_sdf_resolution():
    coarse = _template()
    fine = _template()
    fine.objects.grid = _Grid(gnum=(15, 15, 15), spacing=0.25, extent=2)
    mechanical_grid = build_soft_mechanical_grid(
        coarse, spacing=0.25, shape_function_type=0
    )
    points = np.asarray(
        [[-0.25, -0.25, -0.25], [0.25, 0.25, 0.25]],
        dtype=np.float64,
    )
    topology = select_soft_grid_topology(
        coarse,
        points,
        shape_function_type=0,
        storage="Compact",
        padding_cells=1,
        mechanical_grid=mechanical_grid,
    )
    coarse_support = build_hexahedral_template_support(
        coarse,
        points,
        topology,
        shape_function_type=0,
        mechanical_grid=mechanical_grid,
    )
    fine_support = build_hexahedral_template_support(
        fine,
        points,
        topology,
        shape_function_type=0,
        mechanical_grid=mechanical_grid,
    )

    np.testing.assert_array_equal(
        coarse_support.point_node, fine_support.point_node
    )
    np.testing.assert_allclose(
        coarse_support.point_dshape, fine_support.point_dshape
    )
    assert (
        coarse_support.levelset_node_number
        != fine_support.levelset_node_number
    )


def test_registering_one_template_twice_uploads_one_support_block(
    taichi_runtime,
):
    ti = taichi_runtime
    from src.mpm.soft_particle.SceneFields import (
        register_soft_template_support,
    )

    template = _template()
    mechanical_grid = build_soft_mechanical_grid(
        template, spacing=0.5, shape_function_type=0
    )
    points = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64)
    topology = select_soft_grid_topology(
        template,
        points,
        shape_function_type=0,
        storage="Compact",
        padding_cells=1,
        mechanical_grid=mechanical_grid,
    )
    support = build_hexahedral_template_support(
        template,
        points,
        topology,
        shape_function_type=0,
        mechanical_grid=mechanical_grid,
    )
    scene = SimpleNamespace(
        soft_template_support_registry={},
        softTemplatePointNum=np.zeros(1, dtype=np.int32),
        softTemplateSurfaceNum=np.zeros(1, dtype=np.int32),
        softTemplateSdfNum=np.zeros(1, dtype=np.int32),
        soft_shape_node=ti.field(
            ti.i32, shape=(support.material_point_number, 8)
        ),
        soft_shape=ti.field(
            float, shape=(support.material_point_number, 8)
        ),
        soft_dshape=ti.Vector.field(
            3, float, shape=(support.material_point_number, 8)
        ),
        soft_shape_count=ti.field(
            ti.i32, shape=support.material_point_number
        ),
        surface_shape_node=ti.field(
            ti.i32, shape=(support.surface_node_number, 8)
        ),
        surface_shape=ti.field(
            float, shape=(support.surface_node_number, 8)
        ),
        surface_shape_count=ti.field(
            ti.i32, shape=support.surface_node_number
        ),
        sdf_shape_node=ti.field(
            ti.i32, shape=(support.levelset_node_number, 8)
        ),
        sdf_shape=ti.field(
            float, shape=(support.levelset_node_number, 8)
        ),
        sdf_shape_count=ti.field(
            ti.i32, shape=support.levelset_node_number
        ),
    )

    first = register_soft_template_support(scene, support)
    second = register_soft_template_support(scene, support)
    assert first == second
    assert scene.softTemplatePointNum[0] == support.material_point_number
    assert scene.softTemplateSurfaceNum[0] == support.surface_node_number
    assert scene.softTemplateSdfNum[0] == support.levelset_node_number
