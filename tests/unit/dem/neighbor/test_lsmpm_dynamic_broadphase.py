from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.contact_detection.bounding_volume_hierarchy.AABB import AABB
from src.dem.BaseKernel import (
    reset_verlet_disp_,
    validate_deformable_bounding_sphere_,
)
from src.dem.structs.BaseStruct import (
    BoundingBox,
    DeformableBoundingSphere,
    RigidBody,
)
from src.dem.neighbor.LinkedCell import LinkedCell
from src.dem.neighbor.NeighborBase import NeighborBase
from src.utils.TypeDefination import vec3i


pytestmark = [pytest.mark.unit, pytest.mark.dem, pytest.mark.cpu]


def initialize_taichi() -> None:
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        log_level=ti.ERROR,
    )


def test_radial_growth_triggers_verlet_rebuild_without_center_motion() -> None:
    initialize_taichi()
    try:
        sphere = DeformableBoundingSphere.field(shape=1)

        @ti.kernel
        def initialize():
            sphere[0]._set_deformed_shape(
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Matrix.identity(float, 3),
                ti.Vector([-1.0, -0.5, -0.25]),
                ti.Vector([1.0, 0.5, 0.25]),
                1.0,
                0.1,
            )

        @ti.kernel
        def expand_without_translation():
            sphere[0]._follow_deformed_shape(
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Matrix.identity(float, 3),
                ti.Vector([-1.4, -0.5, -0.25]),
                ti.Vector([1.4, 0.5, 0.25]),
                1.4,
                0.1,
            )

        @ti.kernel
        def shrink_without_translation():
            sphere[0]._follow_deformed_shape(
                ti.Vector([0.0, 0.0, 0.0]),
                ti.Matrix.identity(float, 3),
                ti.Vector([-1.0, -0.5, -0.25]),
                ti.Vector([1.0, 0.5, 0.25]),
                1.0,
                0.1,
            )

        initialize()
        initial_radius = float(sphere.rad.to_numpy()[0])
        np.testing.assert_allclose(initial_radius, 1.1, atol=1.0e-12)
        expand_without_translation()
        expanded_radius = float(sphere.rad.to_numpy()[0])
        np.testing.assert_allclose(sphere.verletDisp.to_numpy()[0], 0.0, atol=1.0e-15)
        assert expanded_radius > initial_radius + 0.2
        assert validate_deformable_bounding_sphere_(0.2**2, 1, sphere) == 1

        reset_verlet_disp_(1, sphere)
        shrink_without_translation()
        assert validate_deformable_bounding_sphere_(0.2**2, 1, sphere) == 0
    finally:
        ti.reset()


def test_lsmpm_bvh_uses_current_shape_aabb_instead_of_sdf_domain_sphere() -> None:
    initialize_taichi()
    try:
        sphere = DeformableBoundingSphere.field(shape=1)
        rigid = RigidBody.field(shape=1)
        box = BoundingBox.field(shape=1)
        aabb = AABB(n_aabbs=1)

        @ti.kernel
        def initialize():
            sphere[0]._add_bounding_sphere(
                ti.Vector([3.0, 4.0, 5.0]), 100.0
            )
            rigid[0].mass_center = ti.Vector([3.0, 4.0, 5.0])
            rigid[0].q = ti.Vector([1.0, 0.0, 0.0, 0.0])
            rigid[0].is_soft = ti.u8(1)
            box[0]._set_bounding_box(
                ti.Vector([-10.0, -10.0, -10.0]),
                ti.Vector([10.0, 10.0, 10.0]),
            )
            box[0]._add_grid(0, 0.1, ti.Vector([201, 201, 201]), 1.0, 4)
            box[0]._set_shape_box(
                ti.Vector([-1.0, -2.0, -0.5]),
                ti.Vector([1.0, 2.0, 0.5]),
            )

        initialize()
        aabb.set_lsmpm_body_aabbs(1, 0, 0.2, sphere, rigid, box)
        expected_min = np.array([3.0, 4.0, 5.0]) - np.array([1.0, 2.0, 0.5]) - 0.3
        expected_max = np.array([3.0, 4.0, 5.0]) + np.array([1.0, 2.0, 0.5]) + 0.3
        np.testing.assert_allclose(aabb.aabbs.min.to_numpy()[0], expected_min)
        np.testing.assert_allclose(aabb.aabbs.max.to_numpy()[0], expected_max)
        assert np.max(aabb.aabbs.max.to_numpy()[0] - aabb.aabbs.min.to_numpy()[0]) < 5.0
    finally:
        ti.reset()


def test_lsmpm_bvh_world_aabb_encloses_all_rotated_obb_corners() -> None:
    initialize_taichi()
    try:
        sphere = DeformableBoundingSphere.field(shape=1)
        rigid = RigidBody.field(shape=1)
        box = BoundingBox.field(shape=1)
        aabb = AABB(n_aabbs=1)
        mass_center = np.array([3.0, -1.5, 2.0])
        shape_min = np.array([-1.0, -2.0, -0.5])
        shape_max = np.array([2.0, 1.0, 0.75])
        axis = np.array([1.0, 2.0, 3.0])
        axis /= np.linalg.norm(axis)
        angle = 0.71
        quaternion = np.r_[axis * np.sin(0.5 * angle), np.cos(0.5 * angle)]

        @ti.kernel
        def initialize():
            rigid[0].mass_center = ti.Vector(
                [mass_center[0], mass_center[1], mass_center[2]]
            )
            rigid[0].q = ti.Vector(
                [quaternion[0], quaternion[1], quaternion[2], quaternion[3]]
            )
            rigid[0].is_soft = ti.u8(1)
            box[0]._set_bounding_box(
                ti.Vector([-10.0, -10.0, -10.0]),
                ti.Vector([10.0, 10.0, 10.0]),
            )
            box[0]._add_grid(0, 0.1, ti.Vector([201, 201, 201]), 1.0, 4)
            box[0]._set_shape_box(
                ti.Vector([shape_min[0], shape_min[1], shape_min[2]]),
                ti.Vector([shape_max[0], shape_max[1], shape_max[2]]),
            )

        initialize()
        verlet_distance = 0.2
        aabb.set_lsmpm_body_aabbs(
            1, 0, verlet_distance, sphere, rigid, box
        )

        cross = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = (
            np.cos(angle) * np.eye(3)
            + (1.0 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * cross
        )
        corners = np.array(
            [
                [x, y, z]
                for x in (shape_min[0], shape_max[0])
                for y in (shape_min[1], shape_max[1])
                for z in (shape_min[2], shape_max[2])
            ]
        )
        world_corners = mass_center + corners @ rotation.T
        padding = verlet_distance + 0.1
        expected_min = world_corners.min(axis=0) - padding
        expected_max = world_corners.max(axis=0) + padding
        actual_min = aabb.aabbs.min.to_numpy()[0]
        actual_max = aabb.aabbs.max.to_numpy()[0]

        np.testing.assert_allclose(actual_min, expected_min, atol=1.0e-12)
        np.testing.assert_allclose(actual_max, expected_max, atol=1.0e-12)
        assert np.all(world_corners >= actual_min - 1.0e-12)
        assert np.all(world_corners <= actual_max + 1.0e-12)
    finally:
        ti.reset()


def test_linked_cell_adapts_within_initialized_capacity() -> None:
    class SimulationStub:
        domain = np.array([10.0, 10.0, 10.0])
        verlet_distance = 0.1
        wall_type = None

        def set_max_bounding_sphere_radius(self, value):
            self.maximum_radius = value

        def set_min_bounding_sphere_radius(self, value):
            self.minimum_radius = value

    class SceneStub:
        radii = (0.5, 1.0)

        def find_bounding_sphere_radius(self, _):
            return self.radii

    linked = LinkedCell.__new__(LinkedCell)
    linked.sims = SimulationStub()
    linked.minimum_grid_size = 1.0
    linked.grid_size = 1.0
    linked.igrid_size = 1.0
    linked.contact_grid_size = 1.0
    linked.cnum = vec3i([10, 10, 10])
    linked.cellSum = 1000
    linked.allocated_cell_sum = 1000
    linked.plane_insert_factor = 0.0
    linked.adaptive_regrid_count = 0
    linked.adaptive_cell_size = True
    scene = SceneStub()

    assert linked.refresh_contact_grid(scene)
    assert linked.grid_size == pytest.approx(2.2)
    np.testing.assert_array_equal(np.asarray(linked.cnum), [4, 4, 4])
    assert linked.cellSum == 64

    scene.radii = (0.1, 0.2)
    assert linked.refresh_contact_grid(scene)
    assert linked.grid_size == pytest.approx(1.0)
    np.testing.assert_array_equal(np.asarray(linked.cnum), [10, 10, 10])
    assert linked.cellSum == linked.allocated_cell_sum

    linked.sims.wall_type = 0
    scene.radii = (0.1, 0.3)
    assert linked.refresh_contact_grid(scene)
    assert linked.grid_size == pytest.approx(1.0)
    assert linked.plane_insert_factor == pytest.approx(0.8)


def test_pure_lsdem_keeps_its_initialized_linked_cell_grid() -> None:
    class SceneStub:
        def find_bounding_sphere_radius(self, _):
            raise AssertionError("pure LSDEM must not run adaptive radius reductions")

    linked = LinkedCell.__new__(LinkedCell)
    linked.adaptive_cell_size = False
    linked.grid_size = 1.25
    linked.cnum = vec3i([8, 8, 8])

    assert not linked.refresh_contact_grid(SceneStub())
    assert linked.grid_size == pytest.approx(1.25)
    np.testing.assert_array_equal(np.asarray(linked.cnum), [8, 8, 8])


def test_lsmpm_verlet_multiplier_uses_physical_particle_radius() -> None:
    neighbor = object.__new__(NeighborBase)
    neighbor.sims = SimpleNamespace(scheme="LSMPM")
    scene = SimpleNamespace(find_particle_min_radius=lambda _sims: 0.5)

    assert neighbor._verlet_reference_radius(scene, 0.9) == 0.5
    neighbor.sims.scheme = "LSDEM"
    assert neighbor._verlet_reference_radius(scene, 0.9) == 0.9
