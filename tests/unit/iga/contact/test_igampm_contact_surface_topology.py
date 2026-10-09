"""Unit checks for coupled contact-surface topology."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

import src.igampm.config as config
from src.igampm.contact.ContactSurface import CouplingContactSurface


@pytest.fixture(autouse=True)
def _initialize_taichi(taichi_runtime):
    config.set_dimension(2)


class _SquarePatch:
    def __init__(self):
        self.control_points = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float64,
        )
        self.weights = np.ones(4, dtype=np.float64)

    def gather_boundary_ctrlpts(self):
        return (
            [2, 2, 2, 2],
            np.asarray([0, 1, 1, 2, 2, 3, 3, 0], dtype=np.int32),
            [np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64) for _ in range(4)],
            [1, 1, 1, 1],
        )


def _two_coincident_patches():
    body = OrderedDict(
        [
            ("left", {"primitive": _SquarePatch()}),
            ("right", {"primitive": _SquarePatch()}),
        ]
    )
    patch = SimpleNamespace(
        primitive=SimpleNamespace(body=body),
        prefix_total_num_ctrlpts=np.asarray([0, 4, 8], dtype=np.int32),
    )
    return SimpleNamespace(patch=patch)


def test_default_keeps_coincident_faces_from_distinct_bodies():
    surface = CouplingContactSurface(_two_coincident_patches())
    assert surface.num_surfaces == 8
    assert len(surface.duplicate_surface_groups) == 4
    assert surface.excluded_surface_keys == []
    assert len(surface.accd_basis) == 1
    assert all(basis is surface.accd_basis[0] for basis in surface.basis)
    assert surface.accd_basis_group_offsets == [0, 8]
    np.testing.assert_array_equal(
        surface.accd_basis_group_surface_ids.to_numpy()[:8],
        np.arange(8, dtype=np.int32),
    )


def test_swept_surface_and_span_bvh_cover_interior_motion_and_clearance():
    config.set_dimension(3)
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    knots = np.array([0.0, 0.0, 1.0, 1.0])
    primitive = SimpleNamespace(
        num_ctrlpts_u=2,
        num_ctrlpts_v=2,
        control_points=points,
        weights=np.ones(4),
        gather_boundary_ctrlpts=lambda: ([4], np.arange(4), [(knots, knots)], [(1, 1)]),
    )
    owner = SimpleNamespace(
        patch=SimpleNamespace(
            primitive=SimpleNamespace(body={"plane": {"primitive": primitive}}), prefix_total_num_ctrlpts=[0]
        )
    )
    surface = CouplingContactSurface(owner)
    direction = ti.field(ti.f64, shape=12)
    direction.from_numpy(np.tile([0.0, 0.0, 3.0], 4))
    surface.update_control_point_direction(direction)
    surface.update_swept_bounds(1.0)
    hit = ti.field(ti.i32, shape=2)

    @ti.kernel
    def query(p: ti.types.vector(3, ti.f64), dp: ti.types.vector(3, ti.f64), clearance: ti.f64):
        lower, upper = ti.min(p, p + dp), ti.max(p, p + dp)
        hit[0] = surface.swept_box_overlap(
            lower, upper, surface.swept_tree_lower[0], surface.swept_tree_upper[0], clearance
        )
        hit[1] = surface.swept_span_overlap(0, lower, upper, clearance)

    query([0.5, 0.5, 2.0], [0.0, 0.0, 0.0], 0.0)
    np.testing.assert_array_equal(hit.to_numpy(), [1, 1])
    query([1.04, 0.5, 2.0], [0.0, 0.0, 0.0], 0.05)
    np.testing.assert_array_equal(hit.to_numpy(), [1, 1])
    query([1.04, 0.5, 2.0], [0.0, 0.0, 0.0], 0.0)
    np.testing.assert_array_equal(hit.to_numpy(), [0, 0])
    surface.update_swept_bounds(0.25)
    query([0.5, 0.5, 2.0], [0.0, 0.0, 0.0], 0.0)
    np.testing.assert_array_equal(hit.to_numpy(), [0, 0])
    direction.fill(0.0)
    surface.update_control_point_direction(direction)
    surface.update_swept_bounds(1.0)
    query([2.0, 0.5, 0.0], [-4.0, 0.0, 0.0], 0.0)
    np.testing.assert_array_equal(hit.to_numpy(), [1, 1])


def test_surface_direction_box_bounds_every_control_and_cancels_translation():
    surface = CouplingContactSurface(_two_coincident_patches())
    direction = ti.field(ti.f64, shape=16)
    bounds = ti.field(ti.f64, shape=surface.num_surfaces)

    @ti.kernel
    def evaluate(point_direction: ti.types.vector(2, ti.f64)):
        for sid in bounds:
            bounds[sid] = surface.relative_motion_upper_bound(sid, point_direction)

    rng = np.random.default_rng(421)
    controls = rng.normal(size=(8, 2))
    direction.from_numpy(controls.ravel())
    surface.update_control_point_direction(direction)
    ids = surface.control_points_id.to_numpy()
    for point_direction in rng.normal(size=(4, 2)):
        evaluate(point_direction)
        actual = bounds.to_numpy()
        for sid in range(surface.num_surfaces):
            face_ids = ids[surface.prefix_num_ctrlpts[sid] : surface.prefix_num_ctrlpts[sid + 1]]
            exact = np.linalg.norm(controls[face_ids] - point_direction, axis=1).max()
            assert actual[sid] >= exact

    translation = np.array([0.1, -0.4])
    direction.from_numpy(np.tile(translation, 8))
    surface.update_control_point_direction(direction)
    evaluate(translation)
    np.testing.assert_array_equal(bounds.to_numpy(), np.zeros(surface.num_surfaces))


def test_explicit_duplicate_ownership_policies():
    canonical = CouplingContactSurface(
        _two_coincident_patches(),
        contact_surface_duplicate_policy="canonical",
    )
    assert canonical.num_surfaces == 4
    assert canonical.surface_keys == [(0, 0), (0, 1), (0, 2), (0, 3)]

    internal = CouplingContactSurface(
        _two_coincident_patches(),
        contact_surface_duplicate_policy="remove_internal",
    )
    assert internal.num_surfaces == 0
    assert len(internal.excluded_surface_keys) == 8
    assert internal.total_ctrlpts == 0


def test_explicit_surface_include_and_exclude():
    included = CouplingContactSurface(
        _two_coincident_patches(),
        contact_surface_include=[(0, 0), "1:2"],
    )
    assert included.surface_keys == [(0, 0), (1, 2)]

    excluded = CouplingContactSurface(
        _two_coincident_patches(),
        contact_surface_exclude=[0, (1, 3)],
    )
    assert excluded.num_surfaces == 6
    assert (0, 0) not in excluded.surface_keys
    assert (1, 3) not in excluded.surface_keys


def test_rectangular_boundary_uses_each_edges_parametric_direction():
    from src.nurbs.BasicSurface import Rectangle

    rectangle = Rectangle()
    rectangle.set_parameters(size=[0.03, 1.0])
    rectangle.generate_knot_u(2, 3)
    rectangle.generate_knot_v(3, 7)
    rectangle.generate_ctrlpts()
    rectangle.generate_weights()
    sizes, _, knots, degrees = rectangle.gather_boundary_ctrlpts()
    assert sizes == [3, 7, 3, 7]
    assert degrees == [2, 3, 2, 3]
    for size, knot, degree in zip(sizes, knots, degrees):
        assert len(knot) - degree - 1 == size
    np.testing.assert_array_equal(knots[1], rectangle.knot_vector_v)
    np.testing.assert_array_equal(knots[3], rectangle.knot_vector_v)

    iga = SimpleNamespace(
        patch=SimpleNamespace(
            primitive=SimpleNamespace(body=OrderedDict(rect={"primitive": rectangle})),
            prefix_total_num_ctrlpts=np.asarray([0, 21], dtype=np.int32),
        )
    )
    surface = CouplingContactSurface(iga, contact_surface_include=[(0, 0), (0, 1)])
    assert surface.num_surfaces == 2
    assert surface.total_ctrlpts == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contact_surface_duplicate_policy": "guess"},
        {"contact_surface_duplicate_tolerance": 0.0},
        {"contact_surface_include": ["bad-key"]},
    ],
)
def test_invalid_surface_topology_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CouplingContactSurface(_two_coincident_patches(), **kwargs)
