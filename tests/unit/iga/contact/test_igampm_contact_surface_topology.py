"""Unit checks for coupled contact-surface topology."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

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
            [
                np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64)
                for _ in range(4)
            ],
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
    assert surface.accd_basis_group_offsets == [0, 8]
    np.testing.assert_array_equal(
        surface.accd_basis_group_surface_ids.to_numpy()[:8],
        np.arange(8, dtype=np.int32),
    )


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
