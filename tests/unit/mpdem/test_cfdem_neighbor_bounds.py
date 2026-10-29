"""CFDEM has no coupling-contact radius; DEM must use its own bodies."""

from types import SimpleNamespace

import pytest

from src.mpdem.mainDEMPM import DEMPM


@pytest.mark.parametrize(
    "bounds, options, expected",
    [
        ((0.0, 0.0), {}, {}),
        ((0.0, 0.0), {"dem_min_bounding_radius": 0.01}, {"min_bounding_radius": 0.01}),
        ((0.02, 0.04), {}, {"min_bounding_radius": 0.02, "max_bounding_radius": 0.04}),
    ],
)
def test_unset_coupling_bounds_do_not_override_dem_radius(bounds, options, expected):
    model = object.__new__(DEMPM)
    model.is_direct_mpm_affine_ipc = lambda: False
    model.is_incompressible_mpm_affine_body = lambda: False
    model.sync_settings = model.load_history_contact = lambda: None
    model.sims = SimpleNamespace(
        coupling_scheme="CFDEM",
        min_bounding_rad=bounds[0],
        max_bounding_rad=bounds[1],
        validate_coupling_configuration=lambda *_: None,
    )
    model.mpm = SimpleNamespace(
        sims=None,
        scene=SimpleNamespace(particleNum=[1], update_coupling_points_number=lambda *_: None),
        add_essentials=lambda **_: None,
    )
    observed = {}

    class DEMInitialized(Exception):
        pass

    def initialize_dem(**kwargs):
        observed.update(kwargs)
        raise DEMInitialized

    model.dem = SimpleNamespace(sims=None, scene=SimpleNamespace(particleNum=[1]), add_essentials=initialize_dem)
    with pytest.raises(DEMInitialized):
        model.add_essentials(**options)
    assert observed == expected
