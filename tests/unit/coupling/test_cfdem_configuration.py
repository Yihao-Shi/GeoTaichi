from types import SimpleNamespace

import pytest

from src.mpdem.Simulation import Simulation
from src.mpdem.Engine import Engine


def _simulation(resolution="Auto"):
    sims = object.__new__(Simulation)
    sims.coupling_scheme = "CFDEM"
    sims.cfdem_resolution = resolution
    sims.enhanced_coupling = False
    sims.dem_timestep = 1.0
    sims.delta = 1.0
    return sims


def _mpm(**overrides):
    values = dict(
        sparse_grid=False,
        material_type="Fluid",
        solver_type="Implicit",
        discretization="FDM",
        dimension=3,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("resolution", "scheme"),
    [("SemiResolved", "DEM"), ("FullyResolved", "LSDEM"), ("Auto", "DEM"), ("Auto", "LSDEM")],
)
def test_incompressible_cfdem_resolution_routes(resolution, scheme):
    _simulation(resolution).validate_coupling_configuration(_mpm(), SimpleNamespace(scheme=scheme))


@pytest.mark.parametrize(
    "overrides",
    [
        {"material_type": "Solid"},
        {"solver_type": "Explicit"},
        {"discretization": "FEM"},
    ],
)
def test_cfdem_rejects_non_incompressible_mpm(overrides):
    with pytest.raises(RuntimeError, match="incompressible semi-implicit MPM"):
        _simulation().validate_coupling_configuration(_mpm(**overrides), SimpleNamespace(scheme="DEM"))


def test_fully_resolved_cfdem_rejects_pure_dem_spheres_and_clumps():
    with pytest.raises(RuntimeError, match="pure DEM Sphere/Clump bodies"):
        _simulation("FullyResolved").validate_coupling_configuration(_mpm(), SimpleNamespace(scheme="DEM"))


def test_semi_resolved_cfdem_rejects_clumps_instead_of_ignoring_them():
    engine = object.__new__(Engine)
    engine.dscene = SimpleNamespace(clumpNum=[1])
    with pytest.raises(RuntimeError, match="supports DEM spheres only"):
        engine.choose_incompressible_dem_sphere_coupling({})
