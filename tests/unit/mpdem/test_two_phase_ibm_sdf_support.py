"""Rotating impactors need a full fluid-cell diagonal inside their SDF halo."""

from types import SimpleNamespace
import math
import pytest
import trimesh

from examples.mmpm.TwoPhaseLSDEMCoupling.sphere_impact_submerged_bed_3d.sphere_impact_submerged_bed_3d import (
    ROOT,
    SPHERE_DIAMETER,
    impactor_grid_parameters,
)


@pytest.mark.parametrize("dx", [0.005, 0.0025])
@pytest.mark.parametrize("is_bunny", [False, True])
def test_impactor_grid_covers_rotated_cell_corners(dx, is_bunny):
    name = "bunny_impact_watertight.ply" if is_bunny else "sphere.stl"
    mesh = trimesh.load(ROOT / "assets/mesh/LSDEM" / name, force="mesh")
    spacing, extent, _ = impactor_grid_parameters(SimpleNamespace(mesh=mesh), dx, is_bunny)
    equivalent_radius = (3 * mesh.volume / (4 * math.pi)) ** (1 / 3)
    scale = 0.5 * SPHERE_DIAMETER / equivalent_radius
    assert extent * spacing * scale >= math.sqrt(3) * dx
