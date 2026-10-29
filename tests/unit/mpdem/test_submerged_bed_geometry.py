"""The initial saturated cylinder must reach its prescribed confining wall."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "examples/mmpm/TwoPhaseLSDEMCoupling/sphere_impact_submerged_bed_3d/sphere_impact_submerged_bed_3d.py"
spec = importlib.util.spec_from_file_location("submerged_bed_geometry", ENTRY)
case = importlib.util.module_from_spec(spec)
spec.loader.exec_module(case)


@pytest.mark.parametrize("dx", [0.005, 0.0025])
@pytest.mark.parametrize("top", [case.BED_HEIGHT, case.WATER_SURFACE])
def test_initial_material_reaches_cylindrical_wall(dx, top):
    region = case.cylinder_region("material", dx, top)
    lower = np.asarray(region["BoundingBoxPoint"])
    upper = lower + region["BoundingBoxSize"]
    center = np.asarray(case.DOMAIN[:2]) * 0.5
    for angle in np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False):
        normal = np.array([np.cos(angle), np.sin(angle)])
        inside = np.r_[center + (case.CONTAINER_RADIUS - 1e-6) * normal, 0.5 * (dx + top)]
        outside = np.r_[center + (case.CONTAINER_RADIUS + 1e-6) * normal, inside[2]]
        assert np.all(inside >= lower) and np.all(inside <= upper)
        assert region["RegionFunction"](inside)
        assert not region["RegionFunction"](outside)
    expected = np.pi * case.CONTAINER_RADIUS**2 * (top - dx)
    assert region["RegionVolume"]() == pytest.approx(expected, rel=1e-14)
