"""IBM fictitious fluid and cut-cell walls have separate geometry ownership."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpm.engines.IncompressibleEngine import IncompressibleEngine

DIMENSIONS = [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))]


@pytest.mark.parametrize("dimension", DIMENSIONS)
@pytest.mark.parametrize("geometry", ["overlap", "disjoint", "partial_contact", "dry_interior"])
def test_ibm_cut_cell_geometry_ownership(taichi_runtime, dimension, geometry):
    shape, offset, center = (5,) * dimension, (-1,) * dimension, (1,) * dimension
    fraction = ti.field(float, shape=shape, offset=offset)
    sdf = ti.field(float, shape=shape, offset=offset)
    sdf.fill(1.0)
    fraction[center] = 0.5 if geometry == "partial_contact" else 1.0
    if geometry in ("overlap", "partial_contact"):
        sdf[center] = -1.0
    else:
        sdf[(0,) * dimension] = -1.0
    cell_type = ti.field(int, shape=shape, offset=offset)
    cell_type.fill(0 if geometry == "dry_interior" else 1)
    scene = SimpleNamespace(
        element=SimpleNamespace(
            ghost_cell=1, cnum=ti.Vector(list(shape)), cell=SimpleNamespace(solid_sdf=sdf, type=cell_type)
        )
    )
    engine = IncompressibleEngine.__new__(IncompressibleEngine)
    engine.ibm_solid_fraction = fraction
    engine.ibm_solid_density = engine.ibm_solid_velocity_cell = engine.ibm_force_cell = object()
    saved = [f.to_numpy() for f in (fraction, sdf, cell_type)]
    if geometry == "overlap":
        with pytest.raises(RuntimeError, match="IBM and cut-cell geometry overlap in 1"):
            engine.validate_ibm_cut_cell_geometry(SimpleNamespace(solid_sdf_cut_cell=True), scene)
    else:
        engine.validate_ibm_cut_cell_geometry(SimpleNamespace(solid_sdf_cut_cell=True), scene)
    for field, original in zip((fraction, sdf, cell_type), saved):
        np.testing.assert_array_equal(field.to_numpy(), original)
