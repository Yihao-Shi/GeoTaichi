from types import SimpleNamespace

import numpy as np
import pytest

from src.mpm import PostPlot
from src.mpm.Recorder import _closed_domain_pressure_offset


@pytest.mark.parametrize("cell_type,expected", [([1, 1, 2], 200), ([1, 0, 2], 0), ([2, 2, 2], 0)])
def test_closed_pressure_offset_preserves_air_reference(cell_type, expected):
    assert _closed_domain_pressure_offset([100, 300, 999], cell_type) == expected


@pytest.mark.parametrize(
    "material,solver,cell_type,offset",
    [
        ("Fluid", "Implicit", [1, 1, 2], 200),
        ("Fluid", "Implicit", [1, 0, 2], 0),
        ("Fluid", "Explicit", [1, 1, 2], 0),
        ("TwoPhaseDoubleLayer", "SemiImplicit", [1, 1, 2], 0),
    ],
)
def test_particle_vtk_uses_grid_gauge_only_for_incompressible_fluid(
    tmp_path, monkeypatch, material, solver, cell_type, offset
):
    (tmp_path / "particles").mkdir()
    (tmp_path / "grids").mkdir()
    particle_file = tmp_path / "particles/MPMParticle000000.npz"
    np.savez(
        particle_file,
        particleID=[0, 1],
        position=[[0, 0, 0], [1, 1, 1]],
        mass=[1, 1],
        bodyID=[0, 0],
        materialID=[1, 1],
        volume=[1, 1],
        velocity=np.zeros((2, 3)),
        pressure=[110.0, 310.0],
    )
    np.savez(tmp_path / "grids/MPMGrid000000.npz", cell_type=cell_type, cell_pressure=[100.0, 300.0, 999.0])
    written = []
    monkeypatch.setattr(PostPlot, "pointsToVTK", lambda *args, **kwargs: written.append(kwargs["data"]))
    PostPlot.write_vtk_file(
        SimpleNamespace(material_type=material, solver_type=solver),
        0,
        1,
        str(tmp_path),
        str(tmp_path / "vtks"),
        {"write_displacement": False},
    )
    np.testing.assert_allclose(written[0]["pressure"], np.array([110.0, 310.0]) - offset)
    assert np.diff(written[0]["pressure"])[0] == 200
    with np.load(particle_file) as original:
        np.testing.assert_array_equal(original["pressure"], [110.0, 310.0])
