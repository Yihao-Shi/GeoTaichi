"""An IBM body retains fictitious fluid and must not also be a cut-cell wall."""

from pathlib import Path
import json
import runpy

import geotaichi as gt
import numpy as np
import pytest
import taichi as ti


@ti.kernel
def duplicate_ibm_geometry(fraction: ti.template(), sdf: ti.template()):
    for I in ti.grouped(fraction):
        if fraction[I] >= 1.0:
            sdf[I] = -1.0


@pytest.mark.parametrize("duplicate_geometry", [False, True])
def test_cylinder_retains_ibm_interior_without_overwriting_wall_sdf(
    taichi_runtime, monkeypatch, tmp_path, duplicate_geometry
):
    monkeypatch.setattr(gt, "init", lambda **kwargs: None)
    for key, value in {
        "DX": ".02",
        "PPC": "3",
        "VALIDATE": "1",
        "SAVE_PATH": str(tmp_path),
    }.items():
        monkeypatch.setenv("GEOTAICHI_CYLINDER_IBM_2D_" + key, value)
    monkeypatch.setenv("GEOTAICHI_SKIP_POSTPROCESS", "1")
    native_factory = gt.MPM
    models = []

    def create():
        mpm = native_factory()
        models.append(mpm)

        def prepare(**callbacks):
            assert "cut_cell_function" not in callbacks
            if duplicate_geometry:
                native = callbacks["ibm_field_function"]

                def conflicting_geometry(sims, scene):
                    native(sims, scene)
                    duplicate_ibm_geometry(mpm.enginer.ibm_solid_fraction, scene.element.cell.solid_sdf)

                callbacks["ibm_field_function"] = conflicting_geometry
            mpm.add_essentials(**callbacks)
            mpm.enginer.pre_calculation(mpm.sims, mpm.scene, mpm.neighbor)

        mpm.run = prepare
        return mpm

    monkeypatch.setattr(gt, "MPM", create)
    entry = (
        Path(__file__).resolve().parents[3] / "examples/cfdem/FullyResolved/IBMFixedCylinder2D/cylinder_flow_ibm_2d.py"
    )
    if duplicate_geometry:
        with pytest.raises(RuntimeError, match="IBM and cut-cell geometry overlap"):
            runpy.run_path(str(entry))
    else:
        namespace = runpy.run_path(str(entry))
    mpm = models[0]
    scene, engine = mpm.scene, mpm.enginer
    ghost = int(scene.element.ghost_cell)
    active = np.array(scene.element.cnum, dtype=int) - 2 * ghost
    assert int(scene.particleNum[0]) == int(np.prod(active)) * 3**2
    if duplicate_geometry:
        return  # Invalid geometry must fail before removing any particle.
    interior = tuple(slice(ghost, ghost + count) for count in active)
    fraction = engine.ibm_solid_fraction.to_numpy()[interior]
    occupied = fraction > 0.5
    assert occupied.any()
    assert np.all(scene.element.cell.type.to_numpy()[interior][occupied] == 1)
    sdf = scene.element.cell.solid_sdf.to_numpy()
    assert np.all(sdf[interior][occupied] > 0)
    namespace["update_cylinder_geometry"](mpm.sims, scene)
    np.testing.assert_array_equal(scene.element.cell.solid_sdf.to_numpy(), sdf)
    # Only ghost cells lie inside the channel walls; the IBM interior stays fluid.
    assert np.all(sdf[interior] >= 0.0) and np.any(sdf < 0.0)
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["passed"] is True
