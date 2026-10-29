"""Restart must preserve the pressure state used by the IBM predictor."""

from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import numpy as np
import pytest
import taichi as ti

import src.utils.GlobalVariable as GV

if __name__ == "__main__":
    GV.DIMENSION = int(sys.argv[1])
from src.mpm.Recorder import WriteFile
from src.mpm.engines.IncompressibleEngine import IncompressibleEngine
from src.mpm.generator.LoadFromFile import BodyReader
from src.mpm.structs.Particle import ParticleCloudIncompressible2D, ParticleCloudIncompressible3D

pytestmark = [pytest.mark.verification, pytest.mark.mpdem, pytest.mark.cpu]


def fixture(dimension):
    GV.DIMENSION = dimension
    shape = (6,) * dimension
    particle_type = ParticleCloudIncompressible2D if dimension == 2 else ParticleCloudIncompressible3D
    particle = particle_type.field(shape=1)
    particle[0].active = particle[0].materialID = particle[0].coupling = 1
    particle[0].x = [1.5] * dimension
    particle[0].m = particle[0].vol = 1.0
    cell = SimpleNamespace()
    for name in ["pressure", "surface_tension", "fluid_sdf"]:
        setattr(cell, name, ti.field(float, shape=shape, offset=(-1,) * dimension))
    cell.type = ti.field(int, shape=shape, offset=(-1,) * dimension)
    cell.type.fill(2)
    for index in np.ndindex((4,) * dimension):
        cell.type[index] = 1
    scene = SimpleNamespace(
        particle=particle,
        particleNum=np.array([1]),
        is_rigid=ti.field(int, shape=1),
        psize=np.full((1, dimension), 0.5),
        material=SimpleNamespace(get_state_vars_dict=lambda **_: {}),
        element=SimpleNamespace(
            cnum=ti.Vector(list(shape)),
            grid_size=ti.Vector([1.0] * dimension),
            igrid_size=ti.Vector([1.0] * dimension),
            ghost_cell=1,
            cell=cell,
        ),
    )
    scene.push_psize = lambda size: setattr(scene, "psize", size)
    sims = SimpleNamespace(
        material_type="IncompressibleFluid",
        dimension=dimension,
        current_time=0.1,
        current_print=0,
        is_continue=True,
        max_particle_num=1,
        neighbor_detection=False,
        coupling=False,
        CurrentTime=ti.field(float, shape=()),
    )
    return sims, scene


def read_and_initialize(path, sims, scene):
    reader = BodyReader.__new__(BodyReader)
    reader.sims = sims
    reader.set_traction = lambda *args: None
    reader.add_npz_body(scene, {"File": str(path), "Restart": True})
    engine = IncompressibleEngine.__new__(IncompressibleEngine)
    engine.initial_pressure_weight = None
    engine.initialize_cell_pressure_from_particles(scene)


def check_legacy_pressure(tmp_path, dimension):
    sims, scene = fixture(dimension)
    writer = WriteFile.__new__(WriteFile)
    output = writer.MonitorParticleBase(sims, scene, 1)
    output["pressure"] = np.array([17.0])
    path = tmp_path / "MPMParticle000000.npz"
    np.savez(path, **output)
    read_and_initialize(path, sims, scene)
    assert scene.particle[0].pressure == pytest.approx(-17.0)
    assert scene.element.cell.pressure[(1,) * dimension] == pytest.approx(17.0)


def check_grid_roundtrip(tmp_path, dimension, coupling):
    sims, scene = fixture(dimension)
    expected_gradient = np.arange(dimension * dimension, dtype=float).reshape(dimension, dimension) + 1.0
    scene.particle[0].velocity_gradient = expected_gradient
    expected = np.random.default_rng(17).normal(size=(6,) * dimension)
    scene.element.cell.pressure.from_numpy(expected)
    writer = WriteFile.__new__(WriteFile)
    writer.particle_path = str(tmp_path)
    writer.visualizeParticle = lambda *args: None
    method = writer.MonitorIncompressibleParticleCoupling if coupling else writer.MonitorIncompressibleParticle
    method(sims, scene)
    scene.element.cell.pressure.fill(123.0)
    read_and_initialize(tmp_path / "MPMParticle000000.npz", sims, scene)
    np.testing.assert_array_equal(scene.element.cell.pressure.to_numpy(), expected)
    np.testing.assert_array_equal(scene.particle[0].velocity_gradient.to_numpy(), expected_gradient)


def check_companion_grid(tmp_path, dimension, case):
    sims, scene = fixture(dimension)
    expected = np.random.default_rng(29).normal(size=(6,) * dimension)
    writer = WriteFile.__new__(WriteFile)
    output = writer.MonitorParticleBase(sims, scene, 1)
    output["pressure"] = np.array([17.0])
    particles, grids = tmp_path / "particles", tmp_path / "grids"
    particles.mkdir()
    grids.mkdir()
    if case in ("invalid_nan", "invalid_geometry"):
        output.update(restart_cell_pressure=expected.copy(), restart_grid_size=np.ones(dimension), restart_ghost_cell=1)
        if case == "invalid_nan":
            output["restart_cell_pressure"].flat[0] = np.nan
        else:
            output["restart_grid_size"][0] *= 2
    else:
        np.savez(
            grids / "MPMGrid000000.npz",
            cell_pressure=expected[..., None],
            t_current=sims.current_time + (1 if case == "invalid_time" else 0),
            coords=np.array([[-1.0] * dimension, [5.0] * dimension]),
        )
    path = particles / "MPMParticle000000.npz"
    np.savez(path, **output)
    if case.startswith("invalid"):
        scene.particle[0].pressure = -123.0
        with pytest.raises(ValueError, match="Incompressible restart"):
            read_and_initialize(path, sims, scene)
        assert scene.particle[0].pressure == -123.0
    else:
        read_and_initialize(path, sims, scene)
        np.testing.assert_array_equal(scene.element.cell.pressure.to_numpy(), expected)


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("case", ["legacy", "particle", "coupling", "legacy_grid"])
def test_incompressible_pressure_restart(tmp_path, dimension, case):
    # Taichi's dimension-dependent kernel annotations are fixed at import time.
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, __file__, str(dimension), case, str(tmp_path)],
        cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root), GEOTAICHI_REAL_DTYPE="float64"),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-6000:]


@pytest.mark.parametrize("case", ["invalid_nan", "invalid_geometry", "invalid_time"])
def test_incompressible_restart_rejects_bad_pressure_before_mutation(tmp_path, case):
    test_incompressible_pressure_restart(tmp_path, 3, case)


if __name__ == "__main__":
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=2, offline_cache=False)
    dimension, case, output = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    if case == "legacy":
        check_legacy_pressure(output, dimension)
    elif case in ("particle", "coupling"):
        check_grid_roundtrip(output, dimension, case == "coupling")
    else:
        check_companion_grid(output, dimension, case)
