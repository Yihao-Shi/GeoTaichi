"""Two-phase checkpoints preserve physical state and reject incompatible files."""

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
from src.mpm.generator.LoadFromFile import BodyReader
from src.mpm.structs.Particle import ParticleCloudTwoPhase, ParticleCloudTwoPhase2D


def run_check(dimension, material_type, folder):
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=2, offline_cache=False)
    particle_type = ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase
    state_type = ti.types.struct(strain=ti.types.vector(6, ti.f64), epdstrain=ti.f64)
    traction_type = ti.types.struct(pid=ti.i32, traction=ti.types.vector(dimension, ti.f64))

    def make_scene():
        p = particle_type.field(shape=5)
        state = state_type.field(shape=5)
        scene = SimpleNamespace(
            particle=p,
            particleNum=np.array([0]),
            psize=np.empty((0, dimension)),
            is_rigid=ti.field(int, shape=1),
            two_phase_fluid_mass_initialized=False,
            material=SimpleNamespace(
                stateVars=state,
                matProps=SimpleNamespace(size=lambda: 2),
                get_state_vars_dict=lambda **kw: {"strain": np.array([99.0, 99.0]), "epdstrain": np.array([0.3, 0.4])},
            ),
            boundary=SimpleNamespace(particle_traction=traction_type.field(shape=3), ptraction_list=np.array([0])),
        )
        grid_pressure = ti.field(ti.f64, shape=(3,) * dimension)
        grid_pressure.from_numpy(np.arange(3**dimension, dtype=float).reshape((3,) * dimension) ** 2)
        scene.element = SimpleNamespace(
            cnum=ti.Vector([3] * dimension),
            grid_size=ti.Vector([0.1] * dimension),
            ghost_cell=0,
            cell=SimpleNamespace(pressure=grid_pressure),
        )
        scene.push_psize = lambda size: setattr(scene, "psize", np.array(size))
        return scene

    scene = make_scene()
    scene.particleNum[0] = 2
    scene.psize = np.full((2, dimension), 0.025)
    scene.two_phase_fluid_mass_initialized = True
    for i in range(2):
        p = scene.particle
        p[i].particleID, p[i].materialID, p[i].active = i + 10, 1, 1
        p[i].phase = i + 1 if material_type == "TwoPhaseDoubleLayer" else 0
        p[i].m, p[i].ms, p[i].mf, p[i].vol, p[i].porosity = 1.0, 0.6, 0.4, 0.1, 0.4
        p[i].x, p[i].v, p[i].vs, p[i].vf = (
            [0.1 + i * 0.1] * dimension,
            [1.0] * dimension,
            [2.0] * dimension,
            [3.0] * dimension,
        )
        p[i].stress, p[i].pressure, p[i].permeability = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 23.0 + i, 0.02
        p[i].solid_velocity_gradient = np.eye(dimension) * (i + 1)
        p[i].fluid_velocity_gradient = np.eye(dimension) * (i + 3)
        scene.material.stateVars[i].strain = np.arange(6) + i + 0.25
        scene.material.stateVars[i].epdstrain = 0.3 + i * 0.1
    scene.boundary.ptraction_list[0] = 1
    scene.boundary.particle_traction[0].pid = 1
    scene.boundary.particle_traction[0].traction = [2.0] * dimension
    sims = SimpleNamespace(
        material_type=material_type,
        dimension=dimension,
        current_time=0.023,
        current_step=23,
        current_print=1,
        free_surface_detection=False,
        visualize=False,
        is_continue=True,
        max_particle_num=5,
        coupling=False,
        delta=0.001,
        CurrentTime=ti.field(float, shape=()),
    )
    writer = WriteFile.__new__(WriteFile)
    writer.particle_path = str(folder)
    writer.visualizeParticle = lambda *a: None
    writer.MonitorParticleTwoPhase(sims, scene)
    path = folder / "MPMParticle000001.npz"
    reader = BodyReader.__new__(BodyReader)
    reader.sims = sims
    reader.set_traction = lambda *a: None
    restored = make_scene()
    for field in restored.particle.field_dict.values():
        field.fill(17)
    reader.add_npz_body(restored, {"Restart": True, "File": str(path)})
    for name in [
        "particleID",
        "bodyID",
        "materialID",
        "active",
        "phase",
        "m",
        "ms",
        "mf",
        "x",
        "v",
        "vs",
        "vf",
        "vol",
        "porosity",
        "stress",
        "pressure",
        "permeability",
        "solid_velocity_gradient",
        "fluid_velocity_gradient",
        "fix_v",
    ]:
        np.testing.assert_array_equal(
            getattr(restored.particle, name).to_numpy()[:2], getattr(scene.particle, name).to_numpy()[:2]
        )
        np.testing.assert_array_equal(getattr(restored.particle, name).to_numpy()[2:], 17)
    for name in ["strain", "epdstrain"]:
        np.testing.assert_array_equal(
            getattr(restored.material.stateVars, name).to_numpy()[:2],
            getattr(scene.material.stateVars, name).to_numpy()[:2],
        )
    assert restored.two_phase_fluid_mass_initialized and sims.current_step == 23 and sims.CurrentTime[None] == 0.023
    np.testing.assert_array_equal(restored.psize, scene.psize)
    np.testing.assert_array_equal(restored.restart_cell_pressure, scene.element.cell.pressure.to_numpy())
    assert restored.boundary.ptraction_list[0] == 1
    np.testing.assert_array_equal(restored.boundary.particle_traction.traction.to_numpy()[0], [2.0] * dimension)
    with np.load(path, allow_pickle=True) as saved:
        data = {key: saved[key] for key in saved.files}
    # Recovery builds used alternate metadata keys; restore those checkpoints too.
    remote = dict(data)
    for current, recovery in [
        ("constitutive_state", "restart_state_vars"),
        ("particle_traction_state", "restart_particle_traction"),
        ("two_phase_fluid_mass_initialized", "fluid_mass_initialized"),
    ]:
        remote[recovery] = remote.pop(current)
    remote["fluid_mass_initialized"] = False
    np.savez(folder / "recovery.npz", **remote)
    recovered = make_scene()
    reader.add_npz_body(recovered, {"Restart": True, "File": str(folder / "recovery.npz")})
    assert not recovered.two_phase_fluid_mass_initialized
    np.testing.assert_array_equal(
        recovered.material.stateVars.strain.to_numpy()[:2], scene.material.stateVars.strain.to_numpy()[:2]
    )
    assert recovered.boundary.ptraction_list[0] == 1
    np.testing.assert_array_equal(recovered.boundary.particle_traction.traction.to_numpy()[0], [2.0] * dimension)
    invalid = dict(data, dimension=5)
    np.savez(folder / "invalid.npz", **invalid)
    empty = make_scene()
    with pytest.raises(ValueError, match="dimension"):
        reader.add_npz_body(empty, {"Restart": True, "File": str(folder / "invalid.npz")})
    assert int(empty.particleNum[0]) == 0 and not empty.two_phase_fluid_mass_initialized
    # Old files require explicit particle sizes but retain separate phase state.
    legacy = {
        key: value
        for key, value in data.items()
        if key
        not in [
            "psize",
            "restart_format_version",
            "dimension",
            "material_type",
            "current_step",
            "two_phase_fluid_mass_initialized",
        ]
    }
    np.savez(folder / "legacy.npz", **legacy)
    reader.add_npz_body(
        empty,
        {"Restart": True, "File": str(folder / "legacy.npz"), "ParticleSize": [0.025] * dimension, "CurrentStep": 23},
    )
    assert empty.two_phase_fluid_mass_initialized and sims.current_step == 23
    if dimension == 3:
        uncoupled_snapshot = {key: value for key, value in legacy.items() if key != "coupling"}
        np.savez(folder / "legacy_coupling.npz", **uncoupled_snapshot)
        sims.coupling = True
        coupled = make_scene()
        reader.add_npz_body(
            coupled,
            {
                "Restart": True,
                "File": str(folder / "legacy_coupling.npz"),
                "ParticleSize": [0.025] * dimension,
                "CurrentStep": 23,
                "FluidMassInitialized": False,
            },
        )
        np.testing.assert_array_equal(
            coupled.particle.coupling.to_numpy()[:2], np.asarray(legacy["phase"] == 1, dtype=np.uint8)
        )
        assert not coupled.two_phase_fluid_mass_initialized
        sims.coupling = False
    wrong_phase = dict(legacy, phase=np.full(2, 0 if material_type == "TwoPhaseDoubleLayer" else 1))
    np.savez(folder / "wrong_phase.npz", **wrong_phase)
    with pytest.raises(ValueError, match="phase"):
        reader.add_npz_body(
            make_scene(),
            {
                "Restart": True,
                "File": str(folder / "wrong_phase.npz"),
                "ParticleSize": [0.025] * dimension,
                "CurrentStep": 23,
            },
        )
    invalid_traction = dict(data)
    invalid_traction["particle_traction_state"] = {"pid": np.array([2]), "traction": np.ones((1, dimension))}
    np.savez(folder / "wrong_traction.npz", **invalid_traction)
    untouched = make_scene()
    with pytest.raises(ValueError, match="traction IDs"):
        reader.add_npz_body(untouched, {"Restart": True, "File": str(folder / "wrong_traction.npz")})
    assert untouched.particleNum[0] == 0
    # Before the first projection the allocated grid is zero; the initial
    # particle pressure must initialize IBM just as it does in a fresh run.
    sims.current_time, sims.current_step = 0.0, 0
    scene.element.cell.pressure.fill(0.0)
    writer.MonitorParticleTwoPhase(sims, scene)
    initial = make_scene()
    reader.add_npz_body(initial, {"Restart": True, "File": str(path)})
    assert initial.restart_cell_pressure is None
    np.testing.assert_array_equal(initial.particle.pressure.to_numpy()[:2], [23.0, 24.0])
    print("roundtrip, raw constitutive state, tractions, validation, legacy routing passed")


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("material_type", ["TwoPhaseSingleLayer", "TwoPhaseDoubleLayer"])
def test_two_phase_checkpoint_roundtrip(tmp_path, dimension, material_type):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.verification.mpdem.test_two_phase_checkpoint_roundtrip",
            str(dimension),
            material_type,
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "GEOTAICHI_REAL_DTYPE": "float64"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout[-12000:]


if __name__ == "__main__":
    run_check(int(sys.argv[1]), sys.argv[2], Path(sys.argv[3]))
