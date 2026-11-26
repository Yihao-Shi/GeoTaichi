"""Native particle snapshots retain raw scalar, vector and tensor material history."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpm.boundaries.BoundaryConstraint import BoundaryConstraints
from src.mpm.Recorder import WriteFile
from src.mpm.generator.LoadFromFile import BodyReader
from src.mpm.structs.ParticleCloud2D import ParticleCloud2D
from src.mpm.structs.ParticleCloud import ParticleCloud
from src.physics_model.constitutive_model.ConstitutiveModelBase import ConstitutiveBase

pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.cpu]


def make_material(capacity):
    material = ConstitutiveBase()
    material.stateVars = ti.Struct.field(
        dict(epdstrain=float, strain=ti.types.vector(6, float), deformation=ti.types.matrix(2, 2, float)),
        shape=capacity,
    )
    for field in material.stateVars.field_dict.values():
        values = field.to_numpy()
        values[:] = np.arange(values.size).reshape(values.shape) * 0.01
        field.from_numpy(values)
    return material


def test_history_restore_preserves_components_and_appended_particles(taichi_runtime):
    material = make_material(5)
    history = {name: field.to_numpy()[:2].copy() for name, field in material.stateVars.field_dict.items()}
    before = material.stateVars.to_numpy()
    material.reload_state_variables(history, particle_count=2, start_index=3)
    for name, field in material.stateVars.field_dict.items():
        np.testing.assert_array_equal(field.to_numpy()[:3], before[name][:3])
        np.testing.assert_array_equal(field.to_numpy()[3:], history[name])
    material.reload_state_variables(np.array(history, dtype=object), particle_count=2)
    for name, field in material.stateVars.field_dict.items():
        np.testing.assert_array_equal(field.to_numpy()[:2], history[name])
        np.testing.assert_array_equal(field.to_numpy()[2:], 0)
    expanded = np.zeros((2, 3, 3))
    expanded[:, [0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2]] = history["strain"]
    expanded[:, [1, 2, 2], [0, 1, 0]] = history["strain"][:, 3:]
    material.reload_state_variables(dict(history, strain=expanded), particle_count=2)
    np.testing.assert_array_equal(material.stateVars.strain.to_numpy()[:2], history["strain"])


@pytest.mark.parametrize("invalid", ["missing", "unknown", "scalar_strain", "nan", "count", "capacity"])
def test_invalid_history_fails_before_uploading_any_field(taichi_runtime, invalid):
    material = make_material(5)
    before = material.stateVars.to_numpy()
    history = {name: values[:2].copy() + 3 for name, values in before.items()}
    if invalid == "missing":
        del history["strain"]
    elif invalid == "unknown":
        history["typo"] = np.zeros(2)
    elif invalid == "scalar_strain":
        history["strain"] = np.zeros(2)
    elif invalid == "nan":
        history["deformation"][0, 0, 0] = np.nan
    elif invalid == "count":
        history["deformation"] = history["deformation"][:1]
    else:
        history = {name: np.repeat(values, 3, axis=0) for name, values in history.items()}
    with pytest.raises(ValueError):
        material.reload_state_variables(history, particle_count=2)
    for name, field in material.stateVars.field_dict.items():
        np.testing.assert_array_equal(field.to_numpy(), before[name])


def test_equivalent_strain_output_is_device_computed_for_selected_slice(taichi_runtime):
    material = make_material(5)
    values = material.stateVars.strain.to_numpy()[1:3]
    tensor = np.zeros((2, 3, 3))
    tensor[:, [0, 1, 2, 0, 1, 0], [0, 1, 2, 1, 2, 2]] = values
    tensor[:, [1, 2, 2], [0, 1, 0]] = values[:, 3:]
    deviator = tensor - np.trace(tensor, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    expected = np.sqrt(2 / 3 * np.sum(deviator**2, axis=(1, 2)))
    actual = material.get_state_vars_dict(["strain"], 1, 3)
    assert set(actual) == {"strain"}
    np.testing.assert_allclose(actual["strain"], expected, rtol=1e-6)


def make_native_snapshot(tmp_path, dimension):
    material = make_material(4)
    particle = (ParticleCloud2D if dimension == 2 else ParticleCloud).field(shape=4)
    particle.active.fill(1)
    particle.materialID.fill(1)
    particle.vol.fill(0.1)
    particle.m.fill(2)
    particle.x[0] = np.arange(dimension) * 0.1 + 0.1
    particle.v[0] = np.arange(dimension) * -0.2
    particle.velocity_gradient[0] = np.arange(dimension**2).reshape(dimension, dimension) * 0.01
    scene = SimpleNamespace(
        boundary=BoundaryConstraints(),
        material=material,
        particle=particle,
        particleNum=np.array([2]),
        element=SimpleNamespace(adaptive=False),
        is_rigid=ti.field(int, shape=1),
        psize=np.full((2, dimension), 0.025),
        restart_cell_pressure=None,
        two_phase_fluid_mass_initialized=False,
    )
    scene.push_psize = lambda sizes: setattr(scene, "psize", sizes)
    sims = SimpleNamespace(
        current_time=0.25,
        current_step=5,
        current_print=0,
        CurrentTime=ti.field(float, shape=()),
        material_type="Solid",
        dimension=dimension,
        coupling=False,
        neighbor_detection=False,
        max_particle_num=4,
        is_continue=True,
    )
    writer = WriteFile.__new__(WriteFile)
    writer.particle_path = str(tmp_path)
    writer.visualizeParticle = lambda *_: None
    writer.monitor_particle(sims, scene)
    path = tmp_path / "MPMParticle000000.npz"
    with np.load(path, allow_pickle=True) as saved:
        snapshot = dict(saved)
    reader = BodyReader.__new__(BodyReader)
    reader.sims = sims
    reader.set_traction = lambda *_: None
    return reader, scene, sims, snapshot, path


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
def test_native_snapshot_roundtrip_uses_raw_history_and_rejects_lost_legacy_tensor(tmp_path, taichi_runtime, dimension):
    reader, scene, sims, snapshot, path = make_native_snapshot(tmp_path, dimension)
    material, particle = scene.material, scene.particle
    assert snapshot["state_vars"].item()["strain"].shape == (2,)
    raw = snapshot["constitutive_state"].item()
    assert raw["strain"].shape == (2, 6)
    for field in material.stateVars.field_dict.values():
        field.fill(-5)
    particle.x.fill(-5)
    particle.v.fill(-5)
    particle.velocity_gradient.fill(-5)
    sims.current_step = 0
    reader.add_npz_body(scene, {"File": str(path), "Restart": True})
    assert snapshot["current_step"] == sims.current_step == 5
    for name, field in material.stateVars.field_dict.items():
        np.testing.assert_array_equal(field.to_numpy()[:2], raw[name])
    for field, key in [
        (particle.x, "position"),
        (particle.v, "velocity"),
        (particle.velocity_gradient, "velocity_gradient"),
    ]:
        np.testing.assert_array_equal(field.to_numpy()[:2], snapshot[key])
    del snapshot["constitutive_state"]
    np.savez(path, **snapshot)
    before = particle.x.to_numpy()
    sims.current_time = 0.0
    with pytest.raises(ValueError, match="components"):
        reader.add_npz_body(scene, {"File": str(path), "Restart": True})
    assert sims.current_time == 0
    np.testing.assert_array_equal(particle.x.to_numpy(), before)


def test_legacy_snapshot_restores_explicit_step_count(tmp_path, taichi_runtime):
    reader, scene, sims, snapshot, path = make_native_snapshot(tmp_path, 2)
    del snapshot["current_step"]
    np.savez(path, **snapshot)
    reader.add_npz_body(scene, {"File": str(path), "Restart": True, "CurrentStep": 25})
    assert sims.current_step == 25


@pytest.mark.parametrize(
    "failure", ["missing_field", "particle_shape", "particle_count", "sizes", "history", "time", "commit"]
)
def test_failed_restart_preserves_particles_history_boundaries_and_clock(tmp_path, taichi_runtime, failure):
    reader, scene, sims, snapshot, path = make_native_snapshot(tmp_path, 2)
    snapshot["position"] = np.full((2, 2), 0.8)
    snapshot["t_current"] = 5.0
    for values in snapshot["constitutive_state"].item().values():
        values[:] += 10
    if failure == "missing_field":
        del snapshot["fix_v"]
    elif failure == "particle_shape":
        snapshot["velocity"] = np.zeros((2, 3))
    elif failure == "particle_count":
        snapshot["body_num"] = 1.5
    elif failure == "sizes":
        snapshot["psize"][0, 0] = np.nan
    elif failure == "history":
        snapshot["constitutive_state"].item()["strain"] = np.zeros(2)
    elif failure == "time":
        snapshot["t_current"] = np.inf
    else:
        scene.boundary.particle_traction = ti.Struct.field(dict(pid=int, force=float), shape=3)
        scene.boundary.particle_traction.force.fill(7)

        def fail_traction(*_args):
            scene.boundary.particle_traction.force.fill(-1)
            scene.boundary.ptraction_list[0] = 2
            raise ValueError("injected traction setup failure")

        reader.set_traction = fail_traction
    np.savez(path, **snapshot)
    before_particles = scene.particle.to_numpy()
    before_history = scene.material.stateVars.to_numpy()
    before_sizes = scene.psize.copy()
    before_rigid = scene.is_rigid.to_numpy()
    old_time, old_clock, old_count = sims.current_time, sims.CurrentTime[None], scene.particleNum[0]
    template = {"File": str(path), "Restart": True}
    if failure == "commit":
        template["Traction"] = {"Pressure": [0, 1]}
    with pytest.raises((KeyError, ValueError)):
        reader.add_npz_body(scene, template)
    for name, field in scene.particle.field_dict.items():
        np.testing.assert_array_equal(field.to_numpy(), before_particles[name])
    for name, field in scene.material.stateVars.field_dict.items():
        np.testing.assert_array_equal(field.to_numpy(), before_history[name])
    np.testing.assert_array_equal(scene.psize, before_sizes)
    np.testing.assert_array_equal(scene.is_rigid.to_numpy(), before_rigid)
    assert (sims.current_time, sims.CurrentTime[None], scene.particleNum[0]) == (old_time, old_clock, old_count)
    assert scene.boundary.ptraction_list[0] == 0
    if failure == "commit":
        np.testing.assert_array_equal(scene.boundary.particle_traction.force.to_numpy(), 7)


@pytest.mark.parametrize("continue_time", [False, True])
def test_public_restart_failure_preserves_output_sequence(tmp_path, taichi_runtime, continue_time):
    from src.mpm.mainMPM import MPM

    model = MPM(log=False)
    model.scene.material = object()
    model.sims.current_print = 3
    model.sims.is_continue = not continue_time
    with pytest.raises(EOFError):
        model.read_restart(12, str(tmp_path), is_continue=continue_time)
    assert model.sims.current_print == 3
    assert model.sims.is_continue == (not continue_time)
