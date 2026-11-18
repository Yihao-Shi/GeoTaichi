"""Two-phase state snapshots preserve phase velocities, masses and tensor history."""

from types import SimpleNamespace

import numpy as np
import pytest
import taichi as ti

from src.mpm.Recorder import WriteFile
from src.mpm.boundaries.BoundaryConstraint import BoundaryConstraints
from src.mpm.generator.LoadFromFile import BodyReader
from src.mpm.structs.ParticleCloudTwoPhase import ParticleCloudTwoPhase
from src.mpm.structs.ParticleCloudTwoPhase2D import ParticleCloudTwoPhase2D
from src.utils.FieldIO import field_to_numpy_prefix

pytestmark = [pytest.mark.unit, pytest.mark.mpm, pytest.mark.cpu]


def scene_and_sim(dimension, capacity=5):
    particle = (ParticleCloudTwoPhase2D if dimension == 2 else ParticleCloudTwoPhase).field(shape=capacity)
    state = ti.Struct.field(
        dict(
            epdstrain=float, strain=ti.types.vector(6, float), deformation=ti.types.matrix(dimension, dimension, float)
        ),
        shape=capacity,
    )
    material = SimpleNamespace(stateVars=state, get_state_vars_dict=lambda **_: {"strain": np.zeros(2)})
    scene = SimpleNamespace(
        particle=particle,
        particleNum=np.array([2]),
        is_rigid=ti.field(int, shape=1),
        psize=np.full((2, dimension), 0.025),
        material=material,
        two_phase_fluid_mass_initialized=True,
        restart_cell_pressure=None,
        boundary=BoundaryConstraints(),
    )
    sims = SimpleNamespace(
        dimension=dimension,
        max_particle_num=capacity,
        current_step=42,
        current_time=0.0042,
        CurrentTime=ti.field(float, shape=()),
        current_print=3,
        free_surface_detection=True,
        visualize=False,
        material_type="TwoPhaseDoubleLayer",
        is_continue=True,
    )
    for name, field in particle.field_dict.items():
        values = field.to_numpy()
        values.fill(0 if np.issubdtype(values.dtype, np.integer) else 0.125)
        field.from_numpy(values)
    particle.materialID.fill(1)
    particle.active.fill(1)
    particle.phase[0], particle.phase[1] = 1, 2
    particle.vs[0], particle.vf[0] = [1.0] * dimension, [-2.0] * dimension
    particle.ms[0], particle.mf[0] = 2.0, 7.0
    for field in state.field_dict.values():
        values = np.arange(field.to_numpy().size, dtype=float).reshape(field.to_numpy().shape) * 0.001
        field.from_numpy(values)
    return scene, sims


def write_snapshot(scene, sims, folder):
    writer = object.__new__(WriteFile)
    writer.particle_path = str(folder)
    writer.visualizeParticle = lambda *_: None
    writer.monitor_particle_two_phase(sims, scene)
    return folder / "MPMParticle000003.npz"


@pytest.mark.parametrize("dimension", [2, pytest.param(3, marks=pytest.mark.isolated_dimension(3))])
@pytest.mark.parametrize(
    "material_type,continue_time",
    [
        ("TwoPhaseDoubleLayer", True),
        ("TwoPhaseDoubleLayer", False),
        ("TwoPhaseSingleLayer", True),
    ],
)
def test_two_phase_roundtrip_preserves_full_history_and_mass(
    tmp_path, taichi_runtime, dimension, material_type, continue_time
):
    scene, sims = scene_and_sim(dimension)
    sims.material_type = material_type
    if material_type == "TwoPhaseSingleLayer":
        scene.particle.phase.fill(0)
    path = write_snapshot(scene, sims, tmp_path)
    with np.load(path, allow_pickle=True) as saved:
        assert saved["state_vars"].item()["strain"].shape == (2,)
        assert saved["constitutive_state"].item()["strain"].shape == (2, 6)
    target, target_sims = scene_and_sim(dimension, capacity=7)
    target_sims.material_type = material_type
    target_sims.current_time, target_sims.current_step = 0.0, 0
    target_sims.is_continue = continue_time
    target.two_phase_fluid_mass_initialized = False
    target.psize = np.empty((0, dimension))
    target.particleNum[0] = 0
    BodyReader(target_sims).add_npz_body(target, {"Restart": True, "File": str(path)})

    assert target.particleNum[0] == 2
    assert target.two_phase_fluid_mass_initialized
    np.testing.assert_array_equal(target.psize, scene.psize)
    for name in scene.particle.field_dict:
        np.testing.assert_array_equal(
            field_to_numpy_prefix(getattr(target.particle, name), 2),
            field_to_numpy_prefix(getattr(scene.particle, name), 2),
        )
    for name in scene.material.stateVars.field_dict:
        np.testing.assert_array_equal(
            field_to_numpy_prefix(getattr(target.material.stateVars, name), 2),
            field_to_numpy_prefix(getattr(scene.material.stateVars, name), 2),
        )
    assert target_sims.current_time == (0.0042 if continue_time else 0.0)
    assert target_sims.current_step == (42 if continue_time else 0)


def test_legacy_restart_requires_original_sizes_and_clock_before_mutation(tmp_path, taichi_runtime):
    scene, sims = scene_and_sim(2)
    path = write_snapshot(scene, sims, tmp_path)
    with np.load(path, allow_pickle=True) as saved:
        legacy = dict(saved)
    del legacy["psize"], legacy["current_step"]
    legacy["state_vars"] = legacy.pop("constitutive_state")
    np.savez(path, **legacy)
    sims.current_time, sims.current_step = 0.0, 0
    reader = BodyReader(sims)
    template = {"Restart": True, "File": str(path)}
    before = scene.particle.x.to_numpy()
    with pytest.raises(ValueError, match="particle_size"):
        reader.add_npz_body(scene, template)
    with pytest.raises(ValueError, match="current_step"):
        reader.add_npz_body(scene, dict(template, ParticleSize=[0.025, 0.025]))
    assert sims.current_time == 0.0
    np.testing.assert_array_equal(scene.particle.x.to_numpy(), before)
    reader.add_npz_body(scene, dict(template, ParticleSize=[0.025, 0.025], CurrentStep=42))
    assert sims.current_time == 0.0042 and sims.current_step == 42


def test_restart_rejects_lost_tensor_history_before_changing_particles(tmp_path, taichi_runtime):
    scene, sims = scene_and_sim(2)
    path = write_snapshot(scene, sims, tmp_path)
    with np.load(path, allow_pickle=True) as saved:
        data = dict(saved)
    data["constitutive_state"].item()["strain"] = np.zeros(2)
    data["position"] = np.full((2, 2), 0.75)
    np.savez(path, **data)
    before = scene.particle.x.to_numpy()
    with pytest.raises(ValueError, match="components"):
        BodyReader(sims).add_npz_body(scene, {"Restart": True, "File": str(path)})
    np.testing.assert_array_equal(scene.particle.x.to_numpy(), before)


@pytest.mark.parametrize("saved_tractions", [False, True])
def test_two_phase_late_failure_restores_checkpoint_state(tmp_path, taichi_runtime, saved_tractions):
    source, source_sims = scene_and_sim(2)
    traction_type = ti.types.struct(pid=ti.i32, traction=ti.types.vector(2, float))
    source.boundary.particle_traction = traction_type.field(shape=3)
    source.boundary.ptraction_list[0] = 1
    source.boundary.particle_traction[0].pid = 1
    source.boundary.particle_traction[0].traction = [1, 2]
    path = write_snapshot(source, source_sims, tmp_path)
    if not saved_tractions:
        with np.load(path, allow_pickle=True) as saved:
            data = dict(saved)
        del data["particle_traction_state"]
        np.savez(path, **data)

    target, sims = scene_and_sim(2, capacity=7)
    target.particle.x.fill(-3)
    for field in target.material.stateVars.field_dict.values():
        field.fill(-4)
    target.particleNum[0] = 1
    target.is_rigid.fill(1)
    target.psize = np.full((1, 2), 0.05)
    target.restart_cell_pressure = np.array([17.0])
    target.two_phase_fluid_mass_initialized = False
    target.boundary.particle_traction = traction_type.field(shape=3)
    target.boundary.particle_traction.traction.fill(7)
    target.boundary.ptraction_list[0] = int(saved_tractions)
    sims.current_time, sims.current_step, sims.CurrentTime[None] = 0.75, 75, 0.75
    before_particle = target.particle.to_numpy()
    before_history = target.material.stateVars.to_numpy()
    before_traction = target.boundary.particle_traction.to_numpy()
    before_essentials = (target.boundary.is_rigid.copy(), target.boundary.psize.copy(), target.boundary.myRegion)
    reader = BodyReader(sims)

    def fail_traction(count, tractions, scene):
        assert scene.particleNum[0] == 0  # New tractions index the restored batch from zero.
        scene.boundary.get_essentials(scene.is_rigid, scene.psize, {"restored": True})
        if tractions:
            scene.boundary.particle_traction.traction.fill(-9)
        scene.boundary.ptraction_list[0] = 2
        sims.current_time, sims.current_step, sims.CurrentTime[None] = 8.0, 800, 8.0
        raise ValueError("injected two-phase traction failure")

    reader.set_traction = fail_traction
    template = {"Restart": True, "File": str(path)}
    if not saved_tractions:
        template["Traction"] = {"Pressure": [0, 1]}
    with pytest.raises(ValueError, match="injected two-phase"):
        reader.add_npz_body(target, template)
    for group, before in [
        (target.particle, before_particle),
        (target.material.stateVars, before_history),
        (target.boundary.particle_traction, before_traction),
    ]:
        for name, field in group.field_dict.items():
            np.testing.assert_array_equal(field.to_numpy(), before[name])
    assert target.particleNum[0] == 1
    np.testing.assert_array_equal(target.psize, np.full((1, 2), 0.05))
    np.testing.assert_array_equal(target.is_rigid.to_numpy(), 1)
    np.testing.assert_array_equal(target.restart_cell_pressure, [17.0])
    assert not target.two_phase_fluid_mass_initialized
    np.testing.assert_array_equal(target.boundary.is_rigid, before_essentials[0])
    np.testing.assert_array_equal(target.boundary.psize, before_essentials[1])
    assert target.boundary.myRegion == before_essentials[2]
    assert target.boundary.ptraction_list[0] == int(saved_tractions)
    assert (sims.current_time, sims.current_step, sims.CurrentTime[None]) == (0.75, 75, 0.75)
