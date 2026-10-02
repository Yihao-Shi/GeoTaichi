from pathlib import Path

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.dem.mainDEM import DEM
from src.fem.mainFEM import FEM
from src.fem.generator import FEMMesh
from src.fedem.mainFEDEM import FEDEM
from src.sdf.BasicShape import polyhedron


pytestmark = [
    pytest.mark.integration,
    pytest.mark.fedem,
    pytest.mark.coupling,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    yield
    ti.reset()


def _coupled_system(
    tmp_path,
    *,
    contact_model="Linear",
    contact_property=None,
    particle_velocity=(0.0, 0.0, 0.0),
    search="LinkedCell",
):
    dem = DEM(log=False)
    dem.set_configuration(
        domain=[2.0, 2.0, 2.0],
        boundary=["Reflect", "Reflect", "Reflect"],
        gravity=[0.0, 0.0, 0.0],
        engine="SymplecticEuler",
        search="LinkedCell",
        log=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_particle_number": 1,
            "max_sphere_number": 1,
            "max_clump_number": 0,
            "verlet_distance_multiplier": 0.1,
        },
        log=False,
    )
    dem.add_attribute(
        materialID=0,
        attribute={
            "Density": 1000.0,
            "ForceLocalDamping": 0.0,
            "TorqueLocalDamping": 0.0,
        },
    )
    dem.create_body(
        {
            "BodyType": "Sphere",
            "Template": [
                {
                    "GroupID": 0,
                    "MaterialID": 0,
                    "InitialVelocity": list(particle_velocity),
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "BodyPoint": [0.5, 0.5, 0.08],
                    "FixVelocity": ["Free", "Free", "Free"],
                    "FixAngularVelocity": ["Free", "Free", "Free"],
                    "Radius": 0.1,
                    "BodyOrientation": "uniform",
                }
            ],
        }
    )
    dem.choose_contact_model(None, None)
    dem.select_save_data()

    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit")
    fem.add_mesh(
        FEMMesh(
            np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 1.5, 0.0]]),
            np.array([[0, 1, 2]], dtype=np.int32),
            "TRI3",
        )
    )
    fem.add_material(
        "StVK",
        density=1000.0,
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        thickness=0.05,
    )
    fem.add_boundary_condition(
        {
            "type": "Dirichlet",
            "nodes": [0, 1, 2],
            "components": "all",
            "value": 0.0,
        }
    )

    coupling = FEDEM(dem, fem, log=False)
    coupling.set_configuration(
        domain=[2.0, 2.0, 2.0],
        gravity=[0.0, 0.0, 0.0],
        search=search,
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": 1.0e-5,
            "SimulationTime": 1.0e-5,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
        },
        log=False,
    )
    coupling.add_surface(modifier={"Orientation": "Parallel", "Direction": [0, 0, 1]})
    coupling.memory_allocate(
        {
            "contact_coordination_number": 8,
            "max_contact_pairs": 8,
            "max_facet_cell_pairs": 128,
        }
    )
    coupling.choose_contact_model(contact_model)
    if contact_property is None:
        contact_property = {
            "NormalStiffness": 1.0e5,
            "TangentialStiffness": 1.0e5,
            "Friction": 0.2,
            "NormalViscousDamping": 0.0,
            "TangentialViscousDamping": 0.0,
        }
    coupling.add_property(
        0,
        0,
        contact_property,
    )
    coupling.add_essentials()
    coupling.enginer.pre_calculate()
    return coupling


def _advance_coupled_step(coupling):
    coupling.enginer.step()
    dt = coupling.sims.delta
    coupling.sims.current_time += dt
    coupling.dem.sims.current_time += dt
    coupling.fem.engine.time = coupling.sims.current_time
    coupling.sims.current_step += 1
    coupling.dem.sims.current_step += 1
    coupling.fem.engine.step_count += 1


def _levelset_coupled_system(tmp_path):
    root = Path(__file__).resolve().parents[3]
    dem = DEM(log=False)
    dem.set_configuration(
        domain=[1.0, 1.0, 1.0],
        scheme="LSDEM",
        engine="VelocityVerlet",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        log=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_rigid_body_number": 1,
            "levelset_grid_number": 16000,
            "surface_node_number": 32,
            "max_sphere_number": 0,
            "max_clump_number": 0,
            "max_plane_number": 0,
            "body_coordination_number": 4,
            "wall_coordination_number": 1,
            "verlet_distance_multiplier": [0.1, 0.1],
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.add_attribute(
        materialID=0,
        attribute={
            "Density": 1200.0,
            "ForceLocalDamping": 0.0,
            "TorqueLocalDamping": 0.0,
        },
    )
    dem.add_template(
        {
            "Name": "restart_levelset_sphere",
            "Object": polyhedron(file=str(root / "assets/mesh/AffineBody/lowpoly_sphere.obj")).grids(
                space=0.2, extent=2
            ),
            "WriteFile": False,
        }
    )
    dem.create_body(
        {
            "BodyType": "RigidBody",
            "Template": [
                {
                    "Name": "restart_levelset_sphere",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [0.5, 0.5, 0.30],
                    "ScaleFactor": 0.11,
                    "InitialVelocity": [0.2, 0.0, 0.0],
                    "InitialAngularVelocity": [0.0, 0.0, 0.0],
                    "FixMotion": ["Free", "Free", "Free"],
                    "BodyOrientation": "constant",
                }
            ],
        }
    )
    dem.choose_contact_model(None, None)

    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit")
    fem.add_mesh(
        FEMMesh(
            np.array(
                [
                    [0.47, 0.47, 0.38],
                    [0.53, 0.47, 0.38],
                    [0.47, 0.53, 0.38],
                ]
            ),
            np.array([[0, 1, 2]], dtype=np.int32),
            "TRI3",
        )
    )
    fem.add_material(
        "StVK",
        density=1000.0,
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        thickness=0.02,
    )
    fem.add_boundary_condition(
        {
            "type": "Dirichlet",
            "nodes": [0, 1, 2],
            "components": "all",
            "value": 0.0,
        }
    )
    coupling = FEDEM(dem, fem, log=False)
    coupling.set_configuration(
        domain=[1.0, 1.0, 1.0],
        gravity=[0.0, 0.0, 0.0],
        search="BVH",
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": 1.0e-5,
            "SimulationTime": 1.0e-4,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
        },
        log=False,
    )
    coupling.add_surface(modifier={"Orientation": "Parallel", "Direction": [0, 0, 1]})
    coupling.memory_allocate(
        {
            "contact_coordination_number": 8,
            "max_contact_pairs": 16,
            "max_levelset_cell_pairs": 256,
        }
    )
    coupling.choose_contact_model("Linear")
    coupling.add_property(
        0,
        0,
        {
            "NormalStiffness": 1.0e4,
            "TangentialStiffness": 5.0e3,
            "Friction": 0.5,
            "NormalViscousDamping": 0.0,
            "TangentialViscousDamping": 0.0,
        },
    )
    coupling.add_essentials()
    coupling.enginer.pre_calculate()
    return coupling


@pytest.mark.parametrize("search", ["LinkedCell", "BVH"])
def test_explicit_fedem_preserves_source_contact_law_and_action_reaction(
    tmp_path,
    search,
):
    coupling = _coupled_system(tmp_path, search=search)
    assert coupling.contactor.neighbor.contact_count == 1

    coupling.enginer.reset_message()
    coupling.enginer.system_resolve()

    contact = coupling.contactor.neighbor.contacts
    normal_force = contact.normal_force.to_numpy()[0]
    dem_force = coupling.dem.scene.particle.contact_force.to_numpy()[0]
    fem_force = coupling.fem.engine.state.external_force.to_numpy().sum(axis=0)
    assert normal_force[2] == pytest.approx(2000.0, rel=5.0e-7)
    assert np.linalg.norm(dem_force + fem_force) < 1.0e-6 * np.linalg.norm(dem_force)

    coupling.enginer.integration()
    assert coupling.enginer.minimum_jacobian == pytest.approx(1.0)

    coupling.sims.set_timestep(10.0)
    coupling.dem.get_critical_timestep = lambda: 4.0
    coupling.fem.engine.stable_time_step = lambda cfl: 2.0 * cfl
    coupling.contactor.critical_timestep = lambda *_: 3.0
    coupling.check_critical_timestep()
    assert coupling.sims.delta == pytest.approx(1.0)
    assert coupling.dem.sims.delta == pytest.approx(1.0)
    assert coupling.fem.engine.dt == pytest.approx(1.0)


def test_fully_fixed_fedem_surface_skips_deformation_work(tmp_path, monkeypatch):
    coupling = _coupled_system(tmp_path)
    engine = coupling.enginer
    fem = coupling.fem.engine
    initial_position = fem.state.position.to_numpy().copy()
    initial_area = coupling.patch.node_area.to_numpy().copy()

    assert engine.static_fem
    assert fem.is_fully_constrained_static
    assert engine.minimum_jacobian == pytest.approx(1.0)

    def unexpected(*_args, **_kwargs):
        raise AssertionError("fixed FEM surface entered deformable-body work")

    monkeypatch.setattr(coupling.patch, "update", unexpected)
    monkeypatch.setattr(fem, "_assemble_internal_device", unexpected)
    monkeypatch.setattr(fem, "_minimum_jacobian_ratio_device", unexpected)
    monkeypatch.setattr(fem.dirichlet, "values", unexpected)
    monkeypatch.setattr(fem.neumann, "force", unexpected)
    engine.step(update_diagnostics=False)

    np.testing.assert_allclose(fem.state.position.to_numpy(), initial_position)
    np.testing.assert_allclose(coupling.patch.node_area.to_numpy(), initial_area)
    assert engine.minimum_jacobian == pytest.approx(1.0)


def test_explicit_fedem_can_decimate_host_safety_flag_reads(tmp_path, monkeypatch):
    coupling = _coupled_system(tmp_path)
    engine = coupling.enginer
    fem = coupling.fem.engine
    # Exercise the dynamic branch even though this compact fixture constrains
    # all three membrane nodes.
    engine.static_fem = False
    engine._bind_fem_runtime_functions()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("decimated step performed a host safety-flag read")

    monkeypatch.setattr(engine.dem_engine, "is_verlet_update", unexpected)
    monkeypatch.setattr(coupling.contactor, "surface_requires_rebuild", unexpected)
    monkeypatch.setattr(fem, "_minimum_jacobian_ratio_device", unexpected)

    class RecordingProfiler:
        def __init__(self):
            self.names = []

        def measure(self, name, function, *args, **kwargs):
            self.names.append(name)
            return function(*args, **kwargs)

    profiler = RecordingProfiler()

    engine.reset_message()
    engine.update_verlet_tables(check_rebuild=False)
    engine.system_resolve(check_rebuild=False, stage_profiler=profiler)
    engine.integration(
        update_diagnostics=False,
        check_jacobian=False,
        stage_profiler=profiler,
    )

    assert np.isfinite(fem.state.position.to_numpy()).all()
    assert profiler.names == [
        "dem_contact_force",
        "fem_lsdem_wall_contact",
        "fem_fem_contact",
        "rigid_body_integration",
        "fem_constitutive_update",
        "fem_internal_force",
        "fem_nodal_integration",
        "fem_boundary_update",
    ]


def test_fem_fem_only_coupling_needs_no_dummy_dem_body(tmp_path):
    dem = DEM(log=False)
    dem.set_configuration(
        domain=[1.0, 1.0, 1.0],
        scheme="LSDEM",
        engine="SymplecticEuler",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        log=False,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_rigid_body_number": 1,
            "levelset_grid_number": 1,
            "surface_node_number": 1,
            "body_coordination_number": 1,
            "wall_coordination_number": 1,
            "verlet_distance_multiplier": [0.1, 0.1],
        },
        log=False,
    )
    dem.add_attribute(
        materialID=0,
        attribute={
            "Density": 1000.0,
            "ForceLocalDamping": 0.0,
            "TorqueLocalDamping": 0.0,
        },
    )
    dem.choose_contact_model(None, None)
    dem.select_save_data(particle=True, surface=True)

    first = FEMMesh(
        np.array(
            [
                [0.20, 0.20, 0.20],
                [0.30, 0.20, 0.20],
                [0.20, 0.30, 0.20],
                [0.20, 0.20, 0.30],
            ]
        ),
        np.array([[0, 1, 2, 3]], dtype=np.int32),
        "TET4",
    )
    second = FEMMesh(
        np.array(
            [
                [0.60, 0.20, 0.20],
                [0.70, 0.20, 0.20],
                [0.60, 0.30, 0.20],
                [0.60, 0.20, 0.30],
            ]
        ),
        np.array([[0, 1, 2, 3]], dtype=np.int32),
        "TET4",
    )
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Explicit", log=False)
    mesh = fem.add_soft_particle(FEMMesh.concatenate([first, second]))
    fem.add_material(
        "NeoHookean",
        density=1000.0,
        young_modulus=1.0e4,
        poisson_ratio=0.3,
    )
    fem.add_soft_particle_contact(
        "Linear",
        search="BVH",
        ContactThickness=0.0,
        NormalStiffness=1.0e5,
        TangentialStiffness=1.0e5,
        Friction=0.0,
        max_point_triangle_pairs=64,
        max_edge_edge_pairs=128,
        contact_history_capacity=256,
    )
    initial_velocity = np.zeros_like(mesh.points)
    initial_velocity[mesh.node_body_ids == 0, 0] = 0.1
    initial_velocity[mesh.node_body_ids == 1, 0] = -0.1

    coupling = FEDEM(dem, fem, log=False)
    coupling.set_configuration(
        domain=[1.0, 1.0, 1.0],
        gravity=[0.0, 0.0, 0.0],
        search="BVH",
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": 1.0e-5,
            "SimulationTime": 1.0e-5,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
            "initial_velocity": initial_velocity,
        },
        log=False,
    )
    coupling.select_save_data(contact=True, checkpoint=True)
    coupling.add_surface(body_ids=[0, 1])
    coupling.memory_allocate(
        {
            "contact_coordination_number": 8,
            "max_contact_pairs": 64,
            "max_levelset_cell_pairs": 64,
            "verlet_distance_multiplier": 0.1,
        }
    )
    coupling.choose_contact_model("Linear")
    for body_id in (0, 1):
        coupling.add_property(
            0,
            body_id,
            {
                "NormalStiffness": 1.0e5,
                "TangentialStiffness": 1.0e5,
                "Friction": 0.0,
                "NormalViscousDamping": 0.0,
                "TangentialViscousDamping": 0.0,
            },
        )
    coupling.add_essentials()
    coupling.enginer.pre_calculate()

    assert coupling.contactor.neighbor is None
    coupling.enginer.step(update_diagnostics=False)
    coupling.save_data()

    contact_files = sorted(
        path for path in (tmp_path / "FEDEMcontacts").glob("*.npz") if not path.name.startswith("._")
    )
    checkpoint_files = sorted(
        path for path in (tmp_path / "checkpoints").glob("*.npz") if not path.name.startswith("._")
    )
    assert len(contact_files) == 1
    assert len(checkpoint_files) == 1
    with np.load(contact_files[0]) as archive:
        assert str(archive["contact_kind"]) == "FEM-FEM"
        assert "point_triangle" in archive
        assert "edge_edge" in archive


def test_explicit_fedem_hertz_mindlin_matches_migrated_equation(tmp_path):
    modulus = 1.0e6
    poisson = 0.25
    coupling = _coupled_system(
        tmp_path,
        contact_model="HertzMindlin",
        contact_property={
            "ShearModulus": modulus,
            "Poisson": poisson,
            "Friction": 0.0,
            "Restitution": 1.0,
        },
    )

    coupling.enginer.reset_message()
    coupling.enginer.system_resolve()

    effective_shear = 0.5 * modulus / (2.0 - poisson)
    effective_young = (4.0 * effective_shear - 2.0 * effective_shear * poisson) / (1.0 - poisson)
    penetration = 0.02
    contact_radius = np.sqrt(0.1 * penetration)
    stiffness = 2.0 * effective_young * contact_radius
    expected = (2.0 / 3.0) * stiffness * penetration
    normal_force = coupling.contactor.neighbor.contacts.normal_force.to_numpy()[0]
    assert normal_force[2] == pytest.approx(expected, rel=5.0e-7)
    assert np.isfinite(coupling.contactor.critical_timestep(coupling.dem.scene, coupling.dem.sims))


def test_explicit_fedem_history_survives_dynamic_list_rebuild(tmp_path):
    coupling = _coupled_system(tmp_path, particle_velocity=(1.0, 0.0, 0.0))
    coupling.enginer.reset_message()
    coupling.enginer.system_resolve()
    contacts = coupling.contactor.neighbor.contacts
    overlap_before = contacts.old_tangential_overlap.to_numpy()[0].copy()
    assert np.linalg.norm(overlap_before) > 0.0

    coupling.contactor.rebuild(coupling.dem.scene)

    assert coupling.contactor.neighbor.contact_count == 1
    overlap_after = contacts.old_tangential_overlap.to_numpy()[0]
    assert overlap_after == pytest.approx(overlap_before)


def test_explicit_fedem_npz_checkpoint_matches_uninterrupted_next_step(tmp_path):
    continuous = _coupled_system(tmp_path / "continuous", particle_velocity=(0.2, 0.0, 0.0))
    _advance_coupled_step(continuous)
    _advance_coupled_step(continuous)
    checkpoint = continuous.save_checkpoint(tmp_path / "state_at_step_2.npz")

    with np.load(checkpoint, allow_pickle=False) as archive:
        assert "metadata_json" in archive.files
        assert "state/fem.state/position" in archive.files
        assert any(name.endswith("/old_tangential_overlap") for name in archive.files)

    _advance_coupled_step(continuous)
    expected = {
        "fem_position": continuous.fem.engine.state.position.to_numpy(),
        "fem_velocity": continuous.fem.engine.state.velocity.to_numpy(),
        "dem_position": continuous.dem.scene.particle.x.to_numpy(),
        "dem_velocity": continuous.dem.scene.particle.v.to_numpy(),
        "overlap": continuous.contactor.neighbor.contacts.old_tangential_overlap.to_numpy(),
    }

    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    restarted = _coupled_system(tmp_path / "restarted", particle_velocity=(0.2, 0.0, 0.0))
    metadata = restarted.read_restart(checkpoint)
    assert metadata["schema"] == "geotaichi.fedem.checkpoint"
    assert restarted.sims.current_step == 2
    assert restarted.sims.current_time == pytest.approx(2.0e-5)

    _advance_coupled_step(restarted)
    np.testing.assert_allclose(
        restarted.fem.engine.state.position.to_numpy(),
        expected["fem_position"],
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        restarted.fem.engine.state.velocity.to_numpy(),
        expected["fem_velocity"],
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        restarted.dem.scene.particle.x.to_numpy(),
        expected["dem_position"],
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        restarted.dem.scene.particle.v.to_numpy(),
        expected["dem_velocity"],
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        restarted.contactor.neighbor.contacts.old_tangential_overlap.to_numpy(),
        expected["overlap"],
        rtol=1.0e-13,
        atol=1.0e-15,
    )


def test_levelset_checkpoint_restores_rigid_and_cross_contact_history(tmp_path):
    continuous = _levelset_coupled_system(tmp_path / "continuous_lsdem")
    _advance_coupled_step(continuous)
    assert continuous.contactor.level_set
    assert continuous.contactor.neighbor.contact_count > 0
    checkpoint_overlap = continuous.contactor.neighbor.contacts.old_tangential_overlap.to_numpy()
    assert np.linalg.norm(checkpoint_overlap) > 0.0
    checkpoint = continuous.save_checkpoint(tmp_path / "lsdem_step_1.npz")

    _advance_coupled_step(continuous)
    expected_position = continuous.dem.scene.rigid.mass_center.to_numpy()
    expected_velocity = continuous.dem.scene.rigid.v.to_numpy()
    expected_quaternion = continuous.dem.scene.rigid.q.to_numpy()
    expected_overlap = continuous.contactor.neighbor.contacts.old_tangential_overlap.to_numpy()

    ti.reset()
    ti.init(
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
    )
    restarted = _levelset_coupled_system(tmp_path / "restarted_lsdem")
    restarted.read_restart(checkpoint)
    np.testing.assert_array_equal(
        restarted.contactor.neighbor.contacts.old_tangential_overlap.to_numpy(),
        checkpoint_overlap,
    )
    _advance_coupled_step(restarted)
    np.testing.assert_allclose(
        restarted.dem.scene.rigid.mass_center.to_numpy(),
        expected_position,
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        restarted.dem.scene.rigid.v.to_numpy(),
        expected_velocity,
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        restarted.dem.scene.rigid.q.to_numpy(),
        expected_quaternion,
        rtol=1.0e-13,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        restarted.contactor.neighbor.contacts.old_tangential_overlap.to_numpy(),
        expected_overlap,
        rtol=1.0e-13,
        atol=1.0e-15,
    )
