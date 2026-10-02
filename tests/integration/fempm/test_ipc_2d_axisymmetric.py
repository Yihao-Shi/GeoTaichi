"""Planar and axisymmetric FEM--MPM IPC integration contracts."""

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem import FEM
from src.fempm import FEMPM
from src.mpm import MPM

pytestmark = [
    pytest.mark.integration,
    pytest.mark.fempm,
    pytest.mark.coupling,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(2),
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


def _coupled_system(tmp_path, axisymmetric, search, mpm_material="NeoHookean"):
    radial_origin = 0.5 if axisymmetric else 0.0
    contact_radius = 1.0 if axisymmetric else 0.5

    fem = FEM(log=False)
    fem.set_configuration(
        dimension=2,
        solver_type="Implicit",
        axisymmetric=axisymmetric,
        axis_offset=0.0,
    )
    fem.add_mesh(
        geometry="rectangle",
        size=(1.0, 0.2),
        divisions=(1, 1),
        origin=(radial_origin, 0.2, 0.0),
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
            "nodes": "all",
            "components": "xy",
            "value": 0.0,
        }
    )

    mpm = MPM(log=False)
    mpm.set_configuration(
        dimension=2,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=[2.0, 1.0],
        gravity=[0.0, 0.0],
        visualize=False,
        axisymmetric=axisymmetric,
        axis_offset=0.0,
        log=False,
    )
    body = mpm.create_body()
    body.add_particles(
        np.array([[contact_radius, 0.18]], dtype=np.float64),
        volume=1.0e-3,
        init_v=[0.0, 0.0],
        name="contact-point",
        grid_size=0.1,
        xmin=[0.0, 0.0],
        xmax=[2.0, 1.0],
        boundary_ids=np.array([0], dtype=np.int32),
        surface_measure=1.0e-2,
    )
    mpm.add_body(body)
    material_parameters = dict(
        model=mpm_material,
        young_modulus=1.0e4,
        poisson_ratio=0.3,
        density=1000.0,
    )
    if mpm_material == "DruckerPrager":
        material_parameters.update(
            FrictionAngle=30.0,
            DilationAngle=30.0,
            Cohesion=1.0,
            dpType="Circumscribed",
        )
    elif mpm_material == "VonMises":
        material_parameters.update(
            YieldStress=1.0,
            HardeningModulus=50.0,
        )
    mpm.add_material(**material_parameters)
    mpm.add_element({"ElementSize": 0.1, "ShapeFunction": "Linear"})

    coupling = FEMPM(fem, mpm, log=False)
    coupling.set_configuration(
        domain=[2.0, 1.0],
        gravity=[0.0, 0.0],
        search=search,
        axisymmetric=axisymmetric,
        axis_offset=0.0,
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": 1.0e-3,
            "SimulationTime": 1.0e-3,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
            "assemble_type": "HashTriplet",
            "linear_solver": "PCG",
            "max_iterations": 8,
            "residual_tolerance": 1.0e-7,
            "visualize": False,
        },
        log=False,
    )
    coupling.add_surface()
    coupling.memory_allocate({})
    coupling.choose_contact_model(
        "IPC",
        dhat=0.05,
        dmin=0.0,
        kappa=100.0,
        friction_coefficient=0.2,
        epsv=1.0e-3,
        friction_mode="lagged",
    )
    coupling.add_ipc_property(0, 0, friction_coefficient=0.2)
    coupling.add_essentials()
    return coupling


@pytest.mark.parametrize(
    "axisymmetric,search",
    [(False, "LinkedCell"), (True, "BVH")],
)
def test_fempm_planar_point_edge_ipc_assembles_symmetric_system(tmp_path, axisymmetric, search):
    coupling = _coupled_system(tmp_path, axisymmetric, search)
    system = coupling.enginer.assemble_system(include_friction=True)

    assert system["active_mpm_dof"] > 0
    assert system["contact_count"] >= 1
    assert coupling.enginer.contact.diagnostics()["active_contacts"] >= 1
    assert coupling.enginer.contact.dimension == 2
    assert coupling.enginer.contact.stencil_size == 3
    assert np.all(coupling.fem.engine.state.constrained.to_numpy().reshape(-1, 3)[:, 2] == 1)
    active_nodes = int(system["active_nodes"])
    matrix = system["matrix"].to_scipy(active_nodes)
    symmetry_error = (matrix - matrix.T).tocoo()
    maximum_error = float(np.max(np.abs(symmetry_error.data))) if symmetry_error.nnz else 0.0
    assert np.all(np.isfinite(matrix.data))
    assert maximum_error < 1.0e-8

    coupling.fem.engine.state.direction.fill(0.0)
    mpm_direction = np.zeros(coupling.mpm.enginer.degree_of_freedom, dtype=np.float64)
    mpm_direction[1 : system["active_mpm_dof"] : 2] = 0.04
    coupling.mpm.enginer.incre_resolution.from_numpy(mpm_direction)
    contact_step = coupling.enginer.contact.maximum_step(
        coupling.fem.engine.state.position,
        coupling.fem.engine.state.direction,
        coupling.mpm.enginer.grid_disp,
        coupling.mpm.enginer.incre_resolution,
    )
    assert contact_step == pytest.approx(0.45, abs=2.0e-10)

    if axisymmetric:
        particle_volume = coupling.mpm.enginer.particle.vol0.to_numpy()[0]
        contact_measure = coupling.mpm.enginer.surface_measure.to_numpy()[0]
        assert particle_volume == pytest.approx(2.0 * np.pi * 1.0e-3)
        assert contact_measure == pytest.approx(2.0 * np.pi * 1.0e-2)


@pytest.mark.parametrize("mpm_material", ["DruckerPrager", "VonMises"])
def test_fempm_axisymmetric_plastic_mpm_uses_three_dimensional_tangent(tmp_path, mpm_material):
    coupling = _coupled_system(
        tmp_path,
        axisymmetric=True,
        search="LinkedCell",
        mpm_material=mpm_material,
    )
    mpm = coupling.mpm.enginer
    mpm.F0.from_numpy(np.array([np.diag([1.18, 0.82, 1.12])], dtype=np.float64))
    system = coupling.enginer.assemble_system(include_friction=True)

    assert coupling.sims.is_axisymmetric
    assert mpm.is_finite_strain_plastic
    assert mpm.material_dimension == 3
    assert mpm.F0.to_numpy().shape == (1, 3, 3)
    assert system["active_mpm_dof"] > 0
    assert system["contact_count"] > 0
    matrix = system["matrix"].to_scipy(system["active_nodes"])
    symmetry_error = (matrix - matrix.T).tocoo()
    maximum_error = float(np.max(np.abs(symmetry_error.data))) if symmetry_error.nnz else 0.0
    assert np.all(np.isfinite(matrix.data))
    assert maximum_error < 1.0e-8
    assert np.all(np.isfinite(coupling.enginer.rhs.to_numpy()[: system["active_dof"]]))


def test_fempm_failed_plastic_step_restores_device_state(tmp_path, monkeypatch):
    coupling = _coupled_system(
        tmp_path,
        axisymmetric=False,
        search="BVH",
        mpm_material="DruckerPrager",
    )
    engine = coupling.enginer
    mpm = coupling.mpm.enginer
    assert mpm.is_plane_strain

    fem_position = coupling.fem.engine.state.position.to_numpy().copy()
    fem_velocity = coupling.fem.engine.state.velocity.to_numpy().copy()
    particle_position = mpm.particle.x.to_numpy().copy()
    particle_velocity = mpm.particle.v.to_numpy().copy()
    particle_acceleration = mpm.particle.a.to_numpy().copy()
    grid_velocity = mpm.grid.v.to_numpy().copy()
    grid_acceleration = mpm.grid.a.to_numpy().copy()
    deformation = mpm.F0.to_numpy().copy()
    equivalent = mpm.material.equivalent_plastic_strain.to_numpy().copy()
    volumetric = mpm.material.volumetric_plastic_strain.to_numpy().copy()
    plastic_inverse = mpm.material.plastic_deformation_inverse.to_numpy().copy()

    def fail_after_snapshot(verbose):
        coupling.fem.engine.state.position.fill(7.0)
        coupling.fem.engine.state.velocity.fill(8.0)
        mpm.particle.x.fill(9.0)
        mpm.particle.v.fill(10.0)
        mpm.F0.fill(2.0)
        mpm.material.equivalent_plastic_strain.fill(3.0)
        mpm.material.volumetric_plastic_strain.fill(4.0)
        mpm.material.plastic_deformation_inverse.fill(5.0)
        raise RuntimeError("synthetic post-snapshot failure")

    monkeypatch.setattr(engine, "_advance_prepared_substep", fail_after_snapshot)
    with pytest.raises(RuntimeError, match="synthetic post-snapshot failure"):
        engine.substep(verbose=False)

    np.testing.assert_array_equal(coupling.fem.engine.state.position.to_numpy(), fem_position)
    np.testing.assert_array_equal(coupling.fem.engine.state.velocity.to_numpy(), fem_velocity)
    np.testing.assert_array_equal(mpm.particle.x.to_numpy(), particle_position)
    np.testing.assert_array_equal(mpm.particle.v.to_numpy(), particle_velocity)
    np.testing.assert_array_equal(mpm.particle.a.to_numpy(), particle_acceleration)
    np.testing.assert_array_equal(mpm.grid.v.to_numpy(), grid_velocity)
    np.testing.assert_array_equal(mpm.grid.a.to_numpy(), grid_acceleration)
    np.testing.assert_array_equal(mpm.F0.to_numpy(), deformation)
    np.testing.assert_array_equal(mpm.material.equivalent_plastic_strain.to_numpy(), equivalent)
    np.testing.assert_array_equal(mpm.material.volumetric_plastic_strain.to_numpy(), volumetric)
    np.testing.assert_array_equal(mpm.material.plastic_deformation_inverse.to_numpy(), plastic_inverse)
    assert np.all(mpm.grid_disp.to_numpy() == 0.0)
    assert engine.time == 0.0
    assert engine.step_count == 0

    def fail_during_preparation():
        mpm.particle.x.fill(11.0)
        mpm.grid.v.fill(12.0)
        raise RuntimeError("synthetic preparation failure")

    monkeypatch.setattr(engine, "_prepare_mpm_step", fail_during_preparation)
    with pytest.raises(RuntimeError, match="synthetic preparation failure"):
        engine.substep(verbose=False)

    np.testing.assert_array_equal(mpm.particle.x.to_numpy(), particle_position)
    np.testing.assert_array_equal(mpm.grid.v.to_numpy(), grid_velocity)
    np.testing.assert_array_equal(mpm.grid.a.to_numpy(), grid_acceleration)
    np.testing.assert_array_equal(mpm.F0.to_numpy(), deformation)
    assert engine.time == 0.0
    assert engine.step_count == 0
