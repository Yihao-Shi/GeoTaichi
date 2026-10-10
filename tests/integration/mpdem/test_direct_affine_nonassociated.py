"""Physical nonassociated DP and lagged friction in ordinary MPM--ABD IPC."""

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import taichi as ti

pytestmark = [
    pytest.mark.integration,
    pytest.mark.mpdem,
    pytest.mark.ipc,
    pytest.mark.serial,
    pytest.mark.isolated_dimension(3),
]


@pytest.fixture(autouse=True)
def runtime():
    ti.reset()
    ti.init(
        arch=ti.cuda if os.environ.get("GEOTAICHI_TEST_ARCH") == "cuda" else ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        device_memory_GB=0.25,
    )
    yield
    ti.reset()


def _case(tmp_path, dilation=0.0, inexact=False, state_dependent=False):
    from geotaichi import MPDEM, polyhedron

    coupling = MPDEM(coupling=False, log=False)
    mpm, dem = coupling.mpm, coupling.dem
    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=[1.0, 1.0, 1.0],
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
        ipc=True,
    )
    body = mpm.create_body()
    body.add_particles(
        np.array([[0.54, 0.53, 0.36]]),
        volume=1e-3,
        init_v=[0.02, 0.0, -0.02],
        grid_size=0.1,
        xmin=[0.0, 0.0, 0.0],
        xmax=[1.0, 1.0, 1.0],
        boundary_ids=np.array([0], dtype=np.int32),
        surface_measure=1e-2,
    )
    mpm.add_body(body)
    mpm.add_material(
        model="StateDependentDruckerPrager" if state_dependent else "DruckerPrager",
        young_modulus=1e4,
        poisson_ratio=0.3,
        density=1000.0,
        Cohesion=1.0,
        FrictionAngle=30.0,
        DilationAngle=dilation,
        **(
            {"e0": 0.62, "e_Tao": 0.9, "lambda_c": 0.119, "ksi": 0.23, "nd": 1.7, "nf": 2.68} if state_dependent else {}
        ),
    )
    mpm.add_element({"ElementSize": 0.1, "ShapeFunction": "Linear"})
    mpm.memory_allocate({"max_material_number": 1, "max_particle_number": 1}, log=False)
    mpm.add_contact(
        "BarrierIPC",
        dhat=0.06,
        dmin=0.001,
        kappa=1e4,
        mu=0.2,
        epsv=1e-3,
        friction_mode="lagged",
        barrier_set=[4, 1],
        friction_set=[4, 1],
    )
    dem.set_configuration(
        domain=[1.0, 1.0, 1.0], scheme="AffineBody", search="BVH", gravity=[0.0, 0.0, 0.0], visualize=False, log=False
    )
    coupling.set_configuration(
        domain=[1.0, 1.0, 1.0],
        coupling_scheme="MPDEM",
        search="BVH",
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        contact_model="BarrierIPC",
        assemble_type="HashTriplet",
        young_modulus=2e4,
        dhat=0.06,
        barrier_stiffness=1e4,
        friction_mode="lagged",
        friction_iterations=-1,
        friction_tolerance=1e-7,
        max_newton_iteration=40,
        line_search_max_iteration=24,
        hessian_shift=0.0,
        max_step=0.02,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": 1,
            "max_rigid_template_number": 1,
            "surface_node_number": 6,
            "body_coordination_number": 2,
            "wall_coordination_number": 0,
            "affine_contact_block_capacity": 256,
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.add_attribute(materialID=0, attribute={"Density": 1200.0})
    dem.add_template(
        {
            "Name": "oct",
            "TemplateType": "AffineBody",
            "Object": polyhedron(file=str(Path(__file__).resolve().parents[2] / "data/affine_octahedron.obj")),
        }
    )
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": {
                "Name": "oct",
                "GroupID": 0,
                "MaterialID": 0,
                "BodyPoint": [0.5, 0.5, 0.45],
                "ScaleFactor": 0.1,
                "InitialVelocity": [0.0, 0.0, 0.0],
            },
        }
    )
    dem.add_property(materialID1=0, materialID2=0, property={"Dhat": 0.06, "BarrierStiffness": 1e4, "Friction": 0.2})
    coupling.set_solver(
        {
            "Timestep": 0.001,
            "SimulationTime": 0.001,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
            "linear_solver_tolerance": 1e-10,
            "linear_solver_max_iters": 1000,
            "scale": 1.0,
            **({"inexact_newton": inexact} if inexact is not None else {}),
        },
        log=False,
    )
    coupling.memory_allocate(
        {"body_coordination_number": 2, "max_point_triangle_pairs": 32, "max_point_edge_pairs": 1}, log=False
    )
    coupling.choose_contact_model(
        "BarrierIPC",
        dhat=0.06,
        dmin=0.001,
        kappa=1e4,
        friction_coefficient=0.2,
        epsv=1e-3,
        friction_mode="lagged",
        friction_iterations=-1,
    )
    coupling.add_ipc_property(
        MPMbody=0,
        AffineBody=0,
        property={"dhat": 0.06, "dmin": 0.001, "kappa": 1e4, "friction_coefficient": 0.2, "epsv": 1e-3},
    )
    coupling.add_essentials()
    coupling.mpm.enginer.initial_simulation()
    coupling.mpm.enginer.prepare_step_device()
    return coupling


def test_nonassociated_jacobian_and_residual_only_friction(tmp_path):
    coupling = _case(tmp_path)
    system = coupling.enginer
    system.mpm.F0.from_numpy(np.array([np.diag([1.18, 0.82, 0.90])]))
    system.begin_step_device(0.001)
    mpm_dof = system.mpm.active_dof
    system.mpm.grid_disp.from_numpy(np.linspace(-2e-5, 3e-5, system.mpm.degree_of_freedom))
    assembled = system.assemble_linearization_device()
    matrix = system.matrix.to_scipy(assembled["active_nodes"]).toarray()
    assert not system.matrix.matrix_symmetric
    assert system.matrix.solver == "BiCGSTAB"
    assert not system.mpm.has_lagged_material
    assert system.mixed.friction_count > 0
    reference = system.physical_rhs.to_numpy().copy()
    diagonal = system.matrix.diag.to_numpy().copy()
    blocks = system.matrix.non_diag.blockH.to_numpy().copy()
    system.assemble_linearization_device(need_matrix=False)
    np.testing.assert_allclose(system.physical_rhs.to_numpy(), reference, rtol=1e-12, atol=1e-10)
    np.testing.assert_array_equal(system.matrix.diag.to_numpy(), diagonal)
    np.testing.assert_array_equal(system.matrix.non_diag.blockH.to_numpy(), blocks)
    y, u = system.affine.y.to_numpy(), system.mpm.grid_disp.to_numpy()
    size = assembled["active_dof"]
    for seed in range(3):
        direction = np.random.default_rng(seed).normal(size=size)
        direction /= np.linalg.norm(direction)
        values = []
        for sign in [-1.0, 1.0]:
            system.affine.y.from_numpy(y + sign * 2e-6 * direction[: 3 * system.affine_controls].reshape(-1, 3))
            trial = u.copy()
            trial[:mpm_dof] += sign * 2e-6 * direction[3 * system.affine_controls :]
            system.mpm.grid_disp.from_numpy(trial)
            system.assemble_linearization_device(need_matrix=False)
            values.append(-system.physical_rhs.to_numpy()[:size].copy())
        numerical = (values[1] - values[0]) / 4e-6
        np.testing.assert_allclose(matrix @ direction, numerical, rtol=1e-5, atol=2e-4)
    system.affine.y.from_numpy(y)
    system.mpm.grid_disp.from_numpy(u)
    system.backup_lagged_friction_for_adjoint_device()
    loss_gradient = np.random.default_rng(3).normal(size=size)
    adjoint = system.solve_adjoint_device(loss_gradient).to_numpy()[:size]
    np.testing.assert_allclose(matrix.T @ adjoint, loss_gradient, rtol=1e-7, atol=1e-7)


def test_nonassociated_strict_and_inexact_lagged_solve(tmp_path):
    coupling = _case(tmp_path)
    system = coupling.enginer
    system.mpm.F0.from_numpy(np.array([np.diag([1.04, 0.96, 0.98])]))
    initial = system.affine.y.to_numpy()
    results = []
    for inexact in [False, True]:
        system.inexact_newton = inexact
        for repeat in range(5):
            system.affine.y.from_numpy(initial)
            start = perf_counter()
            result = system.solve_lagged_equilibrium_device(coupling.dem.sims)
            elapsed = perf_counter() - start
            assert result["friction_converged"]
            assert result["force_residual"] < 1e-6
            assert system.mixed.diagnostics()["minimum_distance"] > 0.001
            if repeat:
                results.append((inexact, elapsed, result, system.mpm.grid_disp.to_numpy(), system.affine.y.to_numpy()))
    for result in results[1:]:
        np.testing.assert_allclose(results[0][3], result[3], rtol=1e-4, atol=1e-8)
        np.testing.assert_allclose(results[0][4], result[4], rtol=1e-7, atol=1e-8)
    print("strict/inexact warm solve:", [(r[0], r[1], r[2]) for r in results])


def test_associated_route_keeps_symmetric_pcg(tmp_path):
    coupling = _case(tmp_path, dilation=30.0, inexact=None)
    system = coupling.enginer
    assert system.matrix.solver == "PCG"
    assert system.matrix.matrix_symmetric
    assert not system.inexact_newton
    result = system.solve_lagged_equilibrium_device(coupling.dem.sims)
    assert result["friction_converged"]


@pytest.mark.parametrize("state_dependent", [False, True])
def test_public_nonassociated_inexact_step_commits_history(tmp_path, state_dependent):
    coupling = _case(tmp_path, inexact=None, state_dependent=state_dependent)
    assert coupling.enginer.inexact_newton
    assert coupling.run(verbose=False)["converged"]
    coupling.enginer.mpm.F0.from_numpy(np.array([np.diag([1.04, 0.96, 0.98])]))
    result = coupling.run(verbose=False)
    assert result["converged"] and result["step"] == 2
    assert result["last_step"]["inexact_newton"]
    assert result["last_step"]["friction_converged"]
    material = coupling.enginer.mpm.material
    assert material.equivalent_plastic_strain[0] > 0.0
    assert np.linalg.det(material.plastic_deformation_inverse.to_numpy()[0]) > 0.0
    if state_dependent:
        assert material.history_state_size == 14
        assert 0.1 <= material.void_ratio[0] <= 1.5
        assert material.state_pressure[0] >= 1000.0
        assert material.committed_jacobian[0] == pytest.approx(np.linalg.det(coupling.enginer.mpm.F0.to_numpy()[0]))
