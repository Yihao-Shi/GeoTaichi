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


def _case(
    tmp_path,
    dilation=0.0,
    inexact=False,
    state_dependent=False,
    newmark=(1.0, 0.5, 1.0),
    material="DruckerPrager",
    ground=False,
    affine_bodies=1,
    shape_function="Linear",
    material_parameters=None,
    gravity=(0.0, 0.0, 0.0),
    fixed_displacement=None,
    mpm_newmark=None,
    particle_count=1,
):
    from geotaichi import MPDEM, polyhedron

    coupling = MPDEM(coupling=False, log=False)
    mpm, dem = coupling.mpm, coupling.dem
    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=[1.0, 1.0, 1.0],
        gravity=gravity,
        visualize=False,
        ipc=True,
    )
    body = mpm.create_body()
    body.add_particles(
        np.array([[0.54, 0.53, 0.36], [0.57, 0.52, 0.37]])[:particle_count],
        volume=1e-3,
        init_v=[0.02, 0.0, -0.02],
        grid_size=0.1,
        xmin=[0.0, 0.0, 0.0],
        xmax=[1.0, 1.0, 1.0],
        boundary_ids=np.arange(particle_count, dtype=np.int32),
        surface_measure=1e-2,
    )
    mpm.add_body(body)
    if ground:
        plane = mpm.create_ground()
        plane.append([0.0, 0.0, 0.31], [0.0, 0.0, 1.0])
        mpm.add_ground(plane)
    material_values = dict(young_modulus=1e4, poisson_ratio=0.3, density=1000.0)
    material_values.update(
        {"YieldStress": 20.0, "HardeningModulus": 100.0}
        if material == "VonMises"
        else {"Cohesion": 1.0, "FrictionAngle": 30.0, "DilationAngle": dilation}
    )
    if material == "ModifiedCamClay":
        material_values = dict(
            density=1000.0,
            PoissonRatio=0.3,
            StressRatio=1.2,
            lambda_star=0.12,
            kappa_star=0.03,
            void_ratio_ref=0.8,
            pc0=200.0,
            InitialPressure=100.0,
        )
    material_values.update(material_parameters or {})
    mpm.add_material(
        model="StateDependentDruckerPrager" if state_dependent else material,
        **material_values,
        **(
            {"e0": 0.62, "e_Tao": 0.9, "lambda_c": 0.119, "ksi": 0.23, "nd": 1.7, "nf": 2.68} if state_dependent else {}
        ),
    )
    mpm.add_element({"ElementSize": 0.1, "ShapeFunction": shape_function})
    if fixed_displacement is not None:
        from src.mpm.boundaries.BoundaryCondition import DirichletBoundary

        boundary = DirichletBoundary()
        boundary.append([[3 * (5 + 11 * (5 + 11 * 3))]], [fixed_displacement])
        mpm.add_boundary_condition(dirichlet=boundary)
    mpm.memory_allocate({"max_material_number": 1, "max_particle_number": particle_count}, log=False)
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
        domain=[1.0, 1.0, 1.0], scheme="AffineBody", search="BVH", gravity=gravity, visualize=False, log=False
    )
    coupling.set_configuration(
        domain=[1.0, 1.0, 1.0],
        coupling_scheme="MPDEM",
        search="BVH",
        gravity=gravity,
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        newmark=newmark,
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
            "max_affine_body_number": affine_bodies,
            "max_rigid_template_number": 1,
            "surface_node_number": 6 * affine_bodies,
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
    if affine_bodies == 2:
        dem.create_body(
            {
                "BodyType": "AffineBody",
                "Template": {
                    "Name": "oct",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [0.515, 0.51, 0.68],
                    "ScaleFactor": 0.1,
                    "InitialVelocity": [0.01, -0.005, 0.0],
                },
            }
        )
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
    if mpm_newmark is not None:
        mpm.direct_solver["newmark"] = mpm_newmark
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


@pytest.mark.parametrize("material", ["DruckerPrager", "VonMises"])
@pytest.mark.parametrize("newmark", [(1.0, 0.5, 1.0), (0.5, 0.25, 0.5)])
def test_associated_route_uses_physical_newton_without_material_outer(tmp_path, newmark, material):
    coupling = _case(tmp_path, material=material, dilation=None, inexact=None, newmark=newmark)
    system = coupling.enginer
    assert system.matrix.solver == "BiCGSTAB"
    assert not system.matrix.matrix_symmetric
    assert not system.mpm.has_lagged_material
    assert not system.inexact_newton
    if material == "DruckerPrager":
        assert system.mpm.material.dilation_is_tied
        assert not system.mpm.material.is_nonassociated
        assert system.mpm.material.use_direct_nonassociated_solve
    result = system.solve_lagged_equilibrium_device(coupling.dem.sims)
    assert result["friction_converged"]
    assert "material_lagged_iterations" not in result
    assert result["linear_solver"] == "BiCGSTAB"


def test_public_nonassociated_default_keeps_fixed_linear_tolerances(tmp_path):
    coupling = _case(tmp_path, inexact=None)
    assert not coupling.enginer.inexact_newton
    assert coupling.enginer.mpm.linear_solver_relative_tolerance == 0.0
    result = coupling.run(verbose=False)
    assert result["converged"] and result["step"] == 1
    assert not result["last_step"]["inexact_newton"]
    assert result["last_step"]["friction_converged"]


@pytest.mark.parametrize("state_dependent", [False, True])
def test_public_nonassociated_inexact_step_commits_history(tmp_path, state_dependent):
    coupling = _case(tmp_path, inexact=True, state_dependent=state_dependent)
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


@pytest.mark.parametrize("material,dilation", [("VonMises", 30.0), ("DruckerPrager", 15.0), ("DruckerPrager", None)])
def test_coupled_trajectory_state_and_history_vjp_matches_fd(tmp_path, material, dilation):
    coupling = _case(tmp_path, material=material, dilation=dilation)
    system, mpm = coupling.enginer, coupling.enginer.mpm
    assert not mpm.has_lagged_material
    system.mixed.pair_friction.fill(0.0)
    mpm.F0.from_numpy(np.array([np.diag([1.08, 0.94, 0.98])]))
    mpm.material.equivalent_plastic_strain[0] = 0.01
    trajectory = coupling.differentiable(steps=3)
    fields = {
        "position": mpm.particle.x,
        "velocity": mpm.particle.v,
        "acceleration": mpm.particle.a,
        "deformation_gradient": mpm.F0,
        "plastic_deformation_inverse": mpm.material.plastic_deformation_inverse,
        "equivalent_plastic_strain": mpm.material.equivalent_plastic_strain,
        "volumetric_plastic_strain": mpm.material.volumetric_plastic_strain,
        "affine_position": system.affine.y,
        "affine_velocity": system.affine.velocity_y,
    }
    initial = {name: field.to_numpy().copy() for name, field in fields.items()}
    rng = np.random.default_rng(731)
    seeds = {name: rng.normal(size=value.shape) * 0.1 for name, value in initial.items()}
    seeds["acceleration"] *= 1e-4
    directions = {name: rng.normal(size=value.shape) * 0.02 for name, value in initial.items()}
    with pytest.raises(RuntimeError, match="record at least one"):
        trajectory.backward(seeds)
    for _ in range(2):
        step = trajectory.step(verbose=False)
        assert step["friction_converged"]
        assert "material_lagged_iterations" not in step
    assert mpm.material.equivalent_plastic_strain[0] > initial["equivalent_plastic_strain"][0] + 1e-8
    assert system.mixed.diagnostics()["active_contacts"] > 0
    terminal = {name: field.to_numpy().copy() for name, field in fields.items()}
    if material == "VonMises":
        np.testing.assert_allclose(
            np.linalg.det(terminal["plastic_deformation_inverse"]),
            np.linalg.det(initial["plastic_deformation_inverse"]),
            rtol=1e-10,
            atol=1e-12,
        )
    elif dilation is None:
        assert terminal["volumetric_plastic_strain"][0] > initial["volumetric_plastic_strain"][0] + 1e-8
    terminal_grid = mpm.grid_disp.to_numpy().copy()
    history = list(mpm.history)
    gradient = trajectory.backward(seeds)
    for name, field in fields.items():
        np.testing.assert_array_equal(field.to_numpy(), terminal[name])
    np.testing.assert_array_equal(mpm.grid_disp.to_numpy(), terminal_grid)
    assert mpm.time == pytest.approx(0.002)
    assert mpm.step_count == coupling.sims.current_step == coupling.dem.sims.current_step == 2
    assert mpm.history == history
    assert all(np.isfinite(value).all() for value in gradient.values())
    analytic = 0.0
    for name, direction in directions.items():
        key = "affine_initial_" + name[7:] if name.startswith("affine_") else "initial_" + name
        analytic += np.sum(gradient[key] * direction)

    def objective(shift):
        for name, field in fields.items():
            field.from_numpy(initial[name] + shift * directions[name])
        mpm.time, mpm.step_count = 0.0, 0
        for _ in range(2):
            trajectory._advance(False)
        return sum(np.sum(seeds[name] * field.to_numpy()) for name, field in fields.items())

    for delta in (1e-5, 2e-6):
        numerical = (objective(delta) - objective(-delta)) / (2.0 * delta)
        assert analytic == pytest.approx(numerical, rel=2e-3, abs=2e-7), (analytic, numerical, material)


def test_coupled_trajectory_mixed_friction_and_failure_restore(tmp_path, monkeypatch):
    coupling = _case(tmp_path, dilation=30.0, material="VonMises")
    mpm, system = coupling.enginer.mpm, coupling.enginer
    trajectory = coupling.differentiable(steps=3)
    seed = {"affine_position": np.tile([0.2, -0.1, 0.3], (4, 1)), "position": np.array([[0.1, 0.2, -0.3]])}
    initial_mu = system.mixed.pair_friction.to_numpy().copy()
    initial_y = system.affine.y.to_numpy().copy()
    rotation = np.cross([0.0, 0.0, 1.0], initial_y - initial_y.mean(axis=0))

    def topology():
        candidates = system.mixed.friction_candidate.to_numpy()[: system.mixed.friction_count]
        return sorted(tuple(row) for row in candidates)

    def restore_initial():
        trajectory.mpm_diff._load_state(0)
        trajectory._load_affine_state(0)
        system.mixed.pair_friction.from_numpy(initial_mu)
        mpm.time, mpm.step_count = 0.0, 0

    # Keep every cache refresh in the original forward, including within a frame.
    def objective(kind, shift):
        restore_initial()
        if kind == "initial_velocity":
            mpm.particle[0].v += ti.Vector([shift, 0.0, 0.0])
        elif kind == "initial_position":
            mpm.particle[0].x += ti.Vector([0.0, 0.0, shift])
        elif kind == "affine_initial_position":
            system.affine.y.from_numpy(initial_y + shift * rotation)
        else:
            system.mixed.pair_friction.from_numpy(initial_mu + shift)
        for record in range(2):
            forward = trajectory._advance(False)
            assert forward["friction_iterations"] == outer_counts[record]
            assert topology() == contacts[record]
        return np.sum(seed["affine_position"] * system.affine.y.to_numpy()) + np.sum(
            seed["position"] * mpm.particle.x.to_numpy()
        )

    for iterations in (-1, 1, 2):
        coupling.dem.sims.affine_friction_iterations = iterations
        restore_initial()
        trajectory.record_count = 0
        contacts, outer_counts = [], []
        for _ in range(2):
            forward = trajectory.step(verbose=False)
            outer_counts.append(forward["friction_iterations"])
            contacts.append(topology())
            assert system.mixed.friction_count > 0
            if iterations < 0:
                assert forward["friction_converged"]
        assert max(outer_counts) >= (2 if iterations != 1 else 1)
        terminal = mpm.particle.x.to_numpy().copy(), system.affine.y.to_numpy().copy(), mpm.F0.to_numpy().copy()
        result = trajectory.backward(seed)
        analytic = {
            "initial_velocity": result["initial_velocity"][0, 0],
            "initial_position": result["initial_position"][0, 2],
            "affine_initial_position": np.sum(result["affine_initial_position"] * rotation),
            "mixed_friction_coefficient": result["mixed_friction_coefficient"][0, 0],
        }
        for kind, value in analytic.items():
            assert abs(value) > 1e-12
            for delta in (1e-5, 2e-6):
                numerical = (objective(kind, delta) - objective(kind, -delta)) / (2.0 * delta)
                assert value == pytest.approx(numerical, rel=3e-3, abs=1e-8), (
                    iterations,
                    kind,
                    delta,
                    value,
                    numerical,
                )

    system.mixed.pair_friction.from_numpy(initial_mu)
    trajectory.mpm_diff._load_state(2)
    trajectory._restore_affine_state(2)
    mpm.time, mpm.step_count = 0.002, 2

    def fail(*args, **kwargs):
        raise RuntimeError("forced coupled adjoint failure")

    with monkeypatch.context() as patch:
        patch.setattr(system, "solve_adjoint_device", fail)
        with pytest.raises(RuntimeError, match="forced coupled adjoint"):
            trajectory.backward(seed)
    np.testing.assert_array_equal(mpm.particle.x.to_numpy(), terminal[0])
    np.testing.assert_array_equal(system.affine.y.to_numpy(), terminal[1])
    np.testing.assert_array_equal(mpm.F0.to_numpy(), terminal[2])
    assert mpm.step_count == coupling.sims.current_step == 2
    assert system.ipc.adjoint_trajectory_record == -1
    commit = system.commit_step_device

    def fail_commit():
        commit()
        raise RuntimeError("forced coupled commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(system, "commit_step_device", fail_commit)
        with pytest.raises(RuntimeError, match="forced coupled commit"):
            trajectory.step(verbose=False)
    np.testing.assert_array_equal(mpm.particle.x.to_numpy(), terminal[0])
    np.testing.assert_array_equal(system.affine.y.to_numpy(), terminal[1])
    np.testing.assert_array_equal(mpm.F0.to_numpy(), terminal[2])
    assert mpm.step_count == trajectory.record_count == 2
    trajectory.step(verbose=False)
    assert mpm.step_count == 3
    with pytest.raises(RuntimeError, match="tape is full"):
        trajectory.step(verbose=False)
    mpm.step_count += 1
    with pytest.raises(RuntimeError, match="only through"):
        trajectory.backward(seed)
