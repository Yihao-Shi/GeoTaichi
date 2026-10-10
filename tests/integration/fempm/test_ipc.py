import os

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem.generator import FEMMesh
from src.fem.mainFEM import FEM
from src.fempm.mainFEMPM import FEMPM
from src.mpm.mainMPM import MPM

pytestmark = [
    pytest.mark.integration,
    pytest.mark.fempm,
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
        arch=ti.cuda if os.environ.get("GEOTAICHI_TEST_ARCH") == "cuda" else ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
        device_memory_GB=0.25,
    )
    yield
    ti.reset()


def _implicit_system(
    tmp_path,
    search="BVH",
    friction=0.2,
    assembly="HashTriplet",
    force_elastoplastic=False,
    mpm_material="NeoHookean",
    contact_model="BarrierIPC",
    dilation=30.0,
    inexact=False,
):
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    fem.add_mesh(
        FEMMesh(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
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

    mpm = MPM(log=False)
    mpm.set_configuration(
        dimension=3,
        mpm_backend="Direct",
        solver_type="Implicit",
        configuration="ULMPM",
        domain=[1.0, 1.0, 1.0],
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
    )
    body = mpm.create_body()
    body.add_particles(
        np.array([[0.25, 0.25, 0.02]], dtype=np.float64),
        volume=1.0e-3,
        init_v=[0.1, 0.0, -0.02],
        name="elastic_point",
        grid_size=0.1,
        xmin=[0.0, 0.0, 0.0],
        xmax=[1.0, 1.0, 1.0],
        boundary_ids=np.array([0], dtype=np.int32),
        surface_measure=1.0e-2,
    )
    mpm.add_body(body)
    material_parameters = {
        "model": mpm_material,
        "young_modulus": 1.0e4,
        "poisson_ratio": 0.3,
        "density": 1000.0,
    }
    if mpm_material in ("DruckerPrager", "StateDependentDruckerPrager"):
        material_parameters.update(
            {
                "FrictionAngle": 30.0,
                "DilationAngle": dilation,
                "Cohesion": 10.0,
                "dpType": "Circumscribed",
            }
        )
        if mpm_material == "StateDependentDruckerPrager":
            material_parameters.update(e0=0.62, e_Tao=0.9, lambda_c=0.119, ksi=0.23, nd=1.7, nf=2.68, fai_c=30.0)
    elif mpm_material == "VonMises":
        material_parameters.update(
            {
                "YieldStress": 100.0,
                "HardeningModulus": 500.0,
            }
        )
    elif mpm_material == "ModifiedCamClay":
        material_parameters.update(
            {
                "StressRatio": 1.2,
                "lambda": 0.20,
                "kappa": 0.05,
                "void_ratio_ref": 0.8,
                "pc0": 200.0,
                "OverConsolidationRatio": 2.0,
            }
        )
    mpm.add_material(**material_parameters)
    mpm.add_element({"ElementSize": 0.1, "ShapeFunction": "Linear"})

    coupling = FEMPM(fem, mpm, log=False)
    coupling.set_configuration(
        domain=[1.0, 1.0, 1.0],
        gravity=[0.0, 0.0, 0.0],
        search=search,
        log=False,
    )
    coupling.set_solver(
        {
            "Timestep": 1.0e-3,
            "SimulationTime": 1.0e-3,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
            "assemble_type": assembly,
            "linear_solver": "PCG",
            "inexact_newton": inexact,
            "max_iterations": 12,
            "residual_tolerance": 1.0e-7,
            "scale": 0.25,
            "visualize": False,
        },
        log=False,
    )
    coupling.add_surface()
    coupling.memory_allocate({})
    coupling.choose_contact_model(
        contact_model,
        dhat=0.05,
        dmin=0.0,
        kappa=100.0,
        friction_coefficient=friction,
        epsv=1.0e-3,
        friction_mode="lagged",
    )
    coupling.add_ipc_property(0, 0, friction_coefficient=friction)
    if force_elastoplastic:
        coupling.fem.scene.material.is_fem_elastoplastic = True
    coupling.add_essentials()
    return coupling


@pytest.mark.parametrize("search", ["LinkedCell", "BVH"])
def test_implicit_ipc_builds_dynamic_candidates_and_cross_hessian(tmp_path, search):
    coupling = _implicit_system(tmp_path, search=search)
    system = coupling.enginer.assemble_system(include_friction=True)

    assert system["active_mpm_dof"] > 0
    assert system["contact_count"] >= 1
    assert coupling.enginer.contact.diagnostics()["active_contacts"] >= 1
    assert int(coupling.enginer.contact_hash.raw_non_diag_count[0]) > 0
    assert coupling.enginer.contact.activate_friction
    if search == "LinkedCell":
        result = coupling.run(steps=1, verbose=False)
        assert result["converged"]
        assert result["time"] == pytest.approx(1.0e-3)
        assert (tmp_path / "vtks" / "FEM000000.vtu").is_file()


def test_implicit_semi_ipc_assembles_cross_hessian(tmp_path):
    coupling = _implicit_system(
        tmp_path,
        search="LinkedCell",
        friction=0.2,
        contact_model="SemiIPC",
    )
    engine = coupling.enginer
    engine.contact.begin_step(
        engine.fem.state.position,
        engine.mpm.grid_disp,
        engine.dt,
    )
    system = engine.assemble_system(include_friction=True)

    assert system["contact_count"] >= 1
    assert engine.contact.friction_count >= 1
    assert engine.contact.diagnostics()["model"] == "SemiIPC"


def test_implicit_ipc_rejects_elastoplastic_fem_before_coupled_solve(tmp_path):
    with pytest.raises(RuntimeError, match="elastic FEM constitutive"):
        _implicit_system(tmp_path, friction=0.0, force_elastoplastic=True)


def test_fempm_nonpotential_plasticity_dispatches_to_residual_merit_line_search():
    from src.fempm.ImplicitEngine import FEMPMImplicitEngine

    engine = object.__new__(FEMPMImplicitEngine)
    engine.mpm_uses_residual_merit = True
    calls = []
    engine._residual_merit_line_search = lambda system, friction, residual: (
        calls.append((system, friction, residual)) or (0.5, 1, 0.25)
    )
    engine._total_energy = lambda *args: pytest.fail(
        "non-potential finite-strain plasticity must not use energy Armijo"
    )

    result = FEMPMImplicitEngine._line_search(
        engine,
        {"active_mpm_dof": 2},
        False,
        3.0,
    )

    assert result == (0.5, 1, 0.25)
    assert calls == [({"active_mpm_dof": 2}, False, 3.0)]


def test_implicit_ipc_accepts_coo_device_assembly(tmp_path):
    coupling = _implicit_system(tmp_path, assembly="COO")
    system = coupling.enginer.assemble_system(include_friction=True)
    assert system["matrix"] is coupling.enginer.monolithic_coo
    assert int(coupling.enginer.coo_count[None]) > 0


@pytest.mark.parametrize("assembly", ["HashTriplet", "COO"])
@pytest.mark.parametrize("dilation", [0.0, 15.0, 30.0])
def test_implicit_ipc_dp_keeps_ccd_and_armijo_pipeline(tmp_path, assembly, dilation):
    coupling = _implicit_system(
        tmp_path,
        assembly=assembly,
        mpm_material="DruckerPrager",
        dilation=dilation,
    )
    result = coupling.run(steps=1, verbose=False)

    assert result["converged"]
    assert coupling.enginer.mpm_uses_residual_merit == (dilation != 30.0)
    assert coupling.enginer.linear_solver == ("PCG" if dilation == 30.0 else "BiCGSTAB")
    if dilation != 30.0:
        assert not coupling.mpm.enginer.has_lagged_material
        assert result["history"][-1]["material_lagged_iterations"] == 0
    record = result["history"][-1]
    assert record["material"] == "FiniteStrainDruckerPragerModel"
    assert record["contact"]["active_contacts"] >= 1
    assert record["friction_iterations"][0]
    assert np.linalg.det(coupling.mpm.enginer.F0.to_numpy()[0]) > 0.0


def test_nonassociated_ipc_physical_jacobian_by_fd(tmp_path):
    coupling = _implicit_system(tmp_path, friction=0.0, mpm_material="DruckerPrager", dilation=0.0)
    engine = coupling.enginer
    mpm = engine.mpm
    mpm.F0.from_numpy(np.array([np.diag([1.18, 0.82, 0.90])]))
    system = engine.assemble_system(include_friction=False)
    active_dof = system["active_dof"]
    offset = 3 * engine.fem_nodes
    free = np.flatnonzero(engine.fixed.to_numpy()[:active_dof] == 0)
    base = np.linspace(-2e-5, 3e-5, mpm.active_dof)
    displacement = mpm.grid_disp.to_numpy()

    def evaluate(values, need_matrix=False):
        displacement[: mpm.active_dof] = values
        mpm.grid_disp.from_numpy(displacement)
        assembled = engine.assemble_system(include_friction=False, need_matrix=need_matrix)
        residual = -engine.physical_rhs.to_numpy()[free]
        matrix = assembled["matrix"].to_scipy(assembled["active_nodes"]).toarray() if need_matrix else None
        return residual, matrix

    _, full = evaluate(base, True)
    analytic = full[np.ix_(free, free)]
    numerical = np.zeros_like(analytic)
    step = 2e-6
    for column, dof in enumerate(free):
        delta = np.zeros(mpm.active_dof)
        delta[dof - offset] = step
        numerical[:, column] = (evaluate(base + delta)[0] - evaluate(base - delta)[0]) / (2 * step)
    assert np.linalg.norm(analytic - numerical) / np.linalg.norm(analytic) < 1e-5
    assert not engine.monolithic_hash.matrix_symmetric
    assert not mpm.has_lagged_material


@pytest.mark.parametrize("assembly", ["HashTriplet", "COO"])
def test_inexact_nonassociated_dp_keeps_force_and_lagged_friction_convergence(tmp_path, assembly):
    from time import perf_counter

    coupling = _implicit_system(tmp_path, assembly=assembly, mpm_material="DruckerPrager", dilation=0.0, inexact=True)
    engine = coupling.enginer
    engine.contact_model.friction_iterations = -1
    # Both policies must satisfy force balance and represented correction.
    engine.mpm.F0.from_numpy(np.array([np.diag([1.04, 0.96, 0.98])]))
    initial = engine.fem.state.position.to_numpy()
    results = []
    for inexact in [False, True]:
        engine.inexact_newton = inexact
        for repeat in range(5):
            engine.fem.state.position.from_numpy(initial)
            engine.mpm.grid_disp.fill(0.0)
            engine.fem.state.build_newmark_prediction(engine.dt, engine.fem.beta)
            engine.contact.begin_step(engine.fem.state.position, engine.mpm.grid_disp, engine.dt)
            start = perf_counter()
            converged, records, _ = engine.solve_lagged_friction_fixed_point(verbose=False)
            elapsed = perf_counter() - start
            assert converged and engine.last_friction_converged
            assert records[-1][-1]["convergence_reason"] == "force_and_correction"
            assert records[-1][-1]["correction_velocity"] <= engine.correction_velocity_tolerance
            assert records[-1][-1]["residual_norm"] <= records[-1][-1]["residual_tolerance"]
            if repeat:
                linear_iterations = sum(
                    r.get("linear_solve", {}).get("iterations", 0) for outer in records for r in outer
                )
                results.append((inexact, elapsed, linear_iterations, engine.mpm.grid_disp.to_numpy()))
    for result in results[1:]:
        np.testing.assert_allclose(results[0][3], result[3], rtol=1e-4, atol=1e-8)
    print("strict/inexact warm FEMPM:", [(r[0], r[1], r[2]) for r in results])


@pytest.mark.parametrize(
    "dilation,inexact,message", [(30.0, True, "requires nonassociated DP"), (0.0, "true", "must be boolean")]
)
def test_inexact_newton_configuration_validation(tmp_path, dilation, inexact, message):
    with pytest.raises(ValueError, match=message):
        _implicit_system(tmp_path, mpm_material="DruckerPrager", dilation=dilation, inexact=inexact)


def test_state_dependent_dp_lagged_friction_and_transactional_commit(tmp_path, monkeypatch):
    coupling = _implicit_system(tmp_path, mpm_material="StateDependentDruckerPrager", inexact=True)
    engine = coupling.enginer
    engine.contact_model.friction_iterations = -1
    engine.mpm.F0.from_numpy(np.array([np.diag([1.04, 0.96, 0.98])]))
    assert engine.nonassociated_newton and engine.linear_solver == "BiCGSTAB"
    assert engine.mpm.material.history_state_size == 14
    result = coupling.run(steps=1, verbose=False)
    assert result["converged"] and engine.last_friction_converged
    assert engine.mpm.material.equivalent_plastic_strain[0] > 0
    assert np.all(np.isfinite(engine.mpm.material.void_ratio.to_numpy()))
    assert not engine.mpm.has_lagged_material
    model = engine.mpm.material
    fields = [
        model.void_ratio,
        model.committed_jacobian,
        model.state_pressure,
        model.plastic_deformation_inverse,
        model.equivalent_plastic_strain,
        model.volumetric_plastic_strain,
        engine.mpm.F0,
    ]
    before = [field.to_numpy().copy() for field in fields]
    commit = engine.mpm.advent_particles

    def fail_after_commit(*args):
        commit(*args)
        raise RuntimeError("injected state-dependent post-commit failure")

    monkeypatch.setattr(engine.mpm, "advent_particles", fail_after_commit)
    with pytest.raises(RuntimeError, match="post-commit failure"):
        engine.substep(verbose=False)
    for field, expected in zip(fields, before):
        np.testing.assert_array_equal(field.to_numpy(), expected)


def test_implicit_ipc_finite_strain_von_mises_pipeline(tmp_path):
    coupling = _implicit_system(
        tmp_path,
        assembly="HashTriplet",
        mpm_material="VonMises",
    )
    result = coupling.run(steps=1, verbose=False)

    assert result["converged"]
    assert not coupling.enginer.mpm_uses_residual_merit
    record = result["history"][-1]
    assert record["material"] == "FiniteStrainVonMisesModel"
    assert record["contact"]["active_contacts"] >= 1
    assert np.linalg.det(coupling.mpm.enginer.F0.to_numpy()[0]) > 0.0


def test_implicit_ipc_finite_strain_modified_cam_clay_pipeline(tmp_path):
    coupling = _implicit_system(
        tmp_path,
        assembly="HashTriplet",
        mpm_material="ModifiedCamClay",
    )
    system = coupling.enginer.assemble_system(include_friction=True)

    assert not coupling.enginer.mpm_uses_residual_merit
    assert coupling.mpm.enginer.material.has_incremental_potential
    assert coupling.mpm.enginer.material.__class__.__name__ == ("FiniteStrainModifiedCamClayModel")
    assert coupling.mpm.enginer.material.history_state_size == 12
    assert system["active_mpm_dof"] > 0
    assert system["contact_count"] >= 1
    matrix = system["matrix"].to_scipy(system["active_nodes"])
    symmetry_error = (matrix - matrix.T).tocoo()
    maximum_error = float(np.max(np.abs(symmetry_error.data))) if symmetry_error.nnz else 0.0
    assert np.all(np.isfinite(matrix.data))
    assert maximum_error < 1.0e-8

    result = coupling.run(steps=1, verbose=False)
    assert result["converged"]
    assert result["history"][-1]["material_lagged_iterations"] >= 1
    assert result["history"][-1]["material_lagged_error"] <= (coupling.mpm.enginer.material_lagged_tolerance)


def test_residual_first_matrix_reuses_body_force_and_contact_query(tmp_path, monkeypatch):
    engine = _implicit_system(tmp_path, search="LinkedCell").enginer
    complete = engine.assemble_system(include_friction=True)
    expected_matrix = complete["matrix"].to_scipy(complete["active_nodes"]).toarray()
    expected_rhs = engine.rhs.to_numpy().copy()
    engine.assemble_system(include_friction=True, need_matrix=False)

    def repeated(*args, **kwargs):
        pytest.fail("unchanged trial must reuse the prepared force and query")

    monkeypatch.setattr(engine.contact, "prepare", repeated)
    monkeypatch.setattr(engine.fem, "_assemble_internal_device", repeated)
    monkeypatch.setattr(engine.mpm, "prepare_material_response", repeated)
    prepared = engine.assemble_system(include_friction=True, residual_prepared=True)
    np.testing.assert_allclose(
        prepared["matrix"].to_scipy(prepared["active_nodes"]).toarray(), expected_matrix, rtol=1e-12, atol=1e-9
    )
    np.testing.assert_allclose(engine.rhs.to_numpy(), expected_rhs, rtol=1e-12, atol=1e-9)
