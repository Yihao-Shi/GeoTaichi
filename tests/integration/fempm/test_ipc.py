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
        arch=ti.cpu,
        default_fp=ti.f64,
        cpu_max_num_threads=1,
        offline_cache=False,
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
    if mpm_material == "DruckerPrager":
        material_parameters.update(
            {
                "FrictionAngle": 30.0,
                "DilationAngle": dilation,
                "Cohesion": 10.0,
                "dpType": "Circumscribed",
            }
        )
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
    assert not coupling.enginer.mpm_uses_residual_merit
    record = result["history"][-1]
    assert record["material"] == "FiniteStrainDruckerPragerModel"
    assert record["contact"]["active_contacts"] >= 1
    assert record["friction_iterations"][0]
    assert np.linalg.det(coupling.mpm.enginer.F0.to_numpy()[0]) > 0.0


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
