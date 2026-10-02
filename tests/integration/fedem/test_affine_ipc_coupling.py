from pathlib import Path

import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from geotaichi import DEM, polyhedron
from src.fem.generator import FEMMesh
from src.fem.mainFEM import FEM
from src.fedem.mainFEDEM import FEDEM

pytestmark = [
    pytest.mark.integration,
    pytest.mark.fedem,
    pytest.mark.ipc,
    pytest.mark.contact,
    pytest.mark.cpu,
    pytest.mark.serial,
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


@pytest.mark.parametrize("contact_model", ["BarrierIPC", "SemiIPC"])
def test_affine_ipc_assembles_mixed_tet_soft_particle_control_system(tmp_path, contact_model):
    dem = DEM(log=False)
    dem.set_configuration(
        domain=[2.0, 2.0, 2.0],
        scheme="AffineBody",
        search="LinkedCell",
        gravity=[0.0, 0.0, 0.0],
        visualize=False,
        log=False,
    )
    dem.set_affine_body_parameters(
        assemble_type="HashTriplet",
        young_modulus=2.0e4,
        local_damping=0.0,
        contact_damping_stiffness=0.0,
        hessian_shift=0.0,
        friction_mode="lagged",
        friction_iterations=1,
    )
    dem.memory_allocate(
        {
            "max_material_number": 1,
            "max_affine_body_number": 1,
            "surface_node_number": 16,
            "body_coordination_number": 8,
            "wall_coordination_number": 1,
            "compaction_ratio": [1.0, 1.0],
        },
        log=False,
    )
    dem.add_attribute(materialID=0, attribute={"Density": 1000.0})
    mesh_file = Path(__file__).parents[2] / "data" / "affine_octahedron.obj"
    dem.add_template(
        {
            "Name": "oct",
            "TemplateType": "AffineBody",
            "Object": polyhedron(file=str(mesh_file)),
        }
    )
    dem.create_body(
        {
            "BodyType": "AffineBody",
            "Template": [
                {
                    "Name": "oct",
                    "GroupID": 0,
                    "MaterialID": 0,
                    "BodyPoint": [0.35, 0.35, 0.2],
                    "ScaleFactor": 0.2,
                    "InitialVelocity": [0.0, 0.0, 0.0],
                }
            ],
        }
    )

    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    fem.add_mesh(
        FEMMesh(
            np.array(
                [
                    [0.0, 0.0, 0.5],
                    [1.0, 0.0, 0.5],
                    [0.0, 1.0, 0.5],
                    [0.0, 0.0, 1.5],
                ]
            ),
            np.array([[0, 1, 2, 3]], dtype=np.int32),
            "TET4",
        )
    )
    fem.add_material(
        "StVK",
        density=1000.0,
        young_modulus=1.0e4,
        poisson_ratio=0.3,
    )
    fem.add_boundary_condition(
        {
            "type": "Dirichlet",
            "nodes": [0, 1, 2, 3],
            "components": "all",
            "value": 0.0,
        }
    )

    coupling = FEDEM(dem, fem, log=False)
    coupling.set_configuration(domain=[2.0, 2.0, 2.0], search="LinkedCell", log=False)
    coupling.set_solver(
        {
            "Timestep": 1.0e-3,
            "SimulationTime": 1.0e-3,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
            "assemble_type": "HashTriplet",
            "linear_solver": "PCG",
        },
        log=False,
    )
    coupling.add_surface()
    coupling.memory_allocate({"max_contact_pairs": 64, "contact_coordination_number": 16})
    coupling.choose_contact_model(
        contact_model,
        dhat=0.25,
        kappa=2.0e4,
        friction_coefficient=0.0,
        epsv=1.0e-3,
        friction_iterations=2,
    )
    coupling.add_ipc_property(
        AffineBody=0,
        FEMbody=0,
        property={"friction_coefficient": 0.4, "epsv": 1.0e-3},
    )
    coupling.add_essentials()

    engine = coupling.enginer
    engine.affine.device_begin_step(engine.dt)
    engine.contact.begin_step(
        engine.fem.state.position,
        engine.fem.state.old_position,
        engine.dt,
    )
    assert engine.contact.diagnostics()["friction_contacts"] > 0
    pt_count = int(engine.contact.friction_pt_count)
    ee_count = int(engine.contact.friction_ee_count)
    pt_coefficient = engine.contact.friction_pt_coefficient.to_numpy()[:pt_count].copy()
    ee_coefficient = engine.contact.friction_ee_coefficient.to_numpy()[:ee_count].copy()
    engine.contact.backup_lagged_friction_for_adjoint_device()
    engine.contact.friction_pt_coefficient.fill(0.0)
    engine.contact.friction_ee_coefficient.fill(0.0)
    engine.contact.restore_lagged_friction_for_adjoint_device()
    np.testing.assert_allclose(engine.contact.friction_pt_coefficient.to_numpy()[:pt_count], pt_coefficient)
    np.testing.assert_allclose(engine.contact.friction_ee_coefficient.to_numpy()[:ee_count], ee_coefficient)
    original_controls = engine.affine.y.to_numpy()
    displaced_controls = original_controls.copy()
    displaced_controls[:, 0] += 5.0e-4
    engine.affine.y.from_numpy(displaced_controls)
    pt_count, ee_count = engine.contact.prepare(engine.fem.state.position)
    engine.contact_hash.reset_system()
    engine.rhs.fill(0.0)
    engine.contact.assemble(
        pt_count,
        ee_count,
        engine.contact_hash,
        engine.rhs,
        True,
    )
    diagnostics = coupling.enginer.contact.diagnostics()
    assert diagnostics["model"] == contact_model
    assert diagnostics["pt_candidates"] + diagnostics["ee_candidates"] > 0
    assert diagnostics["active_contacts"] > 0
    assert diagnostics["friction_contacts"] > 0
    assert diagnostics["friction_energy"] > 0.0
    assert np.isfinite(diagnostics["energy"])
    assert int(engine.contact_hash.overflow[0]) == 0
    assert coupling.enginer.node_count == (coupling.enginer.affine_controls + coupling.enginer.fem_nodes)
    # Coupled source matrices intentionally retain only their diagonal/raw
    # streams.  Exercise the same append-then-reduce path as production rather
    # than finalizing the raw-only contact source directly.
    contact_destination = engine.monolithic_hash
    contact_destination.reset_system()
    contact_destination.append_raw_from(
        engine.contact_hash,
        active_nodes=engine.node_count,
        block_offset=0,
    )
    contact_destination.canonicalize_full_symmetric_input()
    contact_destination.finalize_taichi_assembly()
    contact_matrix = contact_destination.to_scipy(engine.node_count).toarray()
    np.testing.assert_allclose(contact_matrix, contact_matrix.T, rtol=1.0e-10, atol=1.0e-10)
    for block in range(engine.node_count):
        diagonal = contact_matrix[3 * block : 3 * block + 3, 3 * block : 3 * block + 3]
        eigenvalues = np.linalg.eigvalsh(0.5 * (diagonal + diagonal.T))
        assert eigenvalues.min() >= -1.0e-9 * max(1.0, np.abs(eigenvalues).max())
    np.testing.assert_allclose(
        engine.rhs.to_numpy().reshape(-1, 3).sum(axis=0),
        np.zeros(3),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    engine.affine.y.from_numpy(original_controls)
    engine.affine._reconstruct_vertices()

    result = coupling.run(steps=1, verbose=False)
    assert result["converged"]
    assert result["step"] == 1
    assert result["time"] == pytest.approx(1.0e-3)
    assert engine.last_linear_solve["backend"] == "taichi_hash_pcg"
    assert len(result["history"][-1]["friction_iterations"]) == 2
    assert (tmp_path / "vtks" / "FEM000000.vtu").is_file()
    assert (tmp_path / "vtks" / "GraphicAffineBody000000.vtu").is_file()
