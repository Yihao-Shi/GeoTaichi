import numpy as np
import pytest

ti = pytest.importorskip("taichi")

from src.fem.generator import FEMMesh
from src.fem.mainFEM import FEM
from src.fempm.mainFEMPM import FEMPM
from src.mpm.mainMPM import MPM

pytestmark = [pytest.mark.integration, pytest.mark.fempm, pytest.mark.coupling]


@pytest.fixture(autouse=True)
def taichi_cpu_runtime():
    ti.reset()
    ti.init(arch=ti.cpu, default_fp=ti.f64, cpu_max_num_threads=1, offline_cache=False)
    yield
    ti.reset()


@pytest.mark.parametrize(
    "mpm_material,inexact", [("NeoHookean", False), ("DruckerPrager", False), ("DruckerPrager", True)]
)
def test_implicit_ipc_accepts_cloth_surface(tmp_path, mpm_material, inexact):
    fem = FEM(log=False)
    fem.set_configuration(dimension=3, solver_type="Implicit")
    fem.add_mesh(
        FEMMesh(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
            np.asarray([[0, 1, 2]], dtype=np.int32),
            "TRI3",
        )
    )
    fem.add_material(
        "ClothARAP",
        stretch_stiffness=1.0e4,
        compression_stiffness=1.0e4,
        density=1000.0,
        thickness=0.05,
        bending_stiffness=0.0,
        bending_model="None",
    )
    fem.add_boundary_condition({"type": "Dirichlet", "nodes": [0, 1, 2], "components": "all", "value": 0.0})

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
        np.asarray([[0.25, 0.25, 0.02]], dtype=np.float64),
        volume=1.0e-3,
        init_v=[0.02, 0.0, -0.02],
        name="cloth_probe",
        grid_size=0.1,
        xmin=[0.0, 0.0, 0.0],
        xmax=[1.0, 1.0, 1.0],
        boundary_ids=np.asarray([0], dtype=np.int32),
        surface_measure=1.0e-2,
    )
    mpm.add_body(body)
    material = dict(model=mpm_material, young_modulus=1.0e4, poisson_ratio=0.3, density=1000.0)
    if mpm_material == "DruckerPrager":
        material.update(cohesion=1.0, friction_angle=30.0, dilation_angle=0.0)
    mpm.add_material(**material)
    mpm.add_element({"ElementSize": 0.1, "ShapeFunction": "Linear"})

    coupling = FEMPM(fem, mpm, log=False)
    coupling.set_configuration(domain=[1.0, 1.0, 1.0], gravity=[0.0, 0.0, 0.0], search="BVH", log=False)
    coupling.set_solver(
        {
            "Timestep": 1.0e-3,
            "SimulationTime": 1.0e-3,
            "SaveInterval": 1.0,
            "SavePath": str(tmp_path),
            "assemble_type": "HashTriplet",
            "linear_solver": "PCG",
            "inexact_newton": inexact,
            "max_iterations": 8,
            "residual_tolerance": 1.0e-7,
            "linear_solver_tolerance": 1.0e-7,
            "scale": 0.25,
            "visualize": False,
        },
        log=False,
    )
    coupling.add_surface()
    coupling.memory_allocate(
        {
            "max_particle_number": 1,
            "max_contact_pairs": 64,
            "max_point_triangle_pairs": 64,
            "max_point_edge_pairs": 64,
        },
        log=False,
    )
    coupling.choose_contact_model(
        "BarrierIPC",
        dhat=0.05,
        dmin=0.0,
        kappa=100.0,
        friction_coefficient=0.2,
        epsv=1.0e-3,
        friction_iterations=-1,
    )
    coupling.add_ipc_property(0, 0, friction_coefficient=0.2)
    coupling.add_essentials()
    system = coupling.enginer.assemble_system(include_friction=True)
    assert system["contact_count"] >= 1
    result = coupling.run(steps=1, verbose=False)
    assert result["converged"]
    assert coupling.enginer.last_friction_converged
    assert coupling.enginer.last_friction_iterations >= 1
    assert coupling.enginer.contact.friction_count >= 1
    if mpm_material == "DruckerPrager":
        assert coupling.enginer.linear_solver == "BiCGSTAB"
        assert coupling.enginer.mpm_uses_residual_merit
        assert not coupling.enginer.mpm.has_lagged_material
        assert not coupling.enginer.monolithic_hash.matrix_symmetric
        assert not coupling.enginer.fem.cloth_assembler.project_pd
        assert not coupling.enginer.fem.cloth_assembler.project_bending_pd
        if inexact:
            terminal = result["history"][-1]["friction_iterations"][-1][-1]
            assert terminal["convergence_reason"] == "force_and_correction"
            assert terminal["residual_norm"] <= terminal["residual_tolerance"]
            assert terminal["correction_velocity"] <= coupling.enginer.correction_velocity_tolerance
