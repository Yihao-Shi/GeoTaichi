import ast
import os

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.serial]

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))


def test_engine_kernel_function_names_are_unique():
    kernel_path = os.path.join(ROOT, "src", "mpm", "engines", "EngineKernel.py")
    with open(kernel_path, "r", encoding="utf-8") as handle:
        module = ast.parse(handle.read(), filename=kernel_path)

    names = {}
    duplicates = []
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in names:
                duplicates.append((node.name, names[node.name], node.lineno))
            else:
                names[node.name] = node.lineno

    assert duplicates == []


def _run_solid_case(
        configuration, solver_type, material_model, output_root):
    from geotaichi import MPM

    save_path = os.path.join(
        os.fspath(output_root),
        f"solid_dispatch_{configuration}_{solver_type}",
    )
    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.2, 0.2, 0.2],
        gravity=[0.0, 0.0, -9.8],
        alphaPIC=0.0,
        mapping="MUSL",
        shape_function="Linear",
        configuration=configuration,
        solver_type=solver_type,
        material_type="Solid",
        visualize=False,
        log=False,
    )
    if solver_type == "Implicit":
        mpm.set_implicit_solver_parameters(
            assemble_type="MatrixFree",
            max_iteration_number=2,
            displacement_tolerance=1.0e-3,
            residual_tolerance=1.0e-8,
        )
    mpm.set_solver(
        solver={
            "Timestep": 1.0e-5 if solver_type == "Explicit" else 1.0e-3,
            "SimulationTime": 1.0e-5 if solver_type == "Explicit" else 1.0e-3,
            "SaveInterval": 1.0e-5 if solver_type == "Explicit" else 1.0e-3,
            "SavePath": save_path,
        },
        log=False,
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 1,
            "max_particle_number": 64,
            "max_constraint_number": {
                "max_displacement_constraint": 64,
                "max_velocity_constraint": 64,
            },
        },
        log=False,
    )
    mpm.add_material(
        model=material_model,
        material={
            "MaterialID": 1,
            "Density": 1800.0,
            "YoungModulus": 2.0e5,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": [0.1, 0.1, 0.1]})
    mpm.add_region(
        region={
            "Name": "block",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.05, 0.05, 0.05],
            "BoundingBoxSize": [0.1, 0.1, 0.1],
        }
    )
    mpm.add_body(
        body={
            "Template": {
                "RegionName": "block",
                "nParticlesPerCell": 1,
                "BodyID": 0,
                "MaterialID": 1,
                "InitialVelocity": [0.0, 0.0, 0.0],
                "FixVelocity": ["Free", "Free", "Free"],
            }
        }
    )
    if solver_type == "Implicit":
        mpm.add_boundary_condition(
            boundary=[
                {
                    "BoundaryType": "DisplacementConstraint",
                    "Displacement": [0.0, 0.0, 0.0],
                    "StartPoint": [0.0, 0.0, 0.0],
                    "EndPoint": [0.2, 0.2, 0.0],
                }
            ]
        )
    else:
        mpm.add_boundary_condition()
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.run()

    particle_num = int(mpm.scene.particleNum[0])
    assert particle_num > 0
    assert np.isfinite(mpm.scene.particle.x.to_numpy()[:particle_num]).all()
    assert np.isfinite(mpm.scene.particle.v.to_numpy()[:particle_num]).all()


@pytest.mark.isolated_dimension(3)
def test_solid_engine_grid_velocity_dispatch_smoke(tmp_path):
    from geotaichi import init

    init(dim=3, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    _run_solid_case("ULMPM", "Explicit", "LinearElastic", tmp_path)
    _run_solid_case("TLMPM", "Explicit", "NeoHookean", tmp_path)
    _run_solid_case("ULMPM", "Implicit", "LinearElastic", tmp_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
