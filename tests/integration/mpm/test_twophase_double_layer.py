import os
import tempfile

import numpy as np
import pytest
import taichi as ti

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

from geotaichi import MPM, init

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.serial]


def run_case(save_path):
    save_path = os.fspath(save_path)
    init(dim=2, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)

    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.2, 0.2],
        background_damping=0.0,
        gravity=[0.0, -9.8],
        alphaPIC=1.0,
        mapping="USL",
        shape_function="Linear",
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
    )
    mpm.set_solver(
        solver={
            "Timestep": 1.0e-4,
            "SimulationTime": 1.0e-4,
            "SaveInterval": 1.0e-4,
            "SavePath": save_path,
        },
        log=False,
    )
    # The double-layer pressure equation is solved by the dedicated
    # multigrid path.  Relying on the generic PCG default makes the requested
    # solver configuration invalid before the first time step.
    mpm.set_semi_implicit_solver_parameters(
        {
            "pressure_solver": "MGPCG",
            "max_iteration_number": 50,
            "residual_tolerance": 1.0e-8,
            "multilevel": 2,
            "pre_and_post_smoothing": 2,
            "bottom_smoothing": 8,
        }
    )
    mpm.memory_allocate(
        memory={
            "max_material_number": 2,
            "max_particle_number": 128,
            "max_constraint_number": {"max_velocity_constraint": 128},
        },
        log=False,
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 1,
            "SolidDensity": 2650.0,
            "FluidDensity": 1000.0,
            "Porosity": 0.4,
            "FluidBulkModulus": 2.2e8,
            "Permeability": 1.0e-4,
            "FluidViscosity": 1.0e-3,
            "GrainDiameter": 1.0e-2,
            "YoungModulus": 1.0e6,
            "PoissonRatio": 0.3,
        },
    )
    mpm.add_material(
        model="LinearElastic",
        material={
            "MaterialID": 2,
            "SolidDensity": 2200.0,
            "FluidDensity": 980.0,
            "Porosity": 0.5,
            "FluidBulkModulus": 2.2e8,
            "Permeability": 5.0e-5,
            "FluidViscosity": 2.0e-3,
            "GrainDiameter": 8.0e-3,
            "YoungModulus": 8.0e5,
            "PoissonRatio": 0.28,
        },
    )
    mpm.add_element(element={"ElementType": "Q4N2D", "ElementSize": [0.1, 0.1]})
    mpm.add_region(
        region={
            "Name": "sample",
            "Type": "Rectangle2D",
            "BoundingBoxPoint": [0.05, 0.05],
            "BoundingBoxSize": [0.1, 0.1],
        }
    )
    mpm.add_body(
        body={
            "Template": [
                {
                    "RegionName": "sample",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Solid",
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                },
                {
                    "RegionName": "sample",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": "Fluid",
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                },
                {
                    "RegionName": "sample",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 2,
                    "Phase": "Solid",
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                },
                {
                    "RegionName": "sample",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 2,
                    "Phase": "Fluid",
                    "InitialVelocity": [0.0, 0.0],
                    "FixVelocity": ["Free", "Free"],
                },
            ]
        }
    )
    mpm.add_boundary_condition(
        boundary=[
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [None, 0.0],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [0.2, 0.0],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [0.0, 0.0],
                "EndPoint": [0.0, 0.2],
            },
            {
                "BoundaryType": "VelocityConstraint",
                "Velocity": [0.0, None],
                "StartPoint": [0.2, 0.0],
                "EndPoint": [0.2, 0.2],
            },
        ]
    )
    boundary_calls = []

    def update_mac_boundary(sims, scene):
        boundary_calls.append(float(sims.current_time))

    mpm.select_save_data(particle=True, grid=True, object=False)
    mpm.run(mac_boundary_function=update_mac_boundary)

    particle_num = int(mpm.scene.particleNum[0])
    position = mpm.scene.particle.x.to_numpy()[:particle_num]
    velocity = mpm.scene.particle.v.to_numpy()[:particle_num]
    pressure = mpm.scene.particle.pressure.to_numpy()[:particle_num]
    phase = mpm.scene.particle.phase.to_numpy()[:particle_num]
    material_id = mpm.scene.particle.materialID.to_numpy()[:particle_num]
    assert particle_num == 16
    assert set(np.unique(phase).tolist()) == {1, 2}
    assert set(np.unique(material_id).tolist()) == {1, 2}
    assert boundary_calls == [0.0]
    assert np.isfinite(position).all()
    assert np.isfinite(velocity).all()
    assert np.isfinite(pressure).all()
    assert np.any(np.abs(pressure[phase == 1]) > 0.0)
    assert np.any(np.abs(pressure[phase == 2]) > 0.0)
    saved = np.load(os.path.join(save_path, "particles", "MPMParticle000001.npz"))
    assert "particleID" in saved.files
    assert "phase" in saved.files
    saved_phase = saved["phase"]
    saved_pressure = saved["pressure"]
    assert np.any(np.abs(saved_pressure[saved_phase == 1]) > 0.0)
    assert np.any(np.abs(saved_pressure[saved_phase == 2]) > 0.0)
    mpm.postprocessing()
    assert os.path.exists(os.path.join(save_path, "vtks", "GraphicMPMSolidParticle000001.vtu"))
    assert os.path.exists(os.path.join(save_path, "vtks", "GraphicMPMFluidParticle000001.vtu"))
    assert not os.path.exists(os.path.join(save_path, "vtks", "GraphicMPMParticle000001.vtu"))
    saved_grid = np.load(os.path.join(save_path, "grids", "MPMGrid000001.npz"))
    assert "cell_type" in saved_grid.files
    assert "cell_pressure" in saved_grid.files
    assert "cell_fluid_sdf" in saved_grid.files
    assert saved_grid["cell_type"].shape[:2] == tuple(np.asarray(mpm.scene.element.cnum, dtype=np.int64))
    assert set(np.unique(saved_grid["cell_type"]).tolist()).issubset({0, 1, 2})
    grid_vtk = os.path.join(save_path, "vtks", "GraphicMPMGrid000001.vtu")
    assert os.path.exists(grid_vtk)
    with open(grid_vtk, "rb") as file:
        assert b"cell_type" in file.read()


def test_twophase_double_layer_one_step_and_output_round_trip(tmp_path):
    run_case(tmp_path / "double-layer")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="geotaichi-double-layer-") as temporary_directory:
        run_case(os.path.join(temporary_directory, "output"))
