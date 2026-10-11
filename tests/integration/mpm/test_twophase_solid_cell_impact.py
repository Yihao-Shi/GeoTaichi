"""A small saturated block must survive landing on a static SolidCell floor."""

import numpy as np
import pytest

from geotaichi import MPM, init


@pytest.mark.slow
@pytest.mark.serial
@pytest.mark.isolated_dimension(3)
def test_saturated_block_lands_without_porosity_or_volume_collapse(tmp_path):
    init(dim=3, arch="cpu", cpu_max_num_threads=2, offline_cache=False, log=False)
    mpm = MPM(log=False)
    mpm.set_configuration(
        domain=[0.16] * 3,
        gravity=[0, 0, -9.81],
        background_damping=0.0,
        alphaPIC=1.0,
        shape_function="QuadBSpline",
        material_type="TwoPhaseDoubleLayer",
        solver_type="SemiImplicit",
        velocity_projection="Affine",
        delayed_fluid_advection=True,
        particle_shifting=True,
        visualize=False,
    )
    mpm.set_solver({"Timestep": 2e-5, "SimulationTime": 0.04, "SaveInterval": 0.02, "SavePath": str(tmp_path)})
    mpm.set_semi_implicit_solver_parameters(
        {"pressure_solver": "MGPCG", "max_iteration_number": 100, "residual_tolerance": 1e-7, "multilevel": 2}
    )
    mpm.memory_allocate({"max_material_number": 1, "max_particle_number": 256})
    mpm.add_material(
        model="DruckerPrager",
        material={
            "MaterialID": 1,
            "SolidDensity": 7850.0,
            "FluidDensity": 1000.0,
            "Porosity": 0.4,
            "MaximumPorosity": 0.64,
            "FluidBulkModulus": 2.2e9,
            "Permeability": 1e-4,
            "YoungModulus": 1e8,
            "PoissonRatio": 0.25,
            "Cohesion": 0.0,
            "Friction": 26.0,
            "Dilation": 0.0,
        },
    )
    mpm.add_element({"ElementType": "R8N3D", "ElementSize": [0.02] * 3})
    mpm.add_region(
        {
            "Name": "block",
            "Type": "Rectangle",
            "BoundingBoxPoint": [0.06, 0.06, 0.02],
            "BoundingBoxSize": [0.04, 0.04, 0.02],
        }
    )
    mpm.add_body(
        {
            "Template": [
                {
                    "RegionName": "block",
                    "nParticlesPerCell": 2,
                    "BodyID": 0,
                    "MaterialID": 1,
                    "Phase": phase,
                    "InitialVelocity": [0, 0, -0.3],
                    "FixVelocity": ["Free"] * 3,
                }
                for phase in ("Solid", "Fluid")
            ]
        }
    )
    mpm.add_boundary_condition(
        {
            "BoundaryType": "SolidCell",
            "StartPoint": [0, 0, 0.02],
            "EndPoint": [0.16, 0.16, 0.02],
            "Norm": [0, 0, -1],
            "CellThickness": 1,
        }
    )
    mpm.select_save_data(particle=True, grid=False, object=False)
    mpm.run()
    for file_name in sorted((tmp_path / "particles").glob("MPMParticle*.npz")):
        with np.load(file_name) as data:
            solid = (data["active"] > 0) & (data["phase"] == 1)
            for name in ("position", "volume", "porosity", "stress", "solid_velocity"):
                assert np.isfinite(data[name][solid]).all(), (file_name, name)
            assert np.all(data["volume"][solid] > 0)
            assert np.all((data["porosity"][solid] > 0) & (data["porosity"][solid] < 1))
            assert data["position"][solid, 2].min() >= 0.02 - 1e-7
