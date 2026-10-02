import os
import sys

from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *


init(dim=3, arch="cpu", cpu_max_num_threads=4, device_memory_GB=2, debug=False, kernel_profiler=True)

mpm = MPM()

mpm.set_configuration(domain=[0.4, 0.2, 0.25],
                      dimension="3-Dimension",
                      background_damping=0.0,
                      alphaPIC=0.5,
                      mapping="USL",
                      shape_function="QuadBSpline",
                      gravity=[1.0, 0.0, -9.8],
                      material_type="Fluid",
                      velocity_projection="Affine",
                      solver_type="Implicit",
                      discretization="FDM",
                      solid_sdf_cut_cell=True,
                      solid_cut_cell_min_fraction=0.05)

mpm.set_implicit_solver_parameters(linear_solver="MGPCG",
                                   multilevel=3,
                                   pre_and_post_smoothing=2,
                                   bottom_smoothing=20)

mpm.set_solver({
    "Timestep": 2.0e-4,
    "SimulationTime": 2.0e-3,
    "SaveInterval": 1.0e-3,
    "SavePath": "fixed_sdf_coupling_3d"
})

mpm.memory_allocate(memory={
    "max_material_number": 1,
    "max_particle_number": 20000,
    "verlet_distance_multiplier": 1.0,
    "max_constraint_number": {}
})

mpm.add_material(model="Newtonian",
                 material={
                     "MaterialID": 1,
                     "Density": 1000.0,
                     "Modulus": 2.0e6,
                     "Viscosity": 1.0e-3,
                     "ElementLength": 0.025,
                     "cL": 1.5,
                     "cQ": 2.0,
                     "atmospheric_pressure": 0.0,
                     "SurfaceTension": 0.0
                 })

mpm.add_element(element={
    "ElementType": "Staggered",
    "ElementSize": [0.025, 0.025, 0.025],
    "GhostCell": 1
})

mpm.add_region(region=[{
    "Name": "water",
    "Type": "Rectangle",
    "BoundingBoxPoint": [0.025, 0.025, 0.025],
    "BoundingBoxSize": [0.18, 0.15, 0.16],
}])

mpm.add_body(body={
    "Template": [{
        "RegionName": "water",
        "nParticlesPerCell": 2,
        "BodyID": 0,
        "MaterialID": 1,
        "InitialVelocity": [0.0, 0.0, 0.0],
        "FixVelocity": ["Free", "Free", "Free"]
    }]
})

mpm.add_boundary_condition(boundary=[
    {
        "BoundaryType": "SolidCell",
        "Norm": [-1.0, 0.0, 0.0],
        "StartPoint": [0.0, 0.0, 0.0],
        "EndPoint": [0.0, 0.2, 0.25],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [1.0, 0.0, 0.0],
        "StartPoint": [0.4, 0.0, 0.0],
        "EndPoint": [0.4, 0.2, 0.25],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [0.0, -1.0, 0.0],
        "StartPoint": [0.0, 0.0, 0.0],
        "EndPoint": [0.4, 0.0, 0.25],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [0.0, 1.0, 0.0],
        "StartPoint": [0.0, 0.2, 0.0],
        "EndPoint": [0.4, 0.2, 0.25],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [0.0, 0.0, -1.0],
        "StartPoint": [0.0, 0.0, 0.0],
        "EndPoint": [0.4, 0.2, 0.0],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [0.0, 0.0, 1.0],
        "StartPoint": [0.0, 0.0, 0.25],
        "EndPoint": [0.4, 0.2, 0.25],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [0.0, 0.0, 0.0],
        "StartPoint": [0.22, 0.075, 0.025],
        "EndPoint": [0.27, 0.125, 0.17],
        "CellThickness": 1
    }
])

mpm.select_save_data(particle=True, grid=True)

mpm.run(gravity_field=True)

mpm.postprocessing()
