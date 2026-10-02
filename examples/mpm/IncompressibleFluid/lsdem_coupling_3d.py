import os
import sys

from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

os.environ.setdefault("GEOTAICHI_REAL_DTYPE", "float64")

arch = os.environ.get("GT_LSDEM_ARCH", "cpu")
linear_solver = os.environ.get("GT_LSDEM_LINEAR_SOLVER", "MGPCG")
simulation_time = float(os.environ.get("GT_LSDEM_SIMULATION_TIME", "2.0e-3"))
save_interval = float(os.environ.get("GT_LSDEM_SAVE_INTERVAL", "1.0e-3"))
save_path = os.environ.get("GT_LSDEM_SAVE_PATH", "lsdem_coupling_3d")

from geotaichi import DEMPM, init, polyhedron


init(dim=3,
     arch=arch,
     cpu_max_num_threads=4,
     default_fp="float64",
     default_ip="int32",
     offline_cache=True,
     debug=False,
     kernel_profiler=True)

dempm = DEMPM()

# 40 x 20 x 24 cells: every direction is divisible by 2**(multilevel - 1),
# so PCG and MGPCG use the same 0.01 m grid.
domain = [0.4, 0.2, 0.24]

dempm.set_configuration(domain=domain,
                        coupling_scheme="MPDEM",
                        particle_interaction=False,
                        wall_interaction=False,
                        gravity=[1.0, 0.0, -9.8],
                        visualize=False)

dempm.mpm.set_configuration(background_damping=0.0,
                            alphaPIC=0.5,
                            mapping="USL",
                            shape_function="QuadBSpline",
                            gravity=[1.0, 0.0, -9.8],
                            material_type="Fluid",
                            velocity_projection="PIC",
                            solver_type="Implicit",
                            discretization="FDM",
                            visualize=False)

dempm.mpm.set_implicit_solver_parameters(linear_solver=linear_solver,
                                         multilevel=3,
                                         pre_and_post_smoothing=2,
                                         bottom_smoothing=20,
                                         max_iteration_number=200,
                                         residual_tolerance=1.0e-10)

dempm.dem.set_configuration(boundary=["Destroy", "Destroy", "Destroy"],
                            gravity=[1.0, 0.0, -9.8],
                            engine="VelocityVerlet",
                            search="LinkedCell",
                            scheme="LSDEM",
                            visualize=False)

dempm.set_solver({
    "Timestep": 2.0e-4,
    "SimulationTime": simulation_time,
    "SaveInterval": save_interval,
    "SavePath": save_path
})

dempm.dem.memory_allocate(memory={
    "max_material_number": 2,
    "max_rigid_body_number": 1,
    "max_rigid_template_number": 1,
    "levelset_grid_number": 250000,
    "surface_node_number": 5000,
    "max_plane_number": 0,
    "body_coordination_number": 0,
    "wall_coordination_number": 0,
    "verlet_distance_multiplier": [0.15, 0.1],
    "point_coordination_number": [4, 2],
    "compaction_ratio": [0.15, 0.15]
})

dempm.mpm.memory_allocate(memory={
    "max_material_number": 1,
    "max_particle_number": 20000,
    "verlet_distance_multiplier": 1.0,
    "max_constraint_number": {}
})

dempm.memory_allocate(memory={
    "body_coordination_number": 1,
    "wall_coordination_number": 0,
    "compaction_ratio": [0.02, 0.1]
})

dempm.dem.add_attribute(materialID=0,
                        attribute={
                            "Density": 2500.0,
                            "ForceLocalDamping": 0.0,
                            "TorqueLocalDamping": 0.0
                        })

dempm.dem.add_template(template={
    "Name": "fixed_sphere",
    "Object": polyhedron(file=str(Path(ROOT) / "assets/mesh/LSDEM/sphere.stl")).grids(space=0.05, extent=9),
    "WriteFile": False
})

dempm.dem.create_body(body={
    "GenerateType": "Create",
    "BodyType": "RigidBody",
    "Template": [{
        "Name": "fixed_sphere",
        "GroupID": 0,
        "MaterialID": 0,
        "BodyPoint": [0.23, 0.10, 0.10],
        # D / dx = 10 is the starting resolution for IBM force refinement.
        "Radius": 0.05,
        "BodyOrientation": "constant",
        "InitialVelocity": [0.0, 0.0, 0.0],
        "FixMotion": ["Fix", "Fix", "Fix"]
    }]
})

dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                               particle_wall_contact_model=None)

dempm.mpm.add_material(model="Newtonian",
                       material={
                           "MaterialID": 1,
                           "Density": 1000.0,
                           "Modulus": 2.0e6,
                           "Viscosity": 1.0e-3,
                           "ElementLength": 0.01,
                           "cL": 1.5,
                           "cQ": 2.0,
                           "atmospheric_pressure": 0.0,
                           "SurfaceTension": 0.0
                       })

dempm.mpm.add_element(element={
    "ElementType": "Staggered",
    "ElementSize": [0.01, 0.01, 0.01],
    "GhostCell": 1
})

dempm.mpm.add_region(region=[{
    "Name": "water",
    "Type": "Rectangle",
    "BoundingBoxPoint": [0.025, 0.025, 0.025],
    "BoundingBoxSize": [0.23, 0.15, 0.16],
}])

dempm.mpm.add_body(body={
    "Template": [{
        "RegionName": "water",
        "nParticlesPerCell": 1,
        "BodyID": 0,
        "MaterialID": 1,
        "InitialVelocity": [0.0, 0.0, 0.0],
        "FixVelocity": ["Free", "Free", "Free"]
    }]
})

dempm.mpm.add_boundary_condition(boundary=[
    {
        "BoundaryType": "SolidCell",
        "Norm": [-1.0, 0.0, 0.0],
        "StartPoint": [0.0, 0.0, 0.0],
        "EndPoint": [0.0, domain[1], domain[2]],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [1.0, 0.0, 0.0],
        "StartPoint": [0.4, 0.0, 0.0],
        "EndPoint": [domain[0], domain[1], domain[2]],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [0.0, -1.0, 0.0],
        "StartPoint": [0.0, 0.0, 0.0],
        "EndPoint": [domain[0], 0.0, domain[2]],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [0.0, 1.0, 0.0],
        "StartPoint": [0.0, 0.2, 0.0],
        "EndPoint": [domain[0], domain[1], domain[2]],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [0.0, 0.0, -1.0],
        "StartPoint": [0.0, 0.0, 0.0],
        "EndPoint": [domain[0], domain[1], 0.0],
        "CellThickness": 1
    },
    {
        "BoundaryType": "SolidCell",
        "Norm": [0.0, 0.0, 1.0],
        "StartPoint": [0.0, 0.0, domain[2]],
        "EndPoint": domain,
        "CellThickness": 1
    }
])

dempm.mpm.select_save_data(particle=True, grid=True)
dempm.dem.select_save_data(surface=True, grid=True, bounding=True)
dempm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model=None,
                           particle_wall_contact_model=None)

dempm.run(mpm_gravity_field=True)

if os.environ.get("GT_LSDEM_POSTPROCESS", "1") != "0":
    dempm.postprocessing(scheme="LSDEM")
