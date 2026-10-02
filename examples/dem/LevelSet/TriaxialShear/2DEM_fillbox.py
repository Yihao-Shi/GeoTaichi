import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


CASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Fill the triaxial box from a sphere-packing file.")
parser.add_argument(
    "--packing-file",
    default=str(CASE_DIR / "OutputData" / "DEM_Generation" / "BoundingSphere1.txt"),
)
parser.add_argument("--output-dir", default=str(CASE_DIR / "OutputData" / "DEM_Generation"))
arguments = parser.parse_args()
load_path = str(Path(arguments.packing_file).expanduser().resolve())
save_path = str(Path(arguments.output_dir).expanduser().resolve())


scale_fac0 = 1.001891414401244518
scale_fac1 = 1.247387418619341659
scale_fac2 = 1.554954183054590544

scale_fac = scale_fac1
from geotaichi import *

def box_wall(box_point=ti.Vector([0.0125, 0.0125, 0.0125]), box_size=ti.Vector([0.005*scale_fac, 0.005*scale_fac, 0.005*scale_fac]), servo_stress=1.e5,
             expand=1.2,
             limitVelocity = 0.1, 
             servo_fac=0.8):
    
    x0, y0, z0 = box_point
    dx, dy, dz = box_size
    ex = expand - 1.0
    dem.add_wall(body=[
        {
            "WallID": 0,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": ti.Vector([x0 - dx*ex, y0 - dy*ex, z0]),
                "vertice2": ti.Vector([x0 + dx*(1 + ex), y0 - dy*ex, z0]),
                "vertice3": ti.Vector([x0 + dx*(1 + ex), y0 + dy*(1 + ex), z0]),
                "vertice4": ti.Vector([x0 - dx*ex, y0 + dy*(1 + ex), z0])
            },
            "OuterNormal": ti.Vector([0., 0., 1.]),
        },

        {
            "WallID": 1,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": ti.Vector([x0 - dx*ex, y0 - dy*ex, z0 + dz]),
                "vertice2": ti.Vector([x0 + dx*(1 + ex), y0 - dy*ex, z0 + dz]),
                "vertice3": ti.Vector([x0 + dx*(1 + ex), y0 + dy*(1 + ex), z0 + dz]),
                "vertice4": ti.Vector([x0 - dx*ex, y0 + dy*(1 + ex), z0 + dz])
            },
            "OuterNormal": ti.Vector([0., 0., -1.]),  
        },

        {
            "WallID": 2,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": ti.Vector([x0, y0 - dy*ex, z0 - dz*ex]),
                "vertice2": ti.Vector([x0, y0 + dy*(1 + ex), z0 - dz*ex]),
                "vertice3": ti.Vector([x0, y0 + dy*(1 + ex), z0 + dz*(1 + ex)]),
                "vertice4": ti.Vector([x0, y0 - dy*ex, z0 + dz*(1 + ex)]),
            },
            "OuterNormal": ti.Vector([1., 0., 0.]),
        },

        {
            "WallID": 3,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": ti.Vector([x0 + dx, y0 - dy*ex, z0 - dz*ex]),
                "vertice2": ti.Vector([x0 + dx, y0 + dy*(1 + ex), z0 - dz*ex]),
                "vertice3": ti.Vector([x0 + dx, y0 + dy*(1 + ex), z0 + dz*(1 + ex)]),
                "vertice4": ti.Vector([x0 + dx, y0 - dy*ex, z0 + dz*(1 + ex)]),
            },
            "OuterNormal": ti.Vector([-1., 0., 0.]),  
        },

        {
            "WallID": 4,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": ti.Vector([x0 - dx*ex, y0, z0 - dz*ex]),
                "vertice2": ti.Vector([x0 + dx*(1 + ex), y0, z0 - dz*ex]),
                "vertice3": ti.Vector([x0 + dx*(1 + ex), y0, z0 + dz*(1 + ex)]),
                "vertice4": ti.Vector([x0 - dx*ex, y0, z0 + dz*(1 + ex)]),
            },
            "OuterNormal": ti.Vector([0., 1., 0.]),
        },

        {
            "WallID": 5,
            "WallType": "Facet",
            "WallShape": "Polygon",
            "MaterialID": 1,
            "WallVertice": {
                "vertice1": ti.Vector([x0 - dx*ex, y0 + dy, z0 - dz*ex]),
                "vertice2": ti.Vector([x0 + dx*(1 + ex), y0 + dy, z0 - dz*ex]),
                "vertice3": ti.Vector([x0 + dx*(1 + ex), y0 + dy, z0 + dz*(1 + ex)]),
                "vertice4": ti.Vector([x0 - dx*ex, y0 + dy, z0 + dz*(1 + ex)]),
            },
            "OuterNormal": ti.Vector([0., -1., 0.]),
        },
    ])

init(arch='gpu', log=True, debug=False, device_memory_GB=8, kernel_profiler=False)

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.05, 0.05, 0.05]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell", 
                      visualize=True)

dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 13310,
                                "max_sphere_number": 13310,
                                "max_clump_number": 0,
                                # "max_servo_wall_number": 6,
                                "max_facet_number": 12,
                                "body_coordination_number":   25,
                                "wall_coordination_number":   12,
                                "verlet_distance_multiplier": 0.1,
                                "wall_per_cell":              12, 
                                "compaction_ratio":           [0.6, 0.1]
                            })



dem.set_solver({
                "Timestep":         2.5e-7,
                "CFL":              0.5,
                "SimulationTime":   0.1,
                "SaveInterval":     0.02,
                "SavePath":         save_path
               })  

dem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.7,
                                "TorqueLocalDamping": 0.7
                            })

dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": load_path,
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }})

box_wall()

dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")
   
dem.add_property(materialID1=0,
                   materialID2=0,
                   property={
                                "EffectiveModulus":           1.e7,
                                "NormalToShearRatio":         1.2,
                                "Friction":                   0.0,
                                "NormalViscousDamping":       0.00,
                                "TangentialViscousDamping":   0.00
                            })   
                            
dem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                "EffectiveModulus":           1.e8,
                                "NormalToShearRatio":         1.0,
                                "Friction":                   0.0,
                                "NormalViscousDamping":       0.00,
                                "TangentialViscousDamping":   0.00
                            }) 
                                                                                                                                                       

dem.select_save_data(particle=True, surface=False, particle_particle_contact=True)

dem.run(calm=100)
