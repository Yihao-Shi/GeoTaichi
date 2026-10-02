import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


load_path = 'SpherePacking.txt'

save_path = "DEM_Generation1"

from geotaichi import *

def box_wall(box_point=ti.Vector([3.35e-6, 3.35e-6, 3.35e-6]), box_size=ti.Vector([2 * 93.3e-6, 2 * 93.3e-6, 2 * 93.3e-6]), servo_stress=10.e6,
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

init(arch='gpu', log=True, debug=False, device_memory_GB=3, kernel_profiler=False)

dem = DEM()

dem.set_configuration(domain=ti.Vector([2 * 100.e-6, 2 * 100.e-6, 2 * 100.e-6]),
                      boundary=[None, None, None],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="HierarchicalLinkedCell", 
                      visualize=True)

dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 1568562,
                                "max_sphere_number": 1568562,
                                "max_clump_number": 0,
                                # "max_servo_wall_number": 6,
                                "max_facet_number": 12,
                                "hierarchical_level": 2,
                                "hierarchical_size": [1.3540578975e-05, 1.675102e-06],
                                "body_coordination_number":   [30,50],
                                "wall_coordination_number":   12,
                                "verlet_distance_multiplier": 0.15,
                                "wall_per_cell":              12, 
                                "compaction_ratio":           [0.15, 0.1]
                            })



dem.set_solver({
                "Timestep":         1e-10,
                "CFL":              1.0,
                "SimulationTime":   0.00003,
                "SaveInterval":     0.000001,
                "SavePath":         save_path
               })  

dem.add_attribute(materialID=0,
                    attribute={
                                "Density":            715800,
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
                                "EffectiveModulus":           1.4875e+11,
                                "NormalToShearRatio":         1.0,
                                "Friction":                   0.0,
                                "NormalViscousDamping":       0.00,
                                "TangentialViscousDamping":   0.00
                            })   
                            
dem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                "EffectiveModulus":           1.4875e+12,
                                "NormalToShearRatio":         1.0,
                                "Friction":                   0.0,
                                "NormalViscousDamping":       0.00,
                                "TangentialViscousDamping":   0.00
                            }) 

# dem.read_restart(file_number=6, file_path=save_path, 
#                    particle=True, sphere=True, wall=False, servo=False, ppcontact=True, pwcontact=True, is_continue=False)

dem.select_save_data(particle=True, sphere=True, particle_particle_contact = True, particle_wall_contact = True)

dem.run(calm = 500)
