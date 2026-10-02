import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(debug=True, device_memory_GB=20)

dem = DEM()

dem.set_configuration(domain=[0.00036,0.00036,0.014],
                      boundary=["Period", "Period", None],
                      gravity=[0.,0.,0],
                      engine="SymplecticEuler",
                      search="HierarchicalLinkedCell",
                      search_direction="Up")

dem.set_solver({
                "Timestep":         1.7e-10,
                "SimulationTime":   5.1e-5,
                "SaveInterval":     5.1e-6,
                "SavePath":         "Generation"
               })

dem.memory_allocate(memory={
                            "max_material_number": 2,
                            "max_particle_number": 14000000,
                            "max_sphere_number": 14000000,
                            "max_clump_number": 0,
                            "max_facet_number": 4,
                            "max_servo_wall_number": 1,
                            "verlet_distance_multiplier": 0.1,
                            "body_coordination_number": [26, 24],
                            "wall_coordination_number": 2,
                            "hierarchical_level": 2,
                            "hierarchical_size": [5e-06, 7e-05],
                            "compaction_ratio": [0.2, 0.05],
                            "wall_per_cell": [2, 2]
                            }, log=True)                       

dem.add_attribute(materialID=0,
                  attribute={
                            "Density":            7158,
                            "ForceLocalDamping":  0.2,
                            "TorqueLocalDamping": 0.2
                            })
                            
dem.add_attribute(materialID=1,
                  attribute={
                            "Density":            26500,
                            "ForceLocalDamping":  0.,
                            "TorqueLocalDamping": 0.
                            })

dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "File":'SpherePacking.txt',
                               "InitialVelocity": [0.,0.,0.],
                               "InitialAngularVelocity": [0.,0.,0.],
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"],
                               #"ParticleNumber": 12999985
                               }}) 
                               
dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                         particle_wall_contact_model="Linear Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "ShearModulus":               5.95e10,
                            "Poisson":                    0.25,
                            "StaticFriction":             0.4,
                            "DynamicFriction":            0.42,
                            "Restitution":                0.5
                           },
                dType="particle-particle")
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            4.25e5,
                            "TangentialStiffness":        4.25e5,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           },
                dType="particle-wall")
               
dem.add_wall(body=[{
                   "WallID":      0,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0.0, 0.0, 0.00],
                                    "vertice2": [0.00036, 0.0, 0.00],
                                    "vertice3": [0.00036, 0.00036, 0.00],
                                    "vertice4": [0.0, 0.00036, 0.00]
                                   },
                   "OuterNormal": [0., 0., 1.]
                  },
                  
                  {
                   "WallID":      1,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": [0.0, 0.0, 0.014],
                                    "vertice2": [0.00036, 0.0, 0.014],
                                    "vertice3": [0.00036, 0.00036, 0.014],
                                    "vertice4": [0.0, 0.00036, 0.014]
                                   },
                   "OuterNormal": [0., 0., -1.],
                   "ControlType":  "Force",
                   "TargetStress": 2.45e5,
                   "Gain":        1000.,
                   "LimitVelocity": 0.25
                  }])

dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

dem.run()     
