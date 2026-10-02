import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.9, 0.6, 0.3]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 78369,
                                "max_sphere_number": 78369,
                                "max_clump_number": 0,
                                "max_servo_wall_number": 1,
                                "max_facet_number": 24,
                                "body_coordination_number":   28,
                                "wall_coordination_number":   12,
                                "verlet_distance_multiplier": 0.1,
                                "wall_per_cell":              12
                            })   

dem.set_solver({
                "Timestep":         1e-5,
                "SimulationTime":   0.2,
                "SaveInterval":     0.02,
                "SavePath":         "Generation"
               })               

dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2580,
                                "ForceLocalDamping":  0.7,
                                "TorqueLocalDamping": 0.7
                            })
                            
dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "File": "SpherePacking.txt",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"]
                               }}) 
                          
dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            8e4,
                            "TangentialStiffness":        5.4e4,
                            "Friction":                   0.5,
                            "RollingFriction":            0.5, 
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            2e5,
                            "TangentialStiffness":        1.3e5,
                            "Friction":                   0.0,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
                           
dem.add_wall(body=[{
                   "WallID":      0,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.2, 0.1, 0.005]),
                                    "vertice2": ti.Vector([0.7, 0.1, 0.005]),
                                    "vertice3": ti.Vector([0.7, 0.5, 0.005]),
                                    "vertice4": ti.Vector([0.2, 0.5, 0.005])
                                   },
                   "OuterNormal": ti.Vector([0., 0., 1.]),
                   "ControlType":  "Force",
                   "TargetStress": 5.e4,
                   "Alpha":        0.5,
                   "LimitVelocity": 0.025
                  },
                  
                  {
                   "WallID":      1,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.2, 0.1, 0.295]),
                                    "vertice2": ti.Vector([0.7, 0.1, 0.295]),
                                    "vertice3": ti.Vector([0.7, 0.5, 0.295]),
                                    "vertice4": ti.Vector([0.2, 0.5, 0.295])
                                   },
                   "OuterNormal": ti.Vector([0., 0., -1.])
                  },
                  
                  {
                   "WallID":      2,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.3, 0.15, 0.15]),
                                    "vertice2": ti.Vector([0.6, 0.15, 0.15]),
                                    "vertice3": ti.Vector([0.6, 0.15, 0.295]),
                                    "vertice4": ti.Vector([0.3, 0.15, 0.295])
                                   },
                   "OuterNormal": ti.Vector([0., 1., 0.])
                  },
                  
                  {
                   "WallID":      3,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.3, 0.15, 0.005]),
                                    "vertice2": ti.Vector([0.6, 0.15, 0.005]),
                                    "vertice3": ti.Vector([0.6, 0.15, 0.15]),
                                    "vertice4": ti.Vector([0.3, 0.15, 0.15])
                                   },
                   "OuterNormal": ti.Vector([0., 1., 0.])
                  },
                  
                  {
                   "WallID":      4,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.3, 0.45, 0.15]),
                                    "vertice2": ti.Vector([0.6, 0.45, 0.15]),
                                    "vertice3": ti.Vector([0.6, 0.45, 0.295]),
                                    "vertice4": ti.Vector([0.3, 0.45, 0.295])
                                   },
                   "OuterNormal": ti.Vector([0., -1., 0.])
                  },
                  
                  {
                   "WallID":      5,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.3, 0.45, 0.005]),
                                    "vertice2": ti.Vector([0.6, 0.45, 0.005]),
                                    "vertice3": ti.Vector([0.6, 0.45, 0.15]),
                                    "vertice4": ti.Vector([0.3, 0.45, 0.15])
                                   },
                   "OuterNormal": ti.Vector([0., -1., 0.])
                  },
                  
                  {
                   "WallID":      6,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.3, 0.15, 0.15]),
                                    "vertice2": ti.Vector([0.3, 0.15, 0.295]),
                                    "vertice3": ti.Vector([0.3, 0.45, 0.295]),
                                    "vertice4": ti.Vector([0.3, 0.45, 0.15])
                                   },
                   "OuterNormal": ti.Vector([1., 0., 0.])
                  },
                  
                  {
                   "WallID":      7,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.3, 0.15, 0.005]),
                                    "vertice2": ti.Vector([0.3, 0.15, 0.15]),
                                    "vertice3": ti.Vector([0.3, 0.45, 0.15]),
                                    "vertice4": ti.Vector([0.3, 0.45, 0.005])
                                   },
                   "OuterNormal": ti.Vector([1., 0., 0.])
                  },
                  
                  {
                   "WallID":      8,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.6, 0.15, 0.15]),
                                    "vertice2": ti.Vector([0.6, 0.15, 0.295]),
                                    "vertice3": ti.Vector([0.6, 0.45, 0.295]),
                                    "vertice4": ti.Vector([0.6, 0.45, 0.15])
                                   },
                   "OuterNormal": ti.Vector([-1., 0., 0.])
                  },
                  
                  {
                   "WallID":      9,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.6, 0.15, 0.005]),
                                    "vertice2": ti.Vector([0.6, 0.15, 0.15]),
                                    "vertice3": ti.Vector([0.6, 0.45, 0.15]),
                                    "vertice4": ti.Vector([0.6, 0.45, 0.005])
                                   },
                   "OuterNormal": ti.Vector([-1., 0., 0.])
                  },
                  
                  {
                   "WallID":      10,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.6, 0.1, 0.15]),
                                    "vertice2": ti.Vector([0.9, 0.1, 0.15]),
                                    "vertice3": ti.Vector([0.9, 0.5, 0.15]),
                                    "vertice4": ti.Vector([0.6, 0.5, 0.15])
                                   },
                   "OuterNormal": ti.Vector([0., 0., 1.])
                  },
                  
                  {
                   "WallID":      11,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.0, 0.1, 0.15]),
                                    "vertice2": ti.Vector([0.3, 0.1, 0.15]),
                                    "vertice3": ti.Vector([0.3, 0.5, 0.15]),
                                    "vertice4": ti.Vector([0.0, 0.5, 0.15])
                                   },
                   "OuterNormal": ti.Vector([0., 0., -1.])
                  }])
            
dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

dem.run(calm=100)
    
dem.modify_parameters(SimulationTime=0.4, SaveInterval=0.02)

dem.run()

dem.postprocessing(read_path="Generation", write_path="Generation/vtks")
