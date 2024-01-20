from src import *

init()

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.05, 0.05, 0.027]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 28482,
                                "max_sphere_number": 28482,
                                "max_clump_number": 0,
                                "max_servo_wall_number": 6,
                                "max_facet_number": 12,
                                "body_coordination_number":   16,
                                "wall_coordination_number":   12,
                                "verlet_distance_multiplier": 0.1,
                                "wall_per_cell":              12
                            })   

dem.set_solver({
                "Timestep":         5e-7,
                "SimulationTime":   0.1,
                "SaveInterval":     0.01,
                "SavePath":         "Medium/Generation"
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
                            "NormalStiffness":            4.25e4,
                            "TangentialStiffness":        4.25e4,
                            "Friction":                   0.2,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            4.25e4,
                            "TangentialStiffness":        4.25e4,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })
                           
dem.add_wall(body=[{
                   "WallID":      0,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.0, 0.0, 0.001]),
                                    "vertice2": ti.Vector([0.05, 0.0, 0.001]),
                                    "vertice3": ti.Vector([0.05, 0.05, 0.001]),
                                    "vertice4": ti.Vector([0.0, 0.05, 0.001])
                                   },
                   "OuterNormal": ti.Vector([0., 0., 1.]),
                   "ControlType":  "Force",
                   "TargetStress": 2.e5,
                   "Alpha":        0.5,
                   "LimitVelocity": 0.025
                  },
                  
                  {
                   "WallID":      1,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.0, 0.0, 0.026]),
                                    "vertice2": ti.Vector([0.05, 0.0, 0.026]),
                                    "vertice3": ti.Vector([0.05, 0.05, 0.026]),
                                    "vertice4": ti.Vector([0.0, 0.05, 0.026])
                                   },
                   "OuterNormal": ti.Vector([0., 0., -1.]),
                   "ControlType":  "Force",
                   "TargetStress": 2.e5,
                   "Alpha":        0.5,
                   "LimitVelocity": 0.025
                  },
                  
                  {
                   "WallID":      2,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.0125, 0., 0.0]),
                                    "vertice2": ti.Vector([0.0125, 0.05, 0.]),
                                    "vertice3": ti.Vector([0.0125, 0.05, 0.027]),
                                    "vertice4": ti.Vector([0.0125, 0., 0.027])
                                   },
                   "OuterNormal": ti.Vector([1., 0., 0.]),
                   "ControlType":  "Force",
                   "TargetStress": 2.e5,
                   "Alpha":        0.5,
                   "LimitVelocity": 0.025
                  },
                  
                  {
                   "WallID":      3,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0.0375, 0., 0.0]),
                                    "vertice2": ti.Vector([0.0375, 0.05, 0.]),
                                    "vertice3": ti.Vector([0.0375, 0.05, 0.027]),
                                    "vertice4": ti.Vector([0.0375, 0., 0.027])
                                   },
                   "OuterNormal": ti.Vector([-1., 0., 0.]),
                   "ControlType":  "Force",
                   "TargetStress": 2.e5,
                   "Alpha":        0.5,
                   "LimitVelocity": 0.025
                  },
                  
                  {
                   "WallID":      4,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0., 0.0125, 0.]),
                                    "vertice2": ti.Vector([0.05, 0.0125, 0.]),
                                    "vertice3": ti.Vector([0.05, 0.0125, 0.027]),
                                    "vertice4": ti.Vector([0., 0.0125, 0.027])
                                   },
                   "OuterNormal": ti.Vector([0., 1., 0.]),
                   "ControlType":  "Force",
                   "TargetStress": 2.e5,
                   "Alpha":        0.5,
                   "LimitVelocity": 0.025
                  },
                  
                  {
                   "WallID":      5,
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   1,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([0., 0.0375, 0.]),
                                    "vertice2": ti.Vector([0.05, 0.0375, 0.]),
                                    "vertice3": ti.Vector([0.05, 0.0375, 0.027]),
                                    "vertice4": ti.Vector([0., 0.0375, 0.027])
                                   },
                   "OuterNormal": ti.Vector([0., -1., 0.]),
                   "ControlType":  "Force",
                   "TargetStress": 2.e5,
                   "Alpha":        0.5,
                   "LimitVelocity": 0.025
                  }])
            
dem.select_save_data(sphere=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

dem.run(calm=100)
    
dem.modify_parameters(SimulationTime=0.2, SaveInterval=0.01)

dem.run()

dem.postprocessing(read_path="Medium/Generation", write_path="Medium/Generation/vtks")
