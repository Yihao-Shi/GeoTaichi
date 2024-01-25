from geotaichi import *

init()

dem = DEM()

dem.set_configuration(domain=ti.Vector([7.,7.,10.]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0.,0.,-9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell")

dem.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   2,
                "SaveInterval":     0.02
               })

dem.memory_allocate(memory={
                            "max_material_number": 2,
                            "max_particle_number": 45978,
                            "max_sphere_number": 0,
                            "max_clump_number": 22989,
                            "max_plane_number": 6,
                            "body_coordination_number":15,
                            "verlet_distance_multiplier": 0.1
                            }, log=True)
                         

dem.add_attribute(materialID=0,
                  attribute={
                            "Density":            2650,
                            "ForceLocalDamping":  0.3,
                            "TorqueLocalDamping": 0.3
                            })
                           
dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Clump",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.])
                               }})

                            
dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
     
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e5,
                            "TangentialStiffness":        1e5,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.05,
                            "TangentialViscousDamping":   0.05
                           })           
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            5e5,
                            "TangentialStiffness":        5e5,
                            "Friction":                   0.0,
                            "NormalViscousDamping":       0.05,
                            "TangentialViscousDamping":   0.05
                           })                 
                    
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([3.5, 3.5, 0.]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([3.5, 3.5, 10.]),
                   "OuterNormal":  ti.Vector([0., 0., -1.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([7., 3.5, 3.5]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0., 3.5, 3.5]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([3.5, 0., 3.5]),
                   "OuterNormal":  ti.Vector([0., 1., 0.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([3.5, 7., 3.5]),
                   "OuterNormal":  ti.Vector([0., -1., 0.])
                  })
                  
dem.select_save_data(clump=True)
                  
dem.run()            
