import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch="gpu", debug=False, device_memory_GB=18)

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.2,1.0,0.08]),
                      scheme="LSDEM",
                      gravity=ti.Vector([0.,-9.8,0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.set_solver({
                "Timestep":         2e-5,
                "SimulationTime":   6,
                "SaveInterval":     0.3,
                "SavePath":         "OutputData"
               })

dem.memory_allocate(memory={
                            "max_material_number": 2,
                            "max_rigid_body_number": 85000,
                            "max_rigid_template_number": 1,
                            "levelset_grid_number": 52855,
                            "surface_node_number": 127,
                            "max_plane_number": 6,
                            "body_coordination_number":   64,
                            "wall_coordination_number":   4,
                            "verlet_distance_multiplier": [0.05, 0.05],
                            "point_coordination_number":  [6, 4], 
                            "compaction_ratio":           [0.32, 0.25, 0.05, 0.01]
                            }, log=True)                       

dem.add_attribute(materialID=0,
                  attribute={
                            "Density":            1150,
                            "ForceLocalDamping":  0.2,
                            "TorqueLocalDamping": 0.2
                            })
                            
dem.add_attribute(materialID=1,
                  attribute={
                            "Density":            26500,
                            "ForceLocalDamping":  0.,
                            "TorqueLocalDamping": 0.
                            })
                   
dem.add_template(template={
                                "Name":               "Template1",
                                "Object":             polysuperellipsoid(xrad1=5, yrad1=2.5, zrad1=2.5, xrad2=5, yrad2=2.5, zrad2=2.5, epsilon_e=1., epsilon_n=1.).grids(space=0.2, extent=2),
                                "SurfaceNodeNumber":  127,
                                "WriteFile":          True}) 
                                
'''dem.add_template(template={
                                "Name":               "Template1",
                                "Object":             polysuperellipsoid(xrad1=4, yrad1=2, zrad1=4, xrad2=4, yrad2=2, zrad2=4, epsilon_e=1., epsilon_n=1.).grids(space=0.2, extent=2),
                                "SurfaceNodeNumber":  127,
                                "WriteFile":          True}) '''

dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "Name": "Template1",
                               "BodyType": "RigidBody",
                               "File":'BoundingSphere.txt',
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": [0.,0.,0.],
                               "InitialAngularVelocity": [0.,0.,0.]
                               }}) 

dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            2e7,
                            "TangentialStiffness":        2e7,
                            "Friction":                   0.3,
                            "NormalViscousDamping":       0.2,
                            "TangentialViscousDamping":   0.2
                           })    
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            2e7,
                            "TangentialStiffness":        2e7,
                            "Friction":                   0.3,
                            "NormalViscousDamping":       0.2,
                            "TangentialViscousDamping":   0.2
                           }, dType="particle-wall")  
         
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   [0.1, 0.5, 0.],
                   "OuterNormal":  [0., 0., 1]
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   [0.1, 0.5, 0.08],
                   "OuterNormal":  [0., 0., -1]
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   [0.2, 0.5, 0.04],
                   "OuterNormal":  [-1, 0., 0.]
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   [0., 0.5, 0.04],
                   "OuterNormal":  [1, 0., 0.]
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   [0.1, 0., 0.04],
                   "OuterNormal":  [0., 1, 0.]
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   [0.1, 1.0, 0.04],
                   "OuterNormal":  [0., -1, 0.]
                  })
                  
dem.select_save_data(grid=True, bounding=True, surface=True)

dem.run()   
