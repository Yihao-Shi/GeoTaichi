import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32, debug=False, kernel_profiler = False)

from src.dem.mainDEM import DEM

dem = DEM()


dem.set_configuration(domain=ti.Vector([10.,10.,10.]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0.,0.,-9.8]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   1.,
                "SaveInterval":     5e-4
               })

dem.memory_allocate(memory={
                            "max_material_number": 1,
                            "max_particle_number": 4,
                            "max_sphere_number": 0,
                            "max_clump_number": 2,
                            "max_plane_number": 6,
                            "verlet_distance_multiplier": 0.2
                            }, log=True)

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0,0,0]),
                       "BoundingBoxSize": ti.Vector([7.,7.,7.]),
                       "zdirection": ti.Vector([0.,0.,1.])
                       })                            

dem.add_attribute(materialID=0,
                  attribute={
                            "Density":            2650,
                            "ForceLocalDamping":  0.05,
                            "TorqueLocalDamping": 0.
                            })
                           
dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 2,
                                 "Pebble": [{
                                             "Position": ti.Vector([-0.5, 0., 0.]),
                                             "Radius": 1.
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0., 0.]),
                                             "Radius": 1.
                                            }]
                                 })



dem.add_body(body={
                   "GenerateType": "Create",
                   "BodyType": "Clump",
                   "WriteFile": False,
                   "PoissionSampling": False,
                   "Template":{
                               "Name": "clump1",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "Radius": 0.15,
                               "BodyPoint": ti.Vector([5.5,5,1.5]),
                               "BodyOrientation": "constant",
                               "OrientationParameter": ti.Vector([0,0,1]),
                               "InitialVelocity": ti.Vector([-5.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,-50.,0.])}})  
                               
dem.add_body(body={
                   "GenerateType": "Create",
                   "BodyType": "Clump",
                   "WriteFile": False,
                   "PoissionSampling": False,
                   "Template":{
                               "Name": "clump1",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "Radius": 0.15,
                               "BodyPoint": ti.Vector([4.5,5,1.5]),
                               "BodyOrientation": "constant",
                               "OrientationParameter": ti.Vector([1,0,0]),
                               "InitialVelocity": ti.Vector([5.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,-50.,0.])}}) 
                
'''                         
dem.add_body(body={
                   "GenerateType": "Create",
                   "BodyType": "Clump",
                   "Template":{
                               "Name": "clump1",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"],
                               "BodyPoint": ti.Vector([2.5,2.5,0.17]),
                               "Radius": 0.15,
                               "BodyOrientation": "constant",
                               "OrientationParameter": ti.Vector([1., 0., 3.])}})
'''
                           
dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e6,
                            "TangentialStiffness":        1e6,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.2,
                            "TangentialViscousDamping":   0.0
                           })
                    
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   ti.Vector([5., 5., 0.]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   ti.Vector([5., 5., 10.]),
                   "OuterNormal":  ti.Vector([0., 0., -1.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   ti.Vector([10., 5., 5.]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   ti.Vector([0., 5., 5.]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   ti.Vector([5., 0., 5.]),
                   "OuterNormal":  ti.Vector([0., 1., 0.])
                  })
                  
dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   0,
                   "WallCenter":   ti.Vector([5., 10., 5.]),
                   "OuterNormal":  ti.Vector([0., -1., 0.])
                  })

dem.select_save_data()

dem.run()            

import numpy as np        
from third_party.pyevtk.hl import pointsToVTK
position = dem.scene.particle.x.to_numpy()[0:dem.scene.particleNum[None]]
posx, posy, posz = np.ascontiguousarray(position[:, 0]), \
                   np.ascontiguousarray(position[:, 1]), \
                   np.ascontiguousarray(position[:, 2])
bodyID = np.ascontiguousarray(dem.scene.particle.multisphereIndex.to_numpy()[0:dem.scene.particleNum[None]])
groupID = np.ascontiguousarray(dem.scene.particle.groupID.to_numpy()[0:dem.scene.particleNum[None]])
rad = np.ascontiguousarray(dem.scene.particle.rad.to_numpy()[0:dem.scene.particleNum[None]])
pointsToVTK(f'GraphicDEM', posx, posy, posz, data={'bodyID': bodyID, 'group': groupID, "rad": rad})
