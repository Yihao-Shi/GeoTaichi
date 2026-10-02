import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='gpu', log=False, debug=False)

lsdem = DEM()

lsdem.set_configuration(domain=ti.Vector([5.,5.,10.]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., -9.8]))

lsdem.memory_allocate(memory={
                                 "max_material_number": 2,
                                 "max_rigid_body_number": 6300,
                                 "levelset_grid_number": 26825,
                                 "surface_node_number": 127,
                                 "max_plane_number": 6,
                                 "body_coordination_number":   28,
                                 "wall_coordination_number":   3,
                                 "verlet_distance_multiplier": [0.1, 0.1],
                                 "point_coordination_number":  [4, 2], 
                                 "compaction_ratio":           [0.32, 0.1, 0.3, 0.2]
                             })  

lsdem.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   3.,
                "SaveInterval":     0.3,
                "SavePath":         'SandGenerate'
               })  

lsdem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.2,
                                "TorqueLocalDamping": 0.2
                            })
                                
lsdem.add_template(template={
                                "Name":               "sand",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sand.stl').grids(space=5, extent=3),
                                "WriteFile":          True}) 

lsdem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([5.,5.,10.])
                       })  

lsdem.add_body(body={
                            "BodyType": "RigidBody",
                            "GenerateType": "Generate",
                            "RegionName": "region1",
                            "TryNumber": 10000,
                            "Template":[{
                                             "Name": "sand",
                                             "MaxRadius": 0.1,
                                             "MinRadius": 0.1,
                                             "BodyNumber": 6300,
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                                             "BodyOrientation": "uniform"
                                        }]
                        })
                        
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([2.5, 2.5, 0.]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([2.5, 2.5, 10.]),
                   "OuterNormal":  ti.Vector([0., 0., -1.])
                  })
                  
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([5, 2.5, 2.5]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0., 2.5, 2.5]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([2.5, 0., 2.5]),
                   "OuterNormal":  ti.Vector([0., 1., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([2.5, 5, 2.5]),
                   "OuterNormal":  ti.Vector([0., -1., 0.])
                  })

lsdem.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")

                            
lsdem.add_property(materialID1=0,
                   materialID2=0,
                   property={
                                "NormalStiffness":            1e6,
                                "TangentialStiffness":        1e6,
                                "Friction":                   0.5,
                                "NormalViscousDamping":       0.05,
                                "TangentialViscousDamping":   0.05
                            })  
                            
'''lsdem.add_property(materialID1=0,
                   materialID2=0,
                   property={
                                "Stiffness":                  1e8,
                                "NormalCutOff":               0.005,
                                "TangentialCutOff":           0.005,
                                "Friction":                   0.5,
                                "NormalViscousDamping":       0.05,
                                "TangentialViscousDamping":   0.05
                            }, dType='particle-particle')  ''' 
                            
lsdem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                "NormalStiffness":            1e7,
                                "TangentialStiffness":        1e7,
                                "Friction":                   0.0,
                                "NormalViscousDamping":       0.05,
                                "TangentialViscousDamping":   0.05
                            })      

lsdem.select_save_data(surface=True)

lsdem.run()
