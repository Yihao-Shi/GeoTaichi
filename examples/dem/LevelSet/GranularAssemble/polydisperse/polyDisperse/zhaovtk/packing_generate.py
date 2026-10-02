import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='gpu', log=False, debug=False, device_memory_GB=3.9)

lsdem = DEM()

scale=0.1
lsdem.set_configuration(domain=scale*ti.Vector([33.12,33.12,40.12]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., -9.8]),
                        search="HierarchicalLinkedCell",
                        visualize=True)

# verlet_distance_multiplier      point_coordination_number          compaction_ratio                    time                 memory
#       [0.0, 0.0]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]          2273.0660967826843

#       [0.05, 0.0]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          2501.6396946907043
#       [0.05, 0.1]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1434.0126900672913
#       [0.05, 0.2]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1558.7477815151215
#       [0.05, 0.3]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1720.1744408607483            3311
#       [0.05, 0.4]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]          1882.6809649467468
#       [0.05, 0.5]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]          2033.631413936615

#       [0.1, 0.0]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]          2808.7535343170166           
#       [0.1, 0.1]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1702.834220647812        
#       [0.1, 0.2]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1115.5908889770508           
#       [0.1, 0.3]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1227.6707904338837            
#       [0.1, 0.4]                        [4, 2]                 [0.3, 0.1, 0.17, 0.015]          1366.1192960739136   
#       [0.1, 0.5]                        [4, 2]                 [0.3, 0.1, 0.17, 0.015]          1509.3931305408478      

#       [0.15, 0.0]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          3158.558618783951  
#       [0.15, 0.1]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1901.444799900055
#       [0.15, 0.2]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1510.8833665847778
#       [0.15, 0.3]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1128.8836278915405
#       [0.15, 0.4]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]          1273.6373896598816
#       [0.15, 0.5]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]          1377.5707716941833

lsdem.memory_allocate(memory={
                                 "max_material_number": 2,
                                 "max_rigid_body_number": 196001,
                                 "levelset_grid_number": 64155,
                                 "surface_node_number": 127,
                                 "max_plane_number": 6,
                                 "body_coordination_number":   [25, 1],
                                 "wall_coordination_number":   [3, 6],
                                 "verlet_distance_multiplier": [0.1, 0.2],
                                 "point_coordination_number":  [3, 2], 
                                 "hierarchical_level":         2,
                                 "hierarchical_size":          [scale*0.14156, scale*14.156],
                                 "compaction_ratio":           [0.32, 0.1, 0.17, 0.015],
                                 "wall_per_cell":              [3, 6]
                             })  

lsdem.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   3.,
                "SaveInterval":     1.
               })  

lsdem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.5,
                                "TorqueLocalDamping": 0.5
                            })
                   
lsdem.add_template(template={
                                "Name":               "Template1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sand.stl').grids(space=5, extent=4),
                                "WriteFile":          True}) 

lsdem.add_body_from_file(body={
                            "FileType":  "TXT",
                            "Template":[{
                                             "Name": "Template1",
                                             "BodyType": "RigidBody",
                                             "File":'BoundingSphere.txt',
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.])
                                        }]
                        })
                        
lsdem.create_body(body={
                            "BodyType": "RigidBody",
                            "Template":[{
                                             "Name": "Template1",
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0., 0., 0.]),
                                             "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                                             "BodyPoint": scale*ti.Vector([17, 17, 11]),
                                             "Radius": scale*10,
                                             "BodyOrientation": [135, 35.264, 45],
                                             "FixMotion":       ["Fix", "Fix", "Fix"]
                                        }]
                        })
                        
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   scale*ti.Vector([16.56, 16.56, 0.]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   scale*ti.Vector([16.56, 16.56, 40.12]),
                   "OuterNormal":  ti.Vector([0., 0., -1.])
                  })
                  
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   scale*ti.Vector([33.12, 16.56, 20.06]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   scale*ti.Vector([0., 16.56, 20.06]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   scale*ti.Vector([16.56, 0., 20.06]),
                   "OuterNormal":  ti.Vector([0., 1., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   scale*ti.Vector([16.56, 33.12, 20.06]),
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
                                "NormalViscousDamping":       0.1,
                                "TangentialViscousDamping":   0.1
                            })   
                            
lsdem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                "NormalStiffness":            1e7,
                                "TangentialStiffness":        1e7,
                                "Friction":                   0.5,
                                "NormalViscousDamping":       0.1,
                                "TangentialViscousDamping":   0.1
                            })      

lsdem.select_save_data(particle=True, surface=True)

lsdem.run()
