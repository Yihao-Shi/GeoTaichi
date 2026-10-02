import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='gpu', log=False, debug=False)

lsdem = DEM()

lsdem.set_configuration(domain=ti.Vector([26.,26.,32.]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., -9.8]),
                        search="HierarchicalLinkedCell",
                        visualize=False)

# verlet_distance_multiplier      point_coordination_number          compaction_ratio                    time                 memory
#       [0.0, 0.0]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1523.6244761943817

#       [0.05, 0.0]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          1763.22470164299
#       [0.05, 0.1]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          590.2318165302277
#       [0.05, 0.2]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          524.6117603778839
#       [0.05, 0.3]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          591.6413898468018            3311
#       [0.05, 0.4]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]          652.2779026031494
#       [0.05, 0.5]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]          715.5190675258636

#       [0.1, 0.0]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]          2032.3381659984589              
#       [0.1, 0.1]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]          610.5664749145508        
#       [0.1, 0.2]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]          469.0668578147888           
#       [0.1, 0.3]                        [3, 2]                 [0.3, 0.1, 0.17, 0.015]          455.7775876522064            
#       [0.1, 0.4]                        [4, 2]                 [0.3, 0.1, 0.17, 0.015]          501.09127831459045    
#       [0.1, 0.5]                        [4, 2]                 [0.3, 0.1, 0.17, 0.015]          559.5415236949921      

#       [0.15, 0.0]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          2257.206956386566  
#       [0.15, 0.1]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          650.162957906723
#       [0.15, 0.2]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          502.2999668121338
#       [0.15, 0.3]                       [3, 2]                 [0.3, 0.1, 0.17, 0.015]          474.2800860404968
#       [0.15, 0.4]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]          464.1654374599457
#       [0.15, 0.5]                       [4, 2]                 [0.3, 0.1, 0.17, 0.015]          511.8898801803589

lsdem.memory_allocate(memory={
                                 "max_material_number": 2,
                                 "max_rigid_body_number": 200000,
                                 "levelset_grid_number": 64155,
                                 "surface_node_number": 127,
                                 "max_plane_number": 6,
                                 "body_coordination_number":   [25, 1],
                                 "wall_coordination_number":   [3, 6],
                                 "verlet_distance_multiplier": [0.1, 0.3],
                                 "point_coordination_number":  [3, 2], 
                                 "hierarchical_level":         2,
                                 "hierarchical_size":          [0.14155, 14.156],
                                 "compaction_ratio":           [0.32, 0.1, 0.17, 0.015],
                                 "wall_per_cell":              [3, 6]
                             })  

lsdem.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   3.,
                "SaveInterval":     0.15
               })  

lsdem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.2,
                                "TorqueLocalDamping": 0.2
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
                                             "BodyPoint": ti.Vector([12.5, 12.5, 11.5]),
                                             "Radius": 10,
                                             "BodyOrientation": [-45., -35.26438968, 0.],
                                             "FixMotion":       ["Fix", "Fix", "Fix"]
                                        }]
                        })
                        
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([13, 13, 0.]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([13, 13, 32.]),
                   "OuterNormal":  ti.Vector([0., 0., -1.])
                  })
                  
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([26, 13, 16]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0., 13, 16]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([13, 0., 16]),
                   "OuterNormal":  ti.Vector([0., 1., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([13, 26, 16]),
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
                            
lsdem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                "NormalStiffness":            1e7,
                                "TangentialStiffness":        1e7,
                                "Friction":                   0.0,
                                "NormalViscousDamping":       0.05,
                                "TangentialViscousDamping":   0.05
                            })      

lsdem.select_save_data(particle=False, surface=False)

lsdem.run()
