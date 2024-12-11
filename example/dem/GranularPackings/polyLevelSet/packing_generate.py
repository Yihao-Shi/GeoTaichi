from geotaichi import *

init(arch='gpu', log=False, debug=False, device_memory_GB=3.8)

lsdem = DEM()

lsdem.set_configuration(domain=ti.Vector([26.,26.,32.]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., -9.8]),
                        search="HierarchicalLinkedCell",
                        visualize=True)

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
                                 "hierarchical_size":          [0.14155, 14.155],
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
                                "Object":              polyhedron(file='/assets/mesh/LSDEM/sand.stl').grids(space=5, extent=4),
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
                                             "BodyOrientation": [1, 1, -1],
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

lsdem.select_save_data(particle=True, surface=True)

lsdem.run()
