from geotaichi import *

init(arch='gpu', log=False, debug=False, device_memory_GB=4)

lsdem = DEM()

lsdem.set_configuration(domain=ti.Vector([5.,5.,10.]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., -9.8]),
                        track_energy=True,
                        visualize=True)

lsdem.memory_allocate(memory={
                                 "max_material_number": 2,
                                 "max_rigid_body_number": 6300,
                                 "max_rigid_template_number": 7,
                                 "levelset_grid_number": 20493,
                                 "surface_node_number": 1422,
                                 "max_plane_number": 6,
                                 "body_coordination_number":   30,
                                 "wall_coordination_number":   3,
                                 "verlet_distance_multiplier": [0.1, 0.1],
                                 "point_coordination_number":  [4, 2], 
                                 "compaction_ratio":           [0.32, 0.1, 0.3, 0.2]
                             })  

lsdem.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   3,
                "SaveInterval":     0.3
               })  

lsdem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.25,
                                "TorqueLocalDamping": 0.25
                            })
                   
       
lsdem.add_template(template={
                                "Name":               "Template1",
                                "Object":              polyhedron(file='/assets/mesh/LSDEM/Pear.stl').grids(space=0.8, extent=1),
                                "WriteFile":          True}) 
                                
lsdem.add_template(template={
                                "Name":               "Template2",
                                "Object":              polyhedron(file='/assets/mesh/LSDEM/Banana.stl').grids(space=7, extent=1),
                                "WriteFile":          True})                          
                   
lsdem.add_template(template={
                                "Name":               "Template3",
                                "Object":             sphere(1.).grids(space=0.1, extent=1),
                                "SurfaceResolution":  10000, 
                                "WriteFile":          True}) 
                                
cylinder = capped_cylinder([0., 0., -1.], [0., 0., 1.], 0.25)
sdf1 = cylinder.orient([1., 0., 0.]) | cylinder.orient([0., 1., 0.]) | cylinder.orient([0., 0., 1.])                                
lsdem.add_template(template={
                                "Name":               "Template4",
                                "Object":             sdf1.grids(space=0.2, extent=1),
                                "SurfaceResolution":  15000, 
                                "WriteFile":          True}) 
                                
lsdem.add_template(template={
                                "Name":               "Template5",
                                "Object":             torus(1, 0.25).grids(space=0.1, extent=1),
                                "SurfaceResolution":  7000, 
                                "WriteFile":          True}) 
                                
lsdem.add_template(template={
                                "Name":               "Template6",
                                "Object":             polysuperellipsoid(xrad1=0.5, yrad1=0.25, zrad1=0.75, xrad2=0.25, yrad2=0.75, zrad2=0.5, epsilon_e=1.5, epsilon_n=1.5).grids(space=0.1, extent=1),
                                "SurfaceNodeNumber":  1002,
                                "WriteFile":          True}) 
                                
lsdem.add_template(template={
                                "Name":               "Template7",
                                "Object":             polysuperquadrics(xrad1=0.5, yrad1=2.5, zrad1=1.7, xrad2=1.0, yrad2=0.5, zrad2=0.5, epsilon_x=0.5, epsilon_y=1.5, epsilon_z=1.2).grids(space=0.25, extent=1),
                                "SurfaceNodeNumber":  1002,
                                "WriteFile":          True}) 

lsdem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([5.,5.,10.]),
                       "zdirection": ti.Vector([0.,0.,1.])
                       })  

lsdem.add_body(body={
                            "BodyType": "RigidBody",
                            "GenerateType": "Generate",
                            "RegionName": "region1",
                            "TryNumber": 10000,
                            "Template":[
                                        
                                        {
                                             "Name": "Template3",
                                             "BoundingRadius": 0.1,
                                             "BodyNumber": 900,
                                             "GroupID": 2,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                                             "BodyOrientation": "uniform"
                                        },
                                        {
                                             "Name": "Template4",
                                             "BoundingRadius": 0.1,
                                             "BodyNumber": 900,
                                             "GroupID": 2,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                                             "BodyOrientation": "uniform"
                                        },
                                        {
                                             "Name": "Template5",
                                             "BoundingRadius": 0.1,
                                             "BodyNumber": 900,
                                             "GroupID": 2,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                                             "BodyOrientation": "uniform"
                                        },
                                        {
                                             "Name": "Template6",
                                             "BoundingRadius": 0.1,
                                             "BodyNumber": 900,
                                             "GroupID": 2,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                                             "BodyOrientation": "uniform"
                                        },
                                        {
                                             "Name": "Template7",
                                             "BoundingRadius": 0.1,
                                             "BodyNumber": 900,
                                             "GroupID": 2,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                                             "BodyOrientation": "uniform"
                                        },
                                        {
                                             "Name": "Template1",
                                             "BoundingRadius": 0.1,
                                             "BodyNumber": 900,
                                             "GroupID": 2,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                                             "BodyOrientation": "uniform"
                                        },
                                        {
                                             "Name": "Template2",
                                             "BoundingRadius": 0.1,
                                             "BodyNumber": 900,
                                             "GroupID": 2,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                                             "BodyOrientation": "uniform"
                                        }
                                        
                                        ]
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
                                "NormalStiffness":            1e8,
                                "TangentialStiffness":        1e8,
                                "Friction":                   0.5,
                                "NormalViscousDamping":       0.05,
                                "TangentialViscousDamping":   0.05
                            })   
                            
lsdem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                "NormalStiffness":            1e9,
                                "TangentialStiffness":        1e9,
                                "Friction":                   0.5,
                                "NormalViscousDamping":       0.05,
                                "TangentialViscousDamping":   0.05
                            })      

lsdem.select_save_data(surface=True)

lsdem.run()
