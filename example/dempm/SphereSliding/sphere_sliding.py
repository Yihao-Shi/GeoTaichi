from geotaichi import *

init()

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([15., 6., 5.]),
                        coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=False)

dempm.dem.set_configuration(
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([6.929646456, 0., -6.929646456]),
                      engine="SymplecticEuler",
                      search="LinkedCell")
                      
       
dempm.mpm.set_configuration(
                      background_damping=0., 
                      alphaPIC=0.005, 
                      mapping="USF", 
                      shape_function="Linear",
                      gravity=ti.Vector([6.929646456, 0., -6.929646456]))

dempm.set_solver({
                      "Timestep":         6.324555320336759e-05,
                      "SimulationTime":   3.,
                      "SaveInterval":     0.1,
                      "SavePath":         "OutputData/P2PContact/mu=0.1"
                 })

dempm.memory_allocate(memory={
                                  "max_material_number":         1,
                                  "body_coordination_number":    8,
                                  "wall_coordination_number":    6,
                             })
                             
dempm.dem.memory_allocate(memory={
                                "max_material_number": 1,
                                "max_particle_number": 1,
                                "max_sphere_number": 1,
                                "max_clump_number": 0,
                                "max_facet_number": 2,
                                "verlet_distance_multiplier": 0.
                            })            

dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           33552,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   61820
                                                          }
                            })


dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.7,
                                "TorqueLocalDamping": 0.
                            })

dempm.dem.create_body(body={
                   "BodyType": "Sphere",
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0., 0., 0.]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "BodyPoint": ti.Vector([2.0, 3.0, 2.6-0.0031506888]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"],
                               "Radius": 1.6,
                               "BodyOrientation": "uniform"}]})

dempm.dem.add_wall(body={
                   "WallID":      0, 
                   "WallType":    "Facet",
                   "WallShape":   "Polygon",
                   "MaterialID":   0,
                   "WallVertice":  {
                                    "vertice1": ti.Vector([3.6, 0., 0.]),
                                    "vertice2": ti.Vector([3.6, 6., 0.]),
                                    "vertice3": ti.Vector([3.6, 6., 5.]),
                                    "vertice4": ti.Vector([3.6, 0., 5.])
                                   },
                   "OuterNormal": ti.Vector([-1., 0., 0.])
                  })
                          
dempm.dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
dempm.dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e8,
                            "TangentialStiffness":        1e8,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.5,
                            "TangentialViscousDamping":   0.
                           })
 
                  
dempm.dem.select_save_data()

dempm.mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              2650.,
                               "YoungModulus":         1e5,
                               "PoissionRatio":        0.3
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.5, 0.5, 0.5])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0., 0.]),
                            "BoundingBoxSize": ti.Vector([15., 6., 1.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  1,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })

dempm.mpm.add_boundary_condition(boundary=[{
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [0., 0., -1],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [10., 6., 0.]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [-1, 0., 0],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [0., 2., 0.]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [1, 0., 0],
                                        "StartPoint":     [6., 0., 0.],
                                        "EndPoint":       [6., 2., 0.]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [0, -1., 0],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [6., 0., 0.]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [0, 1., 0],
                                        "StartPoint":     [0., 2., 0.],
                                        "EndPoint":       [6., 2., 0.]
                                    }])

dempm.mpm.select_save_data()


dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            1e6,
                                 "TangentialStiffness":        1e6,
                                 "Friction":                   0.,
                                 "NormalViscousDamping":       0.02,
                                 "TangentialViscousDamping":   0.
                            })

dempm.run()

dem.scene.wall[0].active = 0
dem.scene.wall[1].active = 0
dempm.contactor.physpp.surfaceProps[1].ndratio = 0.
dempm.contactor.physpp.surfaceProps[1].mu = 0.1
dem.scene.material[0].fdamp = 0.

dempm.modify_parameters(SimulationTime=5)

dempm.run()

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
