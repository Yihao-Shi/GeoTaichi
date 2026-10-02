from geotaichi import *

init(dim=2, device_memory_GB=7.0)

contact_type = 0   # 0 for MPMContact, 1 for GeoContact, 2 for DEMContact

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([3.6, 9.6]),
                      is_2DAxisy=True,
                      background_damping=0.2,
                      gravity=ti.Vector([0., -9.8]),
                      alphaPIC=0.2, 
                      mapping="USF", 
                      shape_function="QuadBSpline",
                      #stabilize="B-Bar Method",
                      stress_integration="SubStepping",
                      velocity_projection="Taylor"
                      )

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             1,
                           "SaveInterval":               0.02,
                           "SavePath":                   '1_Pile2DAxisy_MCC'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    800000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     134474,
                                                               "max_particle_traction_constraint":   134474,
                                                          }
                            })
                            
if contact_type == 0: mpm.add_contact(contact_type="MPMContact", friction=0.5) 
elif contact_type == 1: mpm.add_contact(contact_type="GeoContact", friction=0.5, penalty=[5./6., 2.])    
elif contact_type == 2: mpm.add_contact(contact_type="DEMContact", materialID=1, stiffness=[5.e4, 5.e4], friction=0.5) 
else: raise RuntimeError("Input correct contact type!")                       

mpm.add_material(model="ModifiedCamClay",
                 material={
                               "MaterialID":                    1,
                               "Density":                       2800,
                               "PoissonRatio":                  0.3,
                               "StressRatio":                   0.984,
                               "lambda":                        0.25,
                               "kappa":                         0.05,
                               "void_ratio_ref":                2.04,
                               "OverConsolidationRatio":        2.,
                               "ThreeInvariants":               True
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               [0.04, 0.04]})

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([3.6, 6.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Cone2D",
                            "BoundingBoxPoint": ti.Vector([0., 6]),
                            "BoundingBoxSize": ti.Vector([0.2, 3.5]),
                            
                      },

                      {
                            "Name": "region3",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0.2, 5.98]),
                            "BoundingBoxSize": ti.Vector([3.4, 0.02]),
                            
                      },
                      ])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-50000, -50000, -50000, 0., 0., 0.])
                                                         },
                                       "Traction":       [{"Pressure": ti.Vector([0, -50000]),
                                                           "RegionName": "region3"}],
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   }]
                   })
                   
if contact_type == 2: 
    mpm.add_polygons(body={"Vertices": "pileMCC.txt", "InitialVelocity":    [0., -3.5]})
else:
    mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":          True,
                                       "Density":            1600,
                                       "InitialVelocity":    [0., -3.5],
                                       "FixVelocity":        ["Fix", "Fix"]
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [3.6, 0.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 9.5]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [3.6, 0.],
                                             "EndPoint":       [3.6, 9.5]
                                        },
                                    ])


mpm.select_save_data(grid=True)

mpm.run(gravity_field=True)

if contact_type == 2: 
    mpm.postprocessing(read_path='1_Pile2DAxisy_MCC', write_background_grid=False, end_file=51)
else:
    mpm.postprocessing(read_path='1_Pile2DAxisy_MCC', write_background_grid=True, end_file=51)

