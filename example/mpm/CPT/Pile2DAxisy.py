from geotaichi import *

init(dim=2, device_memory_GB=7.0)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.6, 2.508]),
                      is_2DAxisy=True,
                      background_damping=0.2,
                      gravity=ti.Vector([0., -9.8]),
                      alphaPIC=0.2, 
                      mapping="USF", 
                      shape_function="CubicBSpline",
                      )

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             10,
                           "SaveInterval":               0.2,
                           "SavePath":                   '1_Pile2DAxisy_SDMC1'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    99500,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     134474,
                                                               "max_particle_traction_constraint":   134474,
                                                          }
                            })

mpm.add_contact(contact_type="MPMContact", friction=0.49)                            

mpm.add_material(model="StateDependentMohrCoulomb",
                 material={
                               "MaterialID":           1,
                               "Density":              1600.,
                               "YoungModulus":         60e6,
                               "PossionRatio":         0.30,
                               "e0":                   0.62,
                               "e_Tao":                0.90,
                               "lambda_c":             0.119,
                               "ksi":                  0.23,
                               "nd":                   1.70,
                               "nf":                   2.68,
                               "fai_c":                30.,
                               "Cohesion":             3000
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.006, 0.006]),
                             "Contact":   {
                                               "ContactDetection":                "MPMContact",
                                               "Friction":                        0.49,
                                          }
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([0.6, 1.5]),
                            "ydirection": ti.Vector([0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Cone2D",
                            "BoundingBoxPoint": ti.Vector([0., 1.5]),
                            "BoundingBoxSize": ti.Vector([0.018, 1.]),
                            "ydirection": ti.Vector([0., 1.])
                      },

                      {
                            "Name": "region3",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0.018, 1.497]),
                            "BoundingBoxSize": ti.Vector([0.582, 0.003]),
                            "ydirection": ti.Vector([0., 1.])
                      },
                      ])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     True,
                                                              "InternalStress":   ti.Vector([-75e3, -150e3, -75e3, 0., 0., 0.])
                                                         },
                                       "Traction":       [{"Pressure": ti.Vector([0, -150e3]),
                                                           "RegionName": "region3"}],
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "RigidBody":          True,
                                       "Density":            1600,
                                       "InitialVelocity":    [0., -0.1],
                                       "FixVelocity":        ["Fix", "Fix"]
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0.6, 0.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 2.508]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0.6, 0.],
                                             "EndPoint":       [0.6, 2.508]
                                        },
                                    ])


mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing(read_path='1_Pile2DAxisy_SDMC1', write_background_grid=True)

