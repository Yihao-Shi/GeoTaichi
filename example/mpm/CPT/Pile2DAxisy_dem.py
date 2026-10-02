from geotaichi import *

init(dim=2, device_memory_GB=7.0)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.6, 2.508]),
                      is_2DAxisy=True,
                      background_damping=0.05,
                      gravity=ti.Vector([0., -9.8]),
                      alphaPIC=0.05,
                      mapping="USF",  #USL
                      shape_function="CubicBSpline"
                      )

mpm.set_solver(solver={
                           "Timestep":                   1.0e-5,
                           "SimulationTime":             10,
                           "SaveInterval":               0.2,
                           "SavePath":                   '1_Pile2DAxisy_SDMC_dem'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    800000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     134474,
                                                               "max_particle_traction_constraint":   134474,
                                                          }
                            })

mpm.add_contact(contact_type="DEMContact", materialID=1, stiffness=[1.e5, 1.e5], friction=0.0)

              
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
                             "ElementSize":               ti.Vector([0.006, 0.006])
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([0.6, 1.5]),
                            "ydirection": ti.Vector([0., 1.])
                      },

                      {
                            "Name": "region3",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0.0, 1.497]),
                            "BoundingBoxSize": ti.Vector([0.6, 0.003]),
                            "ydirection": ti.Vector([0., 1.])
                      },
                      ])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     True,
                                                              "InternalStress":   ti.Vector([-75e3, -150e3, -75e3, 0., 0., 0.])
                                                         },
                                       "Traction":       [{"Pressure": ti.Vector([0, -150e3]),
                                                           "RegionName": "region3"}],
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                   }]
                   })

mpm.add_polygons(body={"Vertices": "pile.txt",
                             "InitialVelocity":    [0., -0.1]}
                         )

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

#mpm.postprocessing(read_path='1_Pile2DAxisy_SDMC_dem', write_background_grid=True)

