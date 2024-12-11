from geotaichi import *

init(debug=True)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([10, 11]),
                      dimension="2-Dimension",
                      is_2DAxisy=True,
                      background_damping=0.0,
                      gravity=ti.Vector([0., 0.]),
                      alphaPIC=0.1,
                      mapping="USL",
                      shape_function="GIMP",)

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.5,
                           "SaveInterval":               0.1,
                           "SavePath":                   '2Bossinesq_miu0.0_a0.1'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    800000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":    134474,
                                                               "max_particle_traction_constraint":   134474
                                                          }
                            })
                            
mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              1500.,
                               "YoungModulus":         1e7,
                               "PossionRatio":         0.30,
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.05, 0.05]),
                             "Contact":                    {}
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([10., 10.]),
                            "ydirection": ti.Vector([0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 9.975]),
                            "BoundingBoxSize": ti.Vector([1., 0.025]),
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
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([-0, -0, -0, 0., 0., 0.])
                                                         },
                                       "Traction":       [{"Pressure": ti.Vector([0, -1e6]),
                                                           "RegionName": "region2"}],
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   },]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [10., 0.],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 11.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [10., 0.],
                                             "EndPoint":       [10., 11.]
                                        },
                                    ])


mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing(read_path='2Bossinesq_miu0.0_a0.1', write_background_grid=True)
