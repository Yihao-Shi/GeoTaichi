from geotaichi import *

init(dim=2, device_memory_GB=2)

mpm = MPM()

mpm.set_configuration(domain=[6., 5.],
                      background_damping=0.00,
                      gravity=[0., -9.8],
                      alphaPIC=0.005,
                      shape_function="QuadBSpline",
                      solver_type="Implicit",
                      )

mpm.set_implicit_solver_parameters(quasi_static=False)

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             5,
                           "SaveInterval":               1e-1
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5,
                                "max_constraint_number":  {
                                                               "max_displacement_constraint": 83000
                                                          }
                            })

mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":      1,
                               "Density":         2500,
                               "YoungModulus":    8.6e5,
                               "PoissionRatio":   0.3,
                               "Friction":        19,
                               "Cohesion":        10,
                               "Dilation":        0
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               [0.025, 0.025]
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": [0., 0.],
                            "BoundingBoxSize": [2., 1.],
                            
                      })

mpm.add_body(body={
                       "Template": {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":[0, 0],
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   }
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "DisplacementConstraint",
                                             "Displacement":   [0., 0],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [6., 0.]
                                        },

                                        {
                                             "BoundaryType":   "DisplacementConstraint",
                                             "Displacement":   [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 5.]
                                        },
                                    ])

mpm.select_save_data()

mpm.run(gravity_field=lambda points: 1. - points[:,1])

mpm.postprocessing()
