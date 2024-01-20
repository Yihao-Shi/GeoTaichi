from src import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([1.7, 0.25, 2.5]), 
                      background_damping=0.00, 
                      alphaPIC=0.002, 
                      mapping="USL", 
                      shape_function="GIMP",
                      material_type="Fluid")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             2.5,
                           "SaveInterval":               0.02
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    8.64e5,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   950000
                                                          }
                            })

mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "Modulus":              3.6e5,
                               "Viscosity":            1e-3
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.01, 0.01, 0.01])
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.05, 0.05, 0.05]),
                            "BoundingBoxSize": ti.Vector([0.6, 0.15, 0.6]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      })

mpm.add_body(body={
                       "Template": {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     True,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.]),
                                                              "Traction":         {}
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [0., 0., -1.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [1.7, 0.25, 0.05]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [-1., 0., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0.05, 0.25, 2.5]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [1., 0., 0.],
                                             "StartPoint":     [1.66, 0., 0.],
                                             "EndPoint":       [1.7, 0.25, 2.5]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [0., -1., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [1.7, 0.05, 2.5]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [0., 1., 0.],
                                             "StartPoint":     [0., 0.2, 0.],
                                             "EndPoint":       [1.7, 0.25, 2.5]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [0., 0., 1],
                                             "StartPoint":     [0., 0., 2.4],
                                             "EndPoint":       [1.7, 0.25, 2.5]
                                        }
                                    ])

mpm.select_save_data()

mpm.run()

mpm.postprocessing()
