try:
    from geotaichi import *
except:
    import os
    import sys
    current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(current_file_path)
    from geotaichi import *
init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.55, 0.2, 0.11]), 
                      background_damping=0.002, 
                      alphaPIC=0.005, 
                      mapping="USF", 
                      stabilize=None,
                      shape_function="GIMP",
                      gauss_number=0)

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.6,
                           "SaveInterval":               0.01
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   134638,
                                                               "max_friction_constraint":   134638
                                                          }
                            })

mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":           1,
                               "Density":              2650.,
                               "YoungModulus":         8.4e5,
                               "PossionRatio":        0.3,
                               "Cohesion":             0.,
                               "Friction":             19.8,
                               "Dilation":             0.,
                               "Tensile":              0.
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.0025, 0.0025, 0.0025]),
                             "Contact":    {
                                                "ContactDetection":                False
                                           }
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.005, 0.005, 0.005]),
                            "BoundingBoxSize": ti.Vector([0.2, 0.05, 0.1]),
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
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., -0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0.55, 0.2, 0.005]
                                        },

                                        {    
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0., 0, 0],
                                             "EndPoint":       [0.005, 0.2, 0.11]
                                        },

                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0.55, 0.005, 0.11],
                                             "Velocity":       [None, 0., None]
                                        },

                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "StartPoint":     [0., 0.055, 0.],
                                             "EndPoint":       [0.55, 0.0575, 0.11],
                                             "Velocity":       [None, 0., None]
                                        }
                                    ])

mpm.select_save_data()

mpm.run()

mpm.postprocessing()
