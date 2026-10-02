from geotaichi import *

init(device_memory_GB=3)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([6.4, 1.4, 8.4]), 
                      background_damping=0.0, 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.00, 
                      mapping="USL", 
                      shape_function="GIMP",
                      stabilize=None)

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             2.4,
                           "SaveInterval":               0.06
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    514000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     12680,
                                                               "max_reflection_constraint":   178540
                                                          }
                            })
                   
mpm.add_contact(contact_type="MPMContact", friction=0.)
                            
mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "YoungModulus":         1e6,
                               "PossionRatio":        0.49,
                               "Cohesion":             100,
                               "Friction":             0.,
                               "Dilation":             0.
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.1, 0.1, 0.1])
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.2, 0.2, 0.2]),
                            "BoundingBoxSize": ti.Vector([6., 1.0, 6.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.2, 0.2, 6.2]),
                            "BoundingBoxSize": ti.Vector([0.6, 1.0, 2.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([0., 0., 0., 0., 0., 0.])
                                                         },
                                       "Traction":       [],
                                       "InitialVelocity":ti.Vector([0., 0., 0.]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":          True,
                                       "Density":            1500,
                                       "ParticleStress":     {},
                                       "Traction":           {},
                                       "InitialVelocity":    ti.Vector([0., 0., -0.00125]),
                                       "FixVelocity":        ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [6.4, 1.4, 0.2],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [-1, 0, 0],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0.2, 1.4, 8.4]
                                        },
                                        
                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [1., 0, 0],
                                             "StartPoint":     [6.2, 0., 0.],
                                             "EndPoint":       [6.4, 1.4, 8.4]
                                        },
                                        
                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [0, -1., 0],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [6.4, 0.2, 8.4]
                                        },
                                        
                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [0, 1., 0],
                                             "StartPoint":     [0., 1.2, 0.],
                                             "EndPoint":       [6.4, 1.4, 8.4]
                                        }
                                    ])


mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing(write_background_grid=True)
