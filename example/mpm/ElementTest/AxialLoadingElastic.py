from geotaichi import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([12, 1, 1]), 
                      background_damping=0.05, 
                      gravity=ti.Vector([0., 0., 0.]), 
                      alphaPIC=0.005, 
                      mapping="USF", 
                      shape_function="Linear",
                      particle_traction_method="Virtual")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             10,
                           "SaveInterval":               0.2
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   396,
                                                          }
                            })

mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              1.,
                               "YoungModulus":         1e3,
                               "PoissonRatio":        0.
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([1.0, 1.0, 1.0])
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0., 0.]),
                            "BoundingBoxSize": ti.Vector([9.5, 1., 1.]),
                            
                      })

mpm.add_body(body={
                       "Template": {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0., 1., 1.]
                                        }
                                    ])

mpm.add_virtual_stress_field(field={"ConfiningPressure": [-1., -1., -1., 0., 0., 0.],
                                    "VirtualForce": [0., 0., 0.]})

mpm.select_save_data()

mpm.run()

mpm.postprocessing()
