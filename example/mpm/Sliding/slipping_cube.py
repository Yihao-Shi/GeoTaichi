from src import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([20., 4., 6.]), 
                      background_damping=0., 
                      gravity=ti.Vector([6.929646456, 0., -6.929646456]),
                      alphaPIC=0.00, 
                      mapping="USF", 
                      shape_function="Linear")

mpm.set_solver(solver={
                           "Timestep":                   1e-3,
                           "SimulationTime":             6,
                           "SaveInterval":               0.1
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5
                            })

mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              2650.,
                               "YoungModulus":         7e9,
                               "PossionRatio":        0.3
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([1., 1., 1.]),
                             "Contact":            {
                                                        "ContactDetection":                True,
                                                        "Friction":                        0.5,
                                                        "CutOff":                          0.74
                                                   }
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([1., 1., 1.]),
                            "BoundingBoxSize": ti.Vector([2., 2., 2.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0., 0.]),
                            "BoundingBoxSize": ti.Vector([20., 4., 1.]),
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
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([5, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":         True,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.]),
                                                              "Traction":         {}
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition()

mpm.select_save_data()

mpm.run()
