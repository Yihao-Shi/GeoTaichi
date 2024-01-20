from src import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([3., 3., 3.]), 
                      background_damping=0., 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.00, 
                      mapping="USL", 
                      shape_function="Linear",
                      gauss_number=2)

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             0.1,
                           "SaveInterval":               0.01
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    80,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   8,
                                                               "max_traction_constraint":   8
                                                          }
                            })

mpm.add_material(model="ModifiedCamClay",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1530,
                               "PossionRatio":                  0.25,
                               "StressRatio":                   1.02,
                               "lambda":                        0.12,
                               "kappa":                         0.023,
                               "void_ratio_ref":                1.7,
                               "ConsolidationPressure":         392000
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([1., 1., 1.]),
                             "Contact":       {
                                                   "ContactDetection":                True,
                                                   "Friction":                        0,
                                                   "CurOff":                          1.6
                                              }
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([1., 1., 1.]),
                            "BoundingBoxSize": ti.Vector([1., 1., 1.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0., 2.]),
                            "BoundingBoxSize": ti.Vector([3., 3., 1.]),
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
                                                              "InternalStress":   ti.Vector([-33000., -33000., -33000., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":         True,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })


mpm.scene.velocity_boundary[0].node = 21
mpm.scene.velocity_boundary[0].level = 0
mpm.scene.velocity_boundary[0].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[0].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[0].velocity = [0, 0, 0]

mpm.scene.velocity_boundary[1].node = 22
mpm.scene.velocity_boundary[1].level = 0
mpm.scene.velocity_boundary[1].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[1].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[1].velocity = [0, 0, 0]

mpm.scene.velocity_boundary[2].node = 25
mpm.scene.velocity_boundary[2].level = 0
mpm.scene.velocity_boundary[2].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[2].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[2].velocity = [0, 0, 0]

mpm.scene.velocity_boundary[3].node = 26
mpm.scene.velocity_boundary[3].level = 0
mpm.scene.velocity_boundary[3].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[3].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[3].velocity = [0, 0, 0]

mpm.scene.velocity_list[0] = 8

p = 33000/4.
mpm.scene.traction_boundary[0].node = 37
mpm.scene.traction_boundary[0].level = 0
mpm.scene.traction_boundary[0].traction = [p, p, -0]

mpm.scene.traction_boundary[1].node = 38
mpm.scene.traction_boundary[1].level = 0
mpm.scene.traction_boundary[1].traction = [-p, p, -0]

mpm.scene.traction_boundary[2].node = 41
mpm.scene.traction_boundary[2].level = 0
mpm.scene.traction_boundary[2].traction = [p, -p, -0]

mpm.scene.traction_boundary[3].node = 42
mpm.scene.traction_boundary[3].level = 0
mpm.scene.traction_boundary[3].traction = [-p, -p, -0]

mpm.scene.traction_boundary[4].node = 21
mpm.scene.traction_boundary[4].level = 0
mpm.scene.traction_boundary[4].traction = [p, p, 0]

mpm.scene.traction_boundary[5].node = 22
mpm.scene.traction_boundary[5].level = 0
mpm.scene.traction_boundary[5].traction = [-p, p, 0]

mpm.scene.traction_boundary[6].node = 25
mpm.scene.traction_boundary[6].level = 0
mpm.scene.traction_boundary[6].traction = [p, -p, 0]

mpm.scene.traction_boundary[7].node = 26
mpm.scene.traction_boundary[7].level = 0
mpm.scene.traction_boundary[7].traction = [-p, -p, 0]

mpm.scene.traction_list[0] = 8

mpm.select_save_data(grid=True)

mpm.run()

mpm.update_particle_properties(property_name='velocity', value=[0., 0., -0.01], bodyID=1)

mpm.modify_parameters(SimulationTime=25, SaveInterval=0.1)

mpm.run()
