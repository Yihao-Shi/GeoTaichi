from geotaichi import *

init(arch='cpu')

pressure=303000

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([3., 3., 3.]), 
                      background_damping=0., 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.00, 
                      mapping="USF", 
                      shape_function="Linear",
                      stress_integration="SubStepping",
                      gauss_number=2)

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             25,
                           "SaveInterval":               0.25,
                           "SavePath":                   f"UndrainedMCC/{int(pressure/1000)}kpa"
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
                               "PoissonRatio":                  0.25,
                               "StressRatio":                   1.02,
                               "lambda":                        0.12,
                               "kappa":                         0.023,
                               "void_ratio_ref":                1.7,
                               "OverConsolidationRatio":        392./(pressure/1000),
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([1., 1., 1.])
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([1., 1., 1.]),
                            "BoundingBoxSize": ti.Vector([1., 1., 1.]),
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
                                                              "InternalStress":   ti.Vector([-pressure, -pressure, -pressure, 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })


mpm.scene.boundary.velocity_boundary[0].node = 37
mpm.scene.boundary.velocity_boundary[0].level = 0
mpm.scene.boundary.velocity_boundary[0].dirs = 0
mpm.scene.boundary.velocity_boundary[0].velocity = -0.0025
mpm.scene.boundary.velocity_boundary[1].node = 37
mpm.scene.boundary.velocity_boundary[1].level = 0
mpm.scene.boundary.velocity_boundary[1].dirs = 1
mpm.scene.boundary.velocity_boundary[1].velocity = -0.0025
mpm.scene.boundary.velocity_boundary[2].node = 37
mpm.scene.boundary.velocity_boundary[2].level = 0
mpm.scene.boundary.velocity_boundary[2].dirs = 2
mpm.scene.boundary.velocity_boundary[2].velocity = -0.005

mpm.scene.boundary.velocity_boundary[3].node = 38
mpm.scene.boundary.velocity_boundary[3].level = 0
mpm.scene.boundary.velocity_boundary[3].dirs = 0
mpm.scene.boundary.velocity_boundary[3].velocity = 0.0025
mpm.scene.boundary.velocity_boundary[4].node = 38
mpm.scene.boundary.velocity_boundary[4].level = 0
mpm.scene.boundary.velocity_boundary[4].dirs = 1
mpm.scene.boundary.velocity_boundary[4].velocity = -0.0025
mpm.scene.boundary.velocity_boundary[5].node = 38
mpm.scene.boundary.velocity_boundary[5].level = 0
mpm.scene.boundary.velocity_boundary[5].dirs = 2
mpm.scene.boundary.velocity_boundary[5].velocity = -0.005

mpm.scene.boundary.velocity_boundary[6].node = 41
mpm.scene.boundary.velocity_boundary[6].level = 0
mpm.scene.boundary.velocity_boundary[6].dirs = 0
mpm.scene.boundary.velocity_boundary[6].velocity = -0.0025
mpm.scene.boundary.velocity_boundary[7].node = 41
mpm.scene.boundary.velocity_boundary[7].level = 0
mpm.scene.boundary.velocity_boundary[7].dirs = 1
mpm.scene.boundary.velocity_boundary[7].velocity = 0.0025
mpm.scene.boundary.velocity_boundary[8].node = 41
mpm.scene.boundary.velocity_boundary[8].level = 0
mpm.scene.boundary.velocity_boundary[8].dirs = 2
mpm.scene.boundary.velocity_boundary[8].velocity = -0.005


mpm.scene.boundary.velocity_boundary[9].node = 42
mpm.scene.boundary.velocity_boundary[9].level = 0
mpm.scene.boundary.velocity_boundary[9].dirs = 0
mpm.scene.boundary.velocity_boundary[9].velocity = 0.0025
mpm.scene.boundary.velocity_boundary[10].node = 42
mpm.scene.boundary.velocity_boundary[10].level = 0
mpm.scene.boundary.velocity_boundary[10].dirs = 1
mpm.scene.boundary.velocity_boundary[10].velocity = 0.0025
mpm.scene.boundary.velocity_boundary[11].node = 42
mpm.scene.boundary.velocity_boundary[11].level = 0
mpm.scene.boundary.velocity_boundary[11].dirs = 2
mpm.scene.boundary.velocity_boundary[11].velocity = -0.005


mpm.scene.boundary.velocity_boundary[12].node = 21
mpm.scene.boundary.velocity_boundary[12].level = 0
mpm.scene.boundary.velocity_boundary[12].dirs = 0
mpm.scene.boundary.velocity_boundary[12].velocity = -0.0025
mpm.scene.boundary.velocity_boundary[13].node = 21
mpm.scene.boundary.velocity_boundary[13].level = 0
mpm.scene.boundary.velocity_boundary[13].dirs = 1
mpm.scene.boundary.velocity_boundary[13].velocity = -0.0025
mpm.scene.boundary.velocity_boundary[14].node = 21
mpm.scene.boundary.velocity_boundary[14].level = 0
mpm.scene.boundary.velocity_boundary[14].dirs = 2
mpm.scene.boundary.velocity_boundary[14].velocity = 0.005


mpm.scene.boundary.velocity_boundary[15].node = 22
mpm.scene.boundary.velocity_boundary[15].level = 0
mpm.scene.boundary.velocity_boundary[15].dirs = 0
mpm.scene.boundary.velocity_boundary[15].velocity = 0.0025
mpm.scene.boundary.velocity_boundary[16].node = 22
mpm.scene.boundary.velocity_boundary[16].level = 0
mpm.scene.boundary.velocity_boundary[16].dirs = 1
mpm.scene.boundary.velocity_boundary[16].velocity = -0.0025
mpm.scene.boundary.velocity_boundary[17].node = 22
mpm.scene.boundary.velocity_boundary[17].level = 0
mpm.scene.boundary.velocity_boundary[17].dirs = 2
mpm.scene.boundary.velocity_boundary[17].velocity = 0.005


mpm.scene.boundary.velocity_boundary[18].node = 25
mpm.scene.boundary.velocity_boundary[18].level = 0
mpm.scene.boundary.velocity_boundary[18].dirs = 0
mpm.scene.boundary.velocity_boundary[18].velocity = -0.0025
mpm.scene.boundary.velocity_boundary[19].node = 25
mpm.scene.boundary.velocity_boundary[19].level = 0
mpm.scene.boundary.velocity_boundary[19].dirs = 1
mpm.scene.boundary.velocity_boundary[19].velocity = 0.0025
mpm.scene.boundary.velocity_boundary[20].node = 25
mpm.scene.boundary.velocity_boundary[20].level = 0
mpm.scene.boundary.velocity_boundary[20].dirs = 2
mpm.scene.boundary.velocity_boundary[20].velocity = 0.005


mpm.scene.boundary.velocity_boundary[21].node = 26
mpm.scene.boundary.velocity_boundary[21].level = 0
mpm.scene.boundary.velocity_boundary[21].dirs = 0
mpm.scene.boundary.velocity_boundary[21].velocity = 0.0025
mpm.scene.boundary.velocity_boundary[22].node = 26
mpm.scene.boundary.velocity_boundary[22].level = 0
mpm.scene.boundary.velocity_boundary[22].dirs = 1
mpm.scene.boundary.velocity_boundary[22].velocity = 0.0025
mpm.scene.boundary.velocity_boundary[23].node = 26
mpm.scene.boundary.velocity_boundary[23].level = 0
mpm.scene.boundary.velocity_boundary[23].dirs = 2
mpm.scene.boundary.velocity_boundary[23].velocity = 0.005
mpm.scene.boundary.velocity_list[0] = 24

# In the case of a single MPM element, the external force applied at each node is taken as one-quarter of the total internal force, assuming equal force distribution among the four nodes to maintain equilibrium.
p = pressure/4.
mpm.scene.boundary.traction_boundary[0].node = 37
mpm.scene.boundary.traction_boundary[0].level = 0
mpm.scene.boundary.traction_boundary[0].dirs = 0
mpm.scene.boundary.traction_boundary[0].traction = p
mpm.scene.boundary.traction_boundary[1].node = 37
mpm.scene.boundary.traction_boundary[1].level = 0
mpm.scene.boundary.traction_boundary[1].dirs = 1
mpm.scene.boundary.traction_boundary[1].traction = p
mpm.scene.boundary.traction_boundary[2].node = 37
mpm.scene.boundary.traction_boundary[2].level = 0
mpm.scene.boundary.traction_boundary[2].dirs = 2
mpm.scene.boundary.traction_boundary[2].traction = -p

mpm.scene.boundary.traction_boundary[3].node = 38
mpm.scene.boundary.traction_boundary[3].level = 0
mpm.scene.boundary.traction_boundary[3].dirs = 0
mpm.scene.boundary.traction_boundary[3].traction = -p
mpm.scene.boundary.traction_boundary[4].node = 38
mpm.scene.boundary.traction_boundary[4].level = 0
mpm.scene.boundary.traction_boundary[4].dirs = 1
mpm.scene.boundary.traction_boundary[4].traction = p
mpm.scene.boundary.traction_boundary[5].node = 38
mpm.scene.boundary.traction_boundary[5].level = 0
mpm.scene.boundary.traction_boundary[5].dirs = 2
mpm.scene.boundary.traction_boundary[5].traction = -p

mpm.scene.boundary.traction_boundary[6].node = 41
mpm.scene.boundary.traction_boundary[6].level = 0
mpm.scene.boundary.traction_boundary[6].dirs = 0
mpm.scene.boundary.traction_boundary[6].traction = p
mpm.scene.boundary.traction_boundary[7].node = 41
mpm.scene.boundary.traction_boundary[7].level = 0
mpm.scene.boundary.traction_boundary[7].dirs = 1
mpm.scene.boundary.traction_boundary[7].traction = -p
mpm.scene.boundary.traction_boundary[8].node = 41
mpm.scene.boundary.traction_boundary[8].level = 0
mpm.scene.boundary.traction_boundary[8].dirs = 2
mpm.scene.boundary.traction_boundary[8].traction = -p

mpm.scene.boundary.traction_boundary[9].node = 42
mpm.scene.boundary.traction_boundary[9].level = 0
mpm.scene.boundary.traction_boundary[9].dirs = 0
mpm.scene.boundary.traction_boundary[9].traction = -p
mpm.scene.boundary.traction_boundary[10].node = 42
mpm.scene.boundary.traction_boundary[10].level = 0
mpm.scene.boundary.traction_boundary[10].dirs = 1
mpm.scene.boundary.traction_boundary[10].traction = -p
mpm.scene.boundary.traction_boundary[11].node = 42
mpm.scene.boundary.traction_boundary[11].level = 0
mpm.scene.boundary.traction_boundary[11].dirs = 2
mpm.scene.boundary.traction_boundary[11].traction = -p

mpm.scene.boundary.traction_boundary[12].node = 21
mpm.scene.boundary.traction_boundary[12].level = 0
mpm.scene.boundary.traction_boundary[12].dirs = 0
mpm.scene.boundary.traction_boundary[12].traction = p
mpm.scene.boundary.traction_boundary[13].node = 21
mpm.scene.boundary.traction_boundary[13].level = 0
mpm.scene.boundary.traction_boundary[13].dirs = 1
mpm.scene.boundary.traction_boundary[13].traction = p
mpm.scene.boundary.traction_boundary[14].node = 21
mpm.scene.boundary.traction_boundary[14].level = 0
mpm.scene.boundary.traction_boundary[14].dirs = 2
mpm.scene.boundary.traction_boundary[14].traction = p

mpm.scene.boundary.traction_boundary[15].node = 22
mpm.scene.boundary.traction_boundary[15].level = 0
mpm.scene.boundary.traction_boundary[15].dirs = 0
mpm.scene.boundary.traction_boundary[15].traction = -p
mpm.scene.boundary.traction_boundary[16].node = 22
mpm.scene.boundary.traction_boundary[16].level = 0
mpm.scene.boundary.traction_boundary[16].dirs = 1
mpm.scene.boundary.traction_boundary[16].traction = p
mpm.scene.boundary.traction_boundary[17].node = 22
mpm.scene.boundary.traction_boundary[17].level = 0
mpm.scene.boundary.traction_boundary[17].dirs = 2
mpm.scene.boundary.traction_boundary[17].traction = p

mpm.scene.boundary.traction_boundary[18].node = 25
mpm.scene.boundary.traction_boundary[18].level = 0
mpm.scene.boundary.traction_boundary[18].dirs = 0
mpm.scene.boundary.traction_boundary[18].traction = p
mpm.scene.boundary.traction_boundary[19].node = 25
mpm.scene.boundary.traction_boundary[19].level = 0
mpm.scene.boundary.traction_boundary[19].dirs = 1
mpm.scene.boundary.traction_boundary[19].traction = -p
mpm.scene.boundary.traction_boundary[20].node = 25
mpm.scene.boundary.traction_boundary[20].level = 0
mpm.scene.boundary.traction_boundary[20].dirs = 2
mpm.scene.boundary.traction_boundary[20].traction = p

mpm.scene.boundary.traction_boundary[21].node = 26
mpm.scene.boundary.traction_boundary[21].level = 0
mpm.scene.boundary.traction_boundary[21].dirs = 0
mpm.scene.boundary.traction_boundary[21].traction = -p
mpm.scene.boundary.traction_boundary[22].node = 26
mpm.scene.boundary.traction_boundary[22].level = 0
mpm.scene.boundary.traction_boundary[22].dirs = 1
mpm.scene.boundary.traction_boundary[22].traction = -p
mpm.scene.boundary.traction_boundary[23].node = 26
mpm.scene.boundary.traction_boundary[23].level = 0
mpm.scene.boundary.traction_boundary[23].dirs = 2
mpm.scene.boundary.traction_boundary[23].traction = p
mpm.scene.boundary.traction_list[0] = 24

mpm.select_save_data()

mpm.run()

