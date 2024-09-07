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

mpm.set_configuration(domain=ti.Vector([3., 3., 3.]), 
                      background_damping=0., 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.00, 
                      mapping="USL", 
                      shape_function="Linear",
                      gauss_number=2)

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             95,
                           "SaveInterval":               0.2
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    80,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   8,
                                                               "max_traction_constraint":   8
                                                          }
                            })

mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":           1,
                               "Density":              2650.,
                               "YoungModulus":         1e7,
                               "PossionRatio":        0.3,
                               "Cohesion":             2500,
                               "Friction":             30.,
                               "Dilation":             0.,
                               "Tensile":              0.
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
                                                              "InternalStress":   ti.Vector([-306000, -306000, -306000, 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })

mpm.scene.velocity_boundary[0].node = 37
mpm.scene.velocity_boundary[0].level = 0
mpm.scene.velocity_boundary[0].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[0].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[0].velocity = [0, 0, -0.005]

mpm.scene.velocity_boundary[1].node = 38
mpm.scene.velocity_boundary[1].level = 0
mpm.scene.velocity_boundary[1].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[1].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[1].velocity = [0, 0, -0.005]

mpm.scene.velocity_boundary[2].node = 41
mpm.scene.velocity_boundary[2].level = 0
mpm.scene.velocity_boundary[2].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[2].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[2].velocity = [0, 0, -0.005]

mpm.scene.velocity_boundary[3].node = 42
mpm.scene.velocity_boundary[3].level = 0
mpm.scene.velocity_boundary[3].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[3].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[3].velocity = [0, 0, -0.005]

mpm.scene.velocity_boundary[4].node = 21
mpm.scene.velocity_boundary[4].level = 0
mpm.scene.velocity_boundary[4].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[4].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[4].velocity = [0, 0, 0.005]

mpm.scene.velocity_boundary[5].node = 22
mpm.scene.velocity_boundary[5].level = 0
mpm.scene.velocity_boundary[5].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[5].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[5].velocity = [0, 0, 0.005]

mpm.scene.velocity_boundary[6].node = 25
mpm.scene.velocity_boundary[6].level = 0
mpm.scene.velocity_boundary[6].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[6].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[6].velocity = [0, 0, 0.005]

mpm.scene.velocity_boundary[7].node = 26
mpm.scene.velocity_boundary[7].level = 0
mpm.scene.velocity_boundary[7].fix_v = [0, 0, 1]
mpm.scene.velocity_boundary[7].unfix_v = [1, 1, 0]
mpm.scene.velocity_boundary[7].velocity = [0, 0, 0.005]

mpm.scene.velocity_list[0] = 8

p = 306000/4.
mpm.scene.traction_boundary[0].node = 37
mpm.scene.traction_boundary[0].level = 0
mpm.scene.traction_boundary[0].traction = [p, p, -p]

mpm.scene.traction_boundary[1].node = 38
mpm.scene.traction_boundary[1].level = 0
mpm.scene.traction_boundary[1].traction = [-p, p, -p]

mpm.scene.traction_boundary[2].node = 41
mpm.scene.traction_boundary[2].level = 0
mpm.scene.traction_boundary[2].traction = [p, -p, -p]

mpm.scene.traction_boundary[3].node = 42
mpm.scene.traction_boundary[3].level = 0
mpm.scene.traction_boundary[3].traction = [-p, -p, -p]

mpm.scene.traction_boundary[4].node = 21
mpm.scene.traction_boundary[4].level = 0
mpm.scene.traction_boundary[4].traction = [p, p, p]

mpm.scene.traction_boundary[5].node = 22
mpm.scene.traction_boundary[5].level = 0
mpm.scene.traction_boundary[5].traction = [-p, p, p]

mpm.scene.traction_boundary[6].node = 25
mpm.scene.traction_boundary[6].level = 0
mpm.scene.traction_boundary[6].traction = [p, -p, p]

mpm.scene.traction_boundary[7].node = 26
mpm.scene.traction_boundary[7].level = 0
mpm.scene.traction_boundary[7].traction = [-p, -p, p]

mpm.scene.traction_list[0] = 8

mpm.select_save_data()

mpm.run()


