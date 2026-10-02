import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='cpu')

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([3., 3., 3.]), 
                      background_damping=0., 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.0, 
                      mapping="USF", 
                      shape_function="Linear",
                      gauss_number=2)

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             25,
                           "SaveInterval":               0.5
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    80,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   24
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
                               "OverConsolidationRatio":        392./300.,
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
                                                              "InternalStress":   ti.Vector([-30300, -30300, -30300, 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })

mpm.scene.boundary.velocity_boundary[0].node = 37
mpm.scene.boundary.velocity_boundary[0].level = 0
mpm.scene.boundary.velocity_boundary[0].dirs = 2
mpm.scene.boundary.velocity_boundary[0].velocity = -0.005
mpm.scene.boundary.velocity_boundary[1].node = 38
mpm.scene.boundary.velocity_boundary[1].level = 0
mpm.scene.boundary.velocity_boundary[1].dirs = 2
mpm.scene.boundary.velocity_boundary[1].velocity = -0.005
mpm.scene.boundary.velocity_boundary[2].node = 41
mpm.scene.boundary.velocity_boundary[2].level = 0
mpm.scene.boundary.velocity_boundary[2].dirs = 2
mpm.scene.boundary.velocity_boundary[2].velocity = -0.005
mpm.scene.boundary.velocity_boundary[3].node = 42
mpm.scene.boundary.velocity_boundary[3].level = 0
mpm.scene.boundary.velocity_boundary[3].dirs = 2
mpm.scene.boundary.velocity_boundary[3].velocity = -0.005


mpm.scene.boundary.velocity_boundary[4].node = 21
mpm.scene.boundary.velocity_boundary[4].level = 0
mpm.scene.boundary.velocity_boundary[4].dirs = 2
mpm.scene.boundary.velocity_boundary[4].velocity = 0.005
mpm.scene.boundary.velocity_boundary[5].node = 22
mpm.scene.boundary.velocity_boundary[5].level = 0
mpm.scene.boundary.velocity_boundary[5].dirs = 2
mpm.scene.boundary.velocity_boundary[5].velocity = 0.005
mpm.scene.boundary.velocity_boundary[6].node = 25
mpm.scene.boundary.velocity_boundary[6].level = 0
mpm.scene.boundary.velocity_boundary[6].dirs = 2
mpm.scene.boundary.velocity_boundary[6].velocity = 0.005
mpm.scene.boundary.velocity_boundary[7].node = 26
mpm.scene.boundary.velocity_boundary[7].level = 0
mpm.scene.boundary.velocity_boundary[7].dirs = 2
mpm.scene.boundary.velocity_boundary[7].velocity = 0.005
mpm.scene.boundary.velocity_list[0] = 8

mpm.select_save_data()

mpm.run()

