import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, device_memory_GB=4)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([5., 3.]), 
                      background_damping=0., 
                      gravity=ti.Vector([0., 0.]),
                      alphaPIC=0., 
                      mapping="USL", 
                      shape_function="GIMP",
                      stress_integration="SubStepping")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.1,
                           "SaveInterval":               0.01
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    3.5e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   5100,
                                                               "max_particle_traction_constraint":   1100,
                                                               "particle_traction_method": "Stable"
                                                          }
                            })

mpm.add_contact(contact_type="MPMContact", friction=1.0)

mpm.add_material(model="ModifiedCamClay",
                 material={
                               "MaterialID":                    1,
                               "Density":                       2000,
                               "PoissonRatio":                  0.3,
                               "StressRatio":                   1.02,
                               "lambda":                        0.1,
                               "kappa":                         0.008,
                               "void_ratio_ref":                1.76,
                               "OverConsolidationRatio":        1.,
                               "ThreeInvariants":               True,
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.05, 0.05])
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([2., 0.]),
                            "BoundingBoxSize": ti.Vector([1., 2.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([1.5, 2.]),
                            "BoundingBoxSize": ti.Vector([2., 0.1]),
                            
                      },
                      
                      {
                            "Name": "region3",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([2., 0.]),
                            "BoundingBoxSize": ti.Vector([0.025, 2.]),
                            
                      },
                      
                      {
                            "Name": "region4",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([2.975, 0.]),
                            "BoundingBoxSize": ti.Vector([0.025, 2.]),
                            
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-100000., -100000., -100000., 0., 0., 0.])
                                                         },
                                       "Traction":         [{
                                                                                    "Pressure": ti.Vector([100000, 0.]),
                                                                                    "RegionName": "region3"
                                                                                   },
                                                                                  
                                                                                   {
                                                                                    "Pressure": ti.Vector([-100000, 0.]),
                                                                                    "RegionName": "region4"
                                                                                   }],
                                       "InitialVelocity":[0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":         True,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-0, -0, -0, 0., 0., 0.])
                                                         },
                                       "InitialVelocity":[0, 0],
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0, 0],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [5., 0.],
                                             "NLevel":         0
                                        }
                                    ])


mpm.select_save_data(grid=True)

mpm.run()

mpm.update_particle_properties(property_name='velocity', value=[0., -0.02], bodyID=1)

mpm.modify_parameters(SimulationTime=15, SaveInterval=0.3)

mpm.run()

mpm.postprocessing()

