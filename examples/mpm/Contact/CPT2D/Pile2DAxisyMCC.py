import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

    
from geotaichi import *
init(dim=2, device_memory_GB=7.0)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.6, 2.508]),
                      is_2DAxisy=True,
                      background_damping=0.2,
                      gravity=ti.Vector([0., -9.8]),
                      alphaPIC=0.2, 
                      mapping="USF", 
                      shape_function="GIMP",
                      #stabilize="B-Bar Method",
                      stress_integration="SubStepping",
                      #velocity_projection="Taylor"
                      )

mpm.set_solver(solver={
                           "Timestep":                   5e-7,
                           "SimulationTime":             10,
                           "SaveInterval":               0.2,
                           "SavePath":                   '1_Pile2DAxisy_MCC1'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    800000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     134474,
                                                               "max_particle_traction_constraint":   134474,
                                                          }
                            })

mpm.add_contact(contact_type="MPMContact", friction=0.49)                            

mpm.add_material(model="ModifiedCamClay",
                 material={
                               "MaterialID":                    1,
                               "Density":                       2800,
                               "PoissonRatio":                  0.3,
                               "StressRatio":                   0.984,
                               "lambda":                        0.25,
                               "kappa":                         0.05,
                               "void_ratio_ref":                2.04,
                               "OverConsolidationRatio":        2.,
                               "ThreeInvariants":               False
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.006, 0.006])})

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([0.6, 1.5]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Cone2D",
                            "BoundingBoxPoint": ti.Vector([0., 1.5]),
                            "BoundingBoxSize": ti.Vector([0.018, 1.]),
                            
                      },

                      {
                            "Name": "region3",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0.018, 1.497]),
                            "BoundingBoxSize": ti.Vector([0.582, 0.003]),
                            
                      },
                      ])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-120e3, -150e3, -120e3, 0., 0., 0.])
                                                         },
                                       "Traction":       [{"Pressure": ti.Vector([0, -150e3]),
                                                           "RegionName": "region3"}],
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":          True,
                                       "Density":            1600,
                                       "InitialVelocity":    [0., -0.1],
                                       "FixVelocity":        ["Fix", "Fix"]
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0.6, 0.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 2.508]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0.6, 0.],
                                             "EndPoint":       [0.6, 2.508]
                                        },
                                    ])


mpm.select_save_data(grid=True)

mpm.run(gravity_field=True)

mpm.postprocessing(read_path='1_Pile2DAxisy_MCC', write_background_grid=True, end_file=51)

