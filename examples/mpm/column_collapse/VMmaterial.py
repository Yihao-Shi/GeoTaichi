import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=3)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.55, 0.2, 0.11]), 
                      background_damping=0.02, 
                      alphaPIC=0.005, 
                      mapping="USF", 
                      shape_function="QuadBSpline",
                      stabilize="F-Bar Method")

mpm.set_solver(solver={
                           "Timestep":                   5e-5,
                           "SimulationTime":             0.6,
                           "SaveInterval":               0.01
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   83000
                                                          }
                            })

mpm.add_material(model="ElasticPerfectlyPlastic",
                 material={
                               "MaterialID":           1,
                               "Density":              2650.,
                               "YoungModulus":         2.5e5,
                               "PoissonRatio":        0.499,
                               "YieldStress":          300
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.0025, 0.0025, 0.0025])
                        })


mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.005, 0.005, 0.005]),
                            "BoundingBoxSize": ti.Vector([0.2, 0.05, 0.1]),
                            
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
                                             "StartPoint":     [0.0025, 0.0025, 0.0025],
                                             "EndPoint":       [0.55, 0.2, 0.005]
                                        },

                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0.0025, 0, 0],
                                             "EndPoint":       [0.005, 0.2, 0.11]
                                        },

                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0., None],
                                             "StartPoint":     [0.0075, 0.0025, 0],
                                             "EndPoint":       [0.55, 0.005, 0.11]
                                        },

                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0., None],
                                             "StartPoint":     [0.0075, 0.055, 0],
                                             "EndPoint":       [0.55, 0.0575, 0.11]
                                        }
                                    ])

mpm.select_save_data()

mpm.run(gravity_field=True)
