import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(default_fp="float64", device_memory_GB=2)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.55, 0.05, 0.11]), 
                      #mode="Lightweight",
                      background_damping=0.00, 
                      alphaPIC=0.00, 
                      mapping="USL", 
                      #stabilize="B-Bar Method",
                      shape_function="GIMP",
                      #velocity_projection="Taylor",
                      sparse_grid=False)

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.3,
                           "SaveInterval":               0.01
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    8e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   110935,
                                                               "max_reflection_constraint":   64638,
                                                               "max_friction_constraint":   53703
                                                          }
                            })

mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":           1,
                               "Density":              2650.,
                               "YoungModulus":         8.4e5,
                               "PoissonRatio":        0.3,
                               "Cohesion":             0.,
                               "Friction":             19.8,
                               "Dilation":             0.,
                               "Tensile":              0.
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.006, 0.006, 0.006]),
                             "Contact":    {
                                                "ContactDetection":                False
                                           }
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.00, 0.00, 0.00]),
                            "BoundingBoxSize": ti.Vector([0.2, 0.05, 0.1]),
                            
                      })

mpm.add_body(body={
                       "Template": {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  5,
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
                                             "EndPoint":       [0.55, 0.05, 0.00]
                                        },

                                        {    
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0., 0, 0],
                                             "EndPoint":       [0.00, 0.05, 0.11]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0.55, 0.00, 0.11],
                                             "Norm":           [0., -1., 0.]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "StartPoint":     [0., 0.05, 0.],
                                             "EndPoint":       [0.55, 0.05, 0.11],
                                             "Norm":           [0., 1., 0.]
                                        }
                                    ])

mpm.select_save_data()

mpm.run(gravity_field=True)

mpm.postprocessing()
