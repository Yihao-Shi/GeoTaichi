import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=5, debug=False)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([5., 5., 5.]), 
                      alphaPIC=0.,  
                      stabilize=None,
                      shape_function="Linear",
                      solver_type="Implicit")
                      
mpm.set_implicit_solver_parameters(quasi_static=False)

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             2,
                           "SaveInterval":               1e-3
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    84000,
                                "max_constraint_number":  {
                                                               "max_displacement_constraint":  18483
                                                          }
                            })

mpm.add_material(model="DruckerParger",
                 material={
                               "MaterialID":           1,
                               "Density":              2500.,
                               "YoungModulus":         2e7,
                               "PoissonRatio":         0.3,
                               "Friction":             30 
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([1, 1, 1])
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2, 2, 0]),
                            "BoundingBoxSize": ti.Vector([1, 1, 1]),
                            
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
                                             "BoundaryType":   "DisplacementConstraint",
                                             "Displacement":   [0., 0., 0],
                                             "StartPoint":     [0., 0, 0],
                                             "EndPoint":       [5., 5., 0.]
                                        }
                                    ])

mpm.select_save_data()

mpm.run()

mpm.postprocessing()
