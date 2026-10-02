import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch="gpu", device_memory_GB=3.7)

mpm = MPM()

mpm.set_configuration(domain=[8., 0.7, 1.],
                      background_damping=0.00,
                      alphaPIC=0.00, 
                      mapping="USL", 
                      shape_function="QuadBSpline",
                      gravity=[0., 0., -9.8],
                      material_type="Fluid")

mpm.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   2.,
                      "SaveInterval":     0.04,
                 }) 
                      
mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           980000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   541681,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            
mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "Modulus":              2e7,
                               "Viscosity":            1e-3,
                               "ElementLength":        0.02,
                               "cL":                   0.7,
                               "cQ":                   2
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               [0.02, 0.02, 0.02]
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": [0.0, 0.0, 0.0],
                            "BoundingBoxSize": [3.5, 0.7, 0.4],
                            
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":[0, 0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
                   

mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., -1., 0.],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [8., 0.0, 1.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 1., 0.],
                                        "StartPoint":     [0, 0.7, 0],
                                        "EndPoint":       [8., 0.7, 1.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [-1., 0., 0.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0., 0.7, 1.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [1., 0., 0.],
                                        "StartPoint":     [8., 0.0, 0],
                                        "EndPoint":       [8., 0.7, 1.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., -1.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [8., 0.7, 0.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., 1.],
                                        "StartPoint":     [0, 0.0, 1.],
                                        "EndPoint":       [8., 0.7, 1.],
                                    }])


mpm.select_save_data(grid=True)

mpm.run(gravity_field=True)

mpm.postprocessing()


