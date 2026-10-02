import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, device_memory_GB=3.7)

mpm = MPM()

mpm.set_configuration(domain=[4., 8.],
                      background_damping=0.,
                      alphaPIC=0.001, 
                      mapping="USL", 
                      shape_function="QuadBSpline",
                      gravity=[0., -9.8],
                      material_type="Fluid",
                      velocity_projection="Taylor",
                      stabilize="F-Bar Method"
                      )

mpm.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   4,
                      "SaveInterval":     1e-1,
                      "SavePath":         'large_tank'
                 }) 
                      
mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           160000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   541681
                                                          }
                            })
                            
mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "Modulus":              2e9,
                               "Viscosity":            1e-3,
                               "ElementLength":        0.2,
                               "cL":                   1.0,
                               "cQ":                   2
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               [0.2, 0.2]
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": [0.0, 0.0],
                            "BoundingBoxSize": [4., 4.],
                            
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
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0., None],
                                        "StartPoint":     [0, 0],
                                        "EndPoint":       [0., 8.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0., None],
                                        "StartPoint":     [4., 0],
                                        "EndPoint":       [4., 8.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0.],
                                        "StartPoint":     [0, 0],
                                        "EndPoint":       [4., 0.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [None, 0.],
                                        "StartPoint":     [0, 8.],
                                        "EndPoint":       [4., 8.],
                                    }])


mpm.select_save_data()

mpm.run(gravity_field=True)

mpm.postprocessing()


