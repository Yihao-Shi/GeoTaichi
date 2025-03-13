from geotaichi import *

init(dim=2, device_memory_GB=3.7)

mpm = MPM()

mpm.set_configuration(domain=[6., 6.],
                      background_damping=0.0,
                      alphaPIC=1.0, 
                      mapping="USL", 
                      shape_function="QuadBSpline",
                      gravity=[0., -9.8],
                      material_type="Fluid",
                      velocity_projection="Affine") #"also support for Taylor PIC"

mpm.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   4,
                      "SaveInterval":     1e-1,
                      "SavePath":         'large_tank'
                 }) 
                      
mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           56000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   541681
                                                          }
                            })
                            
mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "Modulus":              2e6,
                               "Viscosity":            1e-3,
                               "ElementLength":        0.02,
                               "cL":                   1.0,
                               "cQ":                   2
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               [0.02, 0.02]
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": [0.0, 0.0],
                            "BoundingBoxSize": [2.24, 1.12],
                            "ydirection": [0., 1.]
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     True,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":[0, 0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
                   

mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [-1., 0.],
                                        "StartPoint":     [0, 0],
                                        "EndPoint":       [0., 6.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [1., 0.],
                                        "StartPoint":     [6., 0],
                                        "EndPoint":       [6., 6.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., -1.],
                                        "StartPoint":     [0, 0],
                                        "EndPoint":       [6., 0.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 1.],
                                        "StartPoint":     [0, 6.],
                                        "EndPoint":       [6., 6.],
                                    }])


mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing()


