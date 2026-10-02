import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=7)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([1.7, 0.15, 2.5]), 
                      background_damping=0.00, 
                      alphaPIC=0.002, 
                      mapping="USL", 
                      shape_function="QuadBSpline",
                      material_type="Fluid",
                      sparse_grid=False)

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             2.5,
                           "SaveInterval":               0.02
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    9853704,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   1721605
                                                          }
                            })               
mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "Modulus":              3.6e5,
                               "Viscosity":            1e-3,
                               "ElementLength":        0.,
                               "cL":                   0.7,
                               "cQ":                   2
                 })
mpm.scene.activate_particle(mpm.sims)      
                 
mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.0035, 0.0035, 0.0035])
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
                            "BoundingBoxSize": ti.Vector([0.6, 0.15, 0.6]),
                            
                      })

'''mpm.add_body(body={
                       "Template": {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]
                                   },    
                       "WriteFile":       True
                   })'''
                   
mpm.add_body_from_file(body={"FileType":                         "TXT",
                                      "Template": {
                                        "BodyID": 0,
                                        "MaterialID": 1,
                                        "ParticleFile": "Particle.txt",
                                        "InitialVelocity": ti.Vector([0, 0, 0]),
                                        "FixVelocity": ["Free", "Free", "Free"]}
                                     })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [0., 0., -1.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [1.7, 0.15, 0.0]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [-1., 0., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0.0, 0.15, 2.5]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [1., 0., 0.],
                                             "StartPoint":     [1.66, 0., 0.],
                                             "EndPoint":       [1.7, 0.15, 2.5]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [0., -1., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [1.7, 0.0, 2.5]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [0., 1., 0.],
                                             "StartPoint":     [0., 0.15, 0.],
                                             "EndPoint":       [1.7, 0.15, 2.5]
                                        },

                                        {
                                             "BoundaryType":   "ReflectionConstraint",
                                             "Norm":           [0., 0., 1],
                                             "StartPoint":     [0., 0., 2.4],
                                             "EndPoint":       [1.7, 0.15, 2.5]
                                        }
                                    ])

mpm.select_save_data(grid=True)

mpm.run(gravity_field=True)

mpm.postprocessing()
