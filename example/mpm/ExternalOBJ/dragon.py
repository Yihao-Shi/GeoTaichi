import sys
sys.path.append('/home/eleven/work/GeoTaichi')
import numpy as np

from geotaichi import *

init(arch='gpu', log=False, debug=False, device_memory_GB=5.5, kernel_profiler=False)

mpm = MPM()

mpm.set_configuration( 
                      domain=ti.Vector([1.5, 1.2, 0.7]),
                      background_damping=0.005,
                      alphaPIC=0.00, 
                      mapping="USF", 
                      shape_function="Linear",
                      gravity=ti.Vector([0., -9.8, 0.]),
                      material_type="Fluid")

mpm.set_solver({
                "Timestep":         1e-5,
                "SimulationTime":   0.8,
                "SaveInterval":     0.016,
                "SavePath":         "Dragon"
               })
                      
mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           1559361,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   121914,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })


mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "Modulus":              3.6e5,
                               "Viscosity":            1e-3
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.02, 0.02, 0.02])
                        })

mpm.add_body_from_file(body={
                                    "FileType": "OBJ",
                                    "Template":  {
                                                   "BodyID": 0,
                                                   "MaterialID": 1,
                                                   "ParticleFile": "/home/eleven/work/GeoTaichi/assets/Dragon_50k.obj",
                                                   "nParticlesPerCell": 2,
                                                   "Offset": [0.7, 0.4, 0.3]
                                                  }
                   })
                   
mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., -1.],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [1.5, 1.20, 0.]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [-1., 0., 0.],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [0., 1.2, 0.7]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [1., 0., 0.],
                                        "StartPoint":     [1.5, 0., 0.],
                                        "EndPoint":       [1.5, 1.2, 0.7]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., -1., 0.],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [1.5, 0, 0.7]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 1., 0.],
                                        "StartPoint":     [0., 1.2, 0.],
                                        "EndPoint":       [1.5, 1.2, 0.7]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., 1.],
                                        "StartPoint":     [0., 0., 0.7],
                                        "EndPoint":       [1.5, 1.2, 0.7]
                                    }])

mpm.select_save_data()

mpm.run()

