# import sys
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, device_memory_GB=7.0)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.6, 2.508]),
                      is_2DAxisy=True,
                      background_damping=0.05,
                      gravity=ti.Vector([0., -9.8]),
                      alphaPIC=0.06,
                      mapping="USL",
                      shape_function="GIMP",)

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.4,
                           "SaveInterval":               0.2,
                           "SavePath":                   'Pile2DAxisy_Moditfy'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    800000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     134474,
                                                               "max_reflection_constraint":   0
                                                          }
                            })
                            
mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":           1,
                               "Density":              1600.,
                               "YoungModulus":         54e6,
                               "PoissonRatio":         0.3,
                               "Cohesion":             100,
                               "Friction":             30,
                               "Dilation":             0.
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.006, 0.006]),
                             "Contact":   {
                                               "ContactDetection":                "GeoContact",
                                               "Friction":                        0.49,
                                               "CutOff":                          1.0,
                                               "Penalty":                         [0.,25.]
                                          }
                        })

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
                                                              "InternalStress":   ti.Vector([-30000, -30000, -30000, 0., 0., 0.])
                                                         },
                                       "Traction":       [{"Pressure": ti.Vector([0, -30000]),
                                                           "RegionName": "region3"}],
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":          True,
                                       "Density":            1500,
                                       "ParticleStress":     {},
                                       "Traction":           {},
                                       "InitialVelocity":    ti.Vector([0., -0.1]),
                                       "FixVelocity":        ["Fix", "Fix"]
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0.6, 0.],
                                             "NLevel":         0
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

mpm.postprocessing(read_path='Pile2DAxisy_Moditfy', write_background_grid=True)
