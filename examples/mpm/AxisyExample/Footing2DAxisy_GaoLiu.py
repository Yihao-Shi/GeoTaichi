import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([12, 10.4]),
                      is_2DAxisy=True,
                      background_damping=0.0,
                      gravity=ti.Vector([0., -0]),
                      alphaPIC=0.005,
                      mapping="USL",
                      shape_function="GIMP",)

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             4,
                           "SaveInterval":               0.5,
                           "SavePath":                   'Footing2DAxisy_GaoLiu_Test2'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    800000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     134474,
                                                               "max_reflection_constraint":   0,
                                                               "max_particle_traction_constraint": 10293
                                                          }
                            })
                            
mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":           1,
                               "Density":              1500.,
                               "YoungModulus":         28e6,
                               "PoissonRatio":         0.35,
                               "Cohesion":             3000,
                               "Friction":             32.,
                               "Dilation":             9.
                 })

mpm.add_contact(contact_type="GeoContact", friction=1.0, cutoff=0.8, penalty=[1., 2.])

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.1, 0.1])
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([12., 8.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 8.]),
                            "BoundingBoxSize": ti.Vector([1., 2.3]),
                            
                      },

                      {
                            "Name": "region3",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([1., 7.95]),
                            "BoundingBoxSize": ti.Vector([11., 0.05]),
                            
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
                                             "EndPoint":       [12., 0.],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 10.4]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [12., 0.],
                                             "EndPoint":       [12., 10.4]
                                        },
                                    ])


mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing(read_path='Footing2DAxisy_GaoLiu_Test2', write_background_grid=True)
