# import sys
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([55, 10]),
                      dimension="2-Dimension",
                      is_2DAxisy=True,
                      background_damping=0.0,
                      gravity=ti.Vector([0., 0.]),
                      alphaPIC=0.1,
                      mapping="USL",
                      shape_function="2DAxisyGIMP")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.5,
                           "SaveInterval":               0.1,
                           "SavePath":                   'Disk'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    800000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":    134474,
                                                               "max_absorbing_constraint":   134474
                                                          }
                            })
                            
mpm.add_material(model="NeoHookean",
                 material={
                               "MaterialID":           1,
                               "Density":              1500.,
                               "YoungModulus":         2300e6,
                               "PoissonRatio":         0.33,
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([2.5, 2.5]),
                             "Contact":  {
                                             "ContactDetection":                "GeoContact",
                                               "Friction":                        0.5,
                                               "CutOff":                          0.8,
                                               "Penalty":                         [0., 10.]
                                          }
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([50., 10.]),
                            
                      },

                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([50., 0.]),
                            "BoundingBoxSize": ti.Vector([5, 10.]),
                            
                      },
                      ])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "Traction":       [],
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                   },

                                   {
                                       "RegionName": "region2",
                                       "nParticlesPerCell": 2,
                                       "BodyID": 1,
                                       "RigidBody": True,
                                       "Density": 1500,
                                       "ParticleStress": {},
                                       "Traction": {},
                                       "InitialVelocity": ti.Vector([-1., 0.]),
                                       "FixVelocity": ["Fix", "Fix"]

                                   }
                                    ]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [50., 0.],
                                             "NLevel":         0
                                        },

                                        {
                                            "BoundaryType": "VelocityConstraint",
                                            "Velocity":        [0., 0.],
                                            "StartPoint":      [0., 10.],
                                            "EndPoint":        [50., 10.],
                                            "NLevel":           0
                                        },

                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 10.]
                                        },
                                        
                                        # {
                                        #      "BoundaryType":   "VelocityConstraint",
                                        #      "Velocity":       [-1., None],
                                        #      "StartPoint":     [50., 0.],
                                        #      "EndPoint":       [50., 10.]
                                        # },
                                    ])


mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing(read_path='Disk', write_background_grid=True)
