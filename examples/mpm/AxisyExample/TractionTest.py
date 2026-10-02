# import sys
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([10, 11]),
                      is_2DAxisy=True,
                      background_damping=0.05,
                      gravity=ti.Vector([0., 0.]),
                      alphaPIC=0.06,
                      mapping="USL",
                      shape_function="GIMP")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.1,
                           "SaveInterval":               0.1,
                           "SavePath":                   'Bossinesq2DAxisyNoTruncation'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    800000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":    134474,
                                                               "max_absorbing_constraint":   134474
                                                          }
                            })
                            
mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              1500.,
                               "YoungModulus":         1e7,
                               "PoissonRatio":         0.30,
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([2, 2]),
                             "Contact":                    {}
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([10., 10.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 9.]),
                            "BoundingBoxSize": ti.Vector([10., 1.]),
                            
                      },
                      ])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-1e6, -1e6, -1e6, 0., 0., 0.])
                                                         },
                                       "Traction":       [{"Pressure": ti.Vector([0, -1e6]),
                                                           "RegionName": "region2"}],
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   },]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [10., 0.],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 11.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [10., 0.],
                                             "EndPoint":       [10., 11.]
                                        },
                                    ])


mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing(read_path='Bossinesq2DAxisyNoTruncation', write_background_grid=True)
