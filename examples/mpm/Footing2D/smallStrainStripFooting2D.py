import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, device_memory_GB=3)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([12.2, 12.2, 0]),
                      background_damping=0.05, 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.06,
                      mapping="USL", 
                      shape_function="GIMP",
                      stabilize="B-Bar Method")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.4,
                           "SaveInterval":               0.01
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
                               "Density":              1500.,
                               "YoungModulus":         28e6,
                               "PoissonRatio":        0.35,
                               "Cohesion":             3000,
                               "Friction":             32.,
                               "Dilation":             9.
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.1, 0.1]),
                             "Contact":   {
                                               "ContactDetection":                "GeoContact",
                                               "Friction":                        0.5,
                                               "CutOff":                          0.8,
                                               "Penalty":                         [1.,10.]
                                          }
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.1, 0.1, 0.]),
                            "BoundingBoxSize": ti.Vector([12., 8, 0]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.1, 8.1, 0]),
                            "BoundingBoxSize": ti.Vector([1., 4, 0.]),
                            
                      },
                      ])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
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
                                             "EndPoint":       [12.2, 0.1],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0.1, 12.2]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None,],
                                             "StartPoint":     [12.1, 0.],
                                             "EndPoint":       [12.2, 12.2]
                                        },
                                        #
                                        # {
                                        #      "BoundaryType":   "VelocityConstraint",
                                        #      "Velocity":       [None, 0., None],
                                        #      "StartPoint":     [0., 0., 0.],
                                        #      "EndPoint":       [12.2, 0.1, 12.2]
                                        # },
                                        #
                                        # {
                                        #      "BoundaryType":   "VelocityConstraint",
                                        #      "Velocity":       [None, 0., None],
                                        #      "StartPoint":     [0., 1.1, 0.],
                                        #      "EndPoint":       [12.2, 1.2, 12.2]
                                        # }
                                    ])


mpm.select_save_data()

mpm.run()

mpm.postprocessing(write_background_grid=False)
