import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.045, 0.045, 0.06]), 
                      background_damping=0.02, 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.005, 
                      mapping="USF", 
                      shape_function="Linear")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.2,
                           "SaveInterval":               0.002
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   396
                                                          }
                            })

mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":                   1,
                               "SoftType":                     "Linear",
                               "Density":                      2650.,
                               "YoungModulus":                 1e7,
                               "PoissonRatio":                0.3,
                               "Softening":                    True,
                               "Cohesion":                     1000,
                               "Friction":                     0.,
                               "Dilation":                     0.,
                               "ResidualCohesion":             100.,
                               "ResidualFriction":             0.,
                               "ResidualDilation":             0.,
                               "PlasticDevStrain":             0.,
                               "ResidualPlasticDevStrain":     0.001,
                               "Tensile":                      0.
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.005, 0.005, 0.005])
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.01, 0.01, 0.]),
                            "BoundingBoxSize": ti.Vector([0.025, 0.025, 0.05]),
                            
                      })

mpm.add_body(body={
                       "Template": {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0.045, 0.045, 0.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., -0.0005],
                                             "StartPoint":     [0.01, 0.01, 0.05],
                                             "EndPoint":       [0.035, 0.035, 0.05]
                                        }
                                    ])

mpm.select_save_data()

mpm.add_solver()



