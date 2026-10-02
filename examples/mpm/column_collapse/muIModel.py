import os
import sys

import math

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=2)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([2.0, 2.0, 0.3]), 
                      background_damping=0.05, 
                      alphaPIC=0.000, 
                      mapping="USL", 
                      shape_function="GIMP")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.8,
                           "SaveInterval":               0.02,
                           "SavePath":                   "muIModel"
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    804320,
                                "max_constraint_number":  {
                                                               "max_friction_constraint":   121203
                                                          }
                            })

mpm.add_material(model="GranularMaterial",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1500,
                               "YoungModulus":                  2e6,
                               "PoissionRatio":                 0.3,
                               "StaticFriction":                20.9,
                               "DynamicFriction":               20.9,
                               "AverageDiameter":               0.002,
                               "InertialNumber":                0.03,
                               "eps":                           0.01
                 })
                 
'''mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":           1,
                               "RateDependent":        True,
                               "Density":              2650.,
                               "YoungModulus":                  2e6,
                               "PossionRatio":                 0.3,
                               "StaticFriction":                20.9,
                               "DynamicFriction":               20.9,
                               "AverageDiameter":               0.002,
                               "InertialNumber":                0.03,
                               "eps":                           0.01
                 })'''

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.01, 0.01, 0.01])
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Cylinder",
                            "BoundingBoxPoint": ti.Vector([0.6, 0.6, 0.0]),
                            "BoundingBoxSize": ti.Vector([0.8, 0.8, 0.2]),
                            
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
                                             "BoundaryType":   "FrictionConstraint",
                                             "Norm":           [0., 0., 1.],
                                             "Friction":       0.3819,
                                             "StartPoint":     [0.0, 0.0, 0.0],
                                             "EndPoint":       [2.0, 2.0, 0.]
                                        }
                                    ])

mpm.select_save_data()

mpm.run(gravity_field=True)
