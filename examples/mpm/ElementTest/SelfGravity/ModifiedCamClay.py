import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([1., 1., 10]), 
                      background_damping=0., 
                      gravity=ti.Vector([0., 0., -9.8]),
                      alphaPIC=0.005, 
                      mapping="USL", 
                      shape_function="GIMP")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             3,
                           "SaveInterval":               0.1,
                           "SavePath":                   'ModifiedCamClay'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    17424,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     56840,
                                                               "max_particle_traction_constraint": 12000
                                                          }
                            })
                            
mpm.add_material(model="ModifiedCamClay",
                 material={
                                "MaterialID":               1,
                                "Density":                  1350,
                                "PoissionRatio":            0.3,
                                "StressRatio":              0.984,
                                "lambda":                   0.2,
                                "kappa":                    0.05,
                                "OverConsolidationRatio":   3,
                                "void_ratio_ref":           1.8,
                                "pressure_ref":             153
                          })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([1, 1, 1])
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0., 0.]),
                            "BoundingBoxSize": ti.Vector([1., 1., 10.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0., 9.5]),
                            "BoundingBoxSize": ti.Vector([1., 1., 0.3]),
                            
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-459., -459., -459., 0., 0., 0.])
                                                         },
                                       "Traction":       {
                                                             "Pressure":       ti.Vector([0, 0, -459]),
                                                             "RegionName":  "region2"
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [1., 1., 0.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None, None],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0., 1., 10.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None, None],
                                             "StartPoint":     [1., 0., 0.],
                                             "EndPoint":       [1., 1., 10.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0., None],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [1., 0., 10.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0., None],
                                             "StartPoint":     [0., 1., 0.],
                                             "EndPoint":       [1., 1., 10.]
                                        }
                                    ])

mpm.select_save_data()

mpm.run(gravity_field=True)
