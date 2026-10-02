import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2,  device_memory_GB=4)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.5, 0.3]), 
                      background_damping=0., 
                      gravity=ti.Vector([0., 0.]),
                      alphaPIC=0., 
                      mapping="USL", 
                      shape_function="GIMP",
                      stress_integration="SubStepping")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.5,
                           "SaveInterval":               0.01,
                           "SavePath":                   "OedometerMCC2D"
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    3.5e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   5100,
                                                               "max_particle_traction_constraint":   1100,
                                                               "particle_traction_method": "Stable"
                                                          }
                            })

mpm.add_contact(contact_type="MPMContact", friction=0.0)

mpm.add_material(model="ModifiedCamClay",
                 material={
                               "MaterialID":                    1,
                               "Density":                       2000,
                               "PoissonRatio":                  0.3,
                               "StressRatio":                   1.02,
                               "lambda":                        0.1,
                               "kappa":                         0.008,
                               "void_ratio_ref":                1.76,
                               "OverConsolidationRatio":        1.,
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.02, 0.02])
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([0.5, 0.25]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.25]),
                            "BoundingBoxSize": ti.Vector([0.5, 0.05]),
                            
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-2000., -2000., -2000., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":[0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":         True,
                                       "InitialVelocity":[0, -0.01],
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0.5, 0.],
                                             "NLevel":         0
                                        },
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0, None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 0.3],
                                             "NLevel":         0
                                        },
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0, None],
                                             "StartPoint":     [0.5, 0.],
                                             "EndPoint":       [0.5, 0.3],
                                             "NLevel":         0
                                        }
                                    ])

#mpm.add_virtual_stress_field(field={"ConfiningPressure": [-100000., -100000., -100000., 0., 0., 0.],
#                                    "VirtualForce": [0., 0., 0.]})

mpm.select_save_data(grid=True)

mpm.run()

mpm.update_particle_properties(property_name='velocity', value=[0., 0.01], bodyID=1)

mpm.modify_parameters(SimulationTime=0.75, SaveInterval=0.01)

mpm.run()

mpm.postprocessing()

