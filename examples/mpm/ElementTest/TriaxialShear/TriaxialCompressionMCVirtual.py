import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='gpu')

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([5., 5., 2.6]), 
                      background_damping=0., 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.00, 
                      mapping="USF", 
                      shape_function="GIMP",
                      stabilize=None,
                      stress_integration="SubStepping",
                      particle_traction_method="Virtual"
                      )

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.1,
                           "SaveInterval":               0.01
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   306030
                                                          }
                            })

mpm.add_contact(contact_type="MPMContact", friction=1.0)

mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":      1,
                               "Density":         1530,
                               "YoungModulus":    3e7,
                               "PoissionRatio":   0.3,
                               "Friction":        30.5,
                               "Cohesion":        8500,
                                            })

mpm.add_element(element={"ElementType": "R8N3D", "ElementSize": ti.Vector([0.05, 0.05, 0.05])})

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2., 0.2]),
                            "BoundingBoxSize": ti.Vector([1., 1., 2.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([1.5, 1.5, 2.2]),
                            "BoundingBoxSize": ti.Vector([2., 2., 0.1]),
                            
                      },
                      
                      {
                            "Name": "region3",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2., 0.2]),
                            "BoundingBoxSize": ti.Vector([0.02, 1., 2.]),
                            
                      },
                      
                      {
                            "Name": "region4",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2.98, 2., 0.2]),
                            "BoundingBoxSize": ti.Vector([0.02, 1., 2.]),
                            
                      },
                      
                      {
                            "Name": "region5",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2., 0.2]),
                            "BoundingBoxSize": ti.Vector([1., 0.02, 2.]),
                            
                      },
                      
                      {
                            "Name": "region6",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2.98, 0.2]),
                            "BoundingBoxSize": ti.Vector([1., 0.02, 2.]),
                            
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-100000., -100000., -100000., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":         True,
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0, 0, 0],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [5., 5., 0.2],
                                             "NLevel":         0
                                        }
                                    ])

mpm.add_virtual_stress_field(field={"ConfiningPressure": [-100000., -100000., -100000., 0., 0., 0.],
                                    "VirtualForce": [0., 0., 0.]})

mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing()

mpm.update_particle_properties(property_name='velocity', value=[0., 0., -0.04], bodyID=1)

mpm.modify_parameters(SimulationTime=15.1, SaveInterval=0.1)

mpm.run()

mpm.postprocessing()


