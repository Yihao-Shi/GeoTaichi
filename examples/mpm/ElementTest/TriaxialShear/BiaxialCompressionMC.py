import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=4)

dx = 0.05

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([5., 5., 3.]), 
                      background_damping=0., 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.00, 
                      mapping="USF", 
                      shape_function="GIMP",
                      stress_integration="SubStepping",
                      stabilize=None)

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             0.1,
                           "SaveInterval":               0.01
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    3.5e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   181005,
                                                               "max_particle_traction_constraint": 60000
                                                          }
                            })
                            
mpm.add_contact(contact_type="MPMContact", friction=1.)

mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":      1,
                               "Density":         2000,
                               "YoungModulus":    1e6,
                               "PoissionRatio":   0.32,
                               "Friction":        1.,
                               "Cohesion":        4000,
                               "Dilation":        0,
                               "Tensile":         1000,
                                            })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([dx, dx, dx])
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2., 0.]),
                            "BoundingBoxSize": ti.Vector([1., 1., 2.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([1.5, 1.5, 2.]),
                            "BoundingBoxSize": ti.Vector([2., 2., 2*dx]),
                            
                      },
                      
                      {
                            "Name": "region3",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2., 0.]),
                            "BoundingBoxSize": ti.Vector([0.5*dx, 1., 2.]),
                            
                      },
                      
                      {
                            "Name": "region4",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([3.-0.5*dx, 2., 0.]),
                            "BoundingBoxSize": ti.Vector([0.5*dx, 1., 2.]),
                            
                      },
                      
                      {
                            "Name": "region5",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2., 0.]),
                            "BoundingBoxSize": ti.Vector([1., 0.5*dx, 2.]),
                            
                      },
                      
                      {
                            "Name": "region6",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 3.-0.5*dx, 0.]),
                            "BoundingBoxSize": ti.Vector([1., 0.5*dx, 2.]),
                            
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":         True,
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]    
                                       
                                   },
                                   {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-100000., -100000., -100000., 0., 0., 0.])
                                                         },
                                       "Traction":         [{
                                                                                    "Pressure": ti.Vector([100000, 0., 0.]),
                                                                                    "RegionName": "region3"
                                                                                   },
                                                                                  
                                                                                   {
                                                                                    "Pressure": ti.Vector([-100000, 0., 0.]),
                                                                                    "RegionName": "region4"
                                                                                   },
                                                                                   
                                                                                   {
                                                                                    "Pressure": ti.Vector([0., 100000, 0.]),
                                                                                    "RegionName": "region5"
                                                                                   },
                                                                                  
                                                                                   {
                                                                                    "Pressure": ti.Vector([0., -100000, 0.]),
                                                                                    "RegionName": "region6"
                                                                                   }],
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   ]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0, 0, 0],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [5., 5., 0.],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0, None],
                                             "StartPoint":     [0., 2., 0.],
                                             "EndPoint":       [5., 2., 3.],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0, None],
                                             "StartPoint":     [0., 3., 0.],
                                             "EndPoint":       [5., 3., 3.],
                                             "NLevel":         0
                                        }
                                    ])

mpm.select_save_data(grid=True)

mpm.run()

mpm.update_particle_properties(property_name='velocity', value=[0., 0., -0.02], bodyID=1)

mpm.modify_parameters(SimulationTime=15.1, SaveInterval=0.3)

mpm.run()

mpm.postprocessing()
