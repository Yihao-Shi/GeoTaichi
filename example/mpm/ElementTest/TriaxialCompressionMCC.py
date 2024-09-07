try:
    from geotaichi import *
except:
    import os
    import sys
    current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(current_file_path)
    from geotaichi import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([5., 5., 2.6]), 
                      background_damping=0.1, 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.005, 
                      mapping="USF", 
                      shape_function="GIMP",
                      stabilize=None,
                      gauss_number=2)

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             0.1,
                           "SaveInterval":               0.01
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   51005
                                                          }
                            })

mpm.add_material(model="ModifiedCamClay",
                 material={
                               "MaterialID":                    1,
                               "Density":                       2000,
                               "PossionRatio":                  0.3,
                               "StressRatio":                   1.5,
                               "lambda":                        0.1,
                               "kappa":                         0.008,
                               "void_ratio_ref":                1.76,
                               "ConsolidationPressure":         1e5
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.05, 0.05, 0.05]),
                             "Contact":         {
                                                     "ContactDetection":                True,
                                                     "Friction":                        1,
                                                     "CutOff":                          1.6
                                                }
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2., 0.2]),
                            "BoundingBoxSize": ti.Vector([1., 1., 2.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([1.5, 1.5, 2.2]),
                            "BoundingBoxSize": ti.Vector([2., 2., 0.1]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region3",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2., 0.2]),
                            "BoundingBoxSize": ti.Vector([0.02, 1., 2.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region4",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2.98, 2., 0.2]),
                            "BoundingBoxSize": ti.Vector([0.02, 1., 2.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region5",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2., 0.2]),
                            "BoundingBoxSize": ti.Vector([1., 0.02, 2.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region6",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([2., 2.98, 0.2]),
                            "BoundingBoxSize": ti.Vector([1., 0.02, 2.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     False,
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
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":         True,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([-0, -0, -0, 0., 0., 0.])
                                                         },
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

mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing()

mpm.update_particle_properties(property_name='velocity', value=[0., 0., -0.04], bodyID=1)

mpm.modify_parameters(SimulationTime=15.1, SaveInterval=0.1)

mpm.run()

mpm.postprocessing()


