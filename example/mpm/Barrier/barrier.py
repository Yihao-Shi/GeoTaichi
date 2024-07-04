import sys
sys.path.append("/home/eleven/work/GeoTaichi_release")

from geotaichi import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([12, 0.3, 5.5]), 
                      background_damping=0.0, 
                      gravity=ti.Vector([ 0., 0., -9.8]),
                      alphaPIC=0.00, 
                      mapping="USF", 
                      shape_function="GIMP",
                      stabilize=None)

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":           3,
                           "SaveInterval":               0.1
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    2,
                                "max_particle_number":    29600,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     125000,
                                                               "max_reflection_constraint":   0
                                                          }
                            })
                            
mpm.add_material(model="SoilStructureInteraction",
                 material=[{
                               "IsStructure":          False,
                               "MaterialID":           1,
                               "Density":              2500.,
                               "YoungModulus":         2.e7,
                               "PossionRatio":        0.3,
                               "Cohesion":             0.,
                               "Friction":             30.,
                               "Dilation":             0.
                           },
                           {
                               "IsStructure":          True,
                               "MaterialID":           2,
                               "Density":              2500.,
                               "YoungModulus":         2.e7,
                               "PoissionRatio":        0.3
                           }])

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.1, 0.1, 0.1]),
                             "Contact":   {
                                               "ContactDetection":                True,
                                               "GridLevel":                    2,
                                               "Friction":                        0.577,
                                               "CutOff":                          0.8
                                          }
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.1, 0.1, 0.1]),
                            "BoundingBoxSize": ti.Vector([8., 0.1, 4.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([8.1, 0.1, 0.1]),
                            "BoundingBoxSize": ti.Vector([1., 0.1, 5.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     True,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "Traction":       {},
                                       "InitialVelocity":ti.Vector([0., 0., 0.]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "MaterialID":         2,
                                       "ParticleStress": {
                                                              "GravityField":     True,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "Traction":       {},
                                       "InitialVelocity":    ti.Vector([0., 0., 0.]),
                                       "FixVelocity":        ["Free", "Free", "Free"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [12, 0.3, 0.1]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None, None],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0.1, 0.3, 5.2]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0., None],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [12, 0.1, 5.2]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0., None],
                                             "StartPoint":     [0., 0.2, 0.],
                                             "EndPoint":       [12, 0.3, 5.2]
                                        }
                                    ])

mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing(start_file=0, end_file=31, write_background_grid=True)

