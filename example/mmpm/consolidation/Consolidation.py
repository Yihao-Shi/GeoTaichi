from geotaichi import *

init(dim=2)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([0.2, 1.1]),
                      background_damping=0.0,
                      gravity=ti.Vector([0., 0.]),
                      alphaPIC=0.000,
                      mapping="USF",
                      shape_function="GIMP",
                      #stabilize="B-Bar Method",
                      material_type="TwoPhaseSingleLayer")

mpm.set_solver(solver={
                           "Timestep":                   1.0e-5,
                           "SimulationTime":             1.456,
                           "SaveInterval":               0.0728,
                           "SavePath":                   '1_Consolidation'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    10000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":    134474,
                                                               "max_absorbing_constraint":   134474,
                                                               "max_particle_traction_constraint":   10000
                                                          }
                            })
                            
mpm.add_material(model="LinearElastic",   # fluid
                 material={
                               "MaterialID":           1,
                               #"Density":             1500.,
                               "SolidDensity":             2670.,
                               "FluidDensity":             1000.,
                               "Porosity":             0.40,
                               "FluidBulkModulus":                2.2e8,
                               "Permeability":         1e-3,
                               "YoungModulus":         1e7,
                               "PossionRatio":         0.30,
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.05, 0.05]),
                             "Contact":                    {}
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([0.2, 1.]),
                            "ydirection": ti.Vector([0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.975]),
                            "BoundingBoxSize": ti.Vector([0.2, 0.025]),
                            "ydirection": ti.Vector([0., 1.])
                      },
                      ])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([-0, -0, -0, 0., 0., 0.]),
                                                              "PorePressure":     1e4
                                                         },
                                       "Traction":       [{"Pressure": ti.Vector([0, -1e4]),
                                                           "FluidPressure": ti.Vector([0, 0.]),
                                                           "RegionName": "region2"}],
                                        # Traction f
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   },]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0.2, 0.],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 1.1]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0.2, 0.],
                                             "EndPoint":       [0.2, 1.1]
                                        },
                                    ])


mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing(read_path='1_Consolidation', write_background_grid=True)
