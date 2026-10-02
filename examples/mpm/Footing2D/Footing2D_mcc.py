import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([12, 12.2]),
                      dimension="2-Dimension",
                      background_damping=0.0,
                      gravity=ti.Vector([0., -0]),
                      alphaPIC=0.005,
                      mapping="USF",
                      shape_function="GIMP",
                      stress_integration="SubStepping",
                      #stabilize="B-Bar Method"
                      )

mpm.set_solver(solver={
                           "Timestep":                   5e-4,
                           "SimulationTime":             20,
                           "SaveInterval":               0.4,
                           "SavePath":                   '1_Footing_mcc_1'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    800000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     134474,
                                                               "max_reflection_constraint":   0,
                                                               "max_particle_traction_constraint": 10000
                                                          }
                            })

mpm.add_material(model="ModifiedCamClay",
                 material={
                               "MaterialID":              1,
                               "SolidDensity":            2670.,
                               "FluidDensity":            1000.,
                               "Porosity":                0.45,
                               "FluidBulkModulus":        2.2e9,
                               "Permeability":            1e-10,
                               "PoissonRatio":            0.3,
                               "StressRatio":             1.0,
                               "lambda":                  0.1,
                               "kappa":                   0.01,
                               "void_ratio_ref":          0.8182,
                               "OverConsolidationRatio":  1.5
                 })

mpm.add_contact(contact_type="MPMContact", friction=0.1)     

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.1, 0.1])
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([12., 10.]),
                            "ydirection": ti.Vector([0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 10.]),
                            "BoundingBoxSize": ti.Vector([1., 2.0]),
                            "ydirection": ti.Vector([0., 1.])
                      },

                      {
                            "Name": "region3",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([1., 9.95]),
                            "BoundingBoxSize": ti.Vector([11., 0.05]),
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
                                                              "InternalStress":   ti.Vector([-10.e3, -10.e3, -10.e3, 0., 0., 0.])
                                                         },
                                       "Traction":       [{"Pressure": ti.Vector([0, -10.e3]),
                                                           "RegionName": "region3"}],
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
                                             "EndPoint":       [12., 0.],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 12.2]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [12., 0.],
                                             "EndPoint":       [12., 12.2]
                                        },
                                    ])


mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing(read_path='1_Footing_mcc_1', write_background_grid=True)
