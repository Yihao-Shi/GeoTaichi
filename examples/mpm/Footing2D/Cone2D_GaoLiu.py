import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([4, 6.1]),
                      background_damping=0.0,
                      gravity=ti.Vector([0., -9.8]),
                      alphaPIC=0.1,
                      mapping="USL",
                      shape_function="GIMP",)

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             12,
                           "SaveInterval":               0.5,
                           "SavePath":                   'Cone2D_GaoLiu'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    800000,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     134474,
                                                               "max_reflection_constraint":   0
                                                          }
                            })

mpm.add_contact(contact_type="MPMContact", friction=0.4)

mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":           1,
                               "Density":              1200.,
                               "YoungModulus":         11e6,
                               "PoissonRatio":         0.38,
                               "Cohesion":             10000,
                               "Friction":             15.,
                               "Dilation":             3.
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.1, 0.1])
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([4., 4.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Cone2D",
                            "BoundingBoxPoint": ti.Vector([0., 4.]),
                            "BoundingBoxSize": ti.Vector([0.5, 1.]),
                            
                      },
                      ])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  4,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "Traction":       {},
                                       "InitialVelocity":ti.Vector([0., 0.]),
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  4,
                                       "BodyID":             1,
                                       "RigidBody":          True,
                                       "Density":            1500,
                                       "ParticleStress":     {},
                                       "Traction":           {},
                                       "InitialVelocity":    ti.Vector([0., -0.05]),
                                       "FixVelocity":        ["Fix", "Fix"]
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [4., 0.],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 6.1]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [4., 0.],
                                             "EndPoint":       [4., 6.1]
                                        },
                                    ])


mpm.select_save_data(grid=True)

mpm.run(gravity_field=True)

mpm.postprocessing(read_path='Cone2D_GaoLiu', write_background_grid=True)
