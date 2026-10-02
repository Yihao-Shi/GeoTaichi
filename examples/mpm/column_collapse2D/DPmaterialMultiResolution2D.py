import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, debug=False, device_memory_GB=2)

file_path = 'MultiResolution/SameResolution'

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([50.0, 20.0]),
                      is_2DAxisy=False,
                      #mode="Lightweight",
                      background_damping=0.0,
                      gravity=ti.Vector([0., -10.0]),
                      alphaPIC=0.2,
                      mapping="USF", 
                      shape_function="GIMP",
                      velocity_projection="Taylor",
                      )

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             5,
                           "SaveInterval":               0.2,
                           "SavePath":                   file_path
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":   83000,
                                                               "max_friction_constraint": 83000
                                                          }
                            })

mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":           1,
                               "Density":              2650.,
                               "YoungModulus":         8.4e5,
                               "PoissonRatio":         0.3,
                               "Cohesion":             0.,
                               "Friction":             19.8,
                               "Dilation":             0.,
                               "Tensile":              0.
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.2, 0.2]),
                             "Contact":   {}
                        })
                        
mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": [0., 0.],
                            "BoundingBoxSize": [20., 5.],
                            
                      },
                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": [0., 5.],
                            "BoundingBoxSize": [20., 5.],
                            
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  4,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":    [0, 0],
                                       "FixVelocity":        ["Free", "Free"]
                                       
                                   },
                                   {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  4,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":[0, 0],
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [50.0, 0.]
                                        },

                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 20.0]
                                        },

                                        {
                                            "BoundaryType":    "VelocityConstraint",
                                            "Velocity":        [0., None],
                                            "StartPoint":      [50., 0.],
                                            "EndPoint":        [50., 20.0]
                                        },

                                    ])

mpm.select_save_data(grid=False)

mpm.run(gravity_field=lambda points: 10. - points[:,1])

mpm.postprocessing(read_path=file_path)
