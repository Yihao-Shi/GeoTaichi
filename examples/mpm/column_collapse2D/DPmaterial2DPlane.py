import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, arch='cpu', device_memory_GB=2, debug=False)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([6, 2.25]),
                      is_2DAxisy=False,
                      #mode="Lightweight",
                      background_damping=0.0,
                      gravity=ti.Vector([0., -9.8]),
                      alphaPIC=0.010,
                      mapping="USF", 
                      shape_function="GIMP",
                      #velocity_projection="Taylor",
                      )

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             5,
                           "SaveInterval":               0.2,
                           "SavePath":                   'OutputData'
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
                               "Density":              2700.,
                               "YoungModulus":         20e6,
                               "PoissonRatio":         0.2,
                               "Cohesion":             0.,
                               "Friction":             33,
                               "Dilation":             0.,
                               "Tensile":              0.
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.025, 0.025]),
                        })

mpm.add_region(region={
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([2., 1.]),
                            
                      })

mpm.add_body(body={
                       "Template": {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0, 0]),
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   }
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [6., 0.]
                                        },

                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 2.25]
                                        },
                                    ])

mpm.select_save_data(grid=True)

mpm.run(gravity_field=True)

mpm.postprocessing(read_path='DP2DPlane', write_background_grid=True)
