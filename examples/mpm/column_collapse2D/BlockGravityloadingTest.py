import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2, debug=False)

mpm = MPM()

mpm.set_configuration(domain=[15.0, 17.0],
                      background_damping=0.7,
                      gravity=[0., -10.0],
                      mapping="USF",
                      shape_function="CubicBSpline")

mpm.set_solver(solver={
                           "Timestep":                   2.0e-5,
                           "SimulationTime":             6.0,
                           "SaveInterval":               0.3,
                           "SavePath":                   '1_BlockGravityloading'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    14272,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":    134474
                                                          }
                            })
                            
mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":             1685.,
                               "YoungModulus":         2.8e7,
                               "PoissonRatio":         0.35,
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               [1., 1.]
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": [0., 0.],
                            "BoundingBoxSize": [15.0, 15.0],
                            
                         }])


mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":[0., 0.],
                                       "FixVelocity":    ["Free", "Free"]
                                       
                                   },]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [15., 0.],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [0., 0.],
                                             "EndPoint":       [0., 15.],
                                             
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None],
                                             "StartPoint":     [15., 0.],
                                             "EndPoint":       [15., 15.],
                                        },
                                    ])


mpm.select_save_data(grid=True)

mpm.run()
