import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2)

mpm = MPM()

mpm.set_configuration(domain=[5., 5.], 
                      mode="Lightweight",
                      background_damping=0., 
                      gravity=[0., -10.],
                      alphaPIC=0.005, 
                      mapping="USL", 
                      shape_function="Linear")

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             1,
                           "SaveInterval":               0.1,
                           "SavePath":                   'DPmaterial'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    17424,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     15684,
                                                               "max_particle_traction_constraint":  12345
                                                          }
                            })
                            
mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":           1,
                               "Density":              2000.,
                               "YoungModulus":         100e6,
                               "PoissonRatio":         0.3,
                               "Cohesion":             6700.,
                               "Friction":             20.0,
                               "Dilation":             0.,
                               "Tensile":              0.
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               [0.1, 0.1]
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": [1., 2.],
                            "BoundingBoxSize": [1., 1.],
                            
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":[0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [0., 2.],
                                             "EndPoint":       [5., 2.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [1., 0.],
                                             "EndPoint":       [1., 5.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0.],
                                             "StartPoint":     [2., 0.],
                                             "EndPoint":       [2., 5.]
                                        }
                                    ])

mpm.select_save_data()

mpm.run(gravity_field=True)

mpm.postprocessing(read_path='DPmaterial',
                   write_path='DPmaterial'
                  )
