import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([4., 35., 35]), 
                      mode="Lightweight",
                      background_damping=0., 
                      gravity=ti.Vector([0., 0., -0.]),
                      alphaPIC=0.005, 
                      mapping="USL", 
                      shape_function="Linear",
                      gauss_number=2)

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             1,
                           "SaveInterval":               0.1,
                           "SavePath":                   'Elastic'
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    17424,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     15684,
                                                               "max_particle_traction_constraint":  12345
                                                          }
                            })
                            
mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              1500.,
                               "YoungModulus":         2.8e7,
                               "PoissonRatio":        0.35
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([1, 1, 1])
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([1., 1., 1.]),
                            "BoundingBoxSize": ti.Vector([2., 33., 33.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([1., 1., 33.5]),
                            "BoundingBoxSize": ti.Vector([2., 33., 0.3]),
                            
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "InternalStress":   ti.Vector([-1000., -1000., -1000., 0., 0., 0.])
                                                         },
                                       "Traction":       {
                                                             "Pressure":       ti.Vector([0, 0, -1000.]),
                                                             "RegionName":  "region2"
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [4., 35., 1.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None, None],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [1., 35., 35.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., None, None],
                                             "StartPoint":     [3., 0., 0.],
                                             "EndPoint":       [4., 35., 35.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0., None],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [4., 1., 35.]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [None, 0., None],
                                             "StartPoint":     [0., 34., 0.],
                                             "EndPoint":       [4., 35., 35.]
                                        }
                                    ])

mpm.select_save_data()

mpm.run()

mpm.postprocessing(read_path='Elastic',
                   write_path='Elastic'
                  )
