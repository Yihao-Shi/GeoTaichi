import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([30., 6.]),
                      background_damping=0., 
                      gravity=ti.Vector([4.9, -8.487048957]),
                      alphaPIC=0.00, 
                      mapping="USF", 
                      shape_function="Linear",
                      stabilize="B-Bar Method")

mpm.set_solver(solver={
                           "Timestep":                   1e-3,
                           "SimulationTime":             0.1,
                           "SaveInterval":               0.1
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5
                            })

mpm.add_contact(contact_type="GeoContact", friction=0.2, cutoff=0.8, penalty=[1., 2.])

mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              2000.,
                               "YoungModulus":         2e7,
                               "PoissonRatio":         0.33
                 })

mpm.add_element(element={
                             "ElementType":               "Q4N2D",
                             "ElementSize":               ti.Vector([0.5, 0.5])
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([1., 1.]),
                            "BoundingBoxSize": ti.Vector([1., 1.]),
                            
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle2D",
                            "BoundingBoxPoint": ti.Vector([0., 0.]),
                            "BoundingBoxSize": ti.Vector([30., 1.]),
                            
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "RigidBody":         True,
                                       "InitialVelocity":ti.Vector([0, 0]),
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition()

mpm.select_save_data(grid=True)

mpm.run()

mpm.postprocessing(write_background_grid=True)
