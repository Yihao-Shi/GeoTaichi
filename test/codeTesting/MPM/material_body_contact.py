import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, default_ip=ti.i32, debug=False)

from src.mpm.mainMPM import MPM

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([10., 10., 10.]), 
                      background_damping=0.02, 
                      alphaPIC=0.005, 
                      mapping="USL", 
                      shape_function="Linear")

mpm.set_solver(solver={
                           "Timestep":                   1e-4,
                           "SimulationTime":             6,
                           "SaveInterval":               0.1
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    5.12e5
                            })

mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           0,
                               "Density":              2650.,
                               "YoungModulus":         7e5,
                               "PoissionRatio":        0.3
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([1., 1., 1.])
                        })

mpm.add_grid(grid={
                       "ContactDetection":                True
                  })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([4., 4., 4.]),
                            "BoundingBoxSize": ti.Vector([2., 2., 2.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0., 0., 0.]),
                            "BoundingBoxSize": ti.Vector([10., 10., 2.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         0,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.]),
                                                              "Traction":         {}
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             1,
                                       "MaterialID":         -1,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.]),
                                                              "Traction":         {}
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition()

mpm.select_save_data()

mpm.add_solver()
