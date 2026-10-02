from geotaichi import *

init(device_memory_GB=3)

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([10.2, 0.3, 12.2]), 
                      background_damping=0.0, 
                      gravity=ti.Vector([0., 0., 0.]),
                      alphaPIC=0.005, 
                      mapping="USL", 
                      shape_function="GIMP",
                      stabilize="B-Bar Method")

mpm.set_solver(solver={
                           "Timestep":                   1e-5,
                           "SimulationTime":             5,
                           "SaveInterval":               0.25
                      })

mpm.memory_allocate(memory={
                                "max_material_number":    1,
                                "max_particle_number":    652800,
                                "max_constraint_number":  {
                                                               "max_velocity_constraint":     106904
                                                          }
                            })

mpm.add_contact(contact_type="GeoContact", friction=0.1, cutoff=0.8, penalty=[0.78, 2.5])

mpm.add_material(model="MohrCoulomb",
                 material={
                               "MaterialID":           1,
                               "Density":              1000.,
                               "YoungModulus":         1e5,
                               "PossionRatio":         0.49,
                               "Cohesion":             1000,
                               "Friction":             0.,
                               "Dilation":             0.
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.1, 0.1, 0.1])
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.1, 0.1, 0.1]),
                            "BoundingBoxSize": ti.Vector([10., 0.1, 10.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      },
                      
                      {
                            "Name": "region2",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.1, 0.1, 10.1]),
                            "BoundingBoxSize": ti.Vector([1.0, 0.1, 2.]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  4,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     False,
                                                              "InternalStress":   ti.Vector([0., 0., 0., 0., 0., 0.])
                                                         },
                                       "Traction":       [],
                                       "InitialVelocity":ti.Vector([0., 0., 0.]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   },
                                   
                                   {
                                       "RegionName":         "region2",
                                       "nParticlesPerCell":  4,
                                       "BodyID":             1,
                                       "RigidBody":          True,
                                       "Density":            1500,
                                       "ParticleStress":     {},
                                       "Traction":           {},
                                       "InitialVelocity":    ti.Vector([0., 0., -0.02]),
                                       "FixVelocity":        ["Fix", "Fix", "Fix"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":       [0., 0., 0.],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [10.2, 0.3, 0.1],
                                             "NLevel":         0
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":           [0., None, None],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [0.1, 0.3, 12.2]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":           [0., None, None],
                                             "StartPoint":     [10.1, 0., 0.],
                                             "EndPoint":       [10.2, 0.3, 12.2]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":           [None, 0., None],
                                             "StartPoint":     [0., 0., 0.],
                                             "EndPoint":       [10.2, 0.1, 12.2]
                                        },
                                        
                                        {
                                             "BoundaryType":   "VelocityConstraint",
                                             "Velocity":           [None, 0., None],
                                             "StartPoint":     [0., 0.2, 0.],
                                             "EndPoint":       [10.2, 0.3, 12.2]
                                        }
                                    ])

def stepwise():
    ramp = mpm.sims.time
    vel0 = 0.1
    deltat = mpm.sims.CurrentTime[None]
    vel = 0.
    for i in range(mpm.scene.particleNum[0]):
        if mpm.scene.particle[i].bodyID==1:
            mpm.scene.particle[i].v=[0., 0., -(deltat / ramp) * vel0]
    mpm.sims.CurrentTime[None] += mpm.sims.dt[None]

mpm.select_save_data(grid=True)

mpm.run(function=stepwise)

mpm.modify_parameters(SimulationTime=17.5, SaveInterval=0.25)

mpm.run()

mpm.postprocessing(write_background_grid=True, read_path='FootingLargeStrain')
