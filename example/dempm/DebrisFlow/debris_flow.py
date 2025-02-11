import numpy as np

from geotaichi import *

init(device_memory_GB=4.)

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([230.,260.,116.6]),
                        coupling_scheme="MPM",
                        particle_interaction=False,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.005,
                      mode="Lightweight",
                      alphaPIC=0.001, 
                      mapping="USF", 
                      shape_function="QuadBSpline",
                      gravity=ti.Vector([0., 0., -9.8]))

dempm.dem.set_configuration(domain=ti.Vector([230.,260.,116.6]),
                      gravity=ti.Vector([0.,0.,-9.8]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dempm.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   15.2,
                "SaveInterval":     0.4
               })
                      
dempm.dem.memory_allocate(memory={
                            "max_material_number": 1,
                            "max_particle_number": 0,
                            "max_sphere_number": 0,
                            "max_digital_elevation_facet_number": 119600,
                            "verlet_distance_multiplier": 0.4,
                            "body_coordination_number": 0,
                            "wall_coordination_number": 2,
                            "compaction_ratio": [0.25, 0.1]
                            }, log=True)  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           1559361,
                                "verlet_distance_multiplier":    0.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   121914,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    0,
                                  "wall_coordination_number":    2,
                                  "comptaction_ratio": [0.2, 0.15]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            26500,
                                "ForceLocalDamping":  0.15,
                                "TorqueLocalDamping": 0.05
                            })
                 
dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                         particle_wall_contact_model=None)

dempm.dem.add_wall(body={
                   "WallType":    "DigitalElevation",
                   "WallID":       1,
                   "MaterialID":   0,
                   "DigitalElevation":   np.flip(np.loadtxt("dc7_dem_1411slide.txt",  skiprows = 6)  - 1597.4, 0),
                   "CellSize":     1.,
                   "NoData":       -11596.4,
                   "Visualize":    True
                  })
                  
dempm.dem.select_save_data(particle=False)

dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":                    1,
                               "Density":                       2650,
                               "YoungModulus":                  1e5,
                               "PoissionRatio":                 0.3,
                               "Friction":                      22,
                               "Dilation":                      0.0,
                               "Cohesion":                      0.0,
                               "Tensile":                       0.0
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.5, 0.5, 0.5])
                        })

dempm.mpm.add_body_from_file(body={
                                    "FileType": "TXT",
                                    "Template":  {
                                                   "BodyID": 0,
                                                   "MaterialID": 1,
                                                   "ParticleFile": "SpherePacking.txt"
                                                  }
                   })
                   
dempm.mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., -1., 0.],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [230., 0., 116.6]
                                    }])

dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model=None,
                           particle_wall_contact_model="Linear Model")

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            1e6,
                                 "TangentialStiffness":        1e6,
                                 "Friction":                   0.5,
                                 "NormalViscousDamping":       0.5,
                                 "TangentialViscousDamping":   0.5
                            }, dType='particle-wall')

dempm.run()

dempm.mpm.postprocessing()
