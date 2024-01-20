from src import *

init()

dem = DEM()

dem.set_configuration(domain=ti.Vector([1.04, 0.14, 0.5]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., -9.8]),
                      engine="SymplecticEuler",
                      search="LinkedCell",
                      coupling=True)

dem.memory_allocate(memory={
                                "max_material_number": 1,
                                "max_particle_number": 1,
                                "max_sphere_number": 1,
                                "max_clump_number": 0,
                                "max_facet_number": 0,
                                "verlet_distance_multiplier": 0.2
                            })                   

dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            7850,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })

dem.create_body(body={
                   "BodyType": "Sphere",
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([0., 0., -1.12]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "BodyPoint": ti.Vector([0.52, 0.07, 0.3123]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"],
                               "Radius": 0.0223,
                               "BodyOrientation": "uniform"}]})
                          
dem.choose_contact_model(particle_particle_contact_model=None,
                         particle_wall_contact_model=None)
                            
                  
dem.select_save_data()

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([1.04, 0.14, 0.5]), 
                      background_damping=0.05,
                      alphaPIC=0.00005, 
                      mapping="USL", 
                      shape_function="GIMP",
                      gravity=ti.Vector([0., 0., -9.8]),
                      coupling=True,
                      free_surface_detection=False,
                      coupling=True)

mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           240000,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   61820,
                                                               "max_velocity_constraint":   4017
                                                          },
                                "verlet_distance_multiplier":  0.8
                            })

mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":                    1,
                               "Density":                       600,
                               "YoungModulus":                  6.1e7,
                               "PoissionRatio":                 0.2,
                               "Friction":                      16,
                               "Dilation":                      3.0,
                               "Cohesion":                      0.0,
                               "Tensile":                       0.0
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.01, 0.01, 0.01])
                        })


mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.02, 0.02, 0.02]),
                            "BoundingBoxSize": ti.Vector([1.0, 0.1, 0.27]),
                            "zdirection": ti.Vector([0., 0., 1.])
                      }])

mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "ParticleStress": {
                                                              "GravityField":     True,
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[{
                                        "BoundaryType":   "VelocityConstraint",
                                        "Velocity":       [0, 0, 0],
                                        "StartPoint":     [0, 0, 0],
                                        "EndPoint":       [1.02, 0.12, 0.02]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [-1, 0., 0],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [0.02, 0.14, 0.5]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [1, 0., 0],
                                        "StartPoint":     [1.02, 0, 0],
                                        "EndPoint":       [1.04, 0.14, 0.5]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [0, -1., 0],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [1.04, 0.02, 0.5]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [0, 1., 0],
                                        "StartPoint":     [0, 0.12, 0],
                                        "EndPoint":       [1.04, 0.14, 0.5]
                                    }])

mpm.select_save_data()
    

dempm = DEMPM(dem, mpm)

dempm.set_configuration(coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=False)

dempm.set_solver({
                      "Timestep":         1e-6,
                      "SimulationTime":   0.2,
                      "SaveInterval":     0.002,
                      "SavePath":         'OutputData/velz112'
                 })

dempm.memory_allocate(memory={
                                  "body_coordination_number":    50,
                                  "wall_coordination_number":    6
                             })

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            6.5e6,
                                 "TangentialStiffness":        3.3e6,
                                 "Friction":                   0.28,
                                 "NormalViscousDamping":       0.0,
                                 "TangentialViscousDamping":   0.0
                            })

dempm.run()

mpm.postprocessing()
