from geotaichi import *

init()

dem = DEM()

dem.set_configuration(domain=ti.Vector([10., 6., 5.]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
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
                                "Density":            2650,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })

dem.add_body(body={
                   "GenerateType": "Create",
                   "BodyType": "Sphere",
                   "PoissionSampling": False,
                   "TryNumber": 100,
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "InitialVelocity": ti.Vector([2., 0., 0.]),
                               "InitialAngularVelocity": ti.Vector([0., 0., 0.]),
                               "BodyPoint": ti.Vector([4.4, 3.0, 3.0]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"],
                               "Radius": 0.5,
                               "BodyOrientation": "uniform"}]})

                          
dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e8,
                            "TangentialStiffness":        1e8,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.2,
                            "TangentialViscousDamping":   0.
                           })
 
                  
dem.select_save_data(particle_particle_contact=True)


from src.mpm.mainMPM import MPM

mpm = MPM()

mpm.set_configuration(domain=ti.Vector([10., 6., 5.]), 
                      background_damping=0., 
                      alphaPIC=0.005, 
                      mapping="USF", 
                      shape_function="Linear",
                      gravity=ti.Vector([0., 0., 0.]),
                      coupling=True,
                      free_surface_detection=False,
                      coupling=True)

mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           33552,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   61820
                                                          }
                            })

mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              2650.,
                               "YoungModulus":         1e10,
                               "PoissionRatio":        0.3
                 })

mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.1, 0.1, 0.1]),
                             "Contact":                   {
                                                               "ContactDetection":                False
                                                          } 
                        })

mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Spheroid",
                            "BoundingBoxPoint": ti.Vector([5.1, 2.5, 2.5]),
                            "BoundingBoxSize": ti.Vector([1., 1., 1.]),
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
                                                              "InternalStress":   ti.Vector([-0., -0., -0., 0., 0., 0.])
                                                         },
                                       "InitialVelocity":ti.Vector([-2, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })

mpm.add_boundary_condition(boundary=[{
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [0., 0., -1],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [10., 6., 0.]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [-1, 0., 0],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [0., 2., 0.]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [1, 0., 0],
                                        "StartPoint":     [6., 0., 0.],
                                        "EndPoint":       [6., 2., 0.]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [0, -1., 0],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [6., 0., 0.]
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":           [0, 1., 0],
                                        "StartPoint":     [0., 2., 0.],
                                        "EndPoint":       [6., 2., 0.]
                                    }])

mpm.select_save_data()
    

from src.mpdem.mainDEMPM import DEMPM

dempm = DEMPM(dem, mpm)

dempm.set_configuration(coupling_scheme="DEM-MPM",
                        particle_interaction=True,
                        wall_interaction=False)

dempm.set_solver({
                      "Timestep":         1e-6,
                      "SimulationTime":   0.1,
                      "SaveInterval":     0.01
                 })

dempm.memory_allocate(memory={
                                  "max_material_number":         1,
                                  "body_coordination_number":    2500,
                                  "wall_coordination_number":    6,
                             })

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(materialID1=0,
                   materialID2=0,
                   property={
                                 "NormalStiffness":            1e8,
                                 "TangentialStiffness":        1e8,
                                 "Friction":                   0.5,
                                 "NormalViscousDamping":       0.0,
                                 "TangentialViscousDamping":   0.
                            })

dempm.run()

print(dem.scene.particle[0].v)

mpm.postprocessing()

#ti.profiler.print_kernel_profiler_info()
