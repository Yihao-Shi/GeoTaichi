from geotaichi import *

init(device_memory_GB=6.9)

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([0.1, 0.1, 0.2]),
                        coupling_scheme="MPDEM",
                        particl_interaction=True,
                        wall_interaction=False)

dempm.mpm.set_configuration( 
                      background_damping=0.0,
                      alphaPIC=0.00,
                      mapping="USF", 
                      shape_function="QuadBSpline",
                      gravity=ti.Vector([0., 0., -9.8]),
                      material_type="Fluid",
                      velocity_projection="Taylor")

dempm.dem.set_configuration(
                      boundary=[None, None, None],
                      gravity=ti.Vector([0., 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      
dempm.set_solver({
                      "Timestep":         5e-6,
                      "SimulationTime":   0.35,
                      "SaveInterval":     0.007,
                      "SavePath":         'box'
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 1,
                                "levelset_grid_number": 328509,
                                "surface_node_number": 4322,
                                "max_plane_number": 0,
                                "body_coordination_number":   0,
                                "wall_coordination_number":   0,
                                "verlet_distance_multiplier": [0.15, 0.1],
                                "point_coordination_number":  [3, 2], 
                                "compaction_ratio":           [0.3, 0.3, 0.15, 0.15],
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           1300000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   25000,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            
dempm.memory_allocate(memory={    "body_coordination_number":    1,
                                  "wall_coordination_number":    0,
                                  "compaction_ratio": [0.002, 0.1]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2120,
                                "ForceLocalDamping":  0.0,
                                "TorqueLocalDamping": 0.0
                            })
                            
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            8500,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })

dempm.dem.add_template(template={
                                "Name":               "clump1",
                                "Object":              polyhedron(file='/home/eleven/work/GeoTaichi/assets/mesh/LSDEM/box.stl').grids(space=0.02, extent=9),
                                "WriteFile":          False}) 

dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.05, 0.05, 0.13],
                                  "ScaleFactor": 0.02,
                                  "BodyOrientation": "constant",
                                  "InitialVelocity": [0., 0., -0.]
                                  }
                                ]})

dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                               particle_wall_contact_model=None)        
                           

dempm.dem.select_save_data(surface=True)

dempm.mpm.add_material(model="Newtonian",
                 material={
                               "MaterialID":           1,
                               "Density":              996.51,
                               "Modulus":              3.6e5,
                               "Viscosity":            1e-3,
                               "ElementLength":        0.00,
                               "cL":                   0.1,
                               "cQ":                   2
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.002, 0.002, 0.002])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
                            "BoundingBoxSize": ti.Vector([0.1, 0.1, 0.13]),
                            
                      }])

dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
                   

dempm.mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., -1., 0.],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [0.1, 0.0, 0.2],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 1., 0.],
                                        "StartPoint":     [0, 0.1, 0],
                                        "EndPoint":       [0.1, 0.1, 0.2],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [-1., 0., 0.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0., 0.1, 0.2],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [1., 0., 0.],
                                        "StartPoint":     [0.1, 0.0, 0],
                                        "EndPoint":       [0.1, 0.1, 0.2],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., -1.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0.1, 0.1, 0.],
                                    }])


dempm.mpm.select_save_data()

dempm.add_body(check_overlap=True)

dempm.choose_contact_model(particle_particle_contact_model="Fluid Particle",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e4,
                                 "NormalViscousDamping":          0.
                            }, dType='particle-particle')

dempm.run(gravity_field=True)

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
