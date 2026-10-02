import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=5., debug=False)

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([10., 2., 5.]),
                        coupling_scheme="MPDEM",
                        particl_interaction=True,
                        wall_interaction=False)

dempm.mpm.set_configuration( 
                      background_damping=0.0,
                      alphaPIC=0.00,
                      mapping="USL", 
                      shape_function="GIMP",
                      gravity=[0., 0., -0.])

dempm.dem.set_configuration(
                      boundary=["Destroy", "Destroy", "Destroy"],
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM",
                      gravity=[0., 0., -6.9296])
                      
dempm.set_solver({
                      "Timestep":         5e-5,
                      "SimulationTime":   3.,
                      "SaveInterval":     0.1,
                      "SavePath":         "DEMslipping"
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 1,
                                "levelset_grid_number": 166379,
                                "surface_node_number": 126360,
                                "max_plane_number": 0,
                                "body_coordination_number":   0,
                                "wall_coordination_number":   0,
                                "verlet_distance_multiplier": [0., 0.],
                                "point_coordination_number":  [3, 2], 
                                "compaction_ratio":           [1., 1., 1., 1.],
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           500000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   80000,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            
dempm.memory_allocate(memory={    
                                  "body_coordination_number":    1,
                                  "wall_coordination_number":    0,
                                  "compaction_ratio": [1., 1.]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2120,
                                "ForceLocalDamping":  0.5,
                                "TorqueLocalDamping": 0.0
                            })

                                
dempm.dem.add_template(template={
                                "Name":               "clump1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/box.stl').grids(space=0.05, extent=5),
                                "WriteFile":          True}) 

dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1., 1., 1.5],
                                  "ScaleFactor": 1.,
                                  "BodyOrientation": "constant",
                                  "InitialVelocity": [0., 0., -0.],
                                  "FixMotion":    ["Free", "Free", "Free"]     
                                  }
                                ]})

dempm.dem.choose_contact_model(particle_particle_contact_model=None,
                               particle_wall_contact_model=None)        
                           

dempm.dem.select_save_data(surface=True)

dempm.mpm.add_material(model="LinearElastic",
                 material={
                               "MaterialID":           1,
                               "Density":              2000,
                               "YoungModulus":         2e8,
                               "PoissonRatio":         0.3
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               [0.2, 0.2, 0.2]
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": [0., 0., 0.],
                            "BoundingBoxSize": [10., 2., 1.],
                            
                      }])

dempm.mpm.add_body(body={
                       "Template": [{
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":[0, 0, 0],
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                   })
                   
dempm.mpm.add_boundary_condition(boundary=[
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., -1., 0.],
                                        "StartPoint":     [0., 0., 0.],
                                        "EndPoint":       [10., 0.0, 5.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 1., 0.],
                                        "StartPoint":     [0, 2., 0],
                                        "EndPoint":       [10., 2., 5.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [-1., 0., 0.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [0., 2., 5.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [1., 0., 0.],
                                        "StartPoint":     [10., 0.0, 0],
                                        "EndPoint":       [10., 2., 5.],
                                    },
                                    
                                    {
                                        "BoundaryType":   "ReflectionConstraint",
                                        "Norm":       [0., 0., -1.],
                                        "StartPoint":     [0, 0.0, 0],
                                        "EndPoint":       [10., 2., 0.],
                                    }])
                   
dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e8,
                                 "TangentialStiffness":           1e8,
                                 "Friction":                      0.5, 
                                 "NormalViscousDamping":          0.,
                                 "TangentialViscousDamping":      0.
                            }, dType='particle-particle')

dempm.run()

dempm.dem.sims.set_gravity([6.9296, 0., -6.9296])

dempm.dem.update_material_properties(materialID=0, property_name="ForceLocalDamping", value=0.)

dempm.modify_parameters(SimulationTime=5.)

dempm.run()

dempm.mpm.postprocessing(write_contact_force=True)

dempm.dem.postprocessing()
