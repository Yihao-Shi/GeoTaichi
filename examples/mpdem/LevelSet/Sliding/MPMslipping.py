import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=2.)

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([10., 5., 5.]),
                        coupling_scheme="MPDEM",
                        particl_interaction=True,
                        wall_interaction=False)

dempm.mpm.set_configuration( 
                      background_damping=0.0,
                      alphaPIC=0.00,
                      mapping="USL", 
                      shape_function="GIMP",
                      gravity=[0., 0., -6.9296])

dempm.dem.set_configuration(
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=[0., 0., -6.9296],
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      
dempm.set_solver({
                      "Timestep":         5e-5,
                      "SimulationTime":   3.,
                      "SaveInterval":     0.1,
                      "SavePath":         "MPMslipping"
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 1,
                                "levelset_grid_number": 155379,
                                "surface_node_number": 126360,
                                "max_plane_number": 0,
                                "body_coordination_number":   0,
                                "wall_coordination_number":   0,
                                "verlet_distance_multiplier": [0.15, 0.1],
                                "point_coordination_number":  [3, 2], 
                                "compaction_ratio":           [1., 1., 1., 1.],
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           64000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
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
                                "ForceLocalDamping":  0.0,
                                "TorqueLocalDamping": 0.0
                            })

                                
dempm.dem.add_template(template={
                                "Name":               "clump1",
                                "Object":             box((10.,5.,1.)).grids(space=0.5, extent=4).reset(False),
                                "WriteFile":          True}) 

dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [5., 2.5, 0.5],
                                  "ScaleFactor": 1.,
                                  "BodyOrientation": "constant",
                                  "InitialVelocity": [0., 0., -0.],
                                  "FixMotion":    ["Fix", "Fix", "Fix"]     
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
                             "ElementSize":               [0.05, 0.05, 0.05]
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": [0.5, 2., 1.],
                            "BoundingBoxSize": [1., 1., 1.],
                            
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
                   
dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model=None)

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":               1e4,
                                 "TangentialStiffness":           1e4,
                                 "Friction":                      0.5, 
                                 "NormalViscousDamping":          0.2,
                                 "TangentialViscousDamping":      0.0
                            }, dType='particle-particle')

dempm.run()

dempm.mpm.sims.set_gravity([6.9296, 0., -6.9296])

dempm.modify_parameters(SimulationTime=5.)

dempm.run()

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
