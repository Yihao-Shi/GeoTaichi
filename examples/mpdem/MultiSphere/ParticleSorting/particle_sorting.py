import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

dempm = DEMPM()

dempm.set_configuration(coupling_scheme="MPDEM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.dem.set_configuration(domain=ti.Vector([1.5, 0.1, 0.8]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0, 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell")
                      
dempm.mpm.set_configuration(domain=ti.Vector([1.5, 0.1, 0.8]), 
                      background_damping=0.05,
                      alphaPIC=0.00, 
                      mapping="USL", 
                      shape_function="GIMP",
                      gravity=ti.Vector([0., 0., -9.8]))

dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_particle_number": 5600,
                                "max_sphere_number": 5600,
                                "max_clump_number": 0,
                                "max_plane_number": 6,
                                "compaction_ratio": 0.8,
                                "verlet_distance_multiplier":  0.15,
                            })  

dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           450000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                           
dempm.memory_allocate(memory={
                                  "body_coordination_number":    40,
                                  "wall_coordination_number":    4,
                                  "verlet_distance_multiplier":  2.,
                                  "compaction_ratio": 1.0
                             })   
                            
dempm.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   80,
                      "SaveInterval":     0.1,
                      "SavePath":         'OutputData'
                 })
               

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.02,
                                "TorqueLocalDamping": 0.02
                            })
                            
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            8500,
                                "ForceLocalDamping":  0.02,
                                "TorqueLocalDamping": 0.02
                            })
                            
dempm.dem.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.005, 0.0, 0.5]),
                            "BoundingBoxSize": ti.Vector([0.1, 0.1, 0.2]),
                            
                      }])
                 
dempm.dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
dempm.dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            2e5,
                            "TangentialStiffness":        1.5e5,
                            "Friction":                   0.35,
                            "NormalViscousDamping":       0.05,
                            "TangentialViscousDamping":   0.05
                           })           
                           
dempm.dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            8e5,
                            "TangentialStiffness":        5.5e5,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.05,
                            "TangentialViscousDamping":   0.05
                           })                       
                           
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.25, 0.05, 0.25]),
                   "OuterNormal":  ti.Vector([1., 0., 1.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0., 0.05, 0.35]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([1.5, 0.05, 0.35]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.75, 0., 0.35]),
                   "OuterNormal":  ti.Vector([0., 1., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.75, 0.1, 0.35]),
                   "OuterNormal":  ti.Vector([0., -1., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.75, 0.05, 0.]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
dem.select_save_data(clump=True)


dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":                    1,
                               "Density":                       2650,
                               "YoungModulus":                  8e4,
                               "PoissionRatio":                 0.3,
                               "Friction":                      24,
                               "Dilation":                      0.0,
                               "Cohesion":                      0.0,
                               "Tensile":                       0.0
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.005, 0.005, 0.005])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.005, 0.0, 0.5]),
                            "BoundingBoxSize": ti.Vector([0.1, 0.1, 0.2]),
                            
                      }])

dempm.mpm.select_save_data()
  

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            5e3,
                                 "TangentialStiffness":        2.5e3,
                                 "Friction":                   0.35,
                                 "NormalViscousDamping":       0.05,
                                 "TangentialViscousDamping":   0.05
                            })
                            
dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            1e4,
                                 "TangentialStiffness":        7.5e3,
                                 "Friction":                   0.65,
                                 "NormalViscousDamping":       0.05,
                                 "TangentialViscousDamping":   0.05
                            })
                   
dempm.add_body(mpm_body={
                       "Period":     [0, 49, 8],
                       "Template": [
                                    {
                                       "RegionName":         "region1",
                                       "nParticlesPerCell":  2,
                                       "BodyID":             0,
                                       "MaterialID":         1,
                                       "InitialVelocity":ti.Vector([0, 0, 0]),
                                       "FixVelocity":    ["Free", "Free", "Free"]    
                                       
                                   }]
                           }, 
               dem_particle={
                   "GenerateType": "Generate",
                   "BodyType": "Sphere",
                   "RegionName": "region1",
                   "WriteFile":  False,
                   "Period":     [0, 49, 8],
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "MinRadius": 0.0055,
                               "MaxRadius": 0.0055,
                               "BodyNumber": 800,
                               "BodyOrientation": "uniform"}]}, 
               write_file=True, 
               check_overlap=True)    

dempm.run()

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
