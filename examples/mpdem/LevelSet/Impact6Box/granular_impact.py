import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-f', type=int, default=0)
args = parser.parse_args()

start_file = args.f
restart = False if start_file == 0 else True

init(device_memory_GB=4)

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([0.5, 0.1, 0.3]),
                        coupling_scheme="MPDEM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.001,
                      alphaPIC=0.001, 
                      mapping="USL", 
                      shape_function="QuadBSpline",
                      gravity=[0., 0., -9.8])

dempm.dem.set_configuration(
                      #boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., -9.8]),
                      search="LinkedCell",
                      scheme="LSDEM",
                      engine="VelocityVerlet"
                      )
                      

dempm.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   0.401,
                      "SaveInterval":     0.02,
                      "SavePath":         'OutputData'
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 6,
                                "levelset_grid_number": 205379,
                                "surface_node_number": 4322,
                                "max_plane_number": 6,
                                "body_coordination_number":   6,
                                "wall_coordination_number":   3,
                                "verlet_distance_multiplier": [0.15, 0.1],
                                "point_coordination_number":  [3, 2], 
                                "compaction_ratio":           [0.3, 0.3, 0.15, 0.15],
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           200000,
                                "verlet_distance_multiplier":    1.,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   12322,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   0
                                                          }
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    6,
                                  "wall_coordination_number":    3,
                                  "compaction_ratio": [0.2, 0.15]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            850,
                                "ForceLocalDamping":  0.05,
                                "TorqueLocalDamping": 0.05
                            })
                            
dempm.dem.add_attribute(materialID=1,
                  attribute={
                                "Density":            8500,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })

dempm.dem.add_template(template={
                                "Name":               "clump1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/box_finer.stl').grids(space=0.1, extent=5),
                                "WriteFile":          False}) 

dempm.dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")

dempm.dem.add_property(materialID1=0,
                   materialID2=0,
                   property={
                                 "NormalStiffness":            4e6,
                                 "TangentialStiffness":        2e6,
                                 "Friction":                   0.18,
                                 "NormalViscousDamping":       0.25,
                                 "TangentialViscousDamping":   0.25
                            })
                            
dempm.dem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                 "NormalStiffness":            4e7,
                                 "TangentialStiffness":        2e7,
                                 "Friction":                   0.58,
                                 "NormalViscousDamping":       0.25,
                                 "TangentialViscousDamping":   0.25
                            })          

if restart:
    dempm.dem.read_restart(file_number=start_file, file_path="OutputData", 
                           particle=True, wall=True, ppcontact=True, pwcontact=True, is_continue=True)
else:
    dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.02, 0.015],
                                  "ScaleFactor": 0.03,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.05, 0.015],
                                  "ScaleFactor": 0.03,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.08, 0.015],
                                  "ScaleFactor": 0.03,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.035, 0.045],
                                  "ScaleFactor": 0.03,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.065, 0.045],
                                  "ScaleFactor": 0.03,
                                  "BodyOrientation": "constant"
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.05, 0.075],
                                  "ScaleFactor": 0.03,
                                  "BodyOrientation": "constant"
                                  }
                                ]})
                           
    dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.25, 0.05, 0.0]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
    dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.25, 0.05, 0.3]),
                   "OuterNormal":  ti.Vector([0., 0., -1.])
                  })
                  
    dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0., 0.05, 0.15]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
    dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.5, 0.05, 0.15]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
    dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.25, 0.0, 0.15]),
                   "OuterNormal":  ti.Vector([0., 1., 0.])
                  })
                  
    dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.25, 0.1, 0.15]),
                   "OuterNormal":  ti.Vector([0., -1., 0.])
                  })

                  
dempm.dem.select_save_data(particle=True, surface=True, bounding=True, wall=True, particle_particle_contact=True, particle_wall_contact=True)

dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1350,
                               "YoungModulus":                  8e4,
                               "PoissonRatio":                  0.25,
                               "Friction":                      32,
                               "Dilation":                      0.0,
                               "Cohesion":                      0.0,
                               "Tensile":                       0.0
                 })

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([0.005, 0.005, 0.005])
                        })


if restart:
    dempm.mpm.read_restart(file_number=start_file, file_path="OutputData", is_continue=True)
else:
    dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
                            "BoundingBoxSize": ti.Vector([0.1, 0.1, 0.2]),
                            
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


dempm.mpm.select_save_data()

dempm.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")

dempm.add_property(DEMmaterial=0,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            9e3,
                                 "TangentialStiffness":        6e3,
                                 "Friction":                   0.25,
                                 "NormalViscousDamping":       0.15,
                                 "TangentialViscousDamping":   0.15
                            })
                            
dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            5e5,
                                 "TangentialStiffness":        3e5,
                                 "Friction":                   0.1,
                                 "NormalViscousDamping":       0.2,
                                 "TangentialViscousDamping":   0.2
                            })
                            
dempm.select_save_data(particle_particle_contact=True, particle_wall_contact=True)

dempm.run(mpm_gravity_field=True)

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
