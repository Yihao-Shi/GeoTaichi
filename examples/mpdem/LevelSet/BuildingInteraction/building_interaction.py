import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(debug=False)

dempm = DEMPM()

dempm.set_configuration(domain=ti.Vector([850, 850, 400]),
                        coupling_scheme="MPDEM",
                        particle_interaction=True,
                        wall_interaction=True)

dempm.mpm.set_configuration( 
                      background_damping=0.01,
                      alphaPIC=0.001, 
                      mapping="USL", 
                      shape_function="GIMP",
                      gravity=ti.Vector([0., 0., -9.8]))

dempm.dem.set_configuration(
                      gravity=ti.Vector([0., 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      

dempm.set_solver({
                      "Timestep":         5e-3,
                      "SimulationTime":   50,
                      "SaveInterval":     2.5,
                      "SavePath":         'OutputData'
                 }) 
                      
dempm.dem.memory_allocate(memory={
                                "max_material_number": 3,
                                "max_rigid_body_number": 27,
                                "levelset_grid_number": 313875,
                                 "max_rigid_template_number": 2,
                                "surface_node_number": 1184,
                                "max_plane_number": 6,
                                "body_coordination_number":   12,
                                "wall_coordination_number":   6,
                                "verlet_distance_multiplier": [0.15, 0.2],
                                "point_coordination_number":  [3, 4], 
                                "compaction_ratio":           [0.5, 0.5, 0.5, 0.5],
                                "wall_per_cell":              6
                            })  
                 
dempm.mpm.memory_allocate(memory={
                                "max_material_number":           1,
                                "max_particle_number":           254000,
                                "verlet_distance_multiplier":    0.1,
                                "max_constraint_number":  {
                                                               "max_reflection_constraint":   0,
                                                               "max_friction_constraint":   0,
                                                               "max_velocity_constraint":   12322
                                                          }
                            })
                            
dempm.memory_allocate(memory={
                                  "body_coordination_number":    12,
                                  "wall_coordination_number":    3,
                                  "compaction_ratio": [0.2, 0.15]
                             })  
                                          

dempm.dem.add_attribute(materialID=0,
                  attribute={
                                "Density":            1000,
                                "ForceLocalDamping":  0.15,
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
                                "Object":              box((36, 36, 144)).grids(space=3, extent=5).reset(False),
                                "SurfaceResolution":   4902}) 

dempm.dem.add_template(template={
                                "Name":               "clump2",
                                "Object":              box((36, 36, 96)).grids(space=3, extent=5).reset(False),
                                "SurfaceResolution":   4902}) 


dempm.dem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [210, 540, 72],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [480, 150, 72],
                                  "ScaleFactor": 1.,
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [420, 210, 72],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [540, 210, 72],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [480, 270, 72],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [480, 420, 72],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [420, 480, 72],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [540, 480, 72],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [480, 540, 72],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [150, 480, 72],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [210, 420, 72],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [270, 480, 72],
                                  "ScaleFactor": 1.
                                  },

                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [420, 420, 48],
                                  "ScaleFactor": 1.,
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [480, 480, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [540, 540, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [420, 540, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [540, 420, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [150, 420, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [270, 420, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [210, 480, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [150, 540, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [270, 540, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [420, 150, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [540, 150, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [480, 210, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [420, 270, 48],
                                  "ScaleFactor": 1.
                                  },
                                  
                                  {
                                  "Name": "clump2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [540, 270, 48],
                                  "ScaleFactor": 1.
                                  }
                                ]})
                 
dempm.dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
dempm.dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e10,
                            "TangentialStiffness":        1e10,
                            "Friction":                   0.35,
                            "NormalViscousDamping":       0.15,
                            "TangentialViscousDamping":   0.15
                           })           
                           
dempm.dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            8e11,
                            "TangentialStiffness":        7e11,
                            "Friction":                   0.35,
                            "NormalViscousDamping":       0.1,
                            "TangentialViscousDamping":   0.1
                           })  
                           
                           
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([425., 425., 0.0]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0., 425, 200]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([850, 425, 200]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([425, 0., 200]),
                   "OuterNormal":  ti.Vector([0., 1., 0.])
                  })
                  
dempm.dem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([425, 850, 200]),
                   "OuterNormal":  ti.Vector([0., -1., 0.])
                  })
                  
                  
dempm.dem.select_save_data(surface=True)

dempm.mpm.add_material(model="GranularMaterial",
                 material={
                               "MaterialID":                    1,
                               "Density":                       1500,
                               "YoungModulus":                  2e8,
                               "PoissionRatio":                 0.3,
                               "StaticFriction":                      31,
                               "DynamicFriction":                      31,
                               "AverageDiameter":                      0.3,
                               "InertialNumber":                       0.03,
                               "eps":                          0.01
                 })
                 
'''dempm.mpm.add_material(model="DruckerPrager",
                 material={
                               "MaterialID":           1,
                               "RateDependent":        True,
                               "Density":              1500.,
                               "YoungModulus":                  2e8,
                               "PossionRatio":                 0.3,
                               "StaticFriction":                31,
                               "DynamicFriction":               31,
                               "AverageDiameter":               0.3,
                               "InertialNumber":                0.03,
                               "eps":                           0.01
                 })'''

dempm.mpm.add_element(element={
                             "ElementType":               "R8N3D",
                             "ElementSize":               ti.Vector([12, 12, 12])
                        })


dempm.mpm.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
                            "BoundingBoxSize": ti.Vector([360, 360, 360]),
                            
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
                                 "NormalStiffness":            9e7,
                                 "TangentialStiffness":        6e7,
                                 "Friction":                   0.25,
                                 "NormalViscousDamping":       0.15,
                                 "TangentialViscousDamping":   0.15
                            })
                            
dempm.add_property(DEMmaterial=1,
                   MPMmaterial=1,
                   property={
                                 "NormalStiffness":            5e8,
                                 "TangentialStiffness":        3e8,
                                 "Friction":                   0.25,
                                 "NormalViscousDamping":       0.2,
                                 "TangentialViscousDamping":   0.2
                            })
                            

dempm.run(mpm_gravity_field=True)

dempm.mpm.postprocessing()

dempm.dem.postprocessing()
