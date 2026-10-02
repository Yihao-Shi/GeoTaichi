import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=1, debug=False, log=False)

lsdem = DEM()

lsdem.set_configuration(
                      domain=ti.Vector([0.5, 0.1, 0.3]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., -9.8]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      
lsdem.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   0.001,
                      "SaveInterval":     0.00002,
                      "SavePath":         'CubePenetration/272N'
                 }) 
                      
lsdem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 6,
                                "levelset_grid_number": 12167,
                                "surface_node_number": 4372,
                                "max_plane_number": 6,
                                "body_coordination_number":   28,
                                "wall_coordination_number":   3,
                                "verlet_distance_multiplier": [0.15, 0.1],
                                "compaction_ratio":           [0.6, 0.35]
                            })  

lsdem.add_attribute(materialID=0,
                  attribute={
                                "Density":            850,
                                "ForceLocalDamping":  0.0,
                                "TorqueLocalDamping": 0.0
                            })
                            
lsdem.add_attribute(materialID=1,
                  attribute={
                                "Density":            8500,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })

lsdem.add_template(template={
                                "Name":               "clump1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/box.stl').grids(space=0.1, extent=1),
                                "WriteFile":          False}) 

lsdem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "clump1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [0.215, 0.02, 0.005],
                                  "Radius": 0.006203504908994001,
                                  "BodyOrientation": "constant",
                                  "InitialVelocity": [0., 0., -2.],
                                  "FixMotion": ["Fix","Fix","Fix"]
                                  }
                                ]})
                 
lsdem.choose_contact_model(particle_particle_contact_model="Energy Conserving Model",
                         particle_wall_contact_model="Energy Conserving Model")
                            
lsdem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            2e5,
                            "TangentialStiffness":        1e5,
                            "Friction":                   0.0,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })           
                           
lsdem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            1e6,
                            "TangentialStiffness":        1e6,
                            "Friction":                   0.0,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })  

                           
                           
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.25, 0.05, 0.0]),
                   "OuterNormal":  ti.Vector([0., 0., 1.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.25, 0.05, 0.3]),
                   "OuterNormal":  ti.Vector([0., 0., -1.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0., 0.05, 0.15]),
                   "OuterNormal":  ti.Vector([1., 0., 0.])
                  })
                  
lsdem.add_wall(body={
                   "WallType":    "Plane",
                   "MaterialID":   1,
                   "WallCenter":   ti.Vector([0.5, 0.05, 0.15]),
                   "OuterNormal":  ti.Vector([-1., 0., 0.])
                  })
                  
lsdem.select_save_data()

lsdem.run()

print(lsdem.scene.rigid[0].mass_center)
