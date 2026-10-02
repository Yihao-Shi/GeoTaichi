import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=1, debug=False)

lsdem = DEM()

lsdem.set_configuration(
                      domain=ti.Vector([2, 2, 7]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      
lsdem.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   0.01,
                      "SaveInterval":     0.0002,
                      "SavePath":         'SpherePenetration/160N'
                 }) 
                      
lsdem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 6,
                                "levelset_grid_number": 148877,
                                "surface_node_number": 10306,
                                "max_plane_number": 0,
                                "body_coordination_number":   28,
                                "wall_coordination_number":   4,
                                "verlet_distance_multiplier": [0.15, 0.16],
                                "point_coordination_number":  [4, 4],
                                "compaction_ratio":           [1., 1., 0.6, 0.35]
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
                                "Name":               "Template1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sphere05.stl').grids(space=0.05, extent=6),
                                "WriteFile":          True}) 

lsdem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "Template1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.0, 1.0, 0.9875],
                                  "ScaleFactor": 1.,
                                  "BodyOrientation": [0, 0, 0],
                                  "FixMotion": ["Fix","Fix","Fix"]
                                  },
                                  {
                                  "Name": "Template1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.0, 1.0, 2.0125],
                                  "ScaleFactor": 1.,
                                  "BodyOrientation": [0, 0, 0],
                                  "InitialVelocity":  [0., 0., -1.],
                                  "FixMotion": ["Fix","Fix","Fix"]
                                  },
                                ]})
                 
lsdem.choose_contact_model(particle_particle_contact_model="Barrier Model",
                         particle_wall_contact_model=None)
                            
lsdem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "Stiffness":                  1e8,
                            "NormalCutOff":               0.025,
                            "StiffnessRatio":             1.,
                            "Friction":                   0.,
                            "NormalViscousDamping":       0.,
                            "TangentialViscousDamping":   0.0
                           }, dType="particle-particle")                    
                 
'''lsdem.choose_contact_model(particle_particle_contact_model="Energy Conserving Model",
                         particle_wall_contact_model=None)  
                           
lsdem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            2e7,
                            "TangentialStiffness":        0e5,
                            "Friction":                   0.0,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })  '''  
                  
lsdem.select_save_data(clump=True)

lsdem.run()
