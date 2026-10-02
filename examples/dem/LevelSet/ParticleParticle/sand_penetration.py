import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(device_memory_GB=1, debug=False)

mode=3
path='SandPenetration/127N'
files=f'{ROOT}/assets/mesh/LSDEM/sand.stl'
pos=3.22063259
if mode==2:
    path='SandPenetration/502N'
    files=f'{ROOT}/assets/mesh/LSDEM/sand_middle.stl'
    pos=3.22341259
if mode==3:
    path='SandPenetration/2002N'
    files=f'{ROOT}/assets/mesh/LSDEM/sand_finer.stl'
    pos=3.23441259

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
                      "SavePath":         path
                 }) 
                      
lsdem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 6,
                                "levelset_grid_number": 46531,
                                "surface_node_number": 2500,
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
                                "Object":              polyhedron(file=files).grids(space=4, extent=3),
                                "WriteFile":          True}) 

lsdem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "Template1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.0, 1.0, pos],
                                  "ScaleFactor": 0.02,
                                  "BodyOrientation": [0, 0, 0],
                                  "InitialVelocity":  [0., 0., -1.],
                                  "FixMotion": ["Fix","Fix","Fix"]
                                  },{
                                  "Name": "Template1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.0, 1.0, 1.0],
                                  "ScaleFactor": 0.02,
                                  "BodyOrientation": [0, 90, 0],
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
