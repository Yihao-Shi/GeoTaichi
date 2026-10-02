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
                      search="BVH",
                      scheme="LSDEM",
                      track_energy=True)
                      
lsdem.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   0.1,
                      "SaveInterval":     0.001,
                      "SavePath":         "SandImpact"
                 }) 
                      
lsdem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 6,
                                "levelset_grid_number": 21753,
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
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sand_finer.stl').grids(space=5, extent=2),
                                "WriteFile":          True}) 

lsdem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[{
                                  "Name": "Template1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.0, 1.0, 3.205842591487],
                                  "ScaleFactor": 0.02,
                                  "BodyOrientation": [0, 0, 0],
                                  "InitialVelocity":  [0., 0., -1.],
                                  "FixMotion": ["Free","Free","Free"]
                                  },
                                  
                                  {
                                  "Name": "Template1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.0, 1.0, 1.0],
                                  "ScaleFactor": 0.02,
                                  "BodyOrientation": [0, 90, 0],
                                  "InitialVelocity":  [0., 0., 1.],
                                  "FixMotion": ["Free","Free","Free"]
                                  },
                                ]})
                 
lsdem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
'''lsdem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "Stiffness":                  1e5,
                            "NormalCutOff":               0.15,
                            "TangentialCutOff":           0.15,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.15,
                            "TangentialViscousDamping":   0.1
                           }, dType="particle-particle")     '''      
                           
lsdem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            2e8,
                            "TangentialStiffness":        2e8,
                            "Friction":                   0.0,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           })    
                           
lsdem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            8e5,
                            "TangentialStiffness":        8e5,
                            "Friction":                   0.0,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           }, dType="particle-wall")  
                  
lsdem.select_save_data()

lsdem.run()
