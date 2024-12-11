from geotaichi import *

init(device_memory_GB=3, debug=False)

lsdem = DEM()

lsdem.set_configuration(
                      domain=ti.Vector([10, 5, 5]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., -6.929646456]),
                      engine="VelocityVerlet",
                      search="LinkedCell",
                      scheme="LSDEM")
                      
lsdem.set_solver({
                      "Timestep":         1e-5,
                      "SimulationTime":   3,
                      "SaveInterval":     0.1,
                      "SavePath":         'BoxSlide/mu=0.2'
                 }) 
                      
lsdem.memory_allocate(memory={
                                "max_material_number": 2,
                                "max_rigid_body_number": 40,
                                "max_rigid_template_number":  2,
                                "levelset_grid_number": 1005381,
                                "surface_node_number": 126360,
                                "max_plane_number": 0,
                                "body_coordination_number":   28,
                                "wall_coordination_number":   4,
                                "verlet_distance_multiplier": [0.1, 0.1],
                                "point_coordination_number":  [4, 4],
                                "compaction_ratio":           [1., 1., 0.6, 0.35]
                            })  

lsdem.add_attribute(materialID=0,
                  attribute={
                                "Density":            850,
                                "ForceLocalDamping":  0.7,
                                "TorqueLocalDamping": 0.7
                            })
                            
lsdem.add_attribute(materialID=1,
                  attribute={
                                "Density":            8500,
                                "ForceLocalDamping":  0.,
                                "TorqueLocalDamping": 0.
                            })

lsdem.add_template(template={
                                "Name":               "Template1",
                                "Object":              polyhedron(file='assets/mesh/LSDEM/box.stl').grids(space=0.02, extent=4),
                                "WriteFile":          False}) 
                                
lsdem.add_template(template={
                                "Name":               "Template2",
                                "Object":             box((10,5,1)).grids(space=0.05, extent=4).reset(False),
                                "WriteFile":          True}) 

lsdem.create_body(body={
                     "GenerateType": "Create",
                     "BodyType": "RigidBody",
                     "Template":[
                                  {
                                  "Name": "Template1",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [1.5, 2.5, 1.5],
                                  "ScaleFactor": 1.,
                                  "BodyOrientation": [0, 0, 1],
                                  "InitialVelocity":  [0., 0., 0.],
                                  "FixMotion": ["Free","Free","Free"]
                                  },
                                  {
                                  "Name": "Template2",
                                  "GroupID": 0,
                                  "MaterialID": 0,
                                  "BodyPoint": [5, 2.5, 0.5],
                                  "ScaleFactor": 1.,
                                  "BodyOrientation": [0, 0, 1],
                                  "FixMotion": ["Fix","Fix","Fix"]
                                  }
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
                            "TangentialStiffness":        1e8,
                            "Friction":                   0.2,
                            "NormalViscousDamping":       0.,
                            "TangentialViscousDamping":   0.0
                           })    
                           
lsdem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            8e5,
                            "TangentialStiffness":        8e5,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.0,
                            "TangentialViscousDamping":   0.0
                           }, dType="particle-wall")  
                  
lsdem.select_save_data(clump=True)

lsdem.run()

lsdem.modify_parameters(SimulationTime=5)

lsdem.scene.material[0].fdamp=0.
lsdem.scene.material[0].tdamp=0.
lsdem.sims.gravity=[6.929646456,0.,-6.929646456]

lsdem.run()
