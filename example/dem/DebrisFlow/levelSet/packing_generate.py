import numpy as np
from geotaichi import *

init(arch='gpu', log=False, debug=False, device_memory_GB=3.6)

lsdem = DEM()

lsdem.set_configuration(domain=ti.Vector([235.,265.,120.]),
                        scheme="LSDEM",
                        gravity=ti.Vector([0., 0., -9.8]),
                        visualize=False)
              
lsdem.memory_allocate(memory={
                                 "max_material_number": 2,
                                 "max_rigid_body_number": 194920,
                                 "levelset_grid_number": 29600,
                                 "surface_node_number": 127,
                                 "max_digital_elevation_facet_number": 119471,
                                 "body_coordination_number":   25,
                                 "wall_coordination_number":   1,
                                 "verlet_distance_multiplier": [0.05, 0.2],
                                 "point_coordination_number":  [4, 1], 
                                 "compaction_ratio":           [0.32, 0.1, 0.17, 0.02]
                             })  

lsdem.set_solver({
                "Timestep":         1e-3,
                "SimulationTime":   16,
                "SaveInterval":     0.4
               })  

lsdem.add_attribute(materialID=0,
                    attribute={
                                "Density":            2650,
                                "ForceLocalDamping":  0.05,
                                "TorqueLocalDamping": 0.05
                            })
                   
lsdem.add_template(template={
                                "Name":               "Template1",
                                "Object":              polyhedron(file='/assets/mesh/LSDEM/sand.stl').grids(space=5, extent=3),
                                "WriteFile":          True}) 

lsdem.add_body_from_file(body={
                            "FileType":  "TXT",
                            "Template":[{
                                             "Name": "Template1",
                                             "BodyType": "RigidBody",
                                             "File":'BoundingSphere.txt',
                                             "GroupID": 0,
                                             "MaterialID": 0,
                                             "InitialVelocity": ti.Vector([0.,0.,0.]),
                                             "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                                             "ParticleNumber":194920,
                                        }]
                        })
                        
lsdem.add_wall(body={
                   "WallType":    "DigitalElevation",
                   "WallID":       1,
                   "MaterialID":   1,
                   "DigitalElevation":   np.flip(np.loadtxt("dc7_dem_1411slide.txt",  skiprows = 6)  - 1597.4, 0),
                   "CellSize":     1.,
                   "NoData":       -11596.4,
                   "Visualize":    True
                  })
                  
lsdem.static_wall()

lsdem.choose_contact_model(particle_particle_contact_model="Linear Model",
                           particle_wall_contact_model="Linear Model")

                            
lsdem.add_property(materialID1=0,
                   materialID2=0,
                   property={
                                "NormalStiffness":            5e5,
                                "TangentialStiffness":        5e5,
                                "Friction":                   0.5,
                                "NormalViscousDamping":       0.15,
                                "TangentialViscousDamping":   0.15
                            })   
                            
lsdem.add_property(materialID1=0,
                   materialID2=1,
                   property={
                                "NormalStiffness":            1e6,
                                "TangentialStiffness":        1e6,
                                "Friction":                   0.3,
                                "NormalViscousDamping":       0.15,
                                "TangentialViscousDamping":   0.15
                            })    

lsdem.select_save_data(particle=False, surface=False, wall=False)

lsdem.run()

#lsdem.postprocessing()
