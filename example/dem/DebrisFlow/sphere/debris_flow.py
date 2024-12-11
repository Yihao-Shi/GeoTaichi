import numpy as np

from geotaichi import *

init(arch='gpu', log=False, debug=False, device_memory_GB=3.5, kernel_profiler=False)

dem = DEM()

dem.set_configuration(domain=ti.Vector([230.,260.,116.6]),
                      gravity=ti.Vector([0.,0.,-9.8]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.set_solver({
                "Timestep":         1e-3,
                "SimulationTime":   15.2,
                "SaveInterval":     0.4
               })

dem.memory_allocate(memory={
                            "max_material_number": 2,
                            "max_particle_number": 1559361,
                            "max_sphere_number": 1559361,
                            "max_digital_elevation_facet_number": 119600,
                            "verlet_distance_multiplier": 0.4,
                            "body_coordination_number": 16,
                            "wall_coordination_number": 2,
                            "compaction_ratio": [0.25, 0.1]
                            }, log=True)                       

dem.add_attribute(materialID=0,
                  attribute={
                            "Density":            2500,
                            "ForceLocalDamping":  0.2,
                            "TorqueLocalDamping": 0.2
                            })
                            
dem.add_attribute(materialID=1,
                  attribute={
                            "Density":            26500,
                            "ForceLocalDamping":  0.1,
                            "TorqueLocalDamping": 0.1
                            })

dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "TXT",
                   "Template":{
                               "BodyType": "Sphere",
                               "GroupID": 0,
                               "MaterialID": 0,
                               "File":'SpherePacking.txt',
                               "InitialVelocity": ti.Vector([0.,0.,0.]),
                               "InitialAngularVelocity": ti.Vector([0.,0.,0.]),
                               "FixVelocity": ["Free","Free","Free"],
                               "FixAngularVelocity": ["Free","Free","Free"],
                               "ParticleNumber": 1559361
                               }}) 
                               
dem.add_wall(body={
                   "WallType":    "DigitalElevation",
                   "WallID":       1,
                   "MaterialID":   1,
                   "DigitalElevation":   np.flip(np.loadtxt("dc7_dem_1411slide.txt",  skiprows = 6)  - 1597.4, 0),
                   "CellSize":     1.,
                   "NoData":       -11596.4,
                   "Visualize":    True
                  })
                  
dem.static_wall()

dem.choose_contact_model(particle_particle_contact_model="Linear Model",
                         particle_wall_contact_model="Linear Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "NormalStiffness":            1e5,
                            "TangentialStiffness":        1e5,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.5,
                            "TangentialViscousDamping":   0.5
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "NormalStiffness":            1e6,
                            "TangentialStiffness":        1e6,
                            "Friction":                   0.5,
                            "NormalViscousDamping":       0.5,
                            "TangentialViscousDamping":   0.5
                           })

dem.select_save_data(particle=False, wall=False)

dem.run()   

ti.profiler.print_kernel_profiler_info()
