from geotaichi import *

init(debug=False, device_memory_GB=18)

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.2,1,0.2]),
                      scheme="LSDEM",
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0.,-9.8,0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.set_solver({
                "Timestep":         2e-5,
                "SimulationTime":   5,
                "SaveInterval":     0.25,
                "SavePath":         "Static"
               })

dem.memory_allocate(memory={
                            "max_material_number": 3,
                            "max_rigid_body_number": 80011,
                            "max_rigid_template_number": 1,
                            "levelset_grid_number": 52855,
                            "surface_node_number": 500,
                            "max_patch_number": 20000,
                            "body_coordination_number":   64,
                            "wall_coordination_number":   128,
                            "verlet_distance_multiplier": [0.05, 0.05],
                            "point_coordination_number":  [6, 1], 
                            "compaction_ratio":           [0.32, 0.1, 0.05, 0.05]
                            }, log=True)                       

dem.add_attribute(materialID=0,
                  attribute={
                            "Density":            1150,
                            "ForceLocalDamping":  0.,
                            "TorqueLocalDamping": 0.
                            })
                            
dem.add_attribute(materialID=1,
                  attribute={
                            "Density":            26500,
                            "ForceLocalDamping":  0.,
                            "TorqueLocalDamping": 0.
                            })

dem.add_body_from_file(body={
                   "WriteFile": True,
                   "FileType":  "NPZ",
                   "Template":{
                               "RigidFile":'/home/eleven/work/GeoTaichi/examples/dem/LevelSet/RotatingDrum/OutputData/particles/LSDEMRigid000020.npz',
                               "GridFile":'/home/eleven/work/GeoTaichi/examples/dem/LevelSet/RotatingDrum/OutputData/particles/LSDEMGrid000020.npz',
                               "BoundingSphereFile":'/home/eleven/work/GeoTaichi/examples/dem/LevelSet/RotatingDrum/OutputData/particles/LSDEMBoundingSphere000020.npz',
                               "BoundingBoxFile":'/home/eleven/work/GeoTaichi/examples/dem/LevelSet/RotatingDrum/OutputData/particles/LSDEMBoundingBox000020.npz',
                               "SurfaceFile":'/home/eleven/work/GeoTaichi/examples/dem/LevelSet/RotatingDrum/OutputData/particles/LSDEMSurface000020.npz'
                               }}) 

dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                         particle_wall_contact_model="Hertz Mindlin Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "ShearModulus":               4.3e5,
                            "Poisson":                    0.3,
                            "Friction":                   0.3,
                            "Restitution":                0.9
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "ShearModulus":               4.3e6,
                            "Poisson":                    0.3,
                            "Friction":                   0.3,
                            "Restitution":                0.9
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=2,
                 property={
                            "ShearModulus":               4.3e6,
                            "Poisson":                    0.3,
                            "Friction":                   0.9,
                            "Restitution":                0.9
                           })
         
dem.add_wall(body=[{
                   "WallType":    "Patch",
                   "WallID": 0,
                   "WallFile": '/home/eleven/work/GeoTaichi/assets/mesh/Drums/drum_side_raw.stl',
                   "Translation": [0.1, 0.1, 0.1],
                   "RotateCenter": [0.1, 0.1, 0.1],
                   "ScaleFactor": 0.1,
                   "AngularVelocity": [0., 0., 0.],
                   "MaterialID":   2,
                  },
                  {
                   "WallType":    "Patch",
                   "WallID": 1,
                   "WallFile": '/home/eleven/work/GeoTaichi/assets/mesh/Drums/drum_back_raw.stl',
                   "Translation": [0.1, 0.1, 0.1],
                   "ScaleFactor": 0.1,
                   "MaterialID":   1,
                  },
                  {
                   "WallType":    "Patch",
                   "WallID": 2,
                   "WallFile": '/home/eleven/work/GeoTaichi/assets/mesh/Drums/drum_front_raw.stl',
                   "Translation": [0.1, 0.1, 0.1],
                   "ScaleFactor": 0.1,
                   "MaterialID":   1,
                  }])
                  
def region(pos):
    return 0 if (pos[0]-0.1)**2+(pos[1]-0.1)**2<0.0093 and 0.002<pos[2]<0.2-0.002 and pos[1]<0.085 else 1
                      
dem.delete_particles(function=region)
                
dem.select_save_data(grid=True, bounding=True, surface=True, wall=True)

dem.run()            
