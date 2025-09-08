from geotaichi import *

init(debug=False, device_memory_GB=6)

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.2,0.2,0.2]),
                      scheme="LSDEM",
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0.,-9.8,0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.set_solver({
                "Timestep":         2e-5,
                "SimulationTime":   15,
                "SaveInterval":     0.3,
                "SavePath":         "Rotating"
               })

dem.memory_allocate(memory={
                            "max_material_number": 3,
                            "max_rigid_body_number": 25835,
                            "max_rigid_template_number": 1,
                            "levelset_grid_number": 52855,
                            "surface_node_number": 127,
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
                               "RigidFile":'/home/eleven/work/GeoTaichi/examples/dem/LevelSet/RotatingDrum/Static/particles/LSDEMRigid000010.npz',
                               "GridFile":'/home/eleven/work/GeoTaichi/examples/dem/LevelSet/RotatingDrum/Static/particles/LSDEMGrid000010.npz',
                               "BoundingSphereFile":'/home/eleven/work/GeoTaichi/examples/dem/LevelSet/RotatingDrum/Static/particles/LSDEMBoundingSphere000010.npz',
                               "BoundingBoxFile":'/home/eleven/work/GeoTaichi/examples/dem/LevelSet/RotatingDrum/Static/particles/LSDEMBoundingBox000010.npz',
                               "SurfaceFile":'/home/eleven/work/GeoTaichi/examples/dem/LevelSet/RotatingDrum/Static/particles/LSDEMSurface000010.npz'
                               }}) 

dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                         particle_wall_contact_model="Hertz Mindlin Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "ShearModulus":               4.3e6,
                            "Poisson":                    0.3,
                            "Friction":                   0.3,
                            "Restitution":                0.9
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "ShearModulus":               4.3e7,
                            "Poisson":                    0.3,
                            "Friction":                   0.3,
                            "Restitution":                0.9
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=2,
                 property={
                            "ShearModulus":               4.3e7,
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
                   "AngularVelocity": [0., 0., 2.094*3],
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
                
dem.select_save_data(wall=True)

def region(pos):
    return 1 if pos[0]>0.1 else 0

dem.update_particle_properties("groupID", 1, function=region)

dem.run()            
