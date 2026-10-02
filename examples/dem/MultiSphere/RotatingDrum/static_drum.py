import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

CASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Prepare the static MultiSphere rotating-drum packing.")
parser.add_argument("--input-dir", default=str(CASE_DIR / "OutputData"))
parser.add_argument("--input-frame", type=int, default=50)
parser.add_argument("--output-dir", default=str(CASE_DIR / "Static"))
arguments = parser.parse_args()
input_dir = Path(arguments.input_dir).expanduser().resolve() / "particles"

from geotaichi import *

init(debug=False)

dem = DEM()

dem.set_configuration(domain=ti.Vector([2.,2.,2.]),
                      gravity=ti.Vector([0.,-9.8,0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")

dem.set_solver({
                "Timestep":         1e-4,
                "SimulationTime":   5,
                "SaveInterval":     0.25,
                "SavePath":         arguments.output_dir
               })

dem.memory_allocate(memory={
                            "max_material_number": 2,
                            "max_particle_number": 300000,
                            "max_sphere_number": 300000,
                            "max_clump_number": 0,
                            "max_patch_number": 20000,
                            "verlet_distance_multiplier": 0.1,
                            "body_coordination_number": 32,
                            "wall_coordination_number": 128,
                            "compaction_ratio": [0.15, 0.1]
                            }, log=True)                       

dem.add_attribute(materialID=0,
                  attribute={
                            "Density":            2650,
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
                               "ParticleFile": str(input_dir / f'DEMParticle{arguments.input_frame:06d}.npz'),
                               "SphereFile": str(input_dir / f'DEMSphere{arguments.input_frame:06d}.npz')
                               }}) 

dem.choose_contact_model(particle_particle_contact_model="Hertz Mindlin Model",
                         particle_wall_contact_model="Hertz Mindlin Model")
                            
dem.add_property(materialID1=0,
                 materialID2=0,
                 property={
                            "ShearModulus":               4.3e6,
                            "Poisson":                    0.3,
                            "Friction":                   0.5,
                            "Restitution":                0.6
                           })
                           
dem.add_property(materialID1=0,
                 materialID2=1,
                 property={
                            "ShearModulus":               7.9e6,
                            "Poisson":                    0.3,
                            "Friction":                   0.5,
                            "Restitution":                0.6
                           })
         
dem.add_wall(body=[{
                   "WallType":    "Patch",
                   "WallID": 0,
                   "WallFile": f'{ROOT}/assets/mesh/Drums/drum_side_raw.stl',
                   "Translation": [1., 1., 1.],
                   "RotateCenter": [1., 1., 1.],
                   "AngularVelocity": [0., 0., 0.],
                   "MaterialID":   1,
                  },
                  {
                   "WallType":    "Patch",
                   "WallID": 1,
                   "WallFile": f'{ROOT}/assets/mesh/Drums/drum_back_raw.stl',
                   "Translation": [1., 1., 1.],
                   "MaterialID":   1,
                  },
                  {
                   "WallType":    "Patch",
                   "WallID": 2,
                   "WallFile": f'{ROOT}/assets/mesh/Drums/drum_front_raw.stl',
                   "Translation": [1., 1., 1.],
                   "MaterialID":   1,
                  }])
                  
def region(pos, rad):
    return 0 if (pos[0]-1)**2+(pos[1]-1)**2<0.97515625 and 0.0125<pos[2]<2.-0.0125 else 1
                      
dem.delete_particles(function=region)
                
dem.select_save_data(sphere=True, wall=True)

dem.run()            
