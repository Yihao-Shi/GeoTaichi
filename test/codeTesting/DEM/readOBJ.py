import sys
sys.path.append('/home/eleven/work/GeoTaichi')

import taichi as ti
ti.init()

from src.dem.mainDEM import DEM

dem = DEM()

dem.memory_allocate(memory={"max_material_number":1, "max_particle_number": 2, "max_patch_number":12345})

dem.add_wall(body={"WallType": "Patch", "WallID": 0, "MaterialID": 0, 
                   "File": "asserts/mesh/rover_wheels/curiosity_wheel.obj"})


