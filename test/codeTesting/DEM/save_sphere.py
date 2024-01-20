import sys
sys.path.append('/home/eleven/work/GeoTaichi')

import taichi as ti

from src.dem.mainDEM import DEM

dem = DEM()

dem.system_init()

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([12.,12.,12.]),
                       "BoundingBoxSize": ti.Vector([19.,19.,19.]),
                       "ZDirection": ti.Vector([1.,1.,1.])
                       })                            
                          

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Sphere",
                   "WriteFile": True,
                   "PoissionSampling": False,
                   "TryNumber": 100,
                   "Template":{
                               
                               "MaxRadius": 0.1,
                               "MinRadius": 0.06,
                               "BodyNumber": 46337}}) 

