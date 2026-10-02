import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import taichi as ti
ti.init(default_fp=ti.f64, arch=ti.cpu, debug=True)

from src.dem.mainDEM import DEM

dem = DEM()

dem.set_configuration(domain=ti.Vector([10.,10.,10.]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([7.,7.,7.])
                       })          

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Sphere",
                   "WriteFile": True,
                   "PoissonSampling": False,
                   "TryNumber": 1000,
                   "Template":[{
                               "MaxRadius": 0.01,
                               "MinRadius": 0.01,
                               "BodyNumber": 123168,
                               "BodyOrientation": "uniform"},
                               {
                               "MaxRadius": 0.1,
                               "MinRadius": 0.1,
                               "BodyNumber": 6523,
                               "BodyOrientation": "uniform"}]})  

