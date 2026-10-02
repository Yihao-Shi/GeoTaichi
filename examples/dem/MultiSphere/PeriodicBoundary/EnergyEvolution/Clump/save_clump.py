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

                           
dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 2,
                                 "Pebble": [{
                                             "Position": ti.Vector([-0.5, 0., 0.]),
                                             "Radius": 1.
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0., 0.]),
                                             "Radius": 1.
                                            }]
                                 })

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Clump",
                   "WriteFile": True,
                   "PoissonSampling": False,
                   "TryNumber": 1000,
                   "Template":{
                               "Name": "clump1",
                               "MaxRadius": 0.15,
                               "MinRadius": 0.09,
                               "BodyNumber": 23168,
                               "BodyOrientation": "uniform"}})  

