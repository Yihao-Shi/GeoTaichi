import sys
sys.path.append('/home/eleven/work/GeoTaichi')

import taichi as ti

from src.dem.mainDEM import DEM

dem = DEM()

dem.system_init()

dem.set_configuration(domain=ti.Vector([10.,10.,10.]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([7.,7.,7.]),
                       "zdirection": ti.Vector([0.,0.,1.])
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
                   "PoissionSampling": False,
                   "TryNumber": 1000,
                   "Template":{
                               "Name": "clump1",
                               "MaxRadius": 0.15,
                               "MinRadius": 0.09,
                               "BodyNumber": 23168,
                               "BodyOrientation": "uniform"}})  

