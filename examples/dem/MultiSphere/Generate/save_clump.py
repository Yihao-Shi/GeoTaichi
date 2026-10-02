import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *
init(arch="cpu", debug=False)

dem = DEM()

dem.set_configuration(domain=[0.3,0.3,0.3])

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": [0.,0.,0.],
                       "BoundingBoxSize": [0.2,0.2,0.2]
                       })                               

                           
dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 2,
                                 "Pebble": [{
                                             "Position": [-0.5, 0., 0.],
                                             "Radius": 1.
                                            },
                                            {
                                             "Position": [0.5, 0., 0.],
                                             "Radius": 1.
                                            }]
                                 })

dem.add_body(body={
                   "GenerateType": "Distribute",
                   "RegionName": "region1",
                   "BodyType": "Clump",
                   "WriteFile": True,
                   "Porosity":   0.45,
                   "Template":{
                               "Name": "clump1",
                               "MaxRadius": 0.0035,
                               "MinRadius": 0.0035,
                               "BodyNumber": 10000,
                               "BodyOrientation": "uniform"}})  

