import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch="cpu", debug=False)

dem = DEM()

dem.set_configuration(domain=[15, 15, 15],
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=[0., 0., 0.],
                      engine="SymplecticEuler",
                      search="LinkedCell")          

                            
dem.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": [0, 0, 0],
                            "BoundingBoxSize": [15, 15, 15]
                      }])

dem.add_body(body={
                   "GenerateType": "Generate",
                   "BodyType": "Sphere",
                   "RegionName": "region1",
                   "Porosity":   0.45,
                   "WriteFile":  True,
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "MaxRadius": 0.15,
                               "MinRadius": 0.09,
                               "BodyNumber": 20000,
                               "BodyOrientation": "uniform"}]})
