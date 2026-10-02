import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch="gpu", debug=False)

dem = DEM()

dem.set_configuration(domain=ti.Vector([1.1, 1.1, 1.6]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")          

                            
dem.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0125, 0.0125, 0.001]),
                            "BoundingBoxSize": ti.Vector([0.025, 0.025, 0.025])
                      }])

dem.add_body(body={
                   "GenerateType": "Distribute",
                   "BodyType": "Sphere",
                   "RegionName": "region1",
                   "Porosity":   0.45,
                   "WriteFile":  True,
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "MinRadius": 0.00025,
                               "MaxRadius": 0.00055,
                               "BodyOrientation": "uniform"}]})
