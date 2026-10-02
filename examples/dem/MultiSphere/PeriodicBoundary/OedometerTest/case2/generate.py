import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch="gpu", debug=False, device_memory_GB=21)

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.00036,0.00036,0.00052]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")          

                            
dem.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.00]),
                            "BoundingBoxSize": ti.Vector([0.00036,0.00036,0.00052])
                      }])

dem.add_body(body={
                   "GenerateType": "Distribute",
                   "BodyType": "Sphere",
                   "RegionName": "region1",
                   "Porosity":   0.67,
                   "WriteFile":  True,
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "MinRadius": 0.5e-6,
                               "MaxRadius": 0.5e-6,
                               "BodyOrientation": "uniform",
                               "Fraction":0.3},
                               {
                               "GroupID": 1,
                               "MaterialID": 0,
                               "MinRadius": 8.5e-6,
                               "MaxRadius": 8.5e-6,
                               "BodyOrientation": "uniform",
                               "Fraction":0.7}]})
