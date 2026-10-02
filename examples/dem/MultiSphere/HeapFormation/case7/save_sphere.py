import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='cpu')

dem = DEM()


dem.set_configuration(domain=ti.Vector([20.,25.,35.]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([15.,15.,35])
                       })                            
                          

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Sphere",
                   "WriteFile": True,
                   "PoissonSampling": False,
                   "TryNumber": 10000,
                   "Template":{
                               
                               "MaxRadius": 0.075,
                               "MinRadius": 0.075,
                               "BodyNumber": 1600000}}) 

