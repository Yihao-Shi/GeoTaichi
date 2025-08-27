import sys
sys.path.append('/home/eleven/work/GeoTaichi')

from geotaichi import *

init(arch='cpu')

dem = DEM()


dem.set_configuration(domain=ti.Vector([20.,25.,25.]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([20.,25.,25])
                       })                            
                          

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Sphere",
                   "WriteFile": True,
                   "PoissonSampling": False,
                   "TryNumber": 10000,
                   "Template":{
                               
                               "MaxRadius": 0.25,
                               "MinRadius": 0.25,
                               "BodyNumber": 100000}}) 

