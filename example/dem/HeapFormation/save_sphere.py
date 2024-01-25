from geotaichi import *

init()

dem = DEM()


dem.set_configuration(domain=ti.Vector([100.,30.,45.]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([30.,30.,45]),
                       "zdirection": ti.Vector([0.,0.,1.])
                       })                            
                          

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Sphere",
                   "WriteFile": True,
                   "PoissionSampling": False,
                   "TryNumber": 10000,
                   "Template":{
                               
                               "MaxRadius": 0.075,
                               "MinRadius": 0.075,
                               "BodyNumber": 6250000}}) 

