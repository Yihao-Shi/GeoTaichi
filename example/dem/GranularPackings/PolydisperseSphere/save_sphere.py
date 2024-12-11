from geotaichi import *

init(arch='cpu')

dem = DEM()


dem.set_configuration(domain=ti.Vector([20.,25.,25.]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([15.,15.,12]),
                       "zdirection": ti.Vector([0.,0.,1.])
                       })                            
                          

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Sphere",
                   "WriteFile": True,
                   "PoissionSampling": False,
                   "TryNumber": 1000,
                   "Template":[{
                               
                               "Radius": 0.05,
                               "BodyNumber": 400000
                               },
                               
                               {
                               
                               "Radius": 0.1,
                               "BodyNumber": 48000
                               }]
                 }) 

