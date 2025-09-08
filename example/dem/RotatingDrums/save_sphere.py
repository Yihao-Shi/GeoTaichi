from geotaichi import *

init()

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.2,1.6,0.2]),
                      scheme="PolySuperEllipsoid")    

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([0.2,1.6,0.2]),
                       })           

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "RigidBody",
                   "WriteFile": True,
                   "PoissonSampling": True,
                   "TryNumber": 100,
                   "Template":{
                               "Name": "Template1",
                               "MaxBoundingRadius": 0.004,
                               "MinBoundingRadius": 0.004,
                               "BodyNumber": 85000,
                               "BodyOrientation": "uniform"}}) 
