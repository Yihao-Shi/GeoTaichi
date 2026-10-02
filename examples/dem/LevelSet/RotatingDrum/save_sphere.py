import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.2,0.6,0.08]),
                      scheme="PolySuperEllipsoid")    

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([0.2,0.6,0.08]),
                       })           

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "RigidBody",
                   "WriteFile": True,
                   "PoissonSampling": True,
                   "TryNumber": 1000,
                   "Template":{
                               "Name": "Template1",
                               "MaxBoundingRadius": 0.005,
                               "MinBoundingRadius": 0.005,
                               "BodyNumber": 85000,
                               "BodyOrientation": "uniform"}}) 
