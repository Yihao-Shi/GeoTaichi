import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch='cpu')

dem = DEM()

scale=0.1
dem.set_configuration(domain=scale*ti.Vector([33.12,33.12,40.13]), scheme="LSDEM")

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": scale*ti.Vector([0.,0.,21.4]),
                       "BoundingBoxSize": scale*ti.Vector([33.12,33.12,18.72])
                       })                            
               
dem.add_template(template={
                                "Name":               "Template1",
                                "Object":              polyhedron(file=f'{ROOT}/assets/mesh/LSDEM/sand.stl').grids(space=5),
                                "WriteFile":          True})            

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "RigidBody",
                   "WriteFile": True,
                   "PoissonSampling": False,
                   "TryNumber": 1000,
                   "Template":{
                               
                               "Name": "Template1",
                               "MaxRadius": scale*0.1,
                               "MinRadius": scale*0.1,
                               "BodyNumber": 196000,
                               "BodyOrientation": [45.0,  -35.2644,  0.0]}}) 

