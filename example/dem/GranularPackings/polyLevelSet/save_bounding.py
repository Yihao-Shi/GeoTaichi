from geotaichi import *

init(arch='cpu')

dem = DEM()


dem.set_configuration(domain=ti.Vector([26.,26.,32]), scheme="LSDEM")

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,22]),
                       "BoundingBoxSize": ti.Vector([26.,26.,9.5]),
                       "zdirection": ti.Vector([0.,0.,1.])
                       })                            
               
dem.add_template(template={
                                "Name":               "Template1",
                                "Object":              polyhedron(file='/asserts/mesh/LSDEM/sand.stl').grids(space=5),
                                "WriteFile":          True})            

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "RigidBody",
                   "WriteFile": True,
                   "PoissionSampling": True,
                   "TryNumber": 1000,
                   "Template":{
                               
                               "Name": "Template1",
                               "MaxRadius": 0.1,
                               "MinRadius": 0.1,
                               "BodyNumber": 199999,
                               "BodyOrientation": "uniform"}}) 

