from geotaichi import *

init(arch='cpu')

dem = DEM()


dem.set_configuration(domain=ti.Vector([5.,5.,10.]), scheme="LSDEM")

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([5.,5.,10.]),
                       "zdirection": ti.Vector([0.,0.,1.])
                       })                           
               
dem.add_template(template={
                                "Name":               "Template1",
                                "Object":              polyhedron(file='/assets/mesh/LSDEM/Pear.stl').grids(space=0.8, extent=3),
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
                               "BoundingRadius": 0.1,
                               "BodyNumber": 30,
                               "BodyOrientation": "uniform"}}) 

