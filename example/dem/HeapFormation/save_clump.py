try:
    from geotaichi import *
except:
    import os
    import sys
    current_file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(current_file_path)
    from geotaichi import *

init()

dem = DEM()


dem.set_configuration(domain=ti.Vector([20.,25.,25.]))

dem.add_region(region={
                       "Name": "region1",
                       "Type": "Rectangle",
                       "BoundingBoxPoint": ti.Vector([0.,0.,0.]),
                       "BoundingBoxSize": ti.Vector([15.,15.,15.]),
                       "zdirection": ti.Vector([0.,0.,1.])
                       })                            
              
dem.add_template(template={
                                 "Name": "clump1",
                                 "NSphere": 2,
                                 "Pebble": [{
                                             "Position": ti.Vector([-0.5, 0., 0.]),
                                             "Radius": 1.
                                            },
                                            {
                                             "Position": ti.Vector([0.5, 0., 0.]),
                                             "Radius": 1.
                                            }]
                                 })            

dem.add_body(body={
                   "GenerateType": "Generate",
                   "RegionName": "region1",
                   "BodyType": "Clump",
                   "WriteFile": True,
                   "PoissionSampling": False,
                   "TryNumber": 10000,
                   "Template":{
                               "Name": "clump1",
                               "MaxRadius": 0.075,
                               "MinRadius": 0.075,
                               "BodyNumber": 300000,
                               "BodyOrientation": 'uniform'}}) 

