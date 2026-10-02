import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *
import numpy as np
from scipy.stats import lognorm

init(arch="gpu", debug=False)

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.00036,0.00036,0.00085]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")          

                            
dem.add_region(region=[{
                            "Name": "region1",
                            "Type": "Rectangle",
                            "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.00]),
                            "BoundingBoxSize": ti.Vector([0.00036,0.00036,0.00085])
                      }])

def generate_lognormal_particle_sizes(N, mean_linear, std_linear, dmin, dmax):
    oversample = int(N * 1.5)
    diameters = lognorm.rvs(s=std_linear, scale=np.exp(mean_linear), size=oversample)
    valid = diameters[(diameters >= dmin) & (diameters <= dmax)]

    while len(valid) < N:
        extra = lognorm.rvs(s=std_linear, scale=mean_linear, size=(N - len(valid)) * 2)
        valid = np.concatenate([valid, extra[(extra >= dmin) & (extra <= dmax)]])

    return 0.5*np.sort(valid[:N])

dem.add_body(body={
                   "GenerateType": "Generate",
                   "BodyType": "Sphere",
                   "RegionName": "region1",
                   "PoissonSampling": True,
                   "WriteFile":  True,
                   "TryNumber":1000,
                   "Template":[{
                               "GroupID": 0,
                               "MaterialID": 0,
                               "BodyNumber": 13000,
                               "RadiusDistribution": generate_lognormal_particle_sizes(13000,18e-6,0.45e-6,1e-6,70e-6),
                               "BodyOrientation": "uniform"}]})
