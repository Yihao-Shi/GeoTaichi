import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch="cpu")

dem = DEM()


dem.set_configuration(domain=ti.Vector([20.0, 25.0, 25.0]))

dem.add_region(
    region={
        "Name": "region1",
        "Type": "Rectangle",
        "BoundingBoxPoint": ti.Vector([0.0, 0.0, 0.0]),
        "BoundingBoxSize": ti.Vector([15.0, 15.0, 15.0]),
    }
)

dem.add_template(
    template={
        "Name": "clump1",
        "NSphere": 2,
        "Pebble": [
            {"Position": ti.Vector([-0.5, 0.0, 0.0]), "Radius": 1.0},
            {"Position": ti.Vector([0.5, 0.0, 0.0]), "Radius": 1.0},
        ],
    }
)

dem.add_body(
    body={
        "GenerateType": "Generate",
        "RegionName": "region1",
        "BodyType": "Clump",
        "WriteFile": True,
        "PoissonSampling": False,
        "TryNumber": 10000,
        "Template": {
            "Name": "clump1",
            "MaxRadius": 0.075,
            "MinRadius": 0.075,
            "BodyNumber": 300000,
            "BodyOrientation": "uniform",
        },
    }
)
