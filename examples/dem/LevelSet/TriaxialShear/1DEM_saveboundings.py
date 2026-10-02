import argparse
import os
import sys
import shutil
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


CASE_DIR = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description="Generate and save the triaxial sphere packing bounds.")
parser.add_argument(
    "--output-file",
    default=str(CASE_DIR / "OutputData" / "SpherePacking.txt"),
    help="Destination for the generated SpherePacking.txt.",
)
arguments = parser.parse_args()
destination_file = Path(arguments.output_file).expanduser().resolve()
destination_file.parent.mkdir(parents=True, exist_ok=True)
generated_file = Path("SpherePacking.txt").resolve()

if generated_file.exists():
    generated_file.unlink()

from geotaichi import *

scale_fac0 = 1.001891414401244518
scale_fac1 = 1.247387418619341659
scale_fac2 = 1.554954183054590544

scale_fac = scale_fac1

init(arch="gpu", debug=True)

dem = DEM()

dem.set_configuration(domain=ti.Vector([0.1, 0.1, 0.1]),
                      boundary=["Destroy", "Destroy", "Destroy"],
                      gravity=ti.Vector([0., 0., 0.]),
                      engine="SymplecticEuler",
                      search="LinkedCell")          

                            
dem.add_region(region=[{"Name": "region1",
                          "Type": "Rectangle",
                          "BoundingBoxPoint": ti.Vector([0.0125, 0.0125, 0.0125]),
                          "BoundingBoxSize": ti.Vector([0.005*scale_fac, 0.005*scale_fac, 0.005*scale_fac])}])

# dem.add_body(body={
#                    "GenerateType": "Distribute",
#                    "BodyType": "Sphere",
#                    "RegionName": "region1",
#                    "Porosity":   0.5,
#                    "WriteFile":  True,
#                    "Template":[{"GroupID": 0,
#     "MaterialID": 0,
#     "MinRadius": 0.00007*scale_fac,
#     "MaxRadius": 0.00007*scale_fac,
#     "BodyOrientation": "uniform",
#     "Fraction":0.10},
#     {"GroupID": 0,
#     "MaterialID": 0,
#     "MinRadius": 0.000085*scale_fac,
#     "MaxRadius": 0.000085*scale_fac,
#     "BodyOrientation": "uniform",
#     "Fraction":0.20},
#     {"GroupID": 0,
#     "MaterialID": 0,
#     "MinRadius": 0.00012*scale_fac,
#     "MaxRadius": 0.00012*scale_fac,
#     "BodyOrientation": "uniform",
#     "Fraction":0.3},
#     {"GroupID": 0,
#     "MaterialID": 0,
#     "MinRadius": 0.00015*scale_fac,
#     "MaxRadius": 0.00015*scale_fac,
#     "BodyOrientation": "uniform",
#     "Fraction":0.24},
#     {"GroupID": 0,
#     "MaterialID": 0,
#     "MinRadius": 0.000175*scale_fac,
#     "MaxRadius": 0.000175*scale_fac,
#     "BodyOrientation": "uniform",
#     "Fraction":0.16}]})

dem.add_body(body={
                   "GenerateType": "Distribute",
                   "BodyType": "Sphere",
                   "RegionName": "region1",
                   "Porosity":   0.5,
                   "WriteFile":  True,
                   "Template":[{"GroupID": 0,
    "MaterialID": 0,
    "MinRadius": 0.00007*scale_fac,
    "MaxRadius": 0.00007*scale_fac,
    "BodyOrientation": "uniform",
    "Fraction":0.10},
    {"GroupID": 0,
    "MaterialID": 0,
    "MinRadius": 0.000085*scale_fac,
    "MaxRadius": 0.000085*scale_fac,
    "BodyOrientation": "uniform",
    "Fraction":0.20},
    {"GroupID": 0,
    "MaterialID": 0,
    "MinRadius": 0.00012*scale_fac,
    "MaxRadius": 0.00012*scale_fac,
    "BodyOrientation": "uniform",
    "Fraction":0.3},
    {"GroupID": 0,
    "MaterialID": 0,
    "MinRadius": 0.00015*scale_fac,
    "MaxRadius": 0.00015*scale_fac,
    "BodyOrientation": "uniform",
    "Fraction":0.24},
    {"GroupID": 0,
    "MaterialID": 0,
    "MinRadius": 0.000175*scale_fac,
    "MaxRadius": 0.000175*scale_fac,
    "BodyOrientation": "uniform",
    "Fraction":0.16}]})

if destination_file.exists() and destination_file != generated_file:
    destination_file.unlink()
if destination_file != generated_file:
    shutil.move(str(generated_file), str(destination_file))
