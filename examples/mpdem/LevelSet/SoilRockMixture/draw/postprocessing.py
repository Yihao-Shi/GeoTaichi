import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-f", type=str)
args = parser.parse_args()

from geotaichi import *

init()

dempm = DEMPM()

dempm.mpm.postprocessing(
    start_file=0,
    end_file=21,
    total_displacement=True,
    read_path=args.f,
    smooth_setting={"smooth_rad": 2.0, "lower_bound": [0.0, 0.0, 0.0], "upper_bound": [50.0, 50.0, 16.0]},
)
