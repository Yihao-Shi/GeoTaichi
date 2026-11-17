import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init()

dem = DEM()

dem.postprocessing(start_file=1, end_file=90, write_force_chain=True, read_path="Loose/CD", scheme="DEM")
