import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(arch="gpu", log=False, debug=False, device_memory_GB=6.1)

lsdem = DEM()

lsdem.postprocessing(start_file=0, end_file=101, scheme="LSDEM")
