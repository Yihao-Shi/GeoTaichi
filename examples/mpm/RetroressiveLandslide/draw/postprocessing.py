import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import *

init(dim=2)

mpm = MPM()

mpm.postprocessing(start_file=0, end_file=61, read_path="SaintLucdeVincennes")

mpm.postprocessing(start_file=0, end_file=112, read_path="SainteMonique")
