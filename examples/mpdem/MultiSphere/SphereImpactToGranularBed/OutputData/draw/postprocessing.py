import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from geotaichi import MPM, init

init(arch="gpu", debug=False)

mpm = MPM()

mpm.postprocessing(end_file=101, read_path="velz303")
