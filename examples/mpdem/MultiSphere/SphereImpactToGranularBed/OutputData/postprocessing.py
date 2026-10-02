import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


from src import *

init(arch="gpu", debug=False)

from src.package import *

mpm = MPM()

mpm.postprocessing(end_file=101, read_path='velz303')
