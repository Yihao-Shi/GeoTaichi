import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32, debug=False)

from src.mpm.mainMPM import MPM

mpm = MPM()

mpm.postprocessing(end_file=60, write_background_grid=False, read_path="OutputData/velz330", write_path="OutputData/velz330/vtks")
