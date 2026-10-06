import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "../../../.."))
if ROOT not in sys.path:
    sys.path.append(ROOT)


import taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f64, default_ip=ti.i32, debug=False, device_memory_GB=2)

from src.mpm.mainMPM import MPM

mpm = MPM()

mpm.postprocessing(start_file=0, end_file=60, read_path="bbar")
