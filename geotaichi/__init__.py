# Copyright (c) 2023, multiscale geomechanics lab, Zhejiang University
# This file is from the GeoTaichi project, released under the GNU General Public License v3.0

__author__ = "Shi-Yihao, Guo-Ning"
__version__ = "0.1.0"
__license__ = "GNU License"

import taichi as ti
import psutil, platform
import sys, os, datetime  

from src import (
    DEM,
    FEM,
    IGA,
    MPM,
    DEMPM,
    MPDEM,
    FEDEM,
    FEMPM,
    IGAMPM,
)

from src.sdf.BasicShape import arbitrarily, polyhedron, surfacefunction, polysuperquadrics, polysuperellipsoid

from src.sdf.SDFs2D import (
    circle, line, slab,
    rectangle, rounded_rectangle, equilateral_triangle,
    hexagon, rounded_x, cross, polygon
)

from src.sdf.SDFs3D import (
    sphere, plane, slab, 
    box, rounded_box, box_frame, torus, capped_torus, heart,
    link, hexagonal_prism, capsule, cylinder, capped_cylinder, rounded_cylinder,
    cone, capped_cone, rounded_cone, revolved_vesica, octahedron,
    pyramid, rhombus, tetrahedron, dodecahedron, icosahedron
)

from src.sdf.text import (
    image,
    text,
)

from src.sdf.ease import (
    linear,
    in_quad, out_quad, in_out_quad,
    in_cubic, out_cubic, in_out_cubic,
    in_quart, out_quart, in_out_quart,
    in_quint, out_quint, in_out_quint,
    in_sine, out_sine, in_out_sine,
    in_expo, out_expo, in_out_expo,
    in_circ, out_circ, in_out_circ,
    in_elastic, out_elastic, in_out_elastic,
    in_back, out_back, in_out_back,
    in_bounce, out_bounce, in_out_bounce,
    in_square, out_square, in_out_square,
)

from src.utils.GeometryFunction import circle_inside_geometry

import src.utils.GlobalVariable as GlobalVariable
from src.utils.SolverRuntime import python_callback

class Logger(object):
    def __init__(self, filename='Default.log', path='./'):
        self.terminal = sys.stdout
        self.path = os.path.join(path, filename)
        self.log = open(self.path, "a", encoding='utf8')
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        
    def flush(self):
        pass     
       
       
def make_print_to_file(path='./'):
    filename = datetime.datetime.now().strftime('day'+'%Y_%m_%d')
    sys.stdout = Logger(filename+'.log', path=path)
    

def init(dim=3, arch="gpu", cpu_max_num_threads=0, offline_cache=True, debug=False, default_fp="float64", default_ip="int32", device_memory_GB=None, device_memory_fraction=None, kernel_profiler=False, print_ir=False, log=True, random_seed=0):
    """
    Initializes the Taichi runtime environment.
    Args:
        dim (int): The dimensions, Can be either 2 or 3.
        arch (str): The execution architecture. Can be either "cpu" or "gpu".
        cpu_max_num_threads (int): The maximum number of threads to use if the backend is CPU. Defaults to the maximum number of threads available on the CPU.
        offline_cache (bool): Whether to store compiled files. Defaults to True.
        debug (bool): Whether to enable debug mode.
        default_fp (str): The default floating-point type. Can be "float64" or "float32".
        default_ip (str): The default integer type. Can be "int64" or "int32".
        device_memory_GB (int): The pre-allocated GPU memory size in GB. If the device memory is less than 2GB, the default settings will be used.
        device_memory_fraction (float): The fraction of device memory to be used if the backend is GPU.
        kernel_profiler (bool): Whether to enable kernel function profiling.
        log (bool): Whether to enable logging.
    """
    if dim != 2 and dim != 3:
        raise ValueError(f"Keyword:: /dim/ should be either 2 or 3.")
    GlobalVariable.DIMENSION = dim
    
    if default_fp == "float64": default_fp = ti.f64
    elif default_fp == "float32": default_fp = ti.f32
    else: raise RuntimeError("Only ['float64', 'float32'] is available for default type of float")
    
    if default_ip == "int64": default_ip = ti.i64
    elif default_ip == "int32": default_ip = ti.i32
    else: raise RuntimeError("Only ['int64', 'int32'] is available for default type of int")

    if arch == "cpu":
        if (
            platform.system() == "Darwin"
            and platform.machine() == "arm64"
            and os.environ.get("GEOTAICHI_FORCE_CPU", "0") != "1"
        ):
            arch = "metal"
            print("Detected Apple Silicon, using Metal backend.")
        cpu_name = platform.processor()
        cpu_core = psutil.cpu_count(False)
        cpu_logic = psutil.cpu_count(True)
        print(f"Using device {cpu_name} (Core: {cpu_core}, Logic: {cpu_logic})")
        if cpu_max_num_threads == 0:
            ti.init(arch=ti.cpu, offline_cache=offline_cache, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, print_ir=print_ir, log_level=ti.ERROR, random_seed=random_seed)
        else:
            ti.init(arch=ti.cpu, cpu_max_num_threads=cpu_max_num_threads, offline_cache=offline_cache, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, print_ir=print_ir, log_level=ti.ERROR, random_seed=random_seed)
    elif arch == "gpu":
        GlobalVariable.USEGPU = True
        if platform.system() == "Darwin":
            print("Using GPU on macOS (Metal backend).")
            ti.init(arch=ti.metal, offline_cache=offline_cache, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, print_ir=print_ir, log_level=ti.ERROR, random_seed=random_seed)
        else:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_name = pynvml.nvmlDeviceGetName(handle=handle)
            gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle=handle)
            pynvml.nvmlShutdown()
            print(f"Using device {gpu_name} (Total: {bytes_to_GB(gpu_memory.total)}GB, Available: {bytes_to_GB(gpu_memory.free)}GB)")

            if device_memory_GB is None and device_memory_fraction is None:
                ti.init(arch=ti.gpu, offline_cache=offline_cache, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, print_ir=print_ir, log_level=ti.ERROR, random_seed=random_seed)
            elif not device_memory_GB is None:
                device_memory_GB = min(device_memory_GB, bytes_to_GB(gpu_memory.free))
                ti.init(arch=ti.gpu, offline_cache=offline_cache, device_memory_GB=device_memory_GB, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, print_ir=print_ir, log_level=ti.ERROR, random_seed=random_seed)
            elif not device_memory_fraction is None:
                device_memory_GB = min(device_memory_fraction, bytes_to_GB(gpu_memory.free) / bytes_to_GB(gpu_memory.total))
                ti.init(arch=ti.gpu, offline_cache=offline_cache, device_memory_fraction=device_memory_fraction, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, print_ir=print_ir, log_level=ti.ERROR, random_seed=random_seed)
    else:
        raise RuntimeError("arch is not recognized, please choose in the following: ['cpu', 'gpu']")
        
    if log:
        make_print_to_file()


def bytes_to_GB(sizes):
    return round(sizes / (1024 ** 3), 2)
  
