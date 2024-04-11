# Copyright (c) 2023, multiscale geomechanics lab, Zhejiang University
# This file is from the GeoTaichi project, released under the GNU General Public License v3.0

__author__ = "Shi-Yihao, Guo-Ning"
__version__ = "0.1.0"
__license__ = "GNU License"

import taichi as ti
import psutil, pynvml, platform
import sys, os, datetime  

from src import DEM, MPM, DEMPM

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
    

def init(arch="gpu", cpu_max_num_threads=0, offline_cache=True, debug=False, default_fp="float64", default_ip="int32", device_memory_GB=None, device_memory_fraction=None, kernel_profiler=False, log=True):
    """
    Initializes the Taichi runtime environment.
    Args:
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
    初始化函数，用于设置Taichi的运行环境。
    参数:
    arch: 运行架构，可选 "cpu" 或 "gpu"。
    cpu_max_num_threads: 若运行后端为CPU，CPU最大线程数，默认值为该CPU最大的线程数。
    offline_cache: 选择是否储存编译后的文件，默认值为True。
    debug: 是否启用调试模式。
    default_fp: 默认浮点数类型，可选 "float64" 或 "float32"。
    default_ip: 默认整数类型，可选 "int64" 或 "int32"。
    device_memory_GB: 若运行后端为GPU,预先分配GPU内存大小（GB），如果设备内存小于2GB，则使用默认设置。
    device_memory_fraction: 若运行后端为GPU,设备内存占用比例
    kernel_profiler: 是否启用核函数计时。
    log: 是否启用日志记录。
    """
    if default_fp == "float64": default_fp = ti.f64
    elif default_fp == "float32": default_fp = ti.f32
    else: raise RuntimeError("Only ['float64', 'float32'] is available for default type of float")
    
    if default_ip == "int64": default_ip = ti.i64
    elif default_ip == "int32": default_ip = ti.i32
    else: raise RuntimeError("Only ['int64', 'int32'] is available for default type of int")

    if arch == "cpu":
        cpu_name = platform.processor()
        cpu_core = psutil.cpu_count(False)
        cpu_logic = psutil.cpu_count(True)
        print(f"Using device {cpu_name} (Core: {cpu_core}, Logic: {cpu_logic})")
        if cpu_max_num_threads == 0:
            ti.init(arch=ti.cpu, offline_cache=offline_cache, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, log_level=ti.ERROR)
        else:
            ti.init(arch=ti.cpu, cpu_max_num_threads=cpu_max_num_threads, offline_cache=offline_cache, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, log_level=ti.ERROR)
    elif arch == "gpu":
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(handle=handle)
        gpu_memory = pynvml.nvmlDeviceGetMemoryInfo(handle=handle)
        pynvml.nvmlShutdown()
        print(f"Using device {gpu_name} (Total: {bytes_to_GB(gpu_memory.total)}GB, Available: {bytes_to_GB(gpu_memory.free)}GB)")

        if device_memory_GB is None and device_memory_fraction is None:
            ti.init(arch=ti.gpu, offline_cache=offline_cache, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, log_level=ti.ERROR)
        elif not device_memory_GB is None:
            device_memory_GB = min(device_memory_GB, bytes_to_GB(gpu_memory.free))
            ti.init(arch=ti.gpu, offline_cache=offline_cache, device_memory_GB=device_memory_GB, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, log_level=ti.ERROR)
        elif not device_memory_fraction is None:
            device_memory_GB = min(device_memory_GB, bytes_to_GB(gpu_memory.free) / bytes_to_GB(gpu_memory.total))
            ti.init(arch=ti.gpu, offline_cache=offline_cache, device_memory_fraction=device_memory_fraction, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, log_level=ti.ERROR)
    else:
        raise RuntimeError("arch is not recognized, please choose in the following: ['cpu', 'gpu']")
        
    if log:
        make_print_to_file()


def bytes_to_GB(sizes):
    return round(sizes / (1024 ** 3), 2)
  
