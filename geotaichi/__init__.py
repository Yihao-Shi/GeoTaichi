# Copyright (c) 2023, multiscale geomechanics lab, Zhejiang University
# This file is from the GeoTaichi project, released under the GNU General Public License v3.0

__author__ = "Shi-Yihao, Guo-Ning"
__version__ = "0.1.0"
__license__ = "GNU License"

import taichi as ti
import psutil

from src import DEM, MPM, DEMPM

def init(arch="gpu", cpu_max_num_threads=0, offline_cache=True, debug=False, default_fp="float64", default_ip="int32", device_memory_GB=2, device_memory_fraction=0.8, kernel_profiler=False):
    
    
    if default_fp == "float64": default_fp = ti.f64
    elif default_fp == "float32": default_fp = ti.f32
    else: raise RuntimeError("Only ['float64', 'float32'] is available for default type of float")
    
    if default_ip == "int64": default_ip = ti.i64
    elif default_ip == "int32": default_ip = ti.i32
    else: raise RuntimeError("Only ['int64', 'int32'] is available for default type of int")

    if arch == "cpu":
        if cpu_max_num_threads == 0:
            ti.init(arch=ti.cpu, offline_cache=offline_cache, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, log_level=ti.ERROR)
        else:
            ti.init(arch=ti.cpu, cpu_max_num_threads=cpu_max_num_threads, offline_cache=offline_cache, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, log_level=ti.ERROR)
    elif arch == "gpu":
        if device_memory_GB <= 2:
            ti.init(arch=ti.gpu, offline_cache=offline_cache, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, log_level=ti.ERROR)
        elif device_memory_GB < round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2):
            ti.init(arch=ti.gpu, offline_cache=offline_cache, device_memory_GB=device_memory_GB, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, log_level=ti.ERROR)
        else:
            ti.init(arch=ti.gpu, offline_cache=offline_cache, device_memory_fraction=device_memory_fraction, debug=debug, default_fp=default_fp, default_ip=default_ip, kernel_profiler=kernel_profiler, log_level=ti.ERROR)
    else:
        raise RuntimeError("arch is not recognized, please choose in the following: ['cpu', 'gpu']")
       
        

