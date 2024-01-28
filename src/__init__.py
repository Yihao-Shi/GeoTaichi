# Copyright (c) 2023, multiscale geomechanics lab, Zhejiang University
# This file is from the GeoTaichi project, released under the GNU General Public License v3.0

__author__ = "Shi-Yihao, Guo-Ning"
__version__ = "0.1.0"
__license__ = "GNU License"


def DEM():
    from src.dem.mainDEM import DEM 
    return DEM()
    
def MPM():
    from src.mpm.mainMPM import MPM 
    return MPM()
    
def DEMPM():
    from src.dempm.mainDEMPM import DEMPM 
    return DEMPM()
       
        

