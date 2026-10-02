# Copyright (c) 2023, multiscale geomechanics lab, Zhejiang University
# This file is from the GeoTaichi project, released under the GNU General Public License v3.0

__author__ = "Shi-Yihao, Guo-Ning"
__version__ = "0.1.0"
__license__ = "GNU License"
__description__ = 'A High Performance Multiscale and Multiphysics Simulator'


def DEM(title=None, log=True):
    if title is None:
        title = __description__ 
        
    from src.dem.mainDEM import DEM 
    return DEM(title=title, log=log)
    
    
def MPM(title=None, log=True):
    if title is None:
        title = __description__ 
        
    from src.mpm.mainMPM import MPM 
    return MPM(title=title, log=log)
    
    
def DEMPM(dem=None, mpm=None, title=None, coupling="Lagrangian", log=True):
    if title is None:
        title = __description__ 
        
    if dem is None:
        dem = DEM(title='', log=False)
        dem.sims.set_dem_coupling(True)
    if mpm is None:
        mpm = MPM(title='', log=False)
        mpm.sims.set_mpm_coupling(coupling)
    from src.mpdem.mainDEMPM import DEMPM 
    return DEMPM(dem, mpm, log=log)
       
        
def MPDEM(dem=None, mpm=None, coupling="Lagrangian", title=None, log=True):
    if title is None:
        title = __description__ 
        
    if dem is None:
        dem = DEM(title='', log=False)
        dem.sims.set_dem_coupling(True)
    if mpm is None:
        mpm = MPM(title='', log=False)
        mpm.sims.set_mpm_coupling(coupling)
    from src.mpdem.mainDEMPM import DEMPM 
    return DEMPM(dem, mpm, log=log)
