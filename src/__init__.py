# Copyright (c) 2023, multiscale geomechanics lab, Zhejiang University
# This file is from the GeoTaichi project, released under the GNU General Public License v3.0

__author__ = "Shi-Yihao, Guo-Ning"
__version__ = "0.1.0"
__license__ = "GNU License"
__description__ = "A High Performance Multiscale and Multiphysics Simulator"


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


def IGA(title=None, log=True):
    if title is None:
        title = __description__

    from src.iga.mainIGA import IGA

    return IGA(title=title, log=log)


def FEM(title=None, log=True):
    if title is None:
        title = __description__

    from src.fem.mainFEM import FEM

    return FEM(title=title, log=log)


def DEMPM(dem=None, mpm=None, title=None, coupling="Lagrangian", log=True):
    if title is None:
        title = __description__

    if dem is None:
        dem = DEM(title="", log=False)
        dem.sims.set_dem_coupling(True)
    if mpm is None:
        mpm = MPM(title="", log=False)
        mpm.sims.set_mpm_coupling(coupling)
    from src.mpdem.mainDEMPM import DEMPM

    return DEMPM(dem, mpm, log=log)


def MPDEM(dem=None, mpm=None, coupling="Lagrangian", title=None, log=True):
    if title is None:
        title = __description__

    if dem is None:
        dem = DEM(title="", log=False)
        dem.sims.set_dem_coupling(True)
    if mpm is None:
        mpm = MPM(title="", log=False)
        mpm.sims.set_mpm_coupling(coupling)
    from src.mpdem.mainDEMPM import DEMPM

    return DEMPM(dem, mpm, log=log)


def FEDEM(dem=None, fem=None, title=None, log=True):
    if title is None:
        title = __description__
    if dem is None:
        dem = DEM(title="", log=False)
    if fem is None:
        fem = FEM(title="", log=False)
    dem.sims.set_dem_coupling(True)
    from src.fedem.mainFEDEM import FEDEM as FEDEMCoupling

    return FEDEMCoupling(dem, fem, title=title, log=log)


def FEMPM(fem=None, mpm=None, title=None, log=True):
    if title is None:
        title = __description__
    if fem is None:
        fem = FEM(title="", log=False)
    if mpm is None:
        mpm = MPM(title="", log=False)
    # Native explicit FEM--MPM needs ParticleCoupling fields selected before
    # particle allocation.  Direct implicit MPM instead enters the monolithic
    # IPC system through its active grid DOFs and deliberately keeps the
    # standalone MPM coupling flag disabled.
    if not mpm.sims.is_direct_backend():
        mpm.sims.set_mpm_coupling("Lagrangian")
    from src.fempm.mainFEMPM import FEMPM as FEMPMCoupling

    return FEMPMCoupling(fem, mpm, title=title, log=log)


def IGAMPM(iga=None, mpm=None, title=None, log=True, **kwargs):
    if title is None:
        title = __description__

    if iga is None:
        iga = IGA(title="", log=False)
    if mpm is None:
        mpm = MPM(title="", log=False)
    from src.igampm.mainIGAMPM import IGAMPM

    return IGAMPM(iga, mpm, title=title, log=log, **kwargs)
