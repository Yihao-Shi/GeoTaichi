import taichi as ti

from src.levelset.BoundaryConditions import *
from src.levelset.WENO import *


@ti.func
def linear1d(data, posIndex):
    tot = data.shape
    i, ip = get_index_linear(posIndex[0], tot[0])
    return linear_uniform_grid(i, ip, posIndex[0], data[i], data[ip])

@ti.func
def bilinear(data, posIndex):
    tot = data.shape
    i, ip = get_index_linear(posIndex[0], tot[0])
    j, jp = get_index_linear(posIndex[1], tot[1])
    weno1 = linear_uniform_grid(i, ip, posIndex[0], data[i, j], data[ip, j])
    weno2 = linear_uniform_grid(i, ip, posIndex[0], data[i, jp], data[ip, jp])
    return linear_uniform_grid(j, jp, posIndex[1], weno1, weno2)

@ti.func
def trilinear(data, posIndex):
    tot = data.shape
    i, ip = get_index_linear(posIndex[0], tot[0])
    j, jp = get_index_linear(posIndex[1], tot[1])
    k, kp = get_index_linear(posIndex[2], tot[2])
    weno11 = linear_uniform_grid(i, ip, posIndex[0], data[i, j, k], data[ip, j, k])
    weno12 = linear_uniform_grid(i, ip, posIndex[0], data[i, jp, k], data[ip, jp, k])
    weno21 = linear_uniform_grid(i, ip, posIndex[0], data[i, j, kp], data[ip, j, kp])
    weno22 = linear_uniform_grid(i, ip, posIndex[0], data[i, jp, kp], data[ip, jp, kp])
    weno1 = linear_uniform_grid(j, jp, posIndex[1], weno11, weno12)
    weno2 = linear_uniform_grid(j, jp, posIndex[1], weno21, weno22)
    return linear_uniform_grid(k, kp, posIndex[2], weno1, weno2)

@ti.func
def weno31d(data, posIndex):
    tot = data.shape
    im, i, ip = get_index_weno3(posIndex[0], tot[0])
    return weno3_uniform_grid(im, i, ip, posIndex[0], data[im], data[i], data[ip])

@ti.func
def biweno3(data, posIndex):
    tot = data.shape
    im, i, ip = get_index_weno3(posIndex[0], tot[0])
    jm, j, jp = get_index_weno3(posIndex[1], tot[1])
    weno1 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jm], data[i, jm], data[ip, jm])
    weno2 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, j], data[i, j], data[ip, j])
    weno3 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jp], data[i, jp], data[ip, jp])
    return weno3_uniform_grid(jm, j, jp, posIndex[1], weno1, weno2, weno3)

@ti.func
def triweno3(data, posIndex):
    tot = data.shape
    im, i, ip = get_index_weno3(posIndex[0], tot[0])
    jm, j, jp = get_index_weno3(posIndex[1], tot[1])
    km, k, kp = get_index_weno3(posIndex[2], tot[2])
    weno11 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jm, km], data[i, jm, km], data[ip, jm, km])
    weno12 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, j, km], data[i, j, km], data[ip, j, km])
    weno13 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jp, km], data[i, jp, km], data[ip, jp, km])
    weno21 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jm, k], data[i, jm, k], data[ip, jm, k])
    weno22 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, j, k], data[i, j, k], data[ip, j, k])
    weno23 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jp, k], data[i, jp, k], data[ip, jp, k])
    weno31 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jm, kp], data[i, jm, kp], data[ip, jm, kp])
    weno32 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, j, kp], data[i, j, kp], data[ip, j, kp])
    weno33 = weno3_uniform_grid(im, i, ip, posIndex[0], data[im, jp, kp], data[i, jp, kp], data[ip, jp, kp])
    weno1 = weno3_uniform_grid(jm, j, jp, posIndex[1], weno11, weno12, weno13)
    weno2 = weno3_uniform_grid(jm, j, jp, posIndex[1], weno21, weno22, weno23)
    weno3 = weno3_uniform_grid(jm, j, jp, posIndex[1], weno31, weno32, weno33)
    return weno4_uniform_grid(km, k, kp, posIndex[2], weno1, weno2, weno3)

@ti.func
def weno41d(data, posIndex):
    tot = data.shape
    im, i, ip, ipp = get_index_weno4(posIndex[0], tot[0])
    return weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im], data[i], data[ip], data[ipp])

@ti.func
def biweno4(data, posIndex):
    tot = data.shape
    im, i, ip, ipp = get_index_weno4(posIndex[0], tot[0])
    jm, j, jp, jpp = get_index_weno4(posIndex[1], tot[1])
    weno1 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jm], data[i, jm], data[ip, jm], data[ipp, jm])
    weno2 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, j], data[i, j], data[ip, j], data[ipp, j])
    weno3 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jp], data[i, jp], data[ip, jp], data[ipp, jp])
    weno4 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jpp], data[i, jpp], data[ip, jpp], data[ipp, jpp])
    return weno4_uniform_grid(jm, j, jp, jpp, posIndex[1], weno1, weno2, weno3, weno4)

@ti.func
def triweno4(data, posIndex):
    tot = data.shape
    im, i, ip, ipp = get_index_weno4(posIndex[0], tot[0])
    jm, j, jp, jpp = get_index_weno4(posIndex[1], tot[1])
    km, k, kp, kpp = get_index_weno4(posIndex[2], tot[2])
    weno11 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jm, km], data[i, jm, km], data[ip, jm, km], data[ipp, jm, km])
    weno12 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, j, km], data[i, j, km], data[ip, j, km], data[ipp, j, km])
    weno13 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jp, km], data[i, jp, km], data[ip, jp, km], data[ipp, jp, km])
    weno14 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jpp, km], data[i, jpp, km], data[ip, jpp, km], data[ipp, jpp, km])
    weno21 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jm, k], data[i, jm, k], data[ip, jm, k], data[ipp, jm, k])
    weno22 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, j, k], data[i, j, k], data[ip, j, k], data[ipp, j, k])
    weno23 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jp, k], data[i, jp, k], data[ip, jp, k], data[ipp, jp, k])
    weno24 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jpp, k], data[i, jpp, k], data[ip, jpp, k], data[ipp, jpp, k])
    weno31 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jm, kp], data[i, jm, kp], data[ip, jm, kp], data[ipp, jm, kp])
    weno32 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, j, kp], data[i, j, kp], data[ip, j, kp], data[ipp, j, kp])
    weno33 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jp, kp], data[i, jp, kp], data[ip, jp, kp], data[ipp, jp, kp])
    weno34 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jpp, kp], data[i, jpp, kp], data[ip, jpp, kp], data[ipp, jpp, kp])
    weno41 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jm, kpp], data[i, jm, kpp], data[ip, jm, kpp], data[ipp, jm, kpp])
    weno42 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, j, kpp], data[i, j, kpp], data[ip, j, kpp], data[ipp, j, kpp])
    weno43 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jp, kpp], data[i, jp, kpp], data[ip, jp, kpp], data[ipp, jp, kpp])
    weno44 = weno4_uniform_grid(im, i, ip, ipp, posIndex[0], data[im, jpp, kpp], data[i, jpp, kpp], data[ip, jpp, kpp], data[ipp, jpp, kpp])
    weno1 = weno4_uniform_grid(jm, j, jp, jpp, posIndex[1], weno11, weno12, weno13, weno14)
    weno2 = weno4_uniform_grid(jm, j, jp, jpp, posIndex[1], weno21, weno22, weno23, weno24)
    weno3 = weno4_uniform_grid(jm, j, jp, jpp, posIndex[1], weno31, weno32, weno33, weno34)
    weno4 = weno4_uniform_grid(jm, j, jp, jpp, posIndex[1], weno41, weno42, weno43, weno44)
    return weno4_uniform_grid(km, k, kp, kpp, posIndex[2], weno1, weno2, weno3, weno4)

