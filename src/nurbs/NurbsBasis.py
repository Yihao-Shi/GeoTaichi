import numpy as np

from src.nurbs.Nurbs import *
from src.nurbs.Utilities import TOL


def BsplineBasis(ncp, xi, degree, knot_vector):
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL
    return one_basis_function(ncp, xi, degree, knot_vector)

def BsplineBasis1d(xi, degree, knot_vector):
    noFuncs = degree + 1
    N = np.zeros(noFuncs)
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    spanU = find_span(num_ctrlpts, xi, degree, knot_vector)
    Nurbs = basis_functions(spanU, xi, degree, knot_vector)

    kk = 0
    for k in range(degree + 1):
        N[kk] = Nurbs[k]
        kk += 1
    return N

def BsplineBasis2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v):
    noFuncs = (degree_u + 1) * (degree_v + 1)
    N = np.zeros(noFuncs)
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    NurbsU = basis_functions(spanU, xi, degree_u, knot_vector_u)
    NurbsV = basis_functions(spanV, eta, degree_v, knot_vector_v)

    kk = 0
    for j in range(degree_v + 1):
        for k in range(degree_u + 1):
            nmp = NurbsU[k] * NurbsV[j]
            N[kk] = nmp
            kk += 1
    return N

def BsplineBasis3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w):
    noFuncs = (degree_u + 1) * (degree_v + 1) * (degree_w + 1)
    N = np.zeros(noFuncs)
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    NurbsU = basis_functions(spanU, xi, degree_u, knot_vector_u)
    NurbsV = basis_functions(spanV, eta, degree_v, knot_vector_v)
    NurbsW = basis_functions(spanW, zeta, degree_w, knot_vector_w)

    kk = 0
    for i in range(degree_w + 1):
        for j in range(degree_v + 1):
            for k in range(degree_u + 1):
                nmp = NurbsU[k] * NurbsV[j] * NurbsW[i]
                N[kk] = nmp 
                kk += 1
    return N

def BsplineBasisDers(ncp, xi, degree, knot_vector):
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL
    return one_basis_function_ders_1st(ncp, xi, degree, knot_vector)


def BsplineBasisDers1d(xi, degree, knot_vector):
    noFuncs = degree + 1
    N, dNdxi = np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    spanU = find_span(num_ctrlpts, xi, degree, knot_vector)
    dNurbs = basis_function_ders(spanU, xi, degree, knot_vector)

    kk = 0
    for k in range(degree + 1):
        N[kk] = dNurbs[0, k]
        dNdxi[kk] = dNurbs[1, k]
        kk += 1
    return N, dNdxi


def BsplineBasisDers2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v):
    noFuncs = (degree_u + 1) * (degree_v + 1)
    N, dNdxi, dNdeta = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    dNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u)
    dNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v)

    kk = 0
    for j in range(degree_v + 1):
        for k in range(degree_u + 1):
            nmp = dNurbsU[0, k] * dNurbsV[0, j]
            N[kk] = nmp
            dNdxi[kk] = dNurbsU[1, k] * dNurbsV[0, j] 
            dNdeta[kk] = dNurbsU[0, k] * dNurbsV[1, j]
            kk += 1
    return N, dNdxi, dNdeta


def BsplineBasisDers3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w):
    noFuncs = (degree_u + 1) * (degree_v + 1) * (degree_w + 1)
    N, dNdxi, dNdeta, dNdzeta = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    dNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u)
    dNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v)
    dNurbsW = basis_function_ders(spanW, zeta, degree_w, knot_vector_w)

    kk = 0
    for i in range(degree_w + 1):
        for j in range(degree_v + 1):
            for k in range(degree_u + 1):
                nmp = dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i]
                N[kk] = nmp 
                dNdxi[kk] = dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i]
                dNdeta[kk] = dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i]
                dNdzeta[kk] = dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i]
                kk += 1
    return N, dNdxi, dNdeta, dNdzeta


def BsplineBasis2ndDers1d(xi, degree, knot_vector):
    noFuncs = degree + 1
    N, dNdxi, d2Ndxi2 = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    span = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    ddNurbs = basis_function_ders(span, xi, degree, knot_vector, 2)
    dNurbs = basis_function_ders(span, xi, degree, knot_vector)

    kk = 0
    for k in range(degree + 1):
        N[kk] = ddNurbs[0, k]
        dNdxi[kk] = ddNurbs[1, k]
        d2Ndxi2[kk] = ddNurbs[2, k]
        kk += 1
    return N, dNdxi, d2Ndxi2

def BsplineBasis2ndDers2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, weight):
    noFuncs = (degree_u + 1) * (degree_v + 1)
    N, dNdxi, dNdeta = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    d2Ndxi2, d2Ndeta2, d2Ndxe = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    ddNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u, 2)
    ddNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v, 2)

    kk = 0
    for j in range(degree_v + 1):
        for k in range(degree_u + 1):
            N[kk] = ddNurbsU[0, k] * ddNurbsV[0, j]
            dNdxi[kk] = ddNurbsU[1, k] * ddNurbsV[0, j]
            dNdeta[kk] = ddNurbsU[0, k] * ddNurbsV[1, j]
            d2Ndxi2[kk] = ddNurbsU[2, k] * ddNurbsV[0, j]
            d2Ndeta2[kk] = ddNurbsU[0, k] * ddNurbsV[2, j]
            d2Ndxe[kk] = ddNurbsU[1, k] * ddNurbsV[1, j]
            kk += 1
    return N, dNdxi, dNdeta, d2Ndxi2, d2Ndeta2, d2Ndxe


def BsplineBasis2ndDers3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, weight):
    noFuncs = (degree_u + 1) * (degree_v + 1) * (degree_w + 1)
    N, dNdxi, dNdeta, dNdzeta = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    d2Ndxi2, d2Ndeta2, d2Ndzeta2, d2Ndxe, d2Ndez, d2Ndxz = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    ddNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u, 2)
    ddNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v, 2)
    ddNurbsW = basis_function_ders(spanW, zeta, degree_w, knot_vector_w, 2)

    kk = 0
    for i in range(degree_w + 1):
        for j in range(degree_v + 1):
            for k in range(degree_u + 1):
                N[kk] = ddNurbsU[0, k] * ddNurbsV[0, j] * ddNurbsW[0, i]
                dNdxi[kk] = ddNurbsU[1, k] * ddNurbsV[0, j] * ddNurbsW[0, i]
                dNdeta[kk] = ddNurbsU[0, k] * ddNurbsV[1, j] * ddNurbsW[0, i]
                dNdzeta[kk] = ddNurbsU[0, k] * ddNurbsV[0, j] * ddNurbsW[1, i]
                d2Ndxi2[kk] = ddNurbsU[2, k] * ddNurbsV[0, j] * ddNurbsW[0, i]
                d2Ndeta2[kk] = ddNurbsU[0, k] * ddNurbsV[2, j] * ddNurbsW[0, i]
                d2Ndzeta2[kk] = ddNurbsU[0, k] * ddNurbsV[0, j] * ddNurbsW[2, i]
                d2Ndxe[kk] = ddNurbsU[1, k] * ddNurbsV[1, j] * ddNurbsW[0, i]
                d2Ndez[kk] = ddNurbsU[0, k] * ddNurbsV[1, j] * ddNurbsW[1, i]
                d2Ndxz[kk] = ddNurbsU[1, k] * ddNurbsV[0, j] * ddNurbsW[1, i]
                kk += 1
    return N, dNdxi, dNdeta, dNdzeta, d2Ndxi2, d2Ndeta2, d2Ndzeta2, d2Ndxe, d2Ndez, d2Ndxz


def BsplineBasisInterpolations1d(xi, degree, knot_vector, point, weight=None):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    span = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    Nurbs = basis_functions(span, xi, degree, knot_vector)
    
    position = np.zeros(point.shape[1])
    uind = span - degree
    for k in range(degree + 1):
        linear_id = uind + k
        position += Nurbs[k] * point[linear_id]
    return position 


def BsplineBasisInterpolations2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, point, weight=None):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    NurbsU = basis_functions(spanU, xi, degree_u, knot_vector_u)
    NurbsV = basis_functions(spanV, eta, degree_v, knot_vector_v)

    position = np.zeros(point.shape[1])
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            nmp = NurbsU[k] * NurbsV[j]
            position += nmp * point[linear_id] 
    return position


def BsplineBasisInterpolations3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, point, weight=None):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    NurbsU = basis_functions(spanU, xi, degree_u, knot_vector_u)
    NurbsV = basis_functions(spanV, eta, degree_v, knot_vector_v)
    NurbsW = basis_functions(spanW, zeta, degree_w, knot_vector_w)

    position = np.zeros(point.shape[1])
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                nmp = NurbsU[k] * NurbsV[j] * NurbsW[i]
                position += nmp * point[linear_id] 
    return position 


def BsplineBasisInterpolationsDers1d(xi, degree, knot_vector, point, weight=None):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    dNurbs = basis_function_ders(spanU, xi, degree, knot_vector)

    position, dirs = np.zeros(point.shape[1]), np.zeros(point.shape[1])
    uind = spanU - degree
    for k in range(degree + 1):
        linear_id = uind + k 
        nmp = dNurbs[0, k]
        position += nmp * point[linear_id]
        dirs += dNurbs[1, k] * point[linear_id]
    return position, dirs


def BsplineBasisInterpolationsDers2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, point, weight=None):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    dNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u)
    dNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v)

    position, dirs = np.zeros(point.shape[1]), np.zeros((2, point.shape[1]))
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            position += dNurbsU[0, k] * dNurbsV[0, j] * point[linear_id]
            dirs[0] += dNurbsU[1, k] * dNurbsV[0, j] * point[linear_id]
            dirs[1] += dNurbsU[0, k] * dNurbsV[1, j] * point[linear_id]
    return position, dirs


def BsplineBasisInterpolationsDers3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, point, weight=None):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    dNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u)
    dNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v)
    dNurbsW = basis_function_ders(spanW, zeta, degree_w, knot_vector_w)

    position, dirs = np.zeros(point.shape[1]), np.zeros((3, point.shape[1]))
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                nmp = dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i]
                position += nmp * point[linear_id]
                dirs[0] += dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * point[linear_id]
                dirs[1] += dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * point[linear_id]
                dirs[2] += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * point[linear_id]
    return position, dirs


def BsplineBasisInterpolations2ndDers1d(xi, degree, knot_vector, point, weight=None):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    span = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    ddNurbs = basis_function_ders(span, xi, degree, knot_vector, 2)

    position, dirs, ddirs = np.zeros(point.shape[1]), np.zeros((1, point.shape[1])), np.zeros((1, point.shape[1]))
    uind = span - degree
    for k in range(degree + 1):
        linear_id = uind + k 
        position += ddNurbs[0, k] * point[linear_id]
        dirs += ddNurbs[1, k] * point[linear_id]
        ddirs += ddNurbs[2, k] * point[linear_id]
    return position, dirs, ddirs


def BsplineBasisInterpolations2ndDers2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, point, weight=None):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    ddNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u, 2)
    ddNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v, 2)

    position, dirs, ddirs = np.zeros(point.shape[1]), np.zeros((2, point.shape[1])), np.zeros((3, point.shape[1]))
    # dirs[0] -> dNdxi, dirs[1] -> dNdeta
    # ddirs[0] -> d2Ndxi2, ddirs[1] -> d2Ndeta2, ddirs[2] -> d2Ndxideta
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            position += ddNurbsU[0, k] * ddNurbsV[0, j] * point[linear_id]
            dirs[0] += ddNurbsU[1, k] * ddNurbsV[0, j] * point[linear_id]
            dirs[1] += ddNurbsU[0, k] * ddNurbsV[1, j] * point[linear_id]
            ddirs[0] += ddNurbsU[2, k] * ddNurbsV[0, j] * point[linear_id]
            ddirs[1] += ddNurbsU[0, k] * ddNurbsV[2, j] * point[linear_id]
            ddirs[2] += ddNurbsU[1, k] * ddNurbsV[1, j] * point[linear_id]
    return position, dirs, ddirs


def BsplineBasisInterpolations2ndDers3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, point, weight=None):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    ddNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u, 2)
    ddNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v, 2)
    ddNurbsW = basis_function_ders(spanW, zeta, degree_w, knot_vector_w, 2)

    position, dirs, ddirs = np.zeros(point.shape[1]), np.zeros((3, point.shape[1])), np.zeros((6, point.shape[1]))
    # dirs[0] -> dNdxi, dirs[1] -> dNdeta, dirs[2] -> dNdzeta
    # ddirs[0] -> d2Ndxi2, ddirs[1] -> d2Ndeta2, ddirs[2] -> d2Ndzeta, ddirs[3] -> d2Ndxideta, ddirs[4] -> d2Ndetadzeta, ddirs[5] -> d2Ndxidzeta
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                position += ddNurbsU[0, k] * ddNurbsV[0, j] * ddNurbsW[0, i] * point[linear_id]
                dirs[0] += ddNurbsU[1, k] * ddNurbsV[0, j] * ddNurbsW[0, i] * point[linear_id]
                dirs[1] += ddNurbsU[0, k] * ddNurbsV[1, j] * ddNurbsW[0, i] * point[linear_id]
                dirs[2] += ddNurbsU[0, k] * ddNurbsV[0, j] * ddNurbsW[1, i] * point[linear_id]
                ddirs[0] += ddNurbsU[2, k] * ddNurbsV[0, j] * ddNurbsW[0, i] * point[linear_id]
                ddirs[1] += ddNurbsU[0, k] * ddNurbsV[2, j] * ddNurbsW[0, i] * point[linear_id]
                ddirs[2] += ddNurbsU[0, k] * ddNurbsV[0, j] * ddNurbsW[2, i] * point[linear_id]
                ddirs[3] += ddNurbsU[1, k] * ddNurbsV[1, j] * ddNurbsW[0, i] * point[linear_id]
                ddirs[4] += ddNurbsU[0, k] * ddNurbsV[1, j] * ddNurbsW[1, i] * point[linear_id]
                ddirs[5] += ddNurbsU[1, k] * ddNurbsV[0, j] * ddNurbsW[1, i] * point[linear_id]
    return position, dirs, ddirs        


def NurbsBasis(ncp, degree, xi, knot_vector, weight):
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    span = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    dNurbs = basis_function_ders(span, xi, degree, knot_vector)

    w_interp, dw_interp_dxi = 0., 0.
    for c in range(degree + 1):
        w_interp += dNurbs[0][c] * weight[span-degree+c]
        dw_interp_dxi += dNurbs[1][c] * weight[span-degree+c]
	
    Nip, dNip = one_basis_function_ders_1st(ncp, xi, degree, knot_vector)
    Rip = Nip * weight[ncp] / w_interp
    dRipdxi = weight[ncp] * (w_interp * dNip[1] - dw_interp_dxi * Nip) / (w_interp * w_interp)
    return Rip, dRipdxi


def NurbsBasis1d(xi, degree, knot_vector, weight):
    noFuncs = degree + 1
    N = np.zeros(noFuncs)
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    NurbsU = basis_functions(spanU, xi, degree, knot_vector)

    w, kk = 0., 0
    uind = spanU - degree
    for k in range(degree + 1):
        linear_id = uind + k 
        nmp = NurbsU[k] 
        w += nmp * weight[linear_id]
        N[kk] = nmp * weight[linear_id]
        kk += 1
    return N / w


def NurbsBasis2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, weight):
    noFuncs = (degree_u + 1) * (degree_v + 1)
    N = np.zeros(noFuncs)
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    NurbsU = basis_functions(spanU, xi, degree_u, knot_vector_u)
    NurbsV = basis_functions(spanV, eta, degree_v, knot_vector_v)

    w, kk = 0., 0
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            nmp = NurbsU[k] * NurbsV[j]
            w += nmp * weight[linear_id]
            N[kk] = nmp * weight[linear_id]
            kk += 1
    return N / w


def NurbsBasis3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, weight):
    noFuncs = (degree_u + 1) * (degree_v + 1) * (degree_w + 1)
    N = np.zeros(noFuncs)
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    NurbsU = basis_functions(spanU, xi, degree_u, knot_vector_u)
    NurbsV = basis_functions(spanV, eta, degree_v, knot_vector_v)
    NurbsW = basis_functions(spanW, zeta, degree_w, knot_vector_w)

    w, kk = 0., 0
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                nmp = NurbsU[k] * NurbsV[j] * NurbsW[i]
                w += nmp * weight[linear_id]
                N[kk] = nmp * weight[linear_id]
                kk += 1
    return N / w


def NurbsBasisDers1d(xi, degree, knot_vector, weight):
    noFuncs = degree + 1
    N, dNdxi = np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    dNurbs = basis_function_ders(spanU, xi, degree, knot_vector)

    w, dwdxi = 0., 0.
    uind = spanU - degree
    for k in range(degree + 1):
        linear_id = uind + k
        wgt = weight[linear_id]
        w += dNurbs[0, k] * wgt
        dwdxi += dNurbs[1, k] * wgt

    kk = 0
    uind = spanU - degree
    for k in range(degree + 1):
        linear_id = uind + k 
        fac = weight[linear_id] / (w * w)
        nmp = dNurbs[0, k]
        N[kk] = nmp * fac * w
        dNdxi[kk] = (dNurbs[1, k] * w - nmp * dwdxi) * fac
        kk += 1
    return N, dNdxi


def NurbsBasisDers2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, weight):
    noFuncs = (degree_u + 1) * (degree_v + 1)
    N, dNdxi, dNdeta = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    dNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u)
    dNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v)

    w, dwdxi, dwdet = 0., 0., 0.
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            wgt = weight[linear_id]
            w += dNurbsU[0, k] * dNurbsV[0, j] * wgt
            dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * wgt
            dwdet += dNurbsU[0, k] * dNurbsV[1, j] * wgt

    kk = 0
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            fac = weight[linear_id] / (w * w)
            nmp = dNurbsU[0, k] * dNurbsV[0, j]
            N[kk] = nmp * fac * w
            dNdxi[kk] = (dNurbsU[1, k] * dNurbsV[0, j] * w - nmp * dwdxi) * fac
            dNdeta[kk] = (dNurbsU[0, k] * dNurbsV[1, j] * w - nmp * dwdet) * fac
            kk += 1
    return N, dNdxi, dNdeta


def NurbsBasisDers3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, weight):
    noFuncs = (degree_u + 1) * (degree_v + 1) * (degree_w + 1)
    N, dNdxi, dNdeta, dNdzeta = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    dNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u)
    dNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v)
    dNurbsW = basis_function_ders(spanW, zeta, degree_w, knot_vector_w)

    w, dwdxi, dwdet, dwdze = 0., 0., 0., 0.
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                wgt = weight[linear_id]
                w += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
                dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
                dwdet += dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * wgt
                dwdze += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * wgt

    kk = 0
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                fac = weight[linear_id] / (w * w)
                nmp = dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i]
                N[kk] = nmp * fac * w
                dNdxi[kk] = (dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * w - nmp * dwdxi) * fac
                dNdeta[kk] = (dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * w - nmp * dwdet) * fac
                dNdzeta[kk] = (dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * w - nmp * dwdze) * fac
                kk += 1
    return N, dNdxi, dNdeta, dNdzeta
                

def NurbsBasis2ndDers1d(xi, degree, knot_vector, weight):
    noFuncs = degree + 1
    N, dNdxi, d2Ndxi2 = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    span = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    Nurbs = basis_function_ders(span, xi, degree, knot_vector, 2)

    w = 0.0
    dwdxi, d2wdxi = 0.0, 0.0
    uind = span - degree
    for k in range(degree + 1):
        linear_id = uind + k
        wgt = weight[linear_id]
        w += Nurbs[0, k] * wgt
        dwdxi += Nurbs[1, k] * wgt
        d2wdxi += Nurbs[2, k] * wgt

    kk = 0
    inv_w = 1 / w
    inv_w2 = 1 / w / w
    inv_w3 = 1 / w / w / w
    uind = span - degree
    for k in range(degree + 1):
        linear_id = uind + k
        wgt = weight[linear_id]
        fac = wgt / (w * w)
        nmp = Nurbs[0, k]

        N[kk] = nmp * fac * w
        dNdxi[kk] = (Nurbs[1, k] * w - nmp * dwdxi) * fac
        d2Ndxi2[kk] = (Nurbs[2, k] * inv_w - 2 * Nurbs[1, k] * dwdxi * inv_w2 - Nurbs[0, k] * d2wdxi * inv_w2 + 2 * Nurbs[0, k] * dwdxi * dwdxi * inv_w3) * wgt
        kk += 1
    return N, dNdxi, d2Ndxi2


def NurbsBasis2ndDers2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, weight):
    noFuncs = (degree_u + 1) * (degree_v + 1)
    N, dNdxi, dNdeta = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    d2Ndxi2, d2Ndeta2, d2Ndxe = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    NurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u, 2)
    NurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v, 2)

    w = 0.0
    dwdxi, dwdet = 0.0, 0.0
    d2wdxi, d2wdet, d2wdxe = 0.0, 0.0, 0.0
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            wgt = weight[linear_id]
            w += NurbsU[0, k] * NurbsV[0, j] * wgt
            dwdxi += NurbsU[1, k] * NurbsV[0, j] * wgt
            dwdet += NurbsU[0, k] * NurbsV[1, j] * wgt
            d2wdxi += NurbsU[2, k] * NurbsV[0, j] * wgt
            d2wdet += NurbsU[0, k] * NurbsV[2, j] * wgt
            d2wdxe += NurbsU[1, k] * NurbsV[1, j] * wgt

    kk = 0
    inv_w = 1 / w
    inv_w2 = 1 / w / w
    inv_w3 = 1 / w / w / w
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = NurbsU[0, k] * NurbsV[0, j]

            N[kk] = nmp * fac * w
            dNdxi[kk] = (NurbsU[1, k] * NurbsV[0, j] * w - nmp * dwdxi) * fac
            dNdeta[kk] = (NurbsU[0, k] * NurbsV[1, j] * w - nmp * dwdet) * fac
            d2Ndxi2[kk] = (NurbsU[2, k] * NurbsV[0, j] * inv_w - 2 * NurbsU[1, k] * NurbsV[0, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxi * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdxi * inv_w3) * wgt
            d2Ndeta2[kk] = (NurbsU[0, k] * NurbsV[2, j] * inv_w - 2 * NurbsU[0, k] * NurbsV[1, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdet * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdet * dwdet * inv_w3) * wgt
            d2Ndxe[kk] = (NurbsU[1, k]*NurbsV[1, j] * inv_w - NurbsU[1, k] * NurbsV[0, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[1, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxe * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdet * inv_w3) * wgt
            kk += 1
    return N, dNdxi, dNdeta, d2Ndxi2, d2Ndeta2, d2Ndxe


def NurbsBasis2ndDers3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, weight):
    noFuncs = (degree_u + 1) * (degree_v + 1) * (degree_w + 1)
    N, dNdxi, dNdeta, dNdzeta = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    d2Ndxi2, d2Ndeta2, d2Ndzeta2, d2Ndxe, d2Ndez, d2Ndxz = np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs), np.zeros(noFuncs)
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    NurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u, 2)
    NurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v, 2)
    NurbsW = basis_function_ders(spanW, zeta, degree_w, knot_vector_w, 2)

    w     = 0.0
    dwdxi, dwdet, dwdze = 0.0, 0.0, 0.0
    d2wdxi, d2wdet, d2wdze = 0.0, 0.0, 0.0
    d2wdxe, d2wdez, d2wdxz = 0.0, 0.0, 0.0
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                wgt = weight[linear_id]

                w += NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                dwdxi += NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                dwdet += NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * wgt
                dwdze += NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * wgt
                d2wdxi += NurbsU[2, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                d2wdet += NurbsU[0, k] * NurbsV[2, j] * NurbsW[0, i] * wgt
                d2wdze += NurbsU[0, k] * NurbsV[0, j] * NurbsW[2, i] * wgt
                d2wdxe += NurbsU[1, k] * NurbsV[1, j] * wgt
                d2wdez += NurbsV[1, j] * NurbsW[1, i] * wgt
                d2wdxz += NurbsU[1, k] * NurbsW[1, i] * wgt

    kk = 0
    inv_w = 1 / w
    inv_w2 = 1 / w / w
    inv_w3 = 1 / w / w / w
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                wgt = weight[linear_id]
                fac = wgt / (w * w)
                nmp = NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i]

                N[kk] = nmp * fac * w
                dNdxi[kk] = (NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * w - nmp * dwdxi) * fac
                dNdeta[kk] = (NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * w - nmp * dwdet) * fac
                dNdzeta[kk] = (NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * w - nmp * dwdze) * fac
                d2Ndxi2[kk] = (NurbsU[2, k] * NurbsV[0, j] * NurbsW[0, i] * inv_w - 2 * NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdxi * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdxi * dwdxi * inv_w3) * wgt
                d2Ndeta2[kk] = (NurbsU[0, k] * NurbsV[2, j] * NurbsW[0, i] * inv_w - 2 * NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdet * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdet * dwdet * inv_w3) * wgt
                d2Ndzeta2[kk] = (NurbsU[0, k] * NurbsV[0, j] * NurbsW[2, i] * inv_w - 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * dwdze * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdze * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdze * dwdze * inv_w3) * wgt
                d2Ndxe[kk] = (NurbsU[1, k]*NurbsV[1, j] * inv_w - NurbsU[1, k] * NurbsV[0, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[1, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxe * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdet * inv_w3) * wgt
                d2Ndez[kk] = (NurbsV[1, j]*NurbsW[1, i] * inv_w - NurbsV[1, j] * NurbsW[0, i] * dwdxi * inv_w2 - NurbsV[0, j] * NurbsW[1, i] * dwdet * inv_w2 - NurbsV[0, j] * NurbsW[0, i] * d2wdez * inv_w2 + 2 * NurbsV[0, j] * NurbsW[0, i] * dwdet * dwdze * inv_w3) * wgt
                d2Ndxz[kk] = (NurbsU[1, k]*NurbsW[1, i] * inv_w - NurbsW[1, i] * NurbsU[0, k] * dwdet * inv_w2 - NurbsW[0, i] * NurbsU[1, k] * dwdze * inv_w2 - NurbsU[0, k] * NurbsW[0, i] * d2wdxz * inv_w2 + 2 * NurbsU[0, k] * NurbsW[0, i] * dwdxi * dwdze * inv_w3) * wgt
                kk += 1
    return N, dNdxi, dNdeta, dNdzeta, d2Ndxi2, d2Ndeta2, d2Ndzeta2, d2Ndxe, d2Ndez, d2Ndxz


def NurbsBasisInterpolations1d(xi, degree, knot_vector, point, weight):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    span = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    Nurbs = basis_functions(span, xi, degree, knot_vector)
    
    position, w = np.zeros(point.shape[1]), 0.0
    uind = span - degree
    for k in range(degree + 1):
        linear_id = uind + k
        position += Nurbs[k] * weight[linear_id] * point[linear_id]
        w += Nurbs[k] * weight[linear_id]
    return position / w


def NurbsBasisInterpolations2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, field_var, weight):
    field_var = np.asarray(field_var)
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    NurbsU = basis_functions(spanU, xi, degree_u, knot_vector_u)
    NurbsV = basis_functions(spanV, eta, degree_v, knot_vector_v)

    nmp_tensor = np.einsum('i,j->ij', NurbsV, NurbsU)
    uinds = np.arange(degree_u + 1) + (spanU - degree_u)
    vinds = np.arange(degree_v + 1) + (spanV - degree_v)
    V_idx, U_idx = np.meshgrid(vinds, uinds, indexing='ij')
    linear_id = U_idx + V_idx * num_ctrlpts_u
    local_weight = weight[linear_id]
    local_field = field_var[linear_id]
    coef = nmp_tensor * local_weight
    position = np.tensordot(coef, local_field, axes=([0,1], [0,1]))
    wght = np.sum(coef)
    return position / wght


def NurbsBasisInterpolations3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, field_var, weight):
    field_var = np.asarray(field_var)
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    NurbsU = basis_functions(spanU, xi, degree_u, knot_vector_u)
    NurbsV = basis_functions(spanV, eta, degree_v, knot_vector_v)
    NurbsW = basis_functions(spanW, zeta, degree_w, knot_vector_w)

    nmp_tensor = np.einsum('i,j,k->ijk', NurbsW, NurbsV, NurbsU)
    uinds = np.arange(degree_u + 1) + (spanU - degree_u)
    vinds = np.arange(degree_v + 1) + (spanV - degree_v)
    winds = np.arange(degree_w + 1) + (spanW - degree_w)
    W_idx, V_idx, U_idx = np.meshgrid(winds, vinds, uinds, indexing='ij')
    linear_id = U_idx + V_idx * num_ctrlpts_u + W_idx * num_ctrlpts_u * num_ctrlpts_v
    local_weight = weight[linear_id]
    local_field = field_var[linear_id]
    coef = nmp_tensor * local_weight
    position = np.tensordot(coef, local_field, axes=([0,1,2], [0,1,2]))
    wght = np.sum(coef)
    return position / wght


def NurbsBasisInterpolationsDers1d(xi, degree, knot_vector, point, weight):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    dNurbs = basis_function_ders(spanU, xi, degree, knot_vector)

    w, dwdxi = 0., 0.
    uind = spanU - degree
    for k in range(degree + 1):
        linear_id = uind + k
        wgt = weight[linear_id]
        w += dNurbs[0, k] * wgt
        dwdxi += dNurbs[1, k] * wgt

    position, dirs = np.zeros(point.shape[1]), np.zeros(point.shape[1])
    uind = spanU - degree
    for k in range(degree + 1):
        linear_id = uind + k 
        fac = weight[linear_id] / (w * w)
        nmp = dNurbs[0, k]
        position += nmp * fac * w * point[linear_id]
        dirs += (dNurbs[1, k] * w - nmp * dwdxi) * fac * point[linear_id]
    return position, dirs


def NurbsBasisInterpolationsDers2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, point, weight):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    dNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u)
    dNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v)

    w, dwdxi, dwdet = 0., 0., 0.
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            wgt = weight[linear_id]
            w += dNurbsU[0, k] * dNurbsV[0, j] * wgt
            dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * wgt
            dwdet += dNurbsU[0, k] * dNurbsV[1, j] * wgt

    position, dirs = np.zeros(point.shape[1]), np.zeros((2, point.shape[1]))
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            fac = weight[linear_id] / (w * w)
            nmp = dNurbsU[0, k] * dNurbsV[0, j]
            position += nmp * fac * point[linear_id] * w
            dirs[0] += (dNurbsU[1, k] * dNurbsV[0, j] * w - nmp * dwdxi) * fac * point[linear_id]
            dirs[1] += (dNurbsU[0, k] * dNurbsV[1, j] * w - nmp * dwdet) * fac * point[linear_id]
    return position, dirs


def NurbsBasisInterpolationsDers3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, point, weight):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    dNurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u)
    dNurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v)
    dNurbsW = basis_function_ders(spanW, zeta, degree_w, knot_vector_w)

    w, dwdxi, dwdet, dwdze = 0., 0., 0., 0.
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                wgt = weight[linear_id]
                w += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
                dwdxi += dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * wgt
                dwdet += dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * wgt
                dwdze += dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * wgt

    position, dirs = np.zeros(point.shape[1]), np.zeros((3, point.shape[1]))
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                fac = weight[linear_id] / (w * w)
                nmp = dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[0, i]
                position += nmp * fac * w * point[linear_id]
                dirs[0] += (dNurbsU[1, k] * dNurbsV[0, j] * dNurbsW[0, i] * w - nmp * dwdxi) * fac * point[linear_id]
                dirs[1] += (dNurbsU[0, k] * dNurbsV[1, j] * dNurbsW[0, i] * w - nmp * dwdet) * fac * point[linear_id]
                dirs[2] += (dNurbsU[0, k] * dNurbsV[0, j] * dNurbsW[1, i] * w - nmp * dwdze) * fac * point[linear_id]
    return position, dirs


def NurbsBasisInterpolations2ndDers1d(xi, degree, knot_vector, point, weight):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    span = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    dNurbs = basis_function_ders(span, xi, degree, knot_vector, 2)

    w = 0.0
    dwdxi, d2wdxi = 0.0, 0.0
    uind = span - degree
    for k in range(degree + 1):
        linear_id = uind + k
        wgt = weight[linear_id]
        w += dNurbs[0, k] * wgt
        dwdxi += dNurbs[1, k] * wgt
        d2wdxi += dNurbs[2, k] * wgt

    position, dirs, ddirs = np.zeros(point.shape[1]), np.zeros((1, point.shape[1])), np.zeros((1, point.shape[1]))
    inv_w = 1 / w
    inv_w2 = 1 / w / w
    inv_w3 = 1 / w / w / w
    uind = span - degree
    for k in range(degree + 1):
        linear_id = uind + k
        wgt = weight[linear_id]
        fac = wgt / (w * w)
        nmp = dNurbs[0, k]

        position += nmp * fac * w * point[linear_id]
        dirs += (dNurbs[1, k] * w - nmp * dwdxi) * fac * point[linear_id]
        ddirs += (dNurbs[2, k] * inv_w - 2 * dNurbs[1, k] * dwdxi * inv_w2 - dNurbs[0, k] * d2wdxi * inv_w2 + 2 * dNurbs[0, k] * dwdxi * dwdxi * inv_w3) * wgt * point[linear_id]
    return position, dirs, ddirs


def NurbsBasisInterpolations2ndDers2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, point, weight):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    NurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u, 2)
    NurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v, 2)

    w = 0.0
    dwdxi, dwdet = 0.0, 0.0
    d2wdxi, d2wdet, d2wdxe = 0.0, 0.0, 0.0
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            wgt = weight[linear_id]
            w += NurbsU[0, k] * NurbsV[0, j] * wgt
            dwdxi += NurbsU[1, k] * NurbsV[0, j] * wgt
            dwdet += NurbsU[0, k] * NurbsV[1, j] * wgt
            d2wdxi += NurbsU[2, k] * NurbsV[0, j] * wgt
            d2wdet += NurbsU[0, k] * NurbsV[2, j] * wgt
            d2wdxe += NurbsU[1, k] * NurbsV[1, j] * wgt

    position, dirs, ddirs = np.zeros(point.shape[1]), np.zeros((2, point.shape[1])), np.zeros((3, point.shape[1]))
    # dirs[0] -> dNdxi, dirs[1] -> dNdeta
    # ddirs[0] -> d2Ndxi2, ddirs[1] -> d2Ndeta2, ddirs[2] -> d2Ndxideta
    inv_w = 1 / w
    inv_w2 = 1 / w / w
    inv_w3 = 1 / w / w / w
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            wgt = weight[linear_id]
            fac = wgt / (w * w)
            nmp = NurbsU[0, k] * NurbsV[0, j]

            position += nmp * fac * w * point[linear_id]
            dirs[0] += (NurbsU[1, k] * NurbsV[0, j] * w - nmp * dwdxi) * fac * point[linear_id]
            dirs[1] += (NurbsU[0, k] * NurbsV[1, j] * w - nmp * dwdet) * fac * point[linear_id]
            ddirs[0] += (NurbsU[2, k] * NurbsV[0, j] * inv_w - 2 * NurbsU[1, k] * NurbsV[0, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxi * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdxi * inv_w3) * wgt * point[linear_id]
            ddirs[1] += (NurbsU[0, k] * NurbsV[2, j] * inv_w - 2 * NurbsU[0, k] * NurbsV[1, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdet * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdet * dwdet * inv_w3) * wgt * point[linear_id]
            ddirs[2] += (NurbsU[1, k]*NurbsV[1, j] * inv_w - NurbsU[1, k] * NurbsV[0, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[1, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxe * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdet * inv_w3) * wgt * point[linear_id]
    return position, dirs, ddirs


def NurbsBasisInterpolations2ndDers3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, point, weight):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    NurbsU = basis_function_ders(spanU, xi, degree_u, knot_vector_u, 2)
    NurbsV = basis_function_ders(spanV, eta, degree_v, knot_vector_v, 2)
    NurbsW = basis_function_ders(spanW, zeta, degree_w, knot_vector_w, 2)

    w     = 0.0
    dwdxi, dwdet, dwdze = 0.0, 0.0, 0.0
    d2wdxi, d2wdet, d2wdze = 0.0, 0.0, 0.0
    d2wdxe, d2wdez, d2wdxz = 0.0, 0.0, 0.0
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                wgt = weight[linear_id]

                w += NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                dwdxi += NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                dwdet += NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * wgt
                dwdze += NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * wgt
                d2wdxi += NurbsU[2, k] * NurbsV[0, j] * NurbsW[0, i] * wgt
                d2wdet += NurbsU[0, k] * NurbsV[2, j] * NurbsW[0, i] * wgt
                d2wdze += NurbsU[0, k] * NurbsV[0, j] * NurbsW[2, i] * wgt
                d2wdxe += NurbsU[1, k] * NurbsV[1, j] * wgt
                d2wdez += NurbsV[1, j] * NurbsW[1, i] * wgt
                d2wdxz += NurbsU[1, k] * NurbsW[1, i] * wgt

    position, dirs, ddirs = np.zeros(point.shape[1]), np.zeros((3, point.shape[1])), np.zeros((6, point.shape[1]))
    # dirs[0] -> dNdxi, dirs[1] -> dNdeta, dirs[2] -> dNdzeta
    # ddirs[0] -> d2Ndxi2, ddirs[1] -> d2Ndeta2, ddirs[2] -> d2Ndzeta, ddirs[3] -> d2Ndxideta, ddirs[4] -> d2Ndetadzeta, ddirs[5] -> d2Ndxidzeta
    inv_w = 1 / w
    inv_w2 = 1 / w / w
    inv_w3 = 1 / w / w / w
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                wgt = weight[linear_id]
                fac = wgt / (w * w)
                nmp = NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i]

                position += nmp * fac * w * point[linear_id]
                dirs[0] += (NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * w - nmp * dwdxi) * fac * point[linear_id]
                dirs[1] += (NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * w - nmp * dwdet) * fac * point[linear_id]
                dirs[2] += (NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * w - nmp * dwdze) * fac * point[linear_id]
                ddirs[0] += (NurbsU[2, k] * NurbsV[0, j] * NurbsW[0, i] * inv_w - 2 * NurbsU[1, k] * NurbsV[0, j] * NurbsW[0, i] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdxi * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdxi * dwdxi * inv_w3) * wgt * point[linear_id]
                ddirs[1] += (NurbsU[0, k] * NurbsV[2, j] * NurbsW[0, i] * inv_w - 2 * NurbsU[0, k] * NurbsV[1, j] * NurbsW[0, i] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdet * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdet * dwdet * inv_w3) * wgt * point[linear_id]
                ddirs[2] += (NurbsU[0, k] * NurbsV[0, j] * NurbsW[2, i] * inv_w - 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[1, i] * dwdze * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * d2wdze * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * NurbsW[0, i] * dwdze * dwdze * inv_w3) * wgt * point[linear_id]
                ddirs[3] += (NurbsU[1, k]*NurbsV[1, j] * inv_w - NurbsU[1, k] * NurbsV[0, j] * dwdet * inv_w2 - NurbsU[0, k] * NurbsV[1, j] * dwdxi * inv_w2 - NurbsU[0, k] * NurbsV[0, j] * d2wdxe * inv_w2 + 2 * NurbsU[0, k] * NurbsV[0, j] * dwdxi * dwdet * inv_w3) * wgt * point[linear_id]
                ddirs[4] += (NurbsV[1, j]*NurbsW[1, i] * inv_w - NurbsV[1, j] * NurbsW[0, i] * dwdxi * inv_w2 - NurbsV[0, j] * NurbsW[1, i] * dwdet * inv_w2 - NurbsV[0, j] * NurbsW[0, i] * d2wdez * inv_w2 + 2 * NurbsV[0, j] * NurbsW[0, i] * dwdet * dwdze * inv_w3) * wgt * point[linear_id]
                ddirs[5] += (NurbsU[1, k]*NurbsW[1, i] * inv_w - NurbsW[1, i] * NurbsU[0, k] * dwdet * inv_w2 - NurbsW[0, i] * NurbsU[1, k] * dwdze * inv_w2 - NurbsU[0, k] * NurbsW[0, i] * d2wdxz * inv_w2 + 2 * NurbsU[0, k] * NurbsW[0, i] * dwdxi * dwdze * inv_w3) * wgt * point[linear_id]
    return position, dirs, ddirs        


def NurbsDxDweight1d(xi, degree, knot_vector, point, weight):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector = np.asarray(knot_vector)
    numKnot = knot_vector.shape[0]
    num_ctrlpts_u = numKnot - degree - 1
    if abs(xi - knot_vector[numKnot - 1]) < TOL:
        xi = knot_vector[numKnot - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree, knot_vector)
    NurbsU = basis_functions(spanU, xi, degree, knot_vector)

    w = 0.
    position = np.zeros(point.shape[1])
    uind = spanU - degree
    for k in range(degree + 1):
        linear_id = uind + k
        wgt = weight[linear_id]
        w += NurbsU[k] * wgt
        position += NurbsU[k] * wgt * point[linear_id]

    dxdw = np.zeros((degree + 1, point.shape[1]))
    uind = spanU - degree
    kk = 0
    for k in range(degree + 1):
        linear_id = uind + k 
        nmp = NurbsU[k]
        dxdw[kk] = (w * nmp * point[linear_id] - nmp * position) / (w * w)
        kk += 1
    return dxdw


def NurbsDxDweight2d(xi, eta, degree_u, degree_v, knot_vector_u, knot_vector_v, weight):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v = np.array(knot_vector_u), np.array(knot_vector_v)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    NurbsU = basis_functions(spanU, xi, degree_u, knot_vector_u)
    NurbsV = basis_functions(spanV, eta, degree_v, knot_vector_v)

    w = 0.
    position = np.zeros(point.shape[1])
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            wgt = weight[linear_id]
            w += NurbsU[k] * NurbsV[j] * wgt
            position += NurbsU[k] * NurbsV[j] * wgt * point[linear_id]

    dxdw = np.zeros(((degree_v + 1) * (degree_u + 1), point.shape[1]))
    kk = 0
    uind = spanU - degree_u
    for j in range(degree_v + 1):
        vind = spanV - degree_v + j
        for k in range(degree_u + 1):
            linear_id = uind + k + vind * num_ctrlpts_u
            nmp = NurbsU[k] * NurbsV[j]
            dxdw[kk] = (w * nmp * point[linear_id] - nmp * position) / (w * w)
            kk += 1
    return dxdw


def NurbsDxDweight3d(xi, eta, zeta, degree_u, degree_v, degree_w, knot_vector_u, knot_vector_v, knot_vector_w, weight):
    point = np.asarray(point)
    if point.ndim == 1:
        point = point.reshape((-1, 1))
    knot_vector_u, knot_vector_v, knot_vector_w = np.array(knot_vector_u), np.array(knot_vector_v), np.array(knot_vector_w)
    numKnotU = knot_vector_u.shape[0]
    numKnotV = knot_vector_v.shape[0]
    numKnotW = knot_vector_w.shape[0]
    num_ctrlpts_u = numKnotU - degree_u - 1
    num_ctrlpts_v = numKnotV - degree_v - 1
    num_ctrlpts_w = numKnotW - degree_w - 1
    if abs(xi - knot_vector_u[numKnotU - 1]) < TOL:
        xi = knot_vector_u[numKnotU - 1] - TOL
    if abs(eta - knot_vector_v[numKnotV - 1]) < TOL:
        eta = knot_vector_v[numKnotV - 1] - TOL
    if abs(zeta - knot_vector_w[numKnotW - 1]) < TOL:
        zeta = knot_vector_w[numKnotW - 1] - TOL

    spanU = find_span(num_ctrlpts_u, xi, degree_u, knot_vector_u)
    spanV = find_span(num_ctrlpts_v, eta, degree_v, knot_vector_v)
    spanW = find_span(num_ctrlpts_w, zeta, degree_w, knot_vector_w)
    NurbsU = basis_functions(spanU, xi, degree_u, knot_vector_u)
    NurbsV = basis_functions(spanV, eta, degree_v, knot_vector_v)
    NurbsW = basis_functions(spanW, zeta, degree_w, knot_vector_w)

    w = 0.
    position = np.zeros(point.shape[1])
    uind = spanU - degree_u
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                wgt = weight[linear_id]
                w += NurbsU[k] * NurbsV[j] * NurbsW[i] * wgt
                position += NurbsU[k] * NurbsV[j] * NurbsW[i] * wgt * point[linear_id]

    dxdw = np.zeros(((degree_v + 1) * (degree_u + 1), point.shape[1]))
    kk = 0
    for i in range(degree_w + 1):
        wind = spanW - degree_w + i
        for j in range(degree_v + 1):
            vind = spanV - degree_v + j
            for k in range(degree_u + 1):
                linear_id = uind + k + vind * num_ctrlpts_u + wind * num_ctrlpts_u * num_ctrlpts_v
                nmp = NurbsU[k] * NurbsV[j] * NurbsW[i]
                dxdw[kk] = (w * nmp * point[linear_id] - nmp * position) / (w * w)
                kk += 1
    return dxdw