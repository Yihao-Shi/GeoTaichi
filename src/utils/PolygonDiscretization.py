import numpy as np
import random
from shapely.geometry import Polygon, LinearRing


def IsSimplePoly(poly):
    poly_ring = poly.boundary
    if poly_ring.is_ring and list(poly.interiors) == []:
        return True
    else:
        return False


def GetPolyVex(poly):
    return np.asarray(poly.exterior.coords)


def VexCCW(poly):
    return 1 if LinearRing(poly.exterior).is_ccw else -1


def GetDivideVexIdx(poly):
    dividevex_idx_li = [] 
    dividevex_arg_li = []
    vex_arr = GetPolyVex(poly) 
    vex_arr = vex_arr[:-1,:]
    nums = vex_arr.shape[0]
    if nums <= 3:
        return vex_arr, dividevex_idx_li, dividevex_arg_li
    
    pm = VexCCW(poly) 
    for i in range(nums):
        v = vex_arr[i,:]
        l = vex_arr[i-1,:]
        r = vex_arr[(i+1)%nums,:]
        unit_vector = [1., 1., 1.]
        fir_vector = v - l 
        sec_vector = r - v
        A = np.array([unit_vector, fir_vector,sec_vector])
        if pm*np.linalg.det(A) > 0:
            remainvex_arr = np.concatenate([vex_arr[:i,:],vex_arr[i+1:,:]],axis=0)
            remain_poly = Polygon(remainvex_arr)
            tri = Polygon([l,v,r])
            if (remain_poly.is_valid
                and remain_poly.intersection(tri).area < 1e-8 
                and poly.equals(remain_poly.union(tri))):
                
                dividevex_idx_li.append(i) 
                arc = np.arccos(-np.dot(fir_vector,sec_vector)/np.linalg.norm(fir_vector)/np.linalg.norm(sec_vector))
                dividevex_arg_li.append(arc)
    return vex_arr, dividevex_idx_li, dividevex_arg_li


def GetDivTri(poly, tris):
    vex_arr, dv_idx_li, dv_arc_li = GetDivideVexIdx(poly)
    nums = vex_arr.shape[0]
    if nums <= 3: 
        tris.append(poly)
        return tris
    idx = dv_idx_li[np.argmin(np.array(dv_arc_li))]
    v = vex_arr[idx, :]
    l = vex_arr[idx-1, :]
    r = vex_arr[(idx+1)%nums, :]
    tri = Polygon([l,v,r])
    tris.append(tri)
    remain_vex_arr = np.concatenate([vex_arr[:idx,:],vex_arr[idx+1:,:]],axis=0)
    remain_poly = Polygon(remain_vex_arr)
    GetDivTri(remain_poly,tris)
    return tris


def PolyPretreatment(poly_arr):
    temp = poly_arr - np.min(poly_arr,axis=0)
    return temp / np.max(temp), np.max(temp), np.min(poly_arr,axis=0)


def MinAngle(tri):
    point = np.asarray(tri.exterior.coords)
    arc_li = []
    for i in range(3):
        j = (i+1)%3; k=(i+2)%3
        a = np.linalg.norm(point[i,:] - point[j,:])
        b = np.linalg.norm(point[j,:] - point[k,:])
        c = np.linalg.norm(point[k,:] - point[i,:])
        arc = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
        arc_li.append(arc)
    return min(arc_li)


def OptDiv(poly4_vex_arr):
    tri1 = Polygon(poly4_vex_arr[[0,1,2]])
    tri2 = Polygon(poly4_vex_arr[[0,2,3]])
    arc1 = min([MinAngle(tri1),MinAngle(tri2)])

    tri3 = Polygon(poly4_vex_arr[[0,1,3]])
    tri4 = Polygon(poly4_vex_arr[[1,2,3]])
    arc2 = min([MinAngle(tri3),MinAngle(tri4)])

    if arc1 >= arc2:
        return tri1,tri2
    else:
        return tri3,tri4


def OptAlltris(tris):
    random.shuffle(tris)
    nums = len(tris)
    for i in range(nums):
        tri_i = tris[i]
        for j in range(i+1,nums):
            tri_j = tris[j]
            if tri_i.intersection(tri_j).length > 1e-10:
                u = tri_i.union(tri_j)
                vex_arr, dv_vex_li, _=GetDivideVexIdx(u)
                if len(dv_vex_li) == 4:
                    a,b = OptDiv(vex_arr)
                    flag = True
                    for idx in set(range(nums)) - {i,j}:
                        if a.intersection(tris[idx]).area > 0. or b.intersection(tris[idx]).area > 0.:
                            flag = False
                    if flag:
                        tris[i],tris[j] = a,b
    return tris

