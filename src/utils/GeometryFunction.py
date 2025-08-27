import taichi as ti

from src.utils.constants import PI, Threshold
from src.utils.Quaternion import RodriguesRotationMatrix
from src.utils.ScalarFunction import clamp, sgn
from src.utils.TypeDefination import vec3f
from src.utils.VectorFunction import dot2


@ti.func
def SphereTriangleIntersectionArea(position, radius, vertice1, vertice2, vertice3, norm):
    area1 = processSegment(position, radius, vertice1, vertice2, norm)
    area2 = processSegment(position, radius, vertice2, vertice3, norm)
    area3 = processSegment(position, radius, vertice3, vertice1, norm)
    return area1 + area2 + area3


@ti.func
def processSegment(position, radius, initialVertex, finalVertex, norm):
    segmentDisplacement = finalVertex - initialVertex
    segmentLength = segmentDisplacement.norm()
    centerToInitialDisplacement = initialVertex - position
    leftX = centerToInitialDisplacement.dot(segmentDisplacement) / segmentLength
    rightX = leftX + segmentLength
    wedge_product = segmentDisplacement.cross(centerToInitialDisplacement)
    y = wedge_product.dot(norm) / segmentLength
    return processSegmentStandardGeometry(radius, leftX, rightX, y)


@ti.func
def processSegmentStandardGeometry(radius, leftX, rightX, y):
    area = 0.
    if y * y > radius * radius:
        area = processNonIntersectingRegion(radius, leftX, rightX, y)
    else:
        intersectionX = ti.sqrt(radius * radius - y * y)
        if leftX < -intersectionX:
            leftRegionRightEndpoint = ti.min(-intersectionX, rightX)
            area += processNonIntersectingRegion(radius, leftX, leftRegionRightEndpoint, y)
        if intersectionX < rightX:
            rightRegionLeftEndpoint = ti.max(intersectionX, leftX)
            area += processNonIntersectingRegion(radius, rightRegionLeftEndpoint, rightX, y)
        middleRegionLeftEndpoint = ti.max(-intersectionX, leftX)
        middleRegionRightEndpoint = ti.min(intersectionX, rightX)
        middleRegionLength = ti.max(middleRegionRightEndpoint - middleRegionLeftEndpoint, 0)
        area -= 0.5 * middleRegionLength * y
    return area


@ti.func
def processNonIntersectingRegion(radius, leftX, rightX, y):
    initialTheta = ti.atan2(y, leftX)
    finalTheta = ti.atan2(y, rightX)
    deltaTheta = finalTheta - initialTheta
    if deltaTheta < -PI:
        deltaTheta += 2 * PI
    elif deltaTheta > PI:
        deltaTheta -= 2 * PI
    return 0.5 * radius * radius * deltaTheta


@ti.func
def SphereTriangleIntersectionAreaAndCentroid(position, radius, vertice1, vertice2, vertice3, norm):
    area1, centroid1 = processSegmentAndCentroid(position, radius, vertice1, vertice2, norm)
    area2, centroid2 = processSegmentAndCentroid(position, radius, vertice2, vertice3, norm)
    area3, centroid3 = processSegmentAndCentroid(position, radius, vertice3, vertice1, norm)
    area = area1 + area2 + area3
    weighted_centroid = centroid1 + centroid2 + centroid3
    weighted_centroid = weighted_centroid / area if area > 0. else vec3f(0., 0., 0.)
    return area, weighted_centroid


@ti.func
def processSegmentAndCentroid(position, radius, initialVertex, finalVertex, norm):
    segmentDisplacement = finalVertex - initialVertex
    segmentLength = segmentDisplacement.norm()
    centerToInitialDisplacement = initialVertex - position
    leftX = centerToInitialDisplacement.dot(segmentDisplacement) / segmentLength
    rightX = leftX + segmentLength
    wedge_product = segmentDisplacement.cross(centerToInitialDisplacement)
    y = wedge_product.dot(norm) / segmentLength 
    area, centroid = 0., vec3f(0., 0., 0.)
    if y * y > radius * radius:
        darea, center = processNonIntersectingCentroid(radius, leftX, rightX, y)
        AB = centerToInitialDisplacement
        AC = centerToInitialDisplacement + segmentDisplacement
        ABnorm = AB.norm()
        ACnorm = AC.norm()
        centroid += darea * (position + (ACnorm / (ABnorm + ACnorm) * AB + ABnorm / (ACnorm + ABnorm) * AC).normalized() * center)
        area += darea
    else:
        intersectionX = ti.sqrt(radius * radius - y * y)
        if leftX < -intersectionX:
            leftRegionRightEndpoint = ti.min(-intersectionX, rightX)
            darea, center = processNonIntersectingCentroid(radius, leftX, leftRegionRightEndpoint, y)
            AB = centerToInitialDisplacement
            AC = centerToInitialDisplacement + segmentDisplacement / segmentLength * (leftRegionRightEndpoint - leftX)
            ABnorm = AB.norm()
            ACnorm = AC.norm()
            centroid += darea * (position + (ACnorm / (ABnorm + ACnorm) * AB + ABnorm / (ACnorm + ABnorm) * AC).normalized() * center)
            area += darea
        if intersectionX < rightX:
            rightRegionLeftEndpoint = ti.max(intersectionX, leftX)
            darea, center = processNonIntersectingCentroid(radius, rightRegionLeftEndpoint, rightX, y)
            AB = finalVertex - position - segmentDisplacement / segmentLength * (rightX - rightRegionLeftEndpoint)
            AC = finalVertex - position
            ABnorm = AB.norm()
            ACnorm = AC.norm()
            centroid += darea * (position + (ACnorm / (ABnorm + ACnorm) * AB + ABnorm / (ACnorm + ABnorm) * AC).normalized() * center)
            area += darea
        middleRegionLeftEndpoint = ti.max(-intersectionX, leftX)
        middleRegionRightEndpoint = ti.min(intersectionX, rightX)
        middleRegionLength = ti.max(middleRegionRightEndpoint - middleRegionLeftEndpoint, 0)
        darea = -0.5 * middleRegionLength * y
        centroid += darea * 1./3. * (position + initialVertex + segmentDisplacement / segmentLength * (middleRegionLeftEndpoint - leftX) + finalVertex - segmentDisplacement / segmentLength * (rightX - middleRegionRightEndpoint))
        area += darea
    return area, centroid


@ti.func
def processNonIntersectingCentroid(radius, leftX, rightX, y):
    initialTheta = ti.atan2(y, leftX)
    finalTheta = ti.atan2(y, rightX)
    deltaTheta = finalTheta - initialTheta
    if deltaTheta < -PI:
        deltaTheta += 2 * PI
    elif deltaTheta > PI:
        deltaTheta -= 2 * PI
    alpha = 0.5 * deltaTheta
    darea = 0.5 * radius * radius * deltaTheta
    inv_alpha = 1. / alpha if alpha > 0. else 0.
    center = 2./3. * radius * ti.sin(alpha) * inv_alpha
    return darea, center


@ti.func
def IsLineIntersectTriangle(linePoint1, linePoint2, _planePoint1, _planePoint2, _planePoint3):
    is_intersection = 0
    
    u =_planePoint3 - _planePoint1    
    v =_planePoint2 - _planePoint1
    n = u * v
    dir = linePoint2 - linePoint1
    w0 = linePoint1 - _planePoint1
    a = -n.dot(w0)
    b = n.dot(dir)

    if ti.abs(b) > 0.:
        r = a / b
        if r > 0.0:
            I = linePoint1 + dir * r         

            uu = u.dot(u)
            uv = u.dot(v)
            vv = v.dot(v)
            w = I - _planePoint1
            wu = w.dot(u)
            wv = w.dot(v)
            D = uv * uv - uu * vv

            s = (uv * wv - vv * wu) / D
            if s < 0.0 or s > 1.0:     
                is_intersection = 1
            t = (uv * wu - uu * wv) / D
            if t >= 0.0 and (s + t) <= 1.0:  
                is_intersection = 1
    return is_intersection


@ti.func
def IntersectRaySphere(_P, _D, _SphereCenter, _dRadius, _pLength, _pIntersection):
    is_intersection = 0
    m = _P - _SphereCenter
    b = m.dot(_D)
    dC = m.dot(m) - _dRadius * _dRadius
    if dC <= 0 or b <= 0: 
        dDiscr = b * b - dC
        if dDiscr >= 0: 
            _pLength = - b
            if _pLength < 0: _pLength = 0
            _pIntersection = _P + _D * _pLength
            is_intersection = 1
    return is_intersection, _pIntersection


# refers to Arvo, A Simple Method for Box-Sphere Intersection Testing, Graphics Gems, pp. 247-250, 1993
@ti.func
def IsCircleIntersectRectangle(circle_center, radius, recenter, hlength):
    v = ti.abs(circle_center - recenter)
    u = ti.max(v - hlength, 0.)
    return u.dot(u) < radius * radius


@ti.func
def PointProjectionToRectangle(circle_center, recenter, norm, length, width, height):
    p = circle_center - recenter
    rotation_matrix = RodriguesRotationMatrix(vec3f(0, 0, 1), norm)
    localp = rotation_matrix.transpose() @ p

    dist1 = localp.dot(vec3f(1, 0, 0))
    dist2 = localp.dot(vec3f(0, 1, 0))
    dist3 = localp.dot(vec3f(0, 0, 1))
    dist1 = clamp(-length, length, dist1)
    dist2 = clamp(-width, width, dist2)
    dist3 = clamp(-height, height, dist3)
    q = dist1 * vec3f(1, 0, 0) + dist2 * vec3f(0, 1, 0) + dist3 * vec3f(0, 0, 1)
    return rotation_matrix @ q + recenter


@ti.func
def SpheresIntersectionVolume(center1, center2, rad1, rad2):
    dR = ti.max(rad1, rad2)
    dr = ti.min(rad1, rad2)
    dDistance = (center1 - center2).norm()
    
    volume = 0.
    if dDistance + dr < dR: 
        volume = 4./3. * PI * dr * dr * dr
    else:
        db = (dr * dr - dR * dR + dDistance * dDistance) / (2. * dDistance)
        if db >= 0.:
            dh1 = dr - db
            dh2 = dR - dDistance + db
            volume = PI / 3. * dh1 * dh1 * (3 * dr - dh1) + PI / 3. * dh2 * dh2 * (3. * dR - dh2)
        else:
            dh1 = dr - ti.abs(db)
            dh2 = dR - dDistance - ti.abs(db)
            volume = 4. / 3. * PI * dr * dr * dr - PI / 3. * dh1 * dh1 * (3 * dr - dh1) + PI / 3. * dh2 * dh2 * (3 * dR - dh2)
    return volume    


@ti.func
def DistanceFromPointToSegment(point, vertice1, vertice2):
    return ((point - vertice1).cross(point - vertice2)).norm() / (vertice2 - vertice1).norm()


@ti.func
def LineLineIntersection(p1L1, p2L1, p1L2, p2L2):
    dirVecL1 = p2L1 - p1L1
    dirVecL2 = p2L2 - p1L2
    denomX = dirVecL1[1] * dirVecL2[0] - dirVecL2[1] * dirVecL1[0]
    denomY = dirVecL1[0] * dirVecL2[1] - dirVecL2[0] * dirVecL1[1]
    denomZ = dirVecL1[1] * dirVecL2[2] - dirVecL2[1] * dirVecL1[2]
    x = p1L1[0] * dirVecL1[1] * dirVecL2[0] - p1L2[0] * dirVecL2[1] * dirVecL1[0] - p1L1[1] * dirVecL1[0] * dirVecL2[0] + p1L2[1] * dirVecL2[0] * dirVecL1[0]
    y = p1L1[1] * dirVecL1[0] * dirVecL2[1] - p1L2[1] * dirVecL2[0] * dirVecL1[1] - p1L1[0] * dirVecL1[1] * dirVecL2[1] + p1L2[0] * dirVecL2[1] * dirVecL1[1]
    z = p1L1[2] * dirVecL1[1] * dirVecL2[2] - p1L2[2] * dirVecL2[1] * dirVecL1[2] - p1L1[1] * dirVecL1[2] * dirVecL2[2] + p1L2[1] * dirVecL2[2] * dirVecL1[2]
    
    if denomX == 0:
        if x != 0:
            x = 0/0
    else: 
        x /= denomX
    if denomY == 0:
        if y != 0:    
            y = 0/0
    else: 
        y /= denomY
    if denomZ == 0:
        if z != 0:
            z = 0/0
    else:
        z /= denomZ
    return vec3f(x, y, z)


# Refers to https://iquilezles.org/articles/distfunctions/
@ti.func
def DistanceFromPointToTriangle(point, vertice1, vertice2, vertice3, normal):
    ba = vertice2 - vertice1
    cb = vertice3 - vertice2
    ac = vertice1 - vertice3
    pa = point - vertice1
    pb = point - vertice2
    pc = point - vertice3
    norm = ba.cross(ac)
    
    A = sgn((ba.cross(norm).dot(pa))) + sgn((cb.cross(norm).dot(pb))) + sgn((ac.cross(norm).dot(pc)))
    dist = ti.sqrt(ti.min(dot2(ba * clamp(0., 1., ba.dot(pa) / dot2(ba)) - pa), dot2(cb * clamp(0., 1., cb.dot(pb) / dot2(cb)) - pb), dot2(ac * clamp(0., 1., ac.dot(pc) / dot2(ac)) - pc))) \
           if A < 2. else ti.sqrt(norm.dot(pa) * norm.dot(pa) / dot2(norm))
    return -sgn(pa.dot(normal)) * dist


@ti.func
def DistanceFromPointToRectangle(point, vertice1, vertice2, vertice3, vertice4, norm):
    ba = vertice2 - vertice1
    cb = vertice3 - vertice2
    dc = vertice4 - vertice3
    ad = vertice1 - vertice4
    pa = point - vertice1
    pb = point - vertice2
    pc = point - vertice3
    pd = point - vertice4
    # norm = ba.cross(ad)

    A = sgn((ba.cross(norm).dot(pa))) + sgn((cb.cross(norm).dot(pb))) + sgn((dc.cross(norm).dot(pc))) + sgn((ad.cross(norm).dot(pd)))
    dist = ti.sqrt(ti.min(dot2(ba * clamp(0., 1., ba.dot(pa) / dot2(ba)) - pa), dot2(cb * clamp(0., 1., cb.dot(pb) / dot2(cb)) - pb), 
                          dot2(dc * clamp(0., 1., dc.dot(pc) / dot2(dc)) - pc), dot2(ad * clamp(0., 1., ad.dot(pd) / dot2(ad)) - pd))) \
           if A < 3. else ti.sqrt(norm.dot(pa) * norm.dot(pa) / dot2(norm))
    return sgn(pa.dot(-norm)) * dist


# Refers to Philip J. Schneider, and David H. Eberly. Geometric Tools for Computer Graphics. Morgan Kaufmann, 2002
@ti.func
def DistanceFromPointToTriangle2(point, vertice1, vertice2, vertice3):
    vEdge0 = vertice2 - vertice1
    vEdge1 = vertice3 - vertice1
    vD = vertice1 - point

    a = vEdge0.dot(vEdge0)
    b = vEdge0.dot(vEdge1)
    c = vEdge1.dot(vEdge1)
    d = vEdge0.dot(vD)
    e = vEdge1.dot(vD)
    f = dot2(vD)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    if s + t < det:
        if s < 0.:
            if t < 0.:
                if d < 0.:
                    s = clamp(0., 1., -d / a)
                    t = 0.
                else:
                    s = 0.
                    t = clamp(0., 1., -e / c)
            else:
                s = 0.
                t = clamp(0., 1., -e / c)
        elif t < 0.:
            s = clamp(0., 1., -d / a)
            t = 0.
        else:
            invDet = 1. / det
            s *= invDet
            t *= invDet
    else:
        if s < 0.:
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2. * b + c
                s = clamp(0., 1., numer / denom)
                t = 1 - s
            else:
                t = clamp(0., 1., -e / c)
                s = 0
        elif t < 0.:
            if a+d > b+e:
                numer = c + e - b - d
                denom = a - 2. * b + c
                s = clamp(0., 1., numer / denom)
                t = 1 - s
            else:
                s = clamp(0., 1., -e / c)
                t = 0
        else:
            numer = c + e - b - d
            denom = a - 2. * b + c
            s = clamp(0., 1., numer / denom)
            t = 1.0 - s
    return ti.sqrt(a * s * s + 2 * b * s * t + c * t * t + 2 * d * s + 2 * e * t + f)


@ti.func
def PointProjectionDistance(position, center, norm):
    return (position - center).dot(norm)


@ti.func
def ProjectionPointInPlane(position, norm, distance):
    return position - norm * distance


@ti.func
def IsTriangleAntiClockwise(vertice1, vertice2, vertice3, norm):
    return (vertice2 - vertice1).cross(vertice3 - vertice1).dot(norm) > 0.


@ti.func
def IsPointInTriangle(projection_point, vertice1, vertice2, vertice3):
    edge21 = vertice2 - vertice1
    edge31 = vertice3 - vertice1
    W = projection_point - vertice1

    d00 = edge21.dot(edge21)
    d01 = edge21.dot(edge31)
    d11 = edge31.dot(edge31)
    d20 = W.dot(edge21)
    d21 = W.dot(edge31)
    invDenom = 1.0 / (d00 * d11 - d01 * d01)
    gamma = (d11 * d20 - d01 * d21) * invDenom
    betta = (d00 * d21 - d01 * d20) * invDenom
    alpha = 1. - gamma - betta
    return (gamma > 0 and gamma < 1) and (alpha > 0 and alpha < 1) and (betta > 0 and betta < 1)


# GJK contact algorithm
@ti.func
def IsPointInTriangle2(projection_point, vertice1, vertice2, vertice3):
    u = (vertice1 - projection_point).cross(vertice2 - projection_point)
    v = (vertice2 - projection_point).cross(vertice3 - projection_point)
    w = (vertice3 - projection_point).cross(vertice1 - projection_point)
    return u.dot(v) >= 0. and u.dot(w) >= 0.


@ti.func
def IsPointInTetra(point, vertice1, vertice2, vertice3, vertice4):
    pass


@ti.func
def RayTriangleInteraction(origin, direction, vertice1, vertice2, vertice3):
    t, u, v = 0., 0., 0.
    # Reference: T. Möller, B. Trumbore (1997) Fast, minimum storage ray-triangle intersection. J. Graph. Tools, 2, 21-28.
    edge1 = vertice2 - vertice1
    edge2 = vertice3 - vertice1
    pvec = direction.cross(edge2)
    
    det = edge1.dot(pvec)
    if det > Threshold:
        tvec = origin - vertice1
        u = tvec.dot(pvec)
        if 0. <=u <= det:
            qvec = tvec.cross(edge1)
            v = direction.dot(qvec)
            if 0. <=v <= det:
                t = edge2.dot(qvec)
                inv_det = 1. / det
                t *= inv_det
                u *= inv_det
                v *= inv_det
            else:
                u, v,  = 0., 0.
        else:
            u = 0.
    return t, u, v


@ti.func
def IsSphereIntersectTriangle(bound_beg, bound_end, vertice1, vertice2, vertice3, norm, position, radius):
    status = 0
    point = vec3f(0.0, 0.0, 0.0)

    if position[0] <= bound_beg[0] - radius and position[1] <= bound_beg[1] - radius and position[2] <= bound_beg[2] - radius and \
       position[0] >= bound_end[0] + radius and position[1] >= bound_end[1] + radius and position[2] >= bound_end[2] + radius:
        center = (vertice1 + vertice2 + vertice3) / 3.0
        ppd = PointProjectionDistance(position, center, norm)
        if ti.abs(ppd) < radius:
            A = ProjectionPointInPlane(position, norm, ppd)
            if IsPointInTriangle(A, vertice1, vertice2, vertice3):
                status = 1
                point = A
            else:
                edge21 = vertice2 - vertice1
                edge32 = vertice3 - vertice2
                edge13 = vertice1 - vertice3
                lc1 = clamp(0., 1., (A - vertice1).dot(edge21) / dot2(edge21))
                lc2 = clamp(0., 1., (A - vertice2).dot(edge32) / dot2(edge32))
                lc3 = clamp(0., 1., (A - vertice3).dot(edge13) / dot2(edge13))
                C1IsVertice = lc1 == 0 or lc1 == 1
                C2IsVertice = lc2 == 0 or lc2 == 1
                C3IsVertice = lc3 == 0 or lc3 == 1
                C1 = vertice1 + lc1 * edge21
                C2 = vertice2 + lc2 * edge32
                C3 = vertice3 + lc3 * edge13
                sqrLength1 = dot2(C1 - position)
                sqrLength2 = dot2(C2 - position)
                sqrLength3 = dot2(C3 - position)
                if ti.min({ sqrLength1, sqrLength2, sqrLength3 }) < radius * radius:
                    if sqrLength1 <= sqrLength2 and sqrLength1 <= sqrLength3:
                        status = 3 if C1IsVertice else 2
                        point = C1
                    elif (sqrLength2 <= sqrLength3):
                        status = 3 if C2IsVertice else 2
                        point = C2
                    else:
                        status = 3 if C3IsVertice else 2
                        point = C3
    return status, point


@ti.func
def SpheresContactPoint(center1, center2, radius1, radius2):
    return center1 + (center2 - center1) * radius1 / (radius1 + radius2)


@ti.func
def GetBoundingBox(vertice1, vertice2, vertice3):
    coordBeg = vec3f(0, 0, 0)
    coordEnd = vec3f(0, 0, 0)

    coordBeg = ti.min(coordBeg, vertice1)
    coordBeg = ti.min(coordBeg, vertice2)
    coordBeg = ti.min(coordBeg, vertice3)

    coordEnd = ti.max(coordEnd, vertice1)
    coordEnd = ti.max(coordEnd, vertice2)
    coordEnd = ti.max(coordEnd, vertice3)
    return coordBeg, coordEnd


@ti.func
def intersectionOBBs(center1, center2, extent1, extent2, rotate_matrix1, rotate_matrix2):
    cmatrix = rotate_matrix1.transpose() @ rotate_matrix2
    amatrix = ti.abs(cmatrix)
    dvector = center2 - center1

    a0, a1, a2 = 0.5 * extent1
    b0, b1, b2 = 0.5 * extent2
    A0, A1, A2 = rotate_matrix1 @ vec3f(1, 0, 0), rotate_matrix1 @ vec3f(0, 1, 0), rotate_matrix1 @ vec3f(0, 0, 1)
    B0, B1, B2 = rotate_matrix2 @ vec3f(1, 0, 0), rotate_matrix2 @ vec3f(0, 1, 0), rotate_matrix2 @ vec3f(0, 0, 1)
    not_intersect = True
    if not_intersect and ti.abs(A0.dot(dvector)) < a0 + (b0 * amatrix[0, 0] + b1 * amatrix[0, 1] + b2 * amatrix[0, 2]):
        not_intersect = False
    if not_intersect and ti.abs(A1.dot(dvector)) < a1 + (b0 * amatrix[1, 0] + b1 * amatrix[1, 1] + b2 * amatrix[1, 2]):
        not_intersect = False
    if not_intersect and ti.abs(A2.dot(dvector)) < a2 + (b0 * amatrix[2, 0] + b1 * amatrix[2, 1] + b2 * amatrix[2, 2]):
        not_intersect = False
    if not_intersect and ti.abs(B0.dot(dvector)) < b0 + (a0 * amatrix[0, 0] + a1 * amatrix[1, 0] + a2 * amatrix[2, 0]):
        not_intersect = False
    if not_intersect and ti.abs(B1.dot(dvector)) < b1 + (a0 * amatrix[0, 1] + a1 * amatrix[1, 1] + a2 * amatrix[2, 1]):
        not_intersect = False
    if not_intersect and ti.abs(B2.dot(dvector)) < b2 + (a0 * amatrix[0, 2] + a1 * amatrix[1, 2] + a2 * amatrix[2, 2]):
        not_intersect = False
    if not_intersect and ti.abs(cmatrix[1, 0] * A2.dot(dvector) - cmatrix[2, 0] * A1.dot(dvector)) < (a1 * amatrix[2, 0] + a2 * amatrix[1, 0]) + (b1 * amatrix[0, 2] + b2 * amatrix[0, 1]):
        not_intersect = False
    if not_intersect and ti.abs(cmatrix[1, 1] * A2.dot(dvector) - cmatrix[2, 1] * A1.dot(dvector)) < (a1 * amatrix[2, 1] + a2 * amatrix[1, 1]) + (b0 * amatrix[0, 2] + b2 * amatrix[0, 0]):
        not_intersect = False
    if not_intersect and ti.abs(cmatrix[1, 2] * A2.dot(dvector) - cmatrix[2, 2] * A1.dot(dvector)) < (a1 * amatrix[2, 2] + a2 * amatrix[1, 2]) + (b0 * amatrix[0, 1] + b1 * amatrix[0, 0]):
        not_intersect = False
    if not_intersect and ti.abs(cmatrix[2, 0] * A0.dot(dvector) - cmatrix[0, 0] * A2.dot(dvector)) < (a0 * amatrix[2, 0] + a2 * amatrix[0, 0]) + (b1 * amatrix[1, 2] + b2 * amatrix[1, 1]):
        not_intersect = False
    if not_intersect and ti.abs(cmatrix[2, 1] * A0.dot(dvector) - cmatrix[0, 1] * A2.dot(dvector)) < (a0 * amatrix[2, 1] + a2 * amatrix[0, 1]) + (b0 * amatrix[1, 2] + b2 * amatrix[1, 0]):
        not_intersect = False
    if not_intersect and ti.abs(cmatrix[2, 2] * A0.dot(dvector) - cmatrix[0, 2] * A2.dot(dvector)) < (a0 * amatrix[2, 2] + a2 * amatrix[0, 2]) + (b0 * amatrix[1, 1] + b1 * amatrix[1, 0]):
        not_intersect = False
    if not_intersect and ti.abs(cmatrix[0, 0] * A1.dot(dvector) - cmatrix[1, 0] * A0.dot(dvector)) < (a0 * amatrix[1, 0] + a1 * amatrix[0, 0]) + (b1 * amatrix[2, 2] + b2 * amatrix[2, 1]):
        not_intersect = False
    if not_intersect and ti.abs(cmatrix[0, 1] * A1.dot(dvector) - cmatrix[1, 1] * A0.dot(dvector)) < (a0 * amatrix[1, 1] + a1 * amatrix[0, 1]) + (b0 * amatrix[2, 2] + b2 * amatrix[2, 0]):
        not_intersect = False
    if not_intersect and ti.abs(cmatrix[0, 2] * A1.dot(dvector) - cmatrix[1, 2] * A0.dot(dvector)) < (a0 * amatrix[1, 2] + a1 * amatrix[0, 2]) + (b0 * amatrix[2, 1] + b1 * amatrix[2, 0]):
        not_intersect = False
    return not not_intersect


@ti.func
def intersection_triangleOBBs(mass_center, extent, rotate_matrix, vertice1, vertice2, vertice3, norm):
    wall_center = 1./3. * (vertice1 + vertice2 + vertice3)
    vertice1 = rotate_matrix @ (vertice1 - wall_center) + wall_center
    vertice2 = rotate_matrix @ (vertice2 - wall_center) + wall_center
    vertice3 = rotate_matrix @ (vertice3 - wall_center) + wall_center
    edge1, edge2, edge3 = vertice2 - vertice1, vertice3 - vertice1, vertice3 - vertice2
    norm = rotate_matrix @ norm
    dvector = vertice1 - mass_center

    a0, a1, a2 = 0.5 * extent
    A0, A1, A2 = rotate_matrix @ vec3f(1, 0, 0), rotate_matrix @ vec3f(0, 1, 0), rotate_matrix @ vec3f(0, 0, 1)
    not_intersect = True
    if not_intersect:
        R = a0 * ti.abs(norm.dot(A0)) + a1 * ti.abs(norm.dot(A1)) + a2 * ti.abs(norm.dot(A2))
        if -R < norm.dot(dvector) < R: not_intersect = False
    if not_intersect:
        p0 = A0.dot(dvector)
        if -a0 < ti.max(p0, p0 + A0.dot(edge1), p0 + A0.dot(edge2)) < a0: not_intersect = False
    if not_intersect:
        p0 = A1.dot(dvector)
        if -a1 < ti.max(p0, p0 + A1.dot(edge1), p0 + A1.dot(edge2)) < a1: not_intersect = False
    if not_intersect:
        p0 = A2.dot(dvector)
        if -a2 < ti.max(p0, p0 + A2.dot(edge1), p0 + A2.dot(edge2)) < a2: not_intersect = False
    if not_intersect:
        p0 = A0.cross(edge1).dot(dvector)
        R = a1 * ti.abs(A2.dot(edge1)) + a2 * ti.abs(A1.dot(edge1))
        if -R < ti.max(p0, p0 + A0.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A0.cross(edge2).dot(dvector)
        R = a1 * ti.abs(A2.dot(edge2)) + a2 * ti.abs(A1.dot(edge2))
        if -R < ti.max(p0, p0 - A0.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A0.cross(edge3).dot(dvector)
        R = a1 * ti.abs(A2.dot(edge3)) + a2 * ti.abs(A1.dot(edge3))
        if -R < ti.max(p0, p0 + A0.dot(norm), p0 - A0.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A1.cross(edge1).dot(dvector)
        R = a0 * ti.abs(A2.dot(edge1)) + a2 * ti.abs(A0.dot(edge1))
        if -R < ti.max(p0, p0 + A1.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A1.cross(edge2).dot(dvector)
        R = a0 * ti.abs(A2.dot(edge2)) + a2 * ti.abs(A0.dot(edge2))
        if -R < ti.max(p0, p0 - A1.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A1.cross(edge3).dot(dvector)
        R = a0 * ti.abs(A2.dot(edge3)) + a2 * ti.abs(A0.dot(edge3))
        if -R < ti.max(p0, p0 + A1.dot(norm), p0 - A1.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A2.cross(edge1).dot(dvector)
        R = a0 * ti.abs(A1.dot(edge1)) + a1 * ti.abs(A0.dot(edge1))
        if -R < ti.max(p0, p0 + A2.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A2.cross(edge2).dot(dvector)
        R = a0 * ti.abs(A1.dot(edge2)) + a1 * ti.abs(A0.dot(edge2))
        if -R < ti.max(p0, p0 - A2.dot(norm)) < R:
            not_intersect = False
    if not_intersect:
        p0 = A2.cross(edge3).dot(dvector)
        R = a0 * ti.abs(A1.dot(edge3)) + a1 * ti.abs(A0.dot(edge3))
        if -R < ti.max(p0, p0 + A2.dot(norm), p0 - A2.dot(norm)) < R:
            not_intersect = False
    return not not_intersect

@ti.func
def dist3D_Segment_to_Segment(A0,A1,B0,B1):
    u = A1 - A0
    v = B1 - B0
    w = A0 - B0
    a = u.norm_sqr()
    b = u.dot(v)
    c = v.norm_sqr()
    d = u.dot(w)
    e = v.dot(w)
    D = a * c - b * b
    sc, sN, sD = D, D, D
    tc, tN, tD = D, D, D
    if D < 1e-7:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = b * e - c * d
        tN = a * e - b * d
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c
    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if -d + b < 0.0:
            sN = 0.0
        elif -d + b > a:
            sN = sD
        else:
            sN = -d + b
            sD = a

    if ti.abs(sN) < 1e-7:
        sc = 0.0
    else:
        sc = sN / sD
    if ti.abs(tN) < 1e-7:
        tc = 0.0
    else:
        tc = tN / tD
    dP = - w - (sc * u) + (tc * v) # Qc - Pc
    return dP, sc, tc

@ti.func
def dist3D_Point_Triangle(P, V0, V1, V2):
    cord0 = 0.0
    cord1 = 0.0
    cord2 = 0.0
    v = V2 - V0
    u = V1 - V0
    nVec = u.cross(v)
    s_p = (nVec.dot(P - V0)) / (nVec.dot(nVec))
    P0 = P - s_p * nVec # P project to plane
    w = P0 - V0
    n_cross_v = nVec.cross(v)
    n_cross_u = nVec.cross(u)
    s = w.dot(n_cross_v) / (u.dot(n_cross_v))
    t = w.dot(n_cross_u) / (v.dot(n_cross_u))
    if s >= 0.0 and t >= 0.0:
        if s + t <= 1.0:
            cord0 = 1.0 - s - t
            cord1 = s
            cord2 = t
        else:
            q = V2 - V1
            k = (P - V1).dot(q) / (q.dot(q))
            if k > 1.0:
                cord2 = 1.0
            elif k < 0.0:
                cord1 = 1.0
            else:
                cord1 = 1.0 - k
                cord2 = k
    elif s >= 0.0 and t < 0.0:
        k = w.dot(u) / (u.dot(u))
        if k > 1.0:
            cord1 = 1.0
        elif k < 0.0:
            cord0 = 1.0
        else:
            cord0 = 1.0 - k
            cord1 = k
    elif s < 0.0 and t >= 0.0:
        k = w.dot(v) / (v.dot(v))
        if k > 1.0:
            cord2 = 1.0
        elif k < 0.0:
            cord0 = 1.0
        else:
            cord0 = 1.0 - k
            cord2 = k
    else: # s < 0 and t < 0
        cord0 = 1.0
    return cord0, cord1, cord2

